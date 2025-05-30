"""
空调功率优化线性规划问题求解（支持JSON配置）

M_{AC}: \min_{\left \{P_{1},...,P_{T},T_{2}^{i},...,T_{T}^{i}\right \} }\sum_{t=1}^{T} price_{t} P_{t}\Delta t,
subject to for \forall t \in \{1,2,..,T\}:
0 \le P_{t} \le P_{rated},
T_{min} \le T_{t} \le T_{max}

and

室温变化公式：T_{t+1}^{i} = T_{t+1}^{out} - \eta P_{t} R_{t} - (T_{t+1}^{out} - \eta P_{t} R_{t} - T_{t}^{i}) e^{- \Delta t / R C}
"""

import pulp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict
import csv
import pandas as pd
import json
import random
import os
from datetime import datetime

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class ACOptimizer:
    def __init__(self, T=24, delta_t=1.0, P_rated=3.0, T_min=20.0, T_max=26.0,
                 T_max_price_sensitivity_factor=0.05,  # 新增：电价对T_max的敏感度因子
                 eta=0.8, R=2.0, C=5.0, T_initial=22.0):
        """
        初始化空调优化器
        
        参数:
        T: 时间步数 (小时)
        delta_t: 时间步长 (小时)
        P_rated: 额定功率 (kW)
        T_min: 最低温度约束 (°C)
        T_max: 基准最高温度约束 (°C)
        T_max_price_sensitivity_factor: 电价对最高温度影响的敏感度因子
        eta: 空调效率
        R: 热阻 (°C/kW)
        C: 热容 (J/°C)，将自动转换为kWh/°C
        T_initial: 初始室温 (°C)
        """
        self.T = T
        self.delta_t = delta_t
        self.P_rated = P_rated
        self.T_min = T_min
        self.T_max_base = T_max  # 修改：基准最高温度
        self.T_max_price_sensitivity_factor = T_max_price_sensitivity_factor # 新增
        self.eta = eta
        self.R = R
        
        # 单位转换：J/°C → kWh/°C
        # 1 kWh = 3.6e6 J, 所以 C_kWh = C_J / 3.6e6
        self.C = C / 3.6e6  # 转换为 kWh/°C
        self.C_original = C  # 保存原始值用于显示
        
        self.T_initial = T_initial
        
        # 计算指数衰减因子: exp(-Δt/(R*C))
        # 这里 delta_t 是小时，R 是°C/kW，C 是 kWh/°C
        # 所以 R*C 的单位是 (°C/kW) * (kWh/°C) = h
        self.exp_factor = np.exp(-delta_t / (R * self.C))
        
        print(f"热容转换: {C:.1e} J/°C = {self.C:.1e} kWh/°C")
        print(f"时间常数 τ = R*C = {R:.1f} * {self.C:.1e} = {R * self.C:.2f} 小时")
        print(f"指数衰减因子: exp(-Δt/τ) = {self.exp_factor:.6f}")
        
    def set_outdoor_temperature(self, T_out):
        """设置室外温度序列"""
        if isinstance(T_out, (int, float)):
            self.T_out = [T_out] * (self.T + 1)
        else:
            self.T_out = T_out
            
    def set_prices(self, prices):
        """设置电价序列"""
        if isinstance(prices, (int, float)):
            self.prices = [prices] * self.T
        else:
            self.prices = prices
            
    def solve(self):
        """求解线性规划问题"""
        # 检查是否设置了电价，如果没有则使用默认价格
        if not hasattr(self, 'prices'):
            self.prices = [1.0] * self.T  # 默认电价为1.0
            print("警告：未设置电价，使用默认电价 1.0")
        
        # 每次求解时创建新的LpProblem实例，避免状态保留问题
        prob = pulp.LpProblem("AC_Power_Optimization", pulp.LpMinimize)
        
        # 决策变量
        # P_t: 每个时间步的功率
        P = [pulp.LpVariable(f"P_{t}", lowBound=0, upBound=self.P_rated) 
             for t in range(1, self.T + 1)]
        
        # T_i_t: 每个时间步的室内温度
        # 修改：为每个时间步计算动态T_max并定义T_i
        T_i = []
        self.dynamic_T_max_values = [] # 用于存储每个时间步的T_max，以便绘图
        for t_idx in range(self.T): # t_idx from 0 to T-1, 对应T_i_{t_idx+2}
            price_for_step = self.prices[t_idx] # 电价对应P[t_idx]发生的时段
            
            # 计算当前时间步的动态T_max
            # T_max_t = T_max_base * exp(-sensitivity_factor * price)
            # price > 0 (高电价) => exp为负指数 => T_max降低
            # price < 0 (低电价/负电价) => exp为正指数 => T_max升高
            current_T_max = self.T_max_base * np.exp(-self.T_max_price_sensitivity_factor * price_for_step)
            
            # 确保 T_max 不低于 T_min (例如，至少比T_min高0.1°C)
            current_T_max = max(current_T_max, self.T_min + 0.1)
            
            self.dynamic_T_max_values.append(current_T_max)
            T_i.append(pulp.LpVariable(f"T_i_{t_idx+2}", lowBound=self.T_min, upBound=current_T_max))
        
        # 目标函数：最小化总电费成本（包含电价）
        prob += pulp.lpSum([self.prices[t-1] * P[t-1] * self.delta_t for t in range(1, self.T + 1)])
        
        # 约束条件
        # 1. 功率约束（已在变量定义中包含）
        # 2. 温度约束（已在变量定义中包含）
        
        # 3. 室温变化约束（一阶ETP公式）
        for t in range(1, self.T + 1):
            if t == 1:
                # 第一个时间步，使用初始温度
                T_prev = self.T_initial
            else:
                T_prev = T_i[t-2]  # T_i数组从t=2开始，所以t-2对应前一时刻
            
            # 一阶ETP公式的线性化
            # T_{t+1}^{i} = T_{t+1}^{out} - η P_t R - (T_{t+1}^{out} - η P_t R - T_t^{i}) * exp(-Δt/RC)
            # 重新整理为：T_{t+1}^{i} = (1-exp_factor) * (T_{t+1}^{out} - η P_t R) + exp_factor * T_t^{i}
            
            steady_state_temp = self.T_out[t] - self.eta * self.R * P[t-1]
            
            prob += (T_i[t-1] == 
                    (1 - self.exp_factor) * steady_state_temp + 
                    self.exp_factor * T_prev)
        
        # 求解
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 提取结果
        if prob.status == pulp.LpStatusOptimal:
            self.optimal_powers = [P_var.varValue for P_var in P]
            self.optimal_temperatures = [self.T_initial] + [T_var.varValue for T_var in T_i]
            self.total_energy = sum(self.optimal_powers) * self.delta_t
            # 计算总成本（考虑电价）
            self.total_cost = sum([self.prices[t] * self.optimal_powers[t] * self.delta_t for t in range(self.T)])
            self.status = "最优解"
        else:
            self.optimal_powers = [0.0] * self.T # 如果求解失败，将功率设为0
            self.optimal_temperatures = [self.T_initial] * (self.T + 1)
            self.total_energy = 0.0
            self.total_cost = 0.0
            self.status = f"求解失败: {pulp.LpStatus[prob.status]}"
            
        return prob.status == pulp.LpStatusOptimal
    
    def plot_results(self):
        """绘制结果图表"""
        if not hasattr(self, 'optimal_powers'):
            print("请先求解问题")
            return
            
        # 第一个画布：原有的4个时序图
        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        time_steps = list(range(self.T + 1))
        power_time_steps = list(range(1, self.T + 1))
        
        # 绘制功率
        ax1.step(power_time_steps, self.optimal_powers, where='post', linewidth=2, color='blue')
        ax1.set_ylabel('功率 (kW)')
        ax1.set_title('最优空调功率')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, self.P_rated * 1.1)
        
        # 绘制室内温度
        ax2.plot(time_steps, self.optimal_temperatures, 'ro-', linewidth=2, markersize=4, label='室内温度')
        ax2.axhline(y=self.T_min, color='g', linestyle='--', alpha=0.7, label=f'最低温度 {self.T_min}°C')
        # 修改：绘制动态最高温度线
        if hasattr(self, 'dynamic_T_max_values') and len(self.dynamic_T_max_values) == self.T:
            # dynamic_T_max_values 对应 T_optimal_temperatures[1:] 的上限
            # dynamic_T_max_values 的长度为 T，对应 time_steps[1:] (即 1 到 T)
            ax2.plot(time_steps[1:], self.dynamic_T_max_values, color='magenta', linestyle='--', linewidth=1.5, alpha=0.7, label='动态最高温度')
        else:
            # Fallback to base T_max if dynamic values are not available
            ax2.axhline(y=self.T_max_base, color='r', linestyle='--', alpha=0.7, label=f'基准最高温度 {self.T_max_base}°C')
        ax2.set_ylabel('温度 (°C)')
        ax2.set_title('室内温度变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 绘制室外温度
        ax3.plot(time_steps, self.T_out[:self.T+1], 'go-', linewidth=2, markersize=4, label='室外温度')
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('温度 (°C)')
        ax3.set_title('室外温度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 绘制电价
        ax4.step(power_time_steps, self.prices, where='post', linewidth=2, color='orange', alpha=0.7)
        ax4.set_xlabel('时间 (小时)')
        ax4.set_ylabel('电价 (元/kWh)')
        ax4.set_title('电价变化')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def print_results(self):
        """打印结果"""
        if not hasattr(self, 'optimal_powers'):
            print("请先求解问题")
            return
            
        print(f"求解状态: {self.status}")
        if hasattr(self, 'total_energy'):
            print(f"总能耗: {self.total_energy:.2f} kWh")
            print(f"总成本: {self.total_cost:.2f} 元")
            print(f"平均功率: {self.total_energy/self.T:.2f} kW")
            print(f"平均电价: {sum(self.prices)/len(self.prices):.3f} 元/kWh")
            print("\n时间步 | 功率(kW) | 室内温度(°C) | 室外温度(°C) | 电价(元/kWh) | 时段成本(元) | 动态Tmax(°C)")
            print("-" * 100)
            for t in range(self.T):
                cost_t = self.prices[t] * self.optimal_powers[t] * self.delta_t
                dynamic_t_max_val = self.dynamic_T_max_values[t] if hasattr(self, 'dynamic_T_max_values') and t < len(self.dynamic_T_max_values) else self.T_max_base
                print(f"{t+1:6d} | {self.optimal_powers[t]:8.2f} | {self.optimal_temperatures[t+1]:11.2f} | {self.T_out[t+1]:11.2f} | {self.prices[t]:10.3f} | {cost_t:9.3f} | {dynamic_t_max_val:12.2f}")

def load_ac_data(json_file="ac_data.json"):
    """
    从JSON文件加载空调参数数据
    
    参数:
    json_file: JSON文件路径
    
    返回:
    ac_configs: 空调配置列表
    """
    # 检查文件是否存在
    if not os.path.exists(json_file):
        # 尝试在src/mmoe_generate_data目录下寻找
        alt_path = os.path.join("src", "mmoe_generate_data", json_file)
        if os.path.exists(alt_path):
            json_file = alt_path
        else:
            print(f"错误: 无法找到JSON文件 {json_file}")
            return []
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            ac_configs = json.load(f)
        print(f"成功加载 {len(ac_configs)} 个空调配置从 {json_file}")
        return ac_configs
    except Exception as e:
        print(f"加载JSON文件时出错: {e}")
        return []

def select_ac_config(ac_configs, ac_id=None, ac_type=None, random_select=False):
    """
    从空调配置列表中选择一个配置
    
    参数:
    ac_configs: 空调配置列表
    ac_id: 指定的空调ID (如: "AC_S_001")
    ac_type: 指定的空调类型 (如: "small", "medium", "large")
    random_select: 是否随机选择
    
    返回:
    selected_config: 选中的空调配置字典
    """
    if not ac_configs:
        print("空调配置列表为空")
        return None
    
    # 如果指定了ID，直接查找
    if ac_id:
        for config in ac_configs:
            if config.get('id') == ac_id:
                print(f"选择了空调: {config['id']} (类型: {config['type']})")
                return config
        print(f"未找到ID为 {ac_id} 的空调配置")
        return None
    
    # 如果指定了类型，从该类型中选择
    if ac_type:
        type_configs = [config for config in ac_configs if config.get('type') == ac_type]
        if not type_configs:
            print(f"未找到类型为 {ac_type} 的空调配置")
            return None
        
        if random_select:
            selected = random.choice(type_configs)
        else:
            selected = type_configs[0]  # 选择第一个
        
        print(f"选择了空调: {selected['id']} (类型: {selected['type']})")
        return selected
    
    # 如果都没指定，根据random_select决定
    if random_select:
        selected = random.choice(ac_configs)
        print(f"随机选择了空调: {selected['id']} (类型: {selected['type']})")
        return selected
    else:
        selected = ac_configs[0]  # 选择第一个
        print(f"选择了空调: {selected['id']} (类型: {selected['type']})")
        return selected

def create_optimizer_from_config(ac_config, T=24, delta_t=1.0, T_max_price_sensitivity_factor=0.05, T_initial=23.0):
    """
    根据空调配置创建ACOptimizer实例
    
    参数:
    ac_config: 空调配置字典
    T: 时间步数
    delta_t: 时间步长
    T_max_price_sensitivity_factor: 电价敏感度因子
    T_initial: 初始温度
    
    返回:
    optimizer: ACOptimizer实例
    """
    if not ac_config:
        print("空调配置为空，使用默认参数")
        return ACOptimizer()
    
    print(f"\n使用空调配置:")
    print(f"  ID: {ac_config.get('id', 'N/A')}")
    print(f"  类型: {ac_config.get('type', 'N/A')}")
    print(f"  额定功率: {ac_config.get('P_rated', 3.0)} kW")
    print(f"  热阻: {ac_config.get('R', 2.0)} °C/kW")
    print(f"  热容: {ac_config.get('C', 1e7):.1e} J/°C")
    print(f"  温度范围: {ac_config.get('T_min', 20.0)} - {ac_config.get('T_max', 26.0)} °C")
    print(f"  效率: {ac_config.get('eta', 0.98)}")
    
    optimizer = ACOptimizer(
        T=T,
        delta_t=delta_t,
        P_rated=ac_config.get('P_rated', 3.0),
        T_min=ac_config.get('T_min', 20.0),
        T_max=ac_config.get('T_max', 26.0),
        T_max_price_sensitivity_factor=T_max_price_sensitivity_factor,
        eta=ac_config.get('eta', 0.98),  # 从JSON中的eta字段
        R=ac_config.get('R', 2.0),
        C=ac_config.get('C', 1e7),
        T_initial=T_initial
    )
    
    return optimizer

def load_summer_temperature_data(csv_file="data/W2.csv", random_day=True):
    """
    从W2.csv文件中加载夏季温度数据（7月1日到9月30日）
    
    参数:
    csv_file: CSV文件路径
    random_day: 是否随机选择一天的数据，否则使用第一天
    
    返回:
    hourly_temps_celsius: 24小时的温度数据（摄氏度），对应T=0到T=24
    selected_date: 所选择的日期字符串
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 将Time列转换为datetime类型
        df['Time'] = pd.to_datetime(df['Time'])
        
        # 筛选夏季数据（7月1日到9月30日）
        summer_data = df[
            (df['Time'].dt.month >= 7) & 
            (df['Time'].dt.month <= 9) &
            (df['Time'].dt.day <= 30)  # 确保不会超出9月30日
        ].copy()
        
        if summer_data.empty:
            print("警告: 未找到夏季数据，使用默认温度")
            return None, None
        
        # 将华氏度转换为摄氏度: C = (F - 32) * 5/9
        summer_data['Temperature(C)'] = (summer_data['Temperature(F)'] - 32) * 5 / 9
        
        # 获取所有可用的日期
        available_dates = summer_data['Time'].dt.date.unique()
        
        if random_day:
            selected_date = random.choice(available_dates)
        else:
            selected_date = available_dates[0]
        
        print(f"选择的日期: {selected_date}")
        
        # 筛选选中日期的数据
        day_data = summer_data[summer_data['Time'].dt.date == selected_date].copy()
        
        # 确保按时间排序
        day_data = day_data.sort_values('Time')
        
        # 由于数据是每15分钟记录一次，我们需要提取每小时的数据
        # 选择每小时的第一个记录（:00分）或最接近的记录
        day_data['Hour'] = day_data['Time'].dt.hour
        
        # 获取每小时的平均温度
        hourly_data = day_data.groupby('Hour')['Temperature(C)'].mean().reset_index()
        
        # 确保有24小时的数据
        hourly_temps = []
        for hour in range(24):
            if hour in hourly_data['Hour'].values:
                temp = hourly_data[hourly_data['Hour'] == hour]['Temperature(C)'].iloc[0]
            else:
                # 如果某小时没有数据，使用插值或相邻小时的平均值
                if hourly_temps:
                    temp = hourly_temps[-1]  # 使用前一小时的温度
                else:
                    temp = 25.0  # 默认温度
            hourly_temps.append(temp)
        
        # 添加T=24时刻的温度（通常与T=0相同或类似）
        hourly_temps.append(hourly_temps[0])
        
        print(f"成功加载温度数据: {len(hourly_temps)} 个数据点")
        print(f"温度范围: {min(hourly_temps):.1f}°C 到 {max(hourly_temps):.1f}°C")
        
        return hourly_temps, str(selected_date)
        
    except Exception as e:
        print(f"加载温度数据时出错: {e}")
        return None, None

def main():
    """主函数示例"""
    print("=" * 60)
    print("空调功率优化程序 (支持JSON配置)")
    print("=" * 60)

    # 加载空调配置数据
    ac_configs = load_ac_data("ac_data.json")
    
    if ac_configs:
        # 随机选择一个空调配置
        selected_config = select_ac_config(ac_configs, random_select=True)
        
        if selected_config:
            # 使用选中的配置创建优化器
            optimizer = create_optimizer_from_config(
                selected_config,
                T=24,
                delta_t=1.0,
                T_max_price_sensitivity_factor=0.05,
                T_initial=23.0
            )
        else:
            print("无法选择空调配置，使用默认参数")
            optimizer = ACOptimizer()
    else:
        print("无法加载空调配置，使用默认参数")
        optimizer = ACOptimizer(
            T=24,           # 24小时
            delta_t=1.0,    # 1小时时间步
            P_rated=12.0,    # 额定功率
            T_min=21.0,     # 最低温度21°C
            T_max=24.0,     # 基准最高温度24°C
            T_max_price_sensitivity_factor=0.05,
            eta=0.98,       # 效率0.98
            R=3.0,          # 热阻3°C/kW
            C=1.8e7,        # 热容J∕°C
            T_initial=23.0  # 初始温度23°C
        )
    
    # 设置室外温度 - 从W2.csv加载夏季数据
    print("\n" + "=" * 40)
    print("加载室外温度数据...")
    print("=" * 40)
    
    # 尝试从W2.csv加载夏季温度数据
    summer_temps, selected_date = load_summer_temperature_data("data/W2.csv", random_day=True)
    
    if summer_temps is not None:
        print(f"使用{selected_date}的夏季温度数据")
        optimizer.set_outdoor_temperature(summer_temps)
    else:
        # 如果加载失败，使用备用温度数据
        print("使用备用温度数据")
        original_temp = [
            28.0, 28.0, 28.0, 28.0, 28.0, 28.5, 29.0, 29.5, 
            30.0, 30.5, 31.0, 31.5, 32.0, 31.5, 31.0, 30.5, 
            29.0, 27.0, 26.0, 26.0, 27.0, 27.5, 28.0, 28.0, 28.0 
        ]
        optimizer.set_outdoor_temperature(original_temp)
    
    prices = [ # 24个电价数据
    -1.0,  # X=0 (对应第1小时)
    -0.75,  # X=1 (对应第2小时)
    -0.50,  # X=2
    -0.25,  # X=3
    -0.25,  # X=4
    0.25,  # X=5
    0.25,  # X=6
    0.25,  # X=7
    0.25,  # X=8
    0.25,  # X=9
    0.75,  # X=10
    0.75,  # X=11
    1.00,  # X=12
    1.00,  # X=13
    0.75,  # X=14
    0.75,  # X=15
    1.00,  # X=16
    1.00,  # X=17
    0.75,  # X=18
    0.75,  # X=19
    0.50,  # X=20
    0.50,  # X=21
    0.50,  # X=22
    0.50,  # X=23 (对应第24小时)
    ]
    
    optimizer.set_prices(prices)
    
    # 求解问题
    print("\n" + "=" * 40)
    print("开始求解优化问题...")
    print("=" * 40)
    
    if optimizer.solve():
        optimizer.print_results()
        optimizer.plot_results()
    else:
        print(f"求解失败: {optimizer.status}")

if __name__ == "__main__":
    main()