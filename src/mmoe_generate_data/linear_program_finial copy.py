"""
空调功率优化线性规划问题求解

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

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

def load_ac_data(json_file="D:/experiments/ACL_agg_exp/src/mmoe_generate_data/ac_data.json"):
    """
    从JSON文件中加载空调配置数据
    
    参数:
    json_file: JSON文件路径
    
    返回:
    ac_configs: 空调配置列表
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(json_file):
            print(f"警告：找不到空调配置文件 {json_file}")
            return None
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理不同的JSON结构
        ac_configs = []
        
        if isinstance(data, list):
            # 如果JSON文件直接是一个列表
            ac_configs = data
            print(f"检测到JSON文件为列表结构")
        elif isinstance(data, dict):
            # 如果JSON文件是字典结构
            if 'air_conditioners' in data:
                ac_configs = data['air_conditioners']
                print(f"检测到JSON文件为字典结构，使用'air_conditioners'键")
            elif 'acs' in data:
                ac_configs = data['acs']
                print(f"检测到JSON文件为字典结构，使用'acs'键")
            else:
                # 尝试找到第一个列表值
                for key, value in data.items():
                    if isinstance(value, list):
                        ac_configs = value
                        print(f"检测到JSON文件为字典结构，使用'{key}'键")
                        break
                
                if not ac_configs:
                    print(f"警告：在JSON文件中找不到空调配置列表")
                    return None
        else:
            print(f"错误：不支持的JSON文件格式")
            return None
        
        # 验证配置列表
        if not isinstance(ac_configs, list):
            print(f"错误：空调配置不是列表格式")
            return None
        
        print(f"成功加载空调配置文件，共{len(ac_configs)}个空调配置")
        
        # 显示配置概览
        if ac_configs:
            type_counts = {}
            valid_configs = []
            
            for i, config in enumerate(ac_configs):
                if isinstance(config, dict):
                    ac_type = config.get('type', 'unknown')
                    type_counts[ac_type] = type_counts.get(ac_type, 0) + 1
                    valid_configs.append(config)
                else:
                    print(f"警告：第{i+1}个配置不是字典格式，跳过")
            
            print("空调类型统计:")
            for ac_type, count in type_counts.items():
                print(f"  {ac_type}: {count}个")
            
            return valid_configs
        else:
            print("警告：空调配置列表为空")
            return None
        
    except json.JSONDecodeError as e:
        print(f"JSON文件格式错误: {e}")
        return None
    except Exception as e:
        print(f"加载空调配置文件时出错: {e}")
        return None

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
    根据空调配置创建优化器实例
    
    参数:
    ac_config: 空调配置字典
    T: 时间步数
    delta_t: 时间步长
    T_max_price_sensitivity_factor: 电价敏感度因子
    T_initial: 初始温度
    
    返回:
    optimizer: ACOptimizer实例
    ac_params_record: 空调参数记录字典
    """
    try:
        # 从配置中提取参数，支持多种字段名称
        P_rated_original = ac_config.get('P_rated', ac_config.get('rated_power_kw', 12.0))
        T_min = ac_config.get('T_min', ac_config.get('min_temp_c', 21.0))
        T_max = ac_config.get('T_max', ac_config.get('max_temp_c', 24.0))
        
        # 对于efficiency字段，优先使用efficiency，然后是eta（但eta在空调中通常表示COP，不是效率）
        efficiency = ac_config.get('efficiency', 0.98)
        
        # 热阻R
        R_original = ac_config.get('R', ac_config.get('thermal_resistance_c_per_kw', 3.0))
        
        # 热容C
        C = ac_config.get('C', ac_config.get('thermal_capacity_j_per_c', 1.8e7))
        
        # 🔧 修复：增强制冷能力以确保夏季高温下有可行解
        # 根据经验，夏季室外温度可能达到35°C，需要确保能降到24°C以下
        # 所需制冷能力至少: 35 - 24 = 11°C
        min_required_cooling = 12.0  # °C，留出安全余量
        current_cooling_original = efficiency * P_rated_original * R_original
        
        # 初始化修改后的参数（可能会被调整）
        P_rated = P_rated_original
        R = R_original
        
        # 参数调整记录
        params_modified = False
        modification_reason = ""
        
        if current_cooling_original < min_required_cooling:
            params_modified = True
            # 方案1：优先增大额定功率（如果当前功率较小）
            if P_rated_original < 8.0:
                P_rated = min_required_cooling / (efficiency * R_original)
                modification_reason = f"Insufficient rated power: increased from {P_rated_original:.2f}kW to {P_rated:.2f}kW to ensure cooling capacity"
                print(f"    🔧 自动调整额定功率: {P_rated_original:.2f}kW → {P_rated:.2f}kW (确保制冷能力)")
            else:
                # 方案2：增大热阻R（提高制冷效率）
                R = min_required_cooling / (efficiency * P_rated_original)
                modification_reason = f"Insufficient thermal resistance: increased from {R_original:.3f}°C/kW to {R:.3f}°C/kW to ensure cooling capacity"
                print(f"    🔧 自动调整热阻: {R_original:.3f}°C/kW → {R:.3f}°C/kW (确保制冷能力)")
        
        # 计算最终制冷能力
        final_cooling_capacity = efficiency * P_rated * R
        
        # 🆕 创建参数记录
        ac_params_record = {
            'ac_id': ac_config.get('id', 'N/A'),
            'ac_type': ac_config.get('type', 'N/A'),
            'original_params': {
                'P_rated_kw': P_rated_original,
                'R_c_per_kw': R_original,
                'efficiency': efficiency,
                'T_min_c': T_min,
                'T_max_c': T_max,
                'C_j_per_c': C,
                'cooling_capacity_c': current_cooling_original
            },
            'final_params': {
                'P_rated_kw': P_rated,
                'R_c_per_kw': R,
                'efficiency': efficiency,
                'T_min_c': T_min,
                'T_max_c': T_max,
                'C_j_per_c': C,
                'cooling_capacity_c': final_cooling_capacity
            },
            'modification_info': {
                'modified': params_modified,
                'reason': modification_reason,
                'required_cooling_c': min_required_cooling,
                'original_cooling_c': current_cooling_original,
                'final_cooling_c': final_cooling_capacity,
                'cooling_improvement_c': final_cooling_capacity - current_cooling_original
            },
            'thermal_dynamics': {
                'C_kwh_per_c': C / 3.6e6,  # 转换后的热容
                'time_constant_h': R * (C / 3.6e6),  # 时间常数
                'exp_decay_factor': np.exp(-delta_t / (R * (C / 3.6e6)))  # 指数衰减因子
            }
        }
        
        # 创建优化器
        optimizer = ACOptimizer(
            T=T,
            delta_t=delta_t,
            P_rated=P_rated,
            T_min=T_min,
            T_max=T_max,
            T_max_price_sensitivity_factor=T_max_price_sensitivity_factor,
            eta=efficiency,  # 使用efficiency作为eta参数
            R=R,
            C=C,
            T_initial=T_initial
        )
        
        print(f"成功创建空调优化器:")
        print(f"  ID: {ac_config.get('id', 'N/A')}")
        print(f"  类型: {ac_config.get('type', 'N/A')}")
        print(f"  额定功率: {P_rated:.2f} kW")
        print(f"  温度范围: [{T_min:.1f}°C, {T_max:.1f}°C]")
        print(f"  效率: {efficiency:.3f}")
        print(f"  热阻: {R:.3f} °C/kW")
        print(f"  热容: {C:.1e} J/°C")
        print(f"  制冷能力: {final_cooling_capacity:.2f} °C")
        
        return optimizer, ac_params_record
        
    except Exception as e:
        print(f"创建优化器时出错: {e}")
        print(f"配置内容: {ac_config}")
        return None, None

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
            print("-" * 100) # 调整分隔线长度
            for t in range(self.T):
                cost_t = self.prices[t] * self.optimal_powers[t] * self.delta_t
                dynamic_t_max_val = self.dynamic_T_max_values[t] if hasattr(self, 'dynamic_T_max_values') and t < len(self.dynamic_T_max_values) else self.T_max_base
                print(f"{t+1:6d} | {self.optimal_powers[t]:8.2f} | {self.optimal_temperatures[t+1]:11.2f} | {self.T_out[t+1]:11.2f} | {self.prices[t]:10.3f} | {cost_t:9.3f} | {dynamic_t_max_val:12.2f}")

    def generate_price_power_curves_all_hours(self, num_samples=100, save_csv=True, csv_filename="ac_optimization_data.csv", current_date=None, write_header=True, ac_id=None, rolling_hour=None, base_price=None, real_price=None):
        """
        为所有时刻或指定滚动时刻生成电价-功率关系曲线

        参数:
        num_samples: 每个时刻的采样点数量
        save_csv: 是否保存数据到CSV文件
        csv_filename: CSV文件名
        current_date: 当前处理的日期，将添加到CSV中
        write_header: 是否写入CSV头部
        ac_id: 空调ID，将添加到CSV中
        rolling_hour: 如果不为None，则只处理指定的滚动时刻
        base_price: 基准电价，如果为None则使用0.0
        real_price: 真实电价，用于计算下一时刻的真实室内温度

        返回:
        curves_data: 字典，键为时刻索引，值为(c_t, P_t)数据对列表
        """
        # 直接使用-1到+1的电价采样范围
        min_price = -1.0
        max_price = 1.0
        
        # 使用基准电价(如果提供)
        base_price_value = 0.0 if base_price is None else base_price
        print(f"      基准电价: {base_price_value:.2f} 元/kWh")
        
        # 记录真实电价(如果提供)
        real_price_value = base_price_value if real_price is None else real_price
        print(f"      真实电价: {real_price_value:.2f} 元/kWh")
        
        if rolling_hour is not None:
            print(f"滚动优化模式: 只处理第 {rolling_hour+1} 小时")
            hours_to_process = [rolling_hour]  # 只处理指定的滚动时刻
        else:
            print(f"全时段优化模式: 处理所有24小时")
            hours_to_process = list(range(self.T))  # 处理所有时刻
        
        print(f"采样电价范围: {min_price:.3f} - {max_price:.3f} 元/kWh")
        print(f"每个时刻采样点数: {num_samples}")
        print(f"总共需要求解: {len(hours_to_process) * num_samples} 个优化问题")
        
        curves_data = {}
        
        # 创建默认电价序列（用于求解优化问题）
        default_prices = [0.0] * self.T  # 默认电价为0
        
        # 准备CSV数据记录 - 当天所有数据
        daily_csv_data = []
        
        # 为每个需要处理的时刻生成价格-功率曲线
        for hour in hours_to_process:
            if rolling_hour is not None:
                print(f"\n正在处理滚动时刻 {hour+1} 小时...")
            else:
                print(f"\n正在处理第 {hour+1} 小时...")
            
            # 采样不同的价格值 - 均匀采样
            price_samples = np.linspace(min_price, max_price, num_samples)
            
            sampled_prices_list = []
            sampled_powers = []
            
            for i, price_sample_val in enumerate(price_samples):
                # 设置当前时刻的电价
                current_prices = default_prices.copy()
                current_prices[hour] = price_sample_val
                self.set_prices(current_prices)
                
                # 求解优化问题
                if self.solve():
                    # 获取当前时刻的最优功率
                    P_t = self.optimal_powers[hour]
                    sampled_prices_list.append(price_sample_val)
                    sampled_powers.append(P_t)
                    
                    # 记录详细数据到当天CSV数据
                    if save_csv:
                        # 记录该时刻的所有相关数据
                        csv_row = {
                            'Hour': hour + 1,  # 1-24小时
                            'Sampled_Price': price_sample_val,
                            'Base_Price': base_price_value,  # 保留基准电价
                            'Real_Price': real_price_value,  # 保留真实电价信息
                            'AC_Power': P_t,
                            'Outdoor_Temperature': self.T_out[hour] if hour < len(self.T_out) else self.T_out[-1],
                            'Indoor_Temperature': self.optimal_temperatures[hour + 1] if hour + 1 < len(self.optimal_temperatures) else self.optimal_temperatures[-1],
                            'Dynamic_T_max': self.dynamic_T_max_values[hour],
                            'Initial_Temperature': self.T_initial  # 保留初始室内温度
                        }
                        if current_date is not None:
                            csv_row['Date'] = current_date
                        if ac_id is not None:
                            csv_row['AC_ID'] = ac_id
                        if rolling_hour is not None:
                            csv_row['Is_Rolling'] = True
                            csv_row['Rolling_Hour'] = rolling_hour + 1  # 1-24小时
                        daily_csv_data.append(csv_row)
                
                # 显示进度
                if (i + 1) % (num_samples // 5) == 0 or (i+1) == num_samples:
                    print(f"  已完成 {i + 1}/{num_samples} 个采样点")
            
            curves_data[hour] = (sampled_prices_list, sampled_powers)
            print(f"  第 {hour+1} 小时生成了 {len(sampled_prices_list)} 个有效数据点")
            
            # 根据真实电价计算真实的空调功率和下一时刻的室内温度
            # 但不记录到CSV中
            if len(sampled_prices_list) > 0:
                # 将价格和功率转换为numpy数组，以便插值
                prices_array = np.array(sampled_prices_list)
                powers_array = np.array(sampled_powers)
                
                # 根据电价-功率曲线插值得到真实电价下的功率
                # 注意：需要处理真实电价超出采样范围的情况
                if real_price_value <= min(prices_array):
                    # 如果真实电价低于最低采样电价，使用最低电价对应的功率
                    real_power = powers_array[np.argmin(prices_array)]
                    print(f"  真实电价 {real_price_value:.3f} 低于采样范围，使用最低电价的功率: {real_power:.3f} kW")
                elif real_price_value >= max(prices_array):
                    # 如果真实电价高于最高采样电价，使用最高电价对应的功率
                    real_power = powers_array[np.argmax(prices_array)]
                    print(f"  真实电价 {real_price_value:.3f} 高于采样范围，使用最高电价的功率: {real_power:.3f} kW")
                else:
                    # 线性插值获取真实功率
                    # 找到最接近的两个电价点
                    idx = np.searchsorted(prices_array, real_price_value)
                    if idx == 0:
                        idx = 1  # 确保有前一个点
                    
                    # 获取两个最接近的电价点
                    price_low = prices_array[idx-1]
                    price_high = prices_array[idx]
                    power_low = powers_array[idx-1]
                    power_high = powers_array[idx]
                    
                    # 线性插值
                    real_power = power_low + (real_price_value - price_low) * (power_high - power_low) / (price_high - price_low)
                    print(f"  插值计算真实电价 {real_price_value:.3f} 对应的功率: {real_power:.3f} kW")
                
                # 计算真实功率下的下一时刻室内温度
                # 使用ETP模型: T_{t+1}^{i} = (1-exp_factor) * (T_{t+1}^{out} - η P_t R) + exp_factor * T_t^{i}
                
                # 当前室内温度T_t^i
                T_current = self.T_initial
                
                # 室外温度T_{t+1}^{out}
                T_out_next = self.T_out[hour+1] if hour+1 < len(self.T_out) else self.T_out[-1]
                
                # 稳态温度: T_{t+1}^{out} - η P_t R
                steady_state_temp = T_out_next - self.eta * self.R * real_power
                
                # 使用ETP模型计算下一时刻室内温度
                T_next = (1 - self.exp_factor) * steady_state_temp + self.exp_factor * T_current
                
                # 存储真实的下一时刻室内温度，供下一个小时使用
                self.real_next_temperature = T_next
                
                print(f"  真实功率 {real_power:.3f} kW 下，室内温度从 {T_current:.2f}°C 变化到 {T_next:.2f}°C")
                
                # 移除：不再将真实电价对应的功率记录添加到CSV
        
        # 处理完成后，立即保存到CSV文件
        if save_csv and daily_csv_data:
            # 定义列的顺序（英文表头）
            columns_order = ['AC_ID', 'Date', 'Hour', 'Sampled_Price', 'Base_Price', 'Real_Price', 'AC_Power', 
                             'Outdoor_Temperature', 'Indoor_Temperature', 'Dynamic_T_max', 
                             'Initial_Temperature', 'Is_Rolling', 'Rolling_Hour']
            
            df = pd.DataFrame(daily_csv_data)
            
            # 确保列顺序正确，只包含存在的列
            available_columns = [col for col in columns_order if col in df.columns]
            df = df[available_columns]
            
            mode = 'w' if write_header else 'a'
            header = write_header
            
            try:
                df.to_csv(csv_filename, mode=mode, header=header, index=False)
                if rolling_hour is not None:
                    print(f"\n✅ 滚动时刻数据已保存到 {csv_filename} (模式: {'覆盖' if mode == 'w' else '追加'})")
                else:
                    print(f"\n✅ 当天数据已保存到 {csv_filename} (模式: {'覆盖' if mode == 'w' else '追加'})")
                print(f"   本次保存 {len(daily_csv_data)} 行数据")
                if write_header:
                    print(f"   英文表头: {list(df.columns)}")
                
                # 显示当前CSV文件的统计信息
                try:
                    current_df = pd.read_csv(csv_filename)
                    print(f"   CSV文件当前总行数: {len(current_df)}")
                    if 'AC_ID' in current_df.columns:
                        print(f"   包含空调数: {current_df['AC_ID'].nunique()}")
                    if 'Date' in current_df.columns:
                        print(f"   包含日期数: {current_df['Date'].nunique()}")
                except Exception as e:
                    print(f"   无法读取CSV文件统计信息: {e}")
                
            except Exception as e:
                print(f"❌ 保存CSV文件时出错: {e}")
                # 备份保存，使用时间戳
                import datetime
                backup_filename = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{csv_filename}"
                try:
                    df.to_csv(backup_filename, index=False)
                    print(f"   数据已备份保存到: {backup_filename}")
                except Exception as backup_e:
                    print(f"   备份保存也失败: {backup_e}")

        return curves_data
    
    def plot_price_power_curves_all_hours(self, num_samples=100, hours_to_plot=None, save_csv=True, csv_filename="ac_optimization_data.csv"):
        """
        绘制所有时刻或指定时刻的电价-功率关系曲线
        
        参数:
        num_samples: 每个时刻的采样点数量
        hours_to_plot: 要绘制的小时列表，如果为None则绘制所有小时
        save_csv: 是否保存数据到CSV文件
        csv_filename: CSV文件名
        """
        # 生成所有时刻的数据
        curves_data = self.generate_price_power_curves_all_hours(num_samples, save_csv, csv_filename, current_date=None, write_header=True, ac_id=None)
        
        if not curves_data:
            print("无法生成有效的采样数据")
            return
        
        # 确定要绘制的小时
        if hours_to_plot is None:
            hours_to_plot = list(range(self.T))
        
        # 计算子图布局
        num_plots = len(hours_to_plot)
        cols = min(4, num_plots)
        rows = (num_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False) # squeeze=False确保axes总是二维数组
        axes = axes.flatten() # 扁平化以便索引
        
        # 绘制每个时刻的曲线
        for idx, hour in enumerate(hours_to_plot):
            if hour not in curves_data or not curves_data[hour][0]: # 检查数据是否存在且不为空
                if idx < len(axes): # 确保索引在范围内
                    axes[idx].set_title(f'第{hour+1}小时 (无数据)')
                    axes[idx].set_visible(True) # 即使没数据也显示子图标题
                continue
                
            prices_list, powers_list = curves_data[hour] # 使用不同的变量名
            
            ax = axes[idx]
            
            # 按电价排序
            sorted_data = sorted(zip(prices_list, powers_list))
            sorted_prices = [x[0] for x in sorted_data]
            sorted_powers = [x[1] for x in sorted_data]
            
            # 绘制曲线
            ax.plot(sorted_prices, sorted_powers, 'b-', linewidth=2, alpha=0.8)
            ax.scatter(prices_list, powers_list, color='red', s=15, alpha=0.6, zorder=5)
            
            ax.set_xlabel('电价 (元/kWh)')
            ax.set_ylabel('功率 (kW)')
            ax.set_title(f'第{hour+1}小时 电价-功率关系')
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴范围
            if len(prices_list) > 0:
                ax.set_xlim(min(prices_list) * 0.95 if min(prices_list) >= 0 else min(prices_list) * 1.05, 
                            max(prices_list) * 1.05 if max(prices_list) >= 0 else max(prices_list) * 0.95)
                ax.set_ylim(0, max(self.P_rated * 1.1, max(powers_list) * 1.1 if powers_list else 1))
        
        # 隐藏多余的子图
        for i in range(num_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"\n统计信息:")
        for hour in hours_to_plot:
            if hour in curves_data and curves_data[hour][0]:
                prices_list, powers_list = curves_data[hour]
                print(f"第{hour+1}小时: 电价范围 {min(prices_list):.3f}-{max(prices_list):.3f}, "
                      f"功率范围 {min(powers_list):.3f}-{max(powers_list):.3f} kW")
            else:
                print(f"第{hour+1}小时: 无有效数据")


    def plot_combined_price_power_curve(self, num_samples=100, save_csv=True, csv_filename="ac_optimization_data.csv"):
        """
        将所有时刻的电价-功率数据点合并到一个图中绘制折线图
        
        参数:
        num_samples: 每个时刻的采样点数量
        save_csv: 是否保存数据到CSV文件
        csv_filename: CSV文件名
        """
        # 生成所有时刻的数据
        curves_data = self.generate_price_power_curves_all_hours(num_samples, save_csv, csv_filename, current_date=None, write_header=True, ac_id=None)
        
        if not curves_data:
            print("无法生成有效的采样数据")
            return
        
        # 合并所有时刻的数据点
        all_prices_list = [] # 避免与外部prices冲突
        all_powers_list = []
        
        for hour in range(self.T):
            if hour in curves_data and curves_data[hour][0]: # 确保数据存在且不为空
                prices_h, powers_h = curves_data[hour]
                all_prices_list.extend(prices_h)
                all_powers_list.extend(powers_h)
        
        if not all_prices_list: # 检查列表是否为空
            print("没有有效的数据点可供绘制合并曲线")
            return
        
        # 按电价分组，计算每个电价的平均功率
        price_groups = defaultdict(list)
        
        # 将相同电价的功率值分组
        for price_val, power_val in zip(all_prices_list, all_powers_list):
            # 将电价四舍五入到3位小数以便分组
            rounded_price = round(price_val, 3)
            price_groups[rounded_price].append(power_val)
        
        # 计算每个电价的平均功率
        grouped_prices = []
        grouped_powers = []
        grouped_powers_min = []
        grouped_powers_max = []
        
        for price_val, power_list_val in price_groups.items():
            grouped_prices.append(price_val)
            grouped_powers.append(sum(power_list_val) / len(power_list_val))  # 平均功率
            grouped_powers_min.append(min(power_list_val)) # 最小功率
            grouped_powers_max.append(max(power_list_val)) # 最大功率
        
        # 按电价从高到低排序（用于阶梯图）
        sorted_data = sorted(zip(grouped_prices, grouped_powers), reverse=True)
        sorted_data_min = sorted(zip(grouped_prices, grouped_powers_min), reverse=True)
        sorted_data_max = sorted(zip(grouped_prices, grouped_powers_max), reverse=True)
        sorted_prices = [x[0] for x in sorted_data]
        sorted_powers = [x[1] for x in sorted_data]
        sorted_prices_min = [x[0] for x in sorted_data_min]
        sorted_powers_min = [x[1] for x in sorted_data_min]
        sorted_prices_max = [x[0] for x in sorted_data_max]
        sorted_powers_max = [x[1] for x in sorted_data_max]
        
        # 绘制图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 绘制所有原始数据点（较小的点，透明度较低）
        ax.scatter(all_prices_list, all_powers_list, color='gray', s=10, alpha=0.2, label='原始数据点 (所有小时)')
        
        # 绘制分组后的平均值点
        ax.scatter(grouped_prices, grouped_powers, color='blue', s=40, alpha=0.8, label='分组平均值 (按电价)')
        
        # 使用step函数绘制阶梯状折线图
        ax.step(sorted_prices, sorted_powers, where='post', linewidth=2.5, 
                color='darkblue', alpha=0.9, label='阶梯状需求曲线 (平均)')
        # 增加：绘制折线图连接平均值点
        ax.plot(sorted_prices, sorted_powers, linestyle='-', linewidth=1.5,
                color='red', alpha=0.7, label='连接平均值点')
        
        # 绘制最小功率折线图
        ax.plot(sorted_prices_min, sorted_powers_min, linestyle='--', linewidth=1.5,
                color='green', alpha=0.8, label='最小功率')
        
        # 绘制最大功率折线图
        ax.plot(sorted_prices_max, sorted_powers_max, linestyle='--', linewidth=1.5,
                color='purple', alpha=0.8, label='最大功率')
        
        ax.set_xlabel('电价 (元/kWh)', fontsize=12)
        ax.set_ylabel('功率 (kW)', fontsize=12)
        ax.set_title('所有时刻合并的电价-功率关系（阶梯状折线图）', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 设置坐标轴范围
        if all_prices_list: # 确保列表不为空
            min_p = min(all_prices_list)
            max_p = max(all_prices_list)
            ax.set_xlim(min_p * 0.95 if min_p >= 0 else min_p * 1.05, 
                        max_p * 1.05 if max_p >=0 else max_p * 0.95)
            ax.set_ylim(0, max(self.P_rated * 1.1, max(all_powers_list) * 1.1 if all_powers_list else 1))
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        if all_prices_list:
            print(f"\n阶梯图统计信息:")
            print(f"原始数据点数: {len(all_prices_list)}")
            print(f"分组后电价档位数: {len(grouped_prices)}")
            print(f"电价范围: {min(all_prices_list):.3f} - {max(all_prices_list):.3f} 元/kWh")
            print(f"功率范围: {min(all_powers_list):.3f} - {max(all_powers_list):.3f} kW")
            print(f"平均功率 (所有数据点): {np.mean(all_powers_list):.3f} kW")

            print("\n每个电价档位的功率数据点数量:")
            # 按电价从高到低排序后输出数量
            for price_val, power_list_val in sorted(price_groups.items(), reverse=True):
                print(f"  电价 {price_val:.3f} 元/kWh: {len(power_list_val)} 个数据点")

def load_summer_temperature_data(csv_file="data/W2.csv"):
    """
    从W2.csv文件中加载2021年9月的温度数据，以及10月1日数据(用于边界处理)
    
    参数:
    csv_file: CSV文件路径
    
    返回:
    month_data: 包含9月和10月1日数据的DataFrame
    unique_dates: 日期列表
    """
    try:
        import os
        
        # 检查文件路径
        if not os.path.exists(csv_file):
            # 尝试绝对路径
            csv_file = os.path.abspath(csv_file)
            if not os.path.exists(csv_file):
                print(f"错误：找不到文件 {csv_file}")
                return None, None
        
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        print(f"成功读取W2.csv文件，共{len(df)}条记录")
        
        # 将Time列转换为datetime类型
        df['Time'] = pd.to_datetime(df['Time'])
        
        # 筛选2021年9月的数据以及10月1日的数据（用于边界处理）
        month_data = df[
            ((df['Time'].dt.year == 2021) & 
             (df['Time'].dt.month == 9)) |
            ((df['Time'].dt.year == 2021) &
             (df['Time'].dt.month == 10) &
             (df['Time'].dt.day == 1))
        ].copy()
        
        if month_data.empty:
            print("警告: 未找到2021年9月的数据")
            return None, None
        
        print(f"找到2021年9月及10月1日数据：{len(month_data)}条记录")
        
        # 将华氏度转换为摄氏度: C = (F - 32) * 5/9
        month_data['Temperature(C)'] = (month_data['Temperature(F)'] - 32) * 5 / 9
        
        # 获取所有可用的日期
        dates = []
        for time_str in month_data['Time']:
            date_part = time_str.strftime('%Y/%m/%d')
            dates.append(date_part)
        
        # 获取唯一日期并排序
        unique_dates = list(set(dates))
        unique_dates.sort()
        
        print(f"处理总天数: {len(unique_dates)}天")
        print(f"日期范围: {unique_dates[0]} 到 {unique_dates[-1]}")
        
        return month_data, unique_dates
        
    except Exception as e:
        print(f"加载温度数据时出错: {e}")
        return None, None

def extract_daily_temperature(month_data, target_date, ac_config=None):
    """
    从月份数据中提取指定日期的24小时温度数据
    
    参数:
    month_data: 月份数据DataFrame
    target_date: 目标日期字符串（格式：'YYYY/MM/DD'）
    ac_config: 空调配置字典（可选，用于温度调整）
    
    返回:
    hourly_temps: 25个温度数据点（T=0到T=24），单位：摄氏度
    """
    try:
        # 筛选指定日期的数据
        day_data = month_data[month_data['Time'].dt.strftime('%Y/%m/%d') == target_date].copy()
        
        if day_data.empty:
            print(f"警告: 日期 {target_date} 没有数据")
            return None
        
        # 确保按时间排序
        day_data = day_data.sort_values('Time')
        
        # 由于数据是每15分钟记录一次，我们需要提取每小时的数据
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
        
        # 新增：温度调整逻辑
        if ac_config is not None:
            # 获取空调的上限温度
            ac_t_max = ac_config.get('T_max', ac_config.get('max_temp_c', 24.0))
            adjusted_count = 0
            original_min = min(hourly_temps)
            
            # 检查并调整所有温度点
            for i in range(len(hourly_temps)):
                if hourly_temps[i] < ac_t_max:
                    # 温度低于空调上限，调整为上限温度 + 随机增加0.5-1.0度
                    random_increase = random.uniform(0.5, 1.0)
                    hourly_temps[i] = ac_t_max + random_increase
                    adjusted_count += 1
            
            if adjusted_count > 0:
                print(f"    🌡️  温度调整: {adjusted_count}个时刻从低于{ac_t_max:.1f}°C调整为{ac_t_max:.1f}°C+0.5~1.0°C")
                print(f"    📊 调整前温度范围: {original_min:.1f}°C - {max(hourly_temps):.1f}°C")
                print(f"    📊 调整后温度范围: {min(hourly_temps):.1f}°C - {max(hourly_temps):.1f}°C")
            else:
                print(f"    ✅ 所有温度均高于空调上限{ac_t_max:.1f}°C，无需调整")
        
        return hourly_temps
        
    except Exception as e:
        print(f"提取日期 {target_date} 的温度数据时出错: {e}")
        return None

def extract_rolling_temperature(month_data, unique_dates, start_date, start_hour, ac_config=None):
    """
    从月份数据中提取从指定日期和小时开始的滚动24小时温度数据
    
    参数:
    month_data: 月份数据DataFrame
    unique_dates: 可用日期列表
    start_date: 起始日期字符串（格式：'YYYY/MM/DD'）
    start_hour: 起始小时（0-23）
    ac_config: 空调配置字典（可选，用于温度调整）
    
    返回:
    rolling_temps: 从起始时刻开始的25个温度数据点，单位：摄氏度
    """
    try:
        # 获取起始日期的索引
        if start_date not in unique_dates:
            print(f"警告: 起始日期 {start_date} 不在可用日期列表中")
            return None
        
        date_index = unique_dates.index(start_date)
        
        # 计算需要的两个日期
        current_date = start_date
        
        # 计算下一个日期（如果当前不是最后一个日期）
        if date_index < len(unique_dates) - 1:
            next_date = unique_dates[date_index + 1]
        else:
            print(f"警告: {start_date} 是最后一个可用日期，无法获取完整的滚动窗口")
            # 如果没有下一天的数据，可以使用一些默认策略
            # 这里我们返回None，也可以考虑其他策略如复制当天数据
            return None
        
        # 获取当前日期的温度数据
        current_day_temps = extract_daily_temperature(month_data, current_date, None)  # 不应用温度调整
        if current_day_temps is None:
            print(f"警告: 无法获取 {current_date} 的温度数据")
            return None
        
        # 获取下一个日期的温度数据
        next_day_temps = extract_daily_temperature(month_data, next_date, None)  # 不应用温度调整
        if next_day_temps is None:
            print(f"警告: 无法获取 {next_date} 的温度数据")
            return None
        
        # 创建滚动窗口温度数据
        rolling_temps = []
        
        # 从当前日期的start_hour开始
        for i in range(24):
            hour_index = (start_hour + i) % 24
            if i < (24 - start_hour):
                # 使用当前日期的数据
                rolling_temps.append(current_day_temps[hour_index])
            else:
                # 使用下一个日期的数据
                next_day_hour = (start_hour + i) % 24
                rolling_temps.append(next_day_temps[next_day_hour])
        
        # 添加第25个点（与第24个点相同，通常用于边界条件）
        rolling_temps.append(rolling_temps[-1])
        
        # 应用温度调整（如果需要）
        if ac_config is not None:
            # 获取空调的上限温度
            ac_t_max = ac_config.get('T_max', ac_config.get('max_temp_c', 24.0))
            adjusted_count = 0
            original_min = min(rolling_temps)
            
            # 检查并调整所有温度点
            for i in range(len(rolling_temps)):
                if rolling_temps[i] < ac_t_max:
                    # 温度低于空调上限，调整为上限温度 + 随机增加0.5-1.0度
                    random_increase = random.uniform(0.5, 1.0)
                    rolling_temps[i] = ac_t_max + random_increase
                    adjusted_count += 1
            
            if adjusted_count > 0:
                print(f"    🌡️  滚动温度调整: {adjusted_count}个时刻从低于{ac_t_max:.1f}°C调整为{ac_t_max:.1f}°C+0.5~1.0°C")
                print(f"    📊 调整前温度范围: {original_min:.1f}°C - {max(rolling_temps):.1f}°C")
                print(f"    📊 调整后温度范围: {min(rolling_temps):.1f}°C - {max(rolling_temps):.1f}°C")
        
        return rolling_temps
        
    except Exception as e:
        print(f"提取滚动温度数据时出错: {e}")
        return None

def save_ac_params_records(ac_params_records, filename="ac_parameters_record.csv"):
    """
    保存所有空调的参数记录到CSV文件
    
    参数:
    ac_params_records: 空调参数记录列表
    filename: 输出文件名
    """
    try:
        import pandas as pd
        
        # 准备数据列表
        records_data = []
        
        for record in ac_params_records:
            # 展平记录数据
            flat_record = {
                # 基本信息
                'AC_ID': record['ac_id'],
                'AC_Type': record['ac_type'],
                
                # 原始参数
                'Original_P_rated_kW': record['original_params']['P_rated_kw'],
                'Original_R_C_per_kW': record['original_params']['R_c_per_kw'],
                'Original_Efficiency': record['original_params']['efficiency'],
                'Original_T_min_C': record['original_params']['T_min_c'],
                'Original_T_max_C': record['original_params']['T_max_c'],
                'Original_C_J_per_C': record['original_params']['C_j_per_c'],
                'Original_Cooling_Capacity_C': record['original_params']['cooling_capacity_c'],
                
                # 最终参数
                'Final_P_rated_kW': record['final_params']['P_rated_kw'],
                'Final_R_C_per_kW': record['final_params']['R_c_per_kw'],
                'Final_Efficiency': record['final_params']['efficiency'],
                'Final_T_min_C': record['final_params']['T_min_c'],
                'Final_T_max_C': record['final_params']['T_max_c'],
                'Final_C_J_per_C': record['final_params']['C_j_per_c'],
                'Final_Cooling_Capacity_C': record['final_params']['cooling_capacity_c'],
                
                # 修改信息
                'Modified': record['modification_info']['modified'],
                'Modification_Reason': record['modification_info']['reason'],
                'Required_Cooling_C': record['modification_info']['required_cooling_c'],
                'Cooling_Improvement_C': record['modification_info']['cooling_improvement_c'],
                
                # 热力学参数
                'C_kWh_per_C': record['thermal_dynamics']['C_kwh_per_c'],
                'Time_Constant_h': record['thermal_dynamics']['time_constant_h'],
                'Exp_Decay_Factor': record['thermal_dynamics']['exp_decay_factor']
            }
            
            records_data.append(flat_record)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(records_data)
        df.to_csv(filename, index=False)
        
        # 统计信息
        total_acs = len(df)
        modified_acs = len(df[df['Modified'] == True])
        
        print(f"\n💾 空调参数记录已保存到: {filename}")
        print(f"   总空调数: {total_acs}")
        print(f"   被修改的空调数: {modified_acs}")
        print(f"   未修改的空调数: {total_acs - modified_acs}")
        
        if modified_acs > 0:
            print(f"   修改统计:")
            # 统计修改类型
            power_modifications = len(df[df['Modification_Reason'].str.contains('rated power', na=False, case=False)])
            resistance_modifications = len(df[df['Modification_Reason'].str.contains('thermal resistance', na=False, case=False)])
            
            if power_modifications > 0:
                avg_power_increase = df[df['Modification_Reason'].str.contains('rated power', na=False, case=False)]['Final_P_rated_kW'].mean() - \
                                   df[df['Modification_Reason'].str.contains('rated power', na=False, case=False)]['Original_P_rated_kW'].mean()
                print(f"     额定功率调整: {power_modifications}个空调, 平均增加 {avg_power_increase:.2f}kW")
            
            if resistance_modifications > 0:
                avg_resistance_increase = df[df['Modification_Reason'].str.contains('thermal resistance', na=False, case=False)]['Final_R_C_per_kW'].mean() - \
                                        df[df['Modification_Reason'].str.contains('thermal resistance', na=False, case=False)]['Original_R_C_per_kW'].mean()
                print(f"     热阻调整: {resistance_modifications}个空调, 平均增加 {avg_resistance_increase:.3f}°C/kW")
            
            avg_cooling_improvement = df[df['Modified'] == True]['Cooling_Improvement_C'].mean()
            print(f"     平均制冷能力提升: {avg_cooling_improvement:.2f}°C")
        
        return filename
        
    except Exception as e:
        print(f"❌ 保存空调参数记录时出错: {e}")
        return None

def main():
    """主函数：循环处理每个空调和每天的数据，使用滚动预测生成电价-功率关系曲线"""
    print("=" * 80)
    print("多空调多天数据滚动预测电价-功率关系曲线生成程序")
    print("=" * 80)
    
    # 默认的24小时电价变化数组
    # 凌晨和晚上电价为-1（鼓励使用空调）
    # 中午到下午电价为+1（抑制使用空调）
    default_prices = [
        -1.00,  # 0:00 凌晨，鼓励使用
        -1.00,  # 1:00
        -1.00,  # 2:00
        -0.75,  # 3:00
        -0.50,  # 4:00
        -0.25,  # 5:00
        0.00,   # 6:00 早晨，中性
        0.25,   # 7:00
        0.50,   # 8:00
        0.75,   # 9:00
        0.75,   # 10:00
        1.00,   # 11:00 中午，抑制使用
        1.00,   # 12:00
        1.00,   # 13:00
        1.00,   # 14:00 下午，抑制使用
        0.75,   # 15:00
        0.75,   # 16:00
        0.50,   # 17:00
        0.25,   # 18:00
        0.00,   # 19:00 晚上，中性
        -0.25,  # 20:00
        -0.50,  # 21:00
        -0.75,  # 22:00
        -1.00,  # 23:00 深夜，鼓励使用
    ]
    
    # 1. 加载空调配置数据
    print("\n" + "=" * 40)
    print("加载空调配置数据...")
    print("=" * 40)
    
    ac_configs = load_ac_data("D:/experiments/ACL_agg_exp/src/mmoe_generate_data/ac_data.json")
    
    if not ac_configs:
        print("警告：无法加载空调配置，使用默认配置")
        # 创建默认配置
        default_config = {
            'id': 'AC_DEFAULT_001',
            'type': 'default',
            'rated_power_kw': 12.0,
            'min_temp_c': 21.0,
            'max_temp_c': 24.0,
            'efficiency': 0.98,
            'thermal_resistance_c_per_kw': 3.0,
            'thermal_capacity_j_per_c': 1.8e7
        }
        ac_configs = [default_config]
    
    print(f"将处理 {len(ac_configs)} 个空调配置")
    
    # 2. 加载2021年9月温度数据和10月1日数据
    print("\n" + "=" * 40)
    print("加载2021年9月温度数据...")
    print("=" * 40)
    
    month_data, unique_dates = load_summer_temperature_data("data/W2.csv")
    
    if month_data is None or unique_dates is None:
        print("无法加载9月数据，程序退出")
        return
    
    # 3. 设置全局参数
    csv_filename = "all_ac_rolling_optimization_data.csv"
    num_samples = 10  # 每个时刻的采样点数量（可调整）
    total_days = len(unique_dates) - 1  # 减1是因为10月1日仅用于边界处理
    total_acs = len(ac_configs)
    
    print(f"\n全局参数设置:")
    print(f"  空调数量: {total_acs}")
    print(f"  处理天数: {total_days} (不包括10月1日)")
    print(f"  每个时刻采样点数: {num_samples}")
    print(f"  滚动优化: 每个时刻使用未来24小时的温度预测")
    print(f"  输出文件: {csv_filename}")
    print(f"  电价采样范围: -1.0 到 +1.0 元/kWh")
    
    # 打印默认电价数组信息
    print(f"\n默认电价数组 (用作电价采样的基准):")
    print(f"  凌晨和深夜 (0-3时, 21-23时): 负电价，鼓励使用空调")
    print(f"  早晨和晚上 (4-8时, 18-20时): 从负到正过渡")
    print(f"  中午和下午 (11-14时): 高正电价，抑制使用空调")
    print(f"  小时  | 电价(元/kWh)")
    print(f"  ------+------------")
    for h, price in enumerate(default_prices):
        print(f"  {h:2d}:00 | {price:+.2f}")
    
    # 检查CSV文件是否已存在
    csv_exists = os.path.exists(csv_filename)
    if csv_exists:
        try:
            existing_df = pd.read_csv(csv_filename)
            existing_rows = len(existing_df)
            print(f"\n⚠️  发现已存在的CSV文件: {csv_filename}")
            print(f"     文件包含 {existing_rows} 行数据")
            
            if 'AC_ID' in existing_df.columns:
                existing_acs = existing_df['AC_ID'].nunique()
                ac_ids = sorted(existing_df['AC_ID'].unique())
                print(f"     已包含 {existing_acs} 个空调的数据: {ac_ids}")
            
            if 'Date' in existing_df.columns:
                existing_dates = existing_df['Date'].nunique()
                date_range = sorted(existing_df['Date'].unique())
                print(f"     已包含 {existing_dates} 个日期的数据")
                print(f"     日期范围: {date_range[0]} 到 {date_range[-1]}")
            
            print(f"     程序将继续追加新数据（支持断点续传）")
            is_first_write = False  # 文件已存在，不需要写入头部
            
        except Exception as e:
            print(f"⚠️  无法读取现有CSV文件: {e}")
            print(f"     将重新创建文件")
            is_first_write = True
    else:
        print(f"\n📝 将创建新的CSV文件: {csv_filename}")
        is_first_write = True
    
    # 检查空调参数记录文件是否已存在
    params_record_filename = "ac_parameters_record.csv"
    params_exists = os.path.exists(params_record_filename)
    if params_exists:
        try:
            existing_params_df = pd.read_csv(params_record_filename)
            existing_params = len(existing_params_df)
            print(f"\n⚠️  发现已存在的空调参数记录文件: {params_record_filename}")
            print(f"     文件包含 {existing_params} 个空调参数记录")
            print(f"     程序将重新生成空调参数记录文件")
        except Exception as e:
            print(f"⚠️  无法读取现有空调参数记录文件: {e}")
    
    # 4. 循环处理每个空调
    print("\n" + "=" * 40)
    print("开始处理空调配置...")
    print("=" * 40)
    
    # 🆕 创建空调参数记录列表
    all_ac_params_records = []
    
    # 只处理9月的数据（不包括10月1日，它只用于边界处理）
    september_dates = [date for date in unique_dates if date.startswith('2021/09/')]
    
    for ac_idx, ac_config in enumerate(ac_configs):
        print(f"\n{'='*20} 处理空调 {ac_idx + 1}/{total_acs} {'='*20}")
        print(f"空调ID: {ac_config.get('id', 'N/A')}")
        print(f"空调类型: {ac_config.get('type', 'N/A')}")
        
        # 创建当前空调的优化器
        optimizer, ac_params_record = create_optimizer_from_config(
            ac_config,
            T=24,
            delta_t=1.0,
            T_max_price_sensitivity_factor=0.05,
            T_initial=23.0
        )
        
        if optimizer is None or ac_params_record is None:
            print(f"  ❌ 无法创建空调 {ac_config.get('id', 'N/A')} 的优化器，跳过")
            continue
        
        # 🆕 收集参数记录
        all_ac_params_records.append(ac_params_record)
        
        # 5. 循环处理9月每天的数据
        print(f"\n开始处理空调 {ac_config.get('id', 'N/A')} 的9月数据...")
        print(f"📝 注意：滚动优化方式，每天每小时计算一次，使用未来24小时温度窗口")
        
        for day_idx, current_date in enumerate(september_dates):
            print(f"\n  处理第 {day_idx + 1}/{len(september_dates)} 天: {current_date}")
            
            # 循环每个小时进行滚动优化
            for hour in range(24):
                print(f"    处理 {current_date} 的第 {hour+1} 小时")
                
                # 提取滚动预测的温度数据
                rolling_temps = extract_rolling_temperature(
                    month_data, 
                    unique_dates, 
                    current_date, 
                    hour, 
                    ac_config
                )
                
                if rolling_temps is None:
                    print(f"      跳过 {current_date} 的第 {hour+1} 小时（无法获取滚动温度数据）")
                    continue
                
                # 设置滚动窗口的室外温度
                optimizer.set_outdoor_temperature(rolling_temps)
                
                # 记录温度区间
                print(f"      滚动温度范围: {min(rolling_temps):.1f}°C - {max(rolling_temps):.1f}°C")
                
                # 使用默认电价数组进行电价采样
                # 注意：电价采样仍然是在[-1,1]范围内，但默认电价数组用作真实电价
                if hour == 0 and day_idx == 0:
                    # 第一天第一个小时使用默认初始温度
                    current_T_initial = optimizer.T_initial
                else:
                    # 使用上一个时刻根据真实电价计算的结束温度作为本时刻的初始温度
                    if hasattr(optimizer, 'real_next_temperature'):
                        current_T_initial = optimizer.real_next_temperature
                    else:
                        # 如果没有上一个时刻的温度数据，使用默认值
                        current_T_initial = optimizer.T_initial
                
                # 更新优化器的初始温度
                optimizer.T_initial = current_T_initial
                print(f"      当前时刻初始室内温度: {current_T_initial:.2f}°C")
                
                # 使用当前小时的默认电价作为"真实电价"
                real_price = default_prices[hour]
                print(f"      当前时刻真实电价: {real_price:.2f} 元/kWh")
                
                # 生成当前小时的电价-功率关系曲线数据
                write_header = is_first_write and day_idx == 0 and hour == 0  # 只在第一次写入头部
                
                try:
                    # 滚动优化：生成当前时刻的电价-功率关系曲线
                    curves_data = optimizer.generate_price_power_curves_all_hours(
                        num_samples=num_samples,
                        save_csv=True,
                        csv_filename=csv_filename,
                        current_date=current_date,
                        write_header=write_header,
                        ac_id=ac_config.get('id', f'AC_{ac_idx+1}'),
                        rolling_hour=hour,  # 当前处理的小时
                        base_price=real_price,  # 使用默认电价数组中对应小时的电价作为基准和真实电价
                        real_price=real_price  # 新增：传递真实电价
                    )
                    
                    if curves_data:
                        print(f"      ✅ 成功生成滚动预测数据")
                        is_first_write = False  # 第一次写入完成后，后续都是追加
                    else:
                        print(f"      ❌ 生成滚动预测数据失败")
                        
                except Exception as e:
                    print(f"      ❌ 处理日期 {current_date} 第 {hour+1} 小时时出错: {e}")
                    continue
            
            # 显示天进度
            progress = (day_idx + 1) / len(september_dates) * 100
            print(f"    📊 空调进度: {day_idx + 1}/{len(september_dates)} 天已完成 ({progress:.1f}%)")
                
            # 显示当前CSV文件大小
            try:
                if os.path.exists(csv_filename):
                    file_size = os.path.getsize(csv_filename) / (1024 * 1024)  # MB
                    print(f"    📁 CSV文件大小: {file_size:.2f} MB")
            except:
                pass
        
        # 显示空调处理完成的总进度
        ac_progress = (ac_idx + 1) / total_acs * 100
        print(f"\n空调 {ac_config.get('id', 'N/A')} 处理完成！")
        print(f"总进度: {ac_idx + 1}/{total_acs} 个空调已完成 ({ac_progress:.1f}%)")
    
    # 🆕 保存空调参数记录
    if all_ac_params_records:
        print(f"\n" + "=" * 40)
        print("保存空调参数记录...")
        print("=" * 40)
        save_ac_params_records(all_ac_params_records, params_record_filename)
    
    print("\n" + "=" * 80)
    print("所有空调和数据生成完成！")
    print("=" * 80)
    print(f"主要输出文件: {csv_filename}")
    if all_ac_params_records:
        print(f"参数记录文件: {params_record_filename}")
    
    # 6. 显示最终统计信息
    try:
        final_df = pd.read_csv(csv_filename)
        print(f"最终CSV文件包含 {len(final_df)} 行数据")
        
        if 'AC_ID' in final_df.columns:
            print(f"包含的空调数: {final_df['AC_ID'].nunique()}")
            print("空调ID列表:")
            for ac_id in sorted(final_df['AC_ID'].unique()):
                ac_count = len(final_df[final_df['AC_ID'] == ac_id])
                print(f"  {ac_id}: {ac_count} 条记录")
        
        if 'Date' in final_df.columns:
            print(f"包含的日期数: {final_df['Date'].nunique()}")
            date_range = sorted(final_df['Date'].unique())
            print(f"日期范围: {date_range[0]} 到 {date_range[-1]}")
        
        print(f"CSV文件列名: {list(final_df.columns)}")
        
        # 显示前几行和后几行
        print(f"\n前5行数据:")
        print(final_df.head().to_string(index=False))
        print(f"\n后5行数据:")
        print(final_df.tail().to_string(index=False))
        
    except Exception as e:
        print(f"读取最终CSV文件时出错: {e}")

if __name__ == "__main__":
    main()