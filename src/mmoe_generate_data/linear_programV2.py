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

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class ACOptimizer:
    def __init__(self, T=24, delta_t=1.0, P_rated=3.0, T_min=20.0, T_max=26.0,
                 eta=0.8, R=2.0, C=5.0, T_initial=22.0):
        """
        初始化空调优化器
        
        参数:
        T: 时间步数 (小时)
        delta_t: 时间步长 (小时)
        P_rated: 额定功率 (kW)
        T_min, T_max: 温度约束 (°C)
        eta: 空调效率
        R: 热阻 (°C/kW)
        C: 热容 (J/°C)，将自动转换为kWh/°C
        T_initial: 初始室温 (°C)
        """
        self.T = T
        self.delta_t = delta_t
        self.P_rated = P_rated
        self.T_min = T_min
        self.T_max = T_max
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
        T_i = [pulp.LpVariable(f"T_i_{t}", lowBound=self.T_min, upBound=self.T_max) 
               for t in range(2, self.T + 2)]
        
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
        ax2.axhline(y=self.T_max, color='r', linestyle='--', alpha=0.7, label=f'最高温度 {self.T_max}°C')
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
            print("\n时间步 | 功率(kW) | 室内温度(°C) | 室外温度(°C) | 电价(元/kWh) | 时段成本(元)")
            print("-" * 85)
            for t in range(self.T):
                cost_t = self.prices[t] * self.optimal_powers[t] * self.delta_t
                print(f"{t+1:6d} | {self.optimal_powers[t]:8.2f} | {self.optimal_temperatures[t+1]:11.2f} | {self.T_out[t+1]:11.2f} | {self.prices[t]:10.3f} | {cost_t:9.3f}")

    def generate_price_power_curves_all_hours(self, num_samples=100, save_csv=True, csv_filename="ac_optimization_data.csv"):
        """
        为所有时刻生成电价-功率关系曲线
        
        参数:
        num_samples: 每个时刻的采样点数量
        save_csv: 是否保存数据到CSV文件
        csv_filename: CSV文件名
        
        返回:
        curves_data: 字典，键为时刻索引，值为(c_t, P_t)数据对列表
        """
        if not hasattr(self, 'prices'):
            print("错误：请先设置电价序列")
            return {}
        
        # 获取所有时刻的电价范围
        all_prices = self.prices
        min_price = min(all_prices)
        max_price = max(all_prices)
        
        print(f"所有时刻电价范围: {min_price:.3f} - {max_price:.3f} 元/kWh")
        print(f"每个时刻采样点数: {num_samples}")
        print(f"总共需要求解: {self.T * num_samples} 个优化问题")
        
        curves_data = {}
        
        # 保存原始电价
        original_prices = self.prices.copy()
        
        # 准备CSV数据记录
        csv_data = []
        
        # 为每个时刻生成价格-功率曲线
        for hour in range(self.T):
            print(f"\n正在处理第 {hour+1} 小时...")
            
            # 采样不同的价格值
            price_samples = np.linspace(min_price, max_price, num_samples)
            
            sampled_prices = []
            sampled_powers = []
            
            for i, price in enumerate(price_samples):
                # 恢复原始电价
                self.prices = original_prices.copy()
                
                # 设置当前时刻的电价
                self.prices[hour] = price
                
                # 求解优化问题
                if self.solve():
                    # 获取当前时刻的最优功率
                    P_t = self.optimal_powers[hour]
                    sampled_prices.append(price)
                    sampled_powers.append(P_t)
                    
                    # 记录详细数据到CSV
                    if save_csv:
                        # 记录该时刻的所有相关数据
                        csv_row = {
                            'Hour': hour + 1,  # 1-24小时
                            'Sampled_Price': price,
                            'AC_Power': P_t,
                            'Outdoor_Temperature': self.T_out[hour] if hour < len(self.T_out) else self.T_out[-1],
                            'Indoor_Temperature': self.optimal_temperatures[hour + 1] if hour + 1 < len(self.optimal_temperatures) else self.optimal_temperatures[-1],
                            'Original_Price': original_prices[hour],
                            'Total_Energy': self.total_energy,
                            'Total_Cost': self.total_cost
                        }
                        csv_data.append(csv_row)
                
                # 显示进度
                if (i + 1) % 20 == 0:
                    print(f"  已完成 {i + 1}/{num_samples} 个采样点")
            
            curves_data[hour] = (sampled_prices, sampled_powers)
            print(f"  第 {hour+1} 小时生成了 {len(sampled_prices)} 个有效数据点")
        
        # 恢复原始电价
        self.prices = original_prices
        
        # 保存CSV文件
        if save_csv and csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filename, index=False)
            print(f"\n数据已保存到 {csv_filename}")
            print(f"CSV文件包含 {len(csv_data)} 行数据")
            
            # 显示CSV文件的前几行
            print("\nCSV文件前5行预览:")
            print(df.head().to_string(index=False))
        
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
        curves_data = self.generate_price_power_curves_all_hours(num_samples, save_csv, csv_filename)
        
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
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if num_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # 绘制每个时刻的曲线
        for idx, hour in enumerate(hours_to_plot):
            if hour >= len(curves_data):
                continue
                
            prices, powers = curves_data[hour]
            
            if len(prices) == 0:
                continue
            
            ax = axes[idx]
            
            # 按电价排序
            sorted_data = sorted(zip(prices, powers))
            sorted_prices = [x[0] for x in sorted_data]
            sorted_powers = [x[1] for x in sorted_data]
            
            # 绘制曲线
            ax.plot(sorted_prices, sorted_powers, 'b-', linewidth=2, alpha=0.8)
            ax.scatter(prices, powers, color='red', s=15, alpha=0.6, zorder=5)
            
            ax.set_xlabel('电价 (元/kWh)')
            ax.set_ylabel('功率 (kW)')
            ax.set_title(f'第{hour+1}小时 电价-功率关系')
            ax.grid(True, alpha=0.3)
            
            # 设置坐标轴范围
            if len(prices) > 0:
                ax.set_xlim(min(prices) * 0.95, max(prices) * 1.05)
                ax.set_ylim(0, max(powers) * 1.1 if powers else 1)
        
        # 隐藏多余的子图
        for idx in range(num_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"\n统计信息:")
        for hour in hours_to_plot:
            if hour < len(curves_data):
                prices, powers = curves_data[hour]
                if len(prices) > 0:
                    print(f"第{hour+1}小时: 电价范围 {min(prices):.3f}-{max(prices):.3f}, "
                          f"功率范围 {min(powers):.3f}-{max(powers):.3f} kW")

    def plot_combined_price_power_curve(self, num_samples=100, save_csv=True, csv_filename="ac_optimization_data.csv"):
        """
        将所有时刻的电价-功率数据点合并到一个图中绘制折线图
        
        参数:
        num_samples: 每个时刻的采样点数量
        save_csv: 是否保存数据到CSV文件
        csv_filename: CSV文件名
        """
        # 生成所有时刻的数据
        curves_data = self.generate_price_power_curves_all_hours(num_samples, save_csv, csv_filename)
        
        if not curves_data:
            print("无法生成有效的采样数据")
            return
        
        # 合并所有时刻的数据点
        all_prices = []
        all_powers = []
        
        for hour in range(self.T):
            if hour in curves_data:
                prices, powers = curves_data[hour]
                all_prices.extend(prices)
                all_powers.extend(powers)
        
        if len(all_prices) == 0:
            print("没有有效的数据点")
            return
        
        # 按电价分组，计算每个电价的平均功率
        price_groups = defaultdict(list)
        
        # 将相同电价的功率值分组
        for price, power in zip(all_prices, all_powers):
            # 将电价四舍五入到3位小数以便分组
            rounded_price = round(price, 3)
            price_groups[rounded_price].append(power)
        
        # 计算每个电价的平均功率
        grouped_prices = []
        grouped_powers = []
        
        for price, power_list in price_groups.items():
            grouped_prices.append(price)
            grouped_powers.append(sum(power_list) / len(power_list))  # 平均功率
        
        # 按电价从高到低排序（用于阶梯图）
        sorted_data = sorted(zip(grouped_prices, grouped_powers), reverse=True)
        sorted_prices = [x[0] for x in sorted_data]
        sorted_powers = [x[1] for x in sorted_data]
        
        # 绘制图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 绘制所有原始数据点（较小的点，透明度较低）
        ax.scatter(all_prices, all_powers, color='red', s=10, alpha=0.3, label='原始数据点')
        
        # 绘制分组后的平均值点
        ax.scatter(grouped_prices, grouped_powers, color='blue', s=40, alpha=0.8, label='分组平均值')
        
        # 使用step函数绘制阶梯状折线图
        ax.step(sorted_prices, sorted_powers, where='post', linewidth=2.5, 
                color='blue', alpha=0.9, label='阶梯状需求曲线')
        
        ax.set_xlabel('电价 (元/kWh)', fontsize=12)
        ax.set_ylabel('功率 (kW)', fontsize=12)
        ax.set_title('所有时刻的电价-功率关系（阶梯状折线图）', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 设置坐标轴范围
        ax.set_xlim(min(all_prices) * 0.95, max(all_prices) * 1.05)
        ax.set_ylim(0, max(all_powers) * 1.1)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"\n阶梯图统计信息:")
        print(f"原始数据点数: {len(all_prices)}")
        print(f"分组后电价档位数: {len(grouped_prices)}")
        print(f"电价范围: {min(all_prices):.3f} - {max(all_prices):.3f} 元/kWh")
        print(f"功率范围: {min(all_powers):.3f} - {max(all_powers):.3f} kW")
        print(f"平均功率: {np.mean(all_powers):.3f} kW")

def main():
    """主函数示例"""
    # 创建优化器实例
    optimizer = ACOptimizer(
        T=24,           # 24小时
        delta_t=1.0,    # 1小时时间步
        P_rated=12.0,    # 额定功率
        T_min=21.0,     # 最低温度21°C
        T_max=24.0,     # 最高温度24°C
        eta=0.98,       # 效率0.98
        R=3.0,          # 热阻3°C/kW
        C=1.8e7,        # 热容J∕°C
        T_initial=23.0  # 初始温度23°C
    )
    
    # 设置室外温度（直接使用24小时数据）
    original_temp = [
        28.0, 28.0, 28.0, 28.0, 28.0, 28.5, 29.0, 29.5, 
        30.0, 30.5, 31.0, 31.5, 32.0, 31.5, 31.0, 30.5, 
        29.0, 27.0, 26.0, 26.0, 27.0, 27.5, 28.0, 28.0, 28.0
    ]

    prices = [
    -1.0,  # X=0
    -0.75,  # X=1
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
    0.50,  # X=23
    ]
    
    # 直接使用24小时数据
    optimizer.set_outdoor_temperature(original_temp)
    
    # 设置电价
    optimizer.set_prices(prices)
    
    # 求解问题
    if optimizer.solve():
        optimizer.print_results()
        optimizer.plot_results()
    else:
        print(f"求解失败: {optimizer.status}")
    
    print("\n" + "="*60)
    print("开始生成电价-功率关系曲线...")
    print("="*60)
    
    # 生成并绘制合并的电价-功率关系图
    optimizer.plot_combined_price_power_curve(num_samples=100)
    
    # 可选：如果需要查看各个小时的单独图表，取消下面的注释
    # optimizer.plot_price_power_curves_all_hours(num_samples=50, hours_to_plot=[0, 1, 2, 3, 4, 5])

if __name__ == "__main__":
    main() 