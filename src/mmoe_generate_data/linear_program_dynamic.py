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
        all_prices_ref = self.prices # 参考原始电价序列的范围，而不是固定值
        min_price = min(all_prices_ref)
        max_price = max(all_prices_ref)

        # 如果所有原始电价相同，则需要手动设置一个合理的范围进行采样
        if min_price == max_price:
            print(f"警告：原始电价序列中所有电价相同 ({min_price:.3f} 元/kWh)。")
            print(f"将使用电价范围: {min_price - 0.5:.3f} 到 {max_price + 0.5:.3f} 元/kWh 进行采样。")
            min_price -= 0.5
            max_price += 0.5
            if min_price < -1.0: # 限制一下最低价
                 min_price = -1.0
            if max_price > 2.0: # 限制一下最高价
                 max_price = 2.0
        
        print(f"采样电价范围: {min_price:.3f} - {max_price:.3f} 元/kWh")
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
            
            sampled_prices_list = [] # 重命名以避免与外部prices冲突
            sampled_powers = []
            
            for i, price_sample_val in enumerate(price_samples): # 重命名迭代变量
                # 恢复原始电价
                self.prices = original_prices.copy()
                
                # 设置当前时刻的电价
                self.prices[hour] = price_sample_val
                
                # 求解优化问题
                if self.solve(): # solve内部会根据当前的self.prices计算dynamic_T_max_values
                    # 获取当前时刻的最优功率
                    P_t = self.optimal_powers[hour]
                    sampled_prices_list.append(price_sample_val)
                    sampled_powers.append(P_t)
                    
                    # 记录详细数据到CSV
                    if save_csv:
                        # 记录该时刻的所有相关数据
                        csv_row = {
                            'Hour': hour + 1,  # 1-24小时
                            'Sampled_Price': price_sample_val,
                            'AC_Power': P_t,
                            'Outdoor_Temperature': self.T_out[hour] if hour < len(self.T_out) else self.T_out[-1], # 使用对应小时的室外温度
                            'Indoor_Temperature': self.optimal_temperatures[hour + 1] if hour + 1 < len(self.optimal_temperatures) else self.optimal_temperatures[-1],
                            'Dynamic_T_max': self.dynamic_T_max_values[hour] # 记录当前采样价格下的动态T_max
                        }
                        csv_data.append(csv_row)
                
                # 显示进度
                if (i + 1) % (num_samples // 5) == 0 or (i+1) == num_samples : # 更频繁的进度更新
                    print(f"  已完成 {i + 1}/{num_samples} 个采样点")
            
            curves_data[hour] = (sampled_prices_list, sampled_powers)
            print(f"  第 {hour+1} 小时生成了 {len(sampled_prices_list)} 个有效数据点")
        
        # 恢复原始电价
        self.prices = original_prices
        self.solve() # 重新求解一次以恢复原始状态的 dynamic_T_max_values，用于后续可能的绘图
        
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
        curves_data = self.generate_price_power_curves_all_hours(num_samples, save_csv, csv_filename)
        
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
        
        for price_val, power_list_val in price_groups.items():
            grouped_prices.append(price_val)
            grouped_powers.append(sum(power_list_val) / len(power_list_val))  # 平均功率
        
        # 按电价从高到低排序（用于阶梯图）
        sorted_data = sorted(zip(grouped_prices, grouped_powers), reverse=True)
        sorted_prices = [x[0] for x in sorted_data]
        sorted_powers = [x[1] for x in sorted_data]
        
        # 绘制图形
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 绘制所有原始数据点（较小的点，透明度较低）
        ax.scatter(all_prices_list, all_powers_list, color='gray', s=10, alpha=0.2, label='原始数据点 (所有小时)')
        
        # 绘制分组后的平均值点
        ax.scatter(grouped_prices, grouped_powers, color='blue', s=40, alpha=0.8, label='分组平均值 ')
        
        # 使用step函数绘制阶梯状折线图
        ax.step(sorted_prices, sorted_powers, where='post', linewidth=2.5, 
                color='darkblue', alpha=0.9, label='阶梯状需求曲线 (平均)')
        
        ax.set_xlabel('控制信号 ', fontsize=12)
        ax.set_ylabel('功率 (kW)', fontsize=12)
        ax.set_title('所有时刻合并的控制信号-功率关系（阶梯状折线图）', fontsize=14)
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

def main():
    """主函数示例"""
    # 创建优化器实例
    optimizer = ACOptimizer(
        T=24,           # 24小时
        delta_t=1.0,    # 1小时时间步
        P_rated=12.0,    # 额定功率
        T_min=21.0,     # 最低温度21°C
        T_max=24.0,     # 基准最高温度24°C (现在是 T_max_base)
        T_max_price_sensitivity_factor=0.05, # 新增：当电价为1时，T_max降低约5%；电价为-1时，T_max升高约5%
        eta=0.98,       # 效率0.98
        R=3.0,          # 热阻3°C/kW
        C=1.8e7,        # 热容J∕°C
        T_initial=23.0  # 初始温度23°C
    )
    
    # 设置室外温度（直接使用25个数据点，对应T=0到T=24）
    original_temp = [
        28.0, 28.0, 28.0, 28.0, 28.0, 28.5, 29.0, 29.5, 
        30.0, 30.5, 31.0, 31.5, 32.0, 31.5, 31.0, 30.5, 
        29.0, 27.0, 26.0, 26.0, 27.0, 27.5, 28.0, 28.0, 28.0 
    ] # 确保有 T+1 个数据点

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
    
    optimizer.set_outdoor_temperature(original_temp)
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
    optimizer.plot_combined_price_power_curve(num_samples=50) # 减少样本数以加快测试
    
    # 可选：如果需要查看各个小时的单独图表，取消下面的注释
    # optimizer.plot_price_power_curves_all_hours(num_samples=30, hours_to_plot=[0, 6, 12, 18]) # 减少样本数并选择部分小时

if __name__ == "__main__":
    main() 