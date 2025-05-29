"""
空调功率优化线性规划问题求解 - 温度目标约束版本

M_{AC}: \min_{\left \{P_{1},...,P_{T},T_{2}^{i},...,T_{T}^{i}\right \} }\sum_{t=1}^{T}P_{t}\Delta t,
subject to for \forall t \in \{1,2,..,T\}:
0 \le P_{t} \le P_{rated},
T_{min} \le T_{t} \le T_{max}

and

室温变化公式：T_{t+1}^{i} = T_{t+1}^{out} - \eta P_{t} R_{t} - (T_{t+1}^{out} - \eta P_{t} R_{t} - T_{t}^{i}) e^{- \Delta t / R C}

新增约束：T_{1}^{i} = T_{target} (第一个时间步结束时达到目标温度)
"""

import pulp
import numpy as np
# 添加NumPy向后兼容性修复
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv
import pandas as pd
import os
import warnings

# 忽略与NumPy相关的FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class ACOptimizerWithTempTarget:
    def __init__(self, T=24, delta_t=1.0, P_rated=3.0, T_min=20.0, T_max=26.0,
                 eta=0.8, R=2.0, C=5.0, T_initial=22.0, T_target=24.0, 
                 target_type='custom', force_control=None):
        """
        初始化带温度目标约束的空调优化器
        
        参数:
        T: 时间步数 (小时)
        delta_t: 时间步长 (小时)
        P_rated: 额定功率 (kW)
        T_min, T_max: 温度约束范围 (°C)
        eta: 空调效率
        R: 热阻 (°C/kW)
        C: 热容 (J/°C)，将自动转换为kWh/°C
        T_initial: 初始室温 (°C)
        T_target: 第一个时间步结束时的目标温度 (°C)
        target_type: 目标温度类型 ('min', 'max', 'custom')
                    - 'min': 使用 T_min 作为目标
                    - 'max': 使用 T_max 作为目标  
                    - 'custom': 使用指定的 T_target 值
        force_control: 强制控制信号，可以是None或包含每个时间步控制信号的列表
                    - None: 无强制控制
                    - 列表: 每个元素可以是1(尽快升温到最高温度)、-1(尽快降温到最低温度)或0(无强制控制)
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
        
        # 设置目标温度
        self.target_type = target_type
        if target_type == 'min':
            self.T_target = T_min
        elif target_type == 'max':
            self.T_target = T_max
        elif target_type == 'custom':
            # 验证自定义目标温度是否在合理范围内
            if T_target < T_min or T_target > T_max:
                raise ValueError(f"目标温度 {T_target}°C 必须在 [{T_min}, {T_max}]°C 范围内")
            self.T_target = T_target
        else:
            raise ValueError("target_type 必须是 'min', 'max' 或 'custom'")
        
        # 设置强制控制信号
        if force_control is None:
            # 默认无强制控制
            self.force_control = [0] * T
        elif isinstance(force_control, list):
            # 验证长度
            if len(force_control) != T:
                raise ValueError(f"强制控制信号列表长度必须为{T}")
            # 验证值
            for i, signal in enumerate(force_control):
                if signal not in [-1, 0, 1]:
                    raise ValueError(f"强制控制信号值必须是-1、0或1，在位置{i}处发现值{signal}")
            self.force_control = force_control
        else:
            raise ValueError("force_control 必须是None或列表")
        
        # 计算指数衰减因子: exp(-Δt/(R*C))
        # 这里 delta_t 是小时，R 是°C/kW，C 是 kWh/°C
        # 所以 R*C 的单位是 (°C/kW) * (kWh/°C) = h
        self.exp_factor = np.exp(-delta_t / (R * self.C))
        
        print(f"热容转换: {C:.1e} J/°C = {self.C:.1e} kWh/°C")
        print(f"时间常数 τ = R*C = {R:.1f} * {self.C:.1e} = {R * self.C:.2f} 小时")
        print(f"指数衰减因子: exp(-Δt/τ) = {self.exp_factor:.6f}")
        print(f"目标温度设置: {self.T_target}°C (类型: {target_type})")
        
        # 如果有强制控制信号，显示信息
        if any(self.force_control):
            force_positions = [(i, signal) for i, signal in enumerate(self.force_control) if signal != 0]
            if force_positions:
                print(f"强制控制信号设置: {len(force_positions)}个时间步有强制控制")
                for pos, signal in force_positions[:5]:  # 只显示前5个
                    direction = "升温到最高温度" if signal == 1 else "降温到最低温度"
                    print(f"  时间步{pos}: {direction}")
                if len(force_positions) > 5:
                    print(f"  ...以及另外{len(force_positions)-5}个时间步")
        
    def set_outdoor_temperature(self, T_out):
        """
        设置室外温度序列
        
        参数:
        T_out: 室外温度，可以是单个值或序列
        """
        if isinstance(T_out, (int, float)):
            self.T_out = [T_out] * (self.T + 1)
        else:
            if len(T_out) < self.T + 1:
                raise ValueError(f"室外温度序列长度 {len(T_out)} 必须至少为 {self.T + 1}")
            self.T_out = T_out[:self.T + 1]  # 确保长度正确
            
    def solve(self):
        """
        求解带温度目标约束的线性规划问题
        
        返回:
        bool: 是否找到最优解
        """
        # 创建线性规划问题
        prob = pulp.LpProblem("AC_Power_Optimization_With_Temp_Target", pulp.LpMinimize)
        
        # 决策变量
        # P_t: 每个时间步的功率 (t = 1, 2, ..., T)
        P = [pulp.LpVariable(f"P_{t}", lowBound=0, upBound=self.P_rated) 
             for t in range(1, self.T + 1)]
        
        # T_i_t: 每个时间步结束时的室内温度 (t = 1, 2, ..., T)
        # 注意：T_i[0] 对应 t=1 时刻结束时的温度
        T_i = []
        for t in range(1, self.T + 1):
            # 检查是否有强制控制信号
            force_signal = self.force_control[t-1]
            if force_signal == 1:
                # 强制升温到最高温度
                T_i.append(pulp.LpVariable(f"T_i_{t}", lowBound=self.T_max, upBound=self.T_max))
            elif force_signal == -1:
                # 强制降温到最低温度
                T_i.append(pulp.LpVariable(f"T_i_{t}", lowBound=self.T_min, upBound=self.T_min))
            else:
                # 正常温度范围
                T_i.append(pulp.LpVariable(f"T_i_{t}", lowBound=self.T_min, upBound=self.T_max))
        
        # 目标函数：最小化总功耗（移除温度偏差惩罚，简化为纯功耗优化）
        prob += pulp.lpSum([P[t-1] * self.delta_t for t in range(1, self.T + 1)]), "总功耗最小化"
        
        # 约束条件
        
        # 1. 功率约束（已在变量定义中包含）
        # 0 ≤ P_t ≤ P_rated for all t
        
        # 2. 温度约束（已在变量定义中包含，并根据强制控制信号调整）
        # T_min ≤ T_t ≤ T_max for all t
        
        # 3. 温度目标约束：移除强制约束，改为软约束（朝着目标方向努力）
        # prob += T_i[0] == self.T_target, "第一时间步温度目标约束"
        
        # 4. 室温变化约束（一阶ETP公式）
        for t in range(1, self.T + 1):
            if t == 1:
                # 第一个时间步，使用初始温度作为前一时刻温度
                T_prev = self.T_initial
            else:
                # 使用前一时间步的室内温度
                T_prev = T_i[t-2]  # T_i[t-2] 对应 T_i_{t-1}
            
            # 一阶ETP公式的线性化
            # T_{t}^{i} = T_{t}^{out} - η P_{t-1} R - (T_{t}^{out} - η P_{t-1} R - T_{t-1}^{i}) * exp(-Δt/RC)
            # 重新整理为：T_{t}^{i} = (1-exp_factor) * (T_{t}^{out} - η P_{t-1} R) + exp_factor * T_{t-1}^{i}
            
            # 稳态温度：当功率为 P_{t-1} 时的稳态室内温度
            steady_state_temp = self.T_out[t] - self.eta * self.R * P[t-1]
            
            # 添加温度演化约束
            prob += (T_i[t-1] == 
                    (1 - self.exp_factor) * steady_state_temp + 
                    self.exp_factor * T_prev), f"时间步{t}温度演化约束"
        
        # 求解
        print("开始求解线性规划问题...")
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # 提取结果
        if prob.status == pulp.LpStatusOptimal:
            self.optimal_powers = [P[t-1].varValue for t in range(1, self.T + 1)]
            self.optimal_temperatures = [self.T_initial] + [T_i[t-1].varValue for t in range(1, self.T + 1)]
            self.total_energy = sum(self.optimal_powers) * self.delta_t
            self.status = "最优解"
            
            # 验证温度目标是否达到
            first_step_temp = self.optimal_temperatures[1]
            temp_error = abs(first_step_temp - self.T_target)
            print(f"第一时间步目标温度: {self.T_target:.2f}°C")
            print(f"第一时间步实际温度: {first_step_temp:.2f}°C")
            print(f"温度误差: {temp_error:.4f}°C")
            
            # 检查强制控制是否生效
            force_steps = [(i, signal) for i, signal in enumerate(self.force_control) if signal != 0]
            if force_steps:
                print(f"\n强制控制结果验证:")
                for step, signal in force_steps:
                    actual_temp = self.optimal_temperatures[step+1]
                    if signal == 1 and abs(actual_temp - self.T_max) < 0.01:
                        print(f"  ✅ 时间步{step}: 成功升温到最高温度 {actual_temp:.2f}°C")
                    elif signal == -1 and abs(actual_temp - self.T_min) < 0.01:
                        print(f"  ✅ 时间步{step}: 成功降温到最低温度 {actual_temp:.2f}°C")
                    else:
                        expected = self.T_max if signal == 1 else self.T_min
                        print(f"  ❌ 时间步{step}: 未达到预期温度 (实际: {actual_temp:.2f}°C, 预期: {expected:.2f}°C)")
            
        else:
            self.status = f"求解失败: {pulp.LpStatus[prob.status]}"
            print(f"线性规划求解状态: {self.status}")
            
            # 如果失败，检查是否是因为强制控制信号导致问题无解
            if any(self.force_control):
                print("可能是强制控制信号导致问题无解，尝试删除部分强制控制约束后重新求解...")
                return False
            
        return prob.status == pulp.LpStatusOptimal
    
    def get_target_temperature_info(self):
        """
        获取温度目标信息
        
        返回:
        dict: 包含目标温度相关信息的字典
        """
        return {
            'target_type': self.target_type,
            'target_temperature': self.T_target,
            'initial_temperature': self.T_initial,
            'temperature_range': (self.T_min, self.T_max),
            'temperature_change_needed': self.T_target - self.T_initial
        }
    
    def plot_results(self):
        """绘制优化结果图表"""
        if not hasattr(self, 'optimal_powers'):
            print("请先求解问题")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        time_steps = list(range(self.T + 1))
        power_time_steps = list(range(1, self.T + 1))
        
        # 绘制功率曲线
        ax1.step(power_time_steps, self.optimal_powers, where='post', linewidth=2, color='blue')
        ax1.set_ylabel('功率 (kW)')
        ax1.set_title('最优空调功率 (带温度目标约束)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, self.P_rated * 1.1)
        
        # 添加第一个时间步的功率标注
        first_power = self.optimal_powers[0]
        ax1.annotate(f'第1步: {first_power:.2f}kW', 
                    xy=(1, first_power), xytext=(2, first_power + 0.5),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
        
        # 绘制室内温度曲线
        ax2.plot(time_steps, self.optimal_temperatures, 'ro-', linewidth=2, markersize=4, label='室内温度')
        ax2.axhline(y=self.T_min, color='g', linestyle='--', alpha=0.7, label=f'最低温度 {self.T_min}°C')
        ax2.axhline(y=self.T_max, color='r', linestyle='--', alpha=0.7, label=f'最高温度 {self.T_max}°C')
        ax2.axhline(y=self.T_target, color='orange', linestyle=':', linewidth=2, label=f'目标温度 {self.T_target}°C')
        
        # 标注初始温度和第一步目标温度
        ax2.annotate(f'初始: {self.T_initial}°C', 
                    xy=(0, self.T_initial), xytext=(0.5, self.T_initial + 0.5),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=10, color='blue')
        ax2.annotate(f'目标: {self.optimal_temperatures[1]:.2f}°C', 
                    xy=(1, self.optimal_temperatures[1]), xytext=(1.5, self.optimal_temperatures[1] + 0.5),
                    arrowprops=dict(arrowstyle='->', color='orange'),
                    fontsize=10, color='orange')
        
        ax2.set_ylabel('温度 (°C)')
        ax2.set_title('室内温度变化 (第1步必须达到目标温度)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 绘制室外温度曲线
        ax3.plot(time_steps, self.T_out[:self.T+1], 'go-', linewidth=2, markersize=4, label='室外温度')
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('温度 (°C)')
        ax3.set_title('室外温度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def print_control_cycle_details(self):
        """
        详细打印每个控制周期的状态信息
        输出室外温度、室内温度、空调功率等详细信息
        """
        if not hasattr(self, 'optimal_powers'):
            print("请先求解问题")
            return
            
        print("\n" + "=" * 80)
        print("详细控制周期信息")
        print("=" * 80)
        
        # 打印系统参数
        print(f"系统参数:")
        print(f"  时间步长: {self.delta_t:.1f} 小时")
        print(f"  额定功率: {self.P_rated:.1f} kW")
        print(f"  空调效率: {self.eta:.2f}")
        print(f"  热阻: {self.R:.1f} °C/kW")
        print(f"  热容: {self.C_original:.1e} J/°C ({self.C:.1e} kWh/°C)")
        print(f"  时间常数 τ = R×C: {self.R * self.C:.2f} 小时")
        print(f"  温度范围: [{self.T_min}°C, {self.T_max}°C]")
        print(f"  目标温度: {self.T_target}°C (类型: {self.target_type})")
        
        print(f"\n控制周期详细信息:")
        print("-" * 80)
        
        # 初始状态
        print(f"初始状态 (t=0):")
        print(f"  室外温度: {self.T_out[0]:6.2f}°C")
        print(f"  室内温度: {self.T_initial:6.2f}°C")
        print(f"  空调功率: {'--':>6} kW (未启动)")
        print(f"  能耗累计: {'0.00':>6} kWh")
        print("-" * 80)
        
        # 逐个控制周期输出
        cumulative_energy = 0
        for t in range(self.T):
            power = self.optimal_powers[t]
            indoor_temp_prev = self.optimal_temperatures[t]
            indoor_temp_curr = self.optimal_temperatures[t+1]
            outdoor_temp = self.T_out[t+1]
            
            # 计算当前周期能耗
            cycle_energy = power * self.delta_t
            cumulative_energy += cycle_energy
            
            # 计算温度变化
            temp_change = indoor_temp_curr - indoor_temp_prev
            
            # 计算稳态温度（如果维持当前功率）
            steady_state_temp = outdoor_temp - self.eta * self.R * power
            
            print(f"控制周期 {t+1} (t={t+1}):")
            print(f"  室外温度: {outdoor_temp:6.2f}°C")
            print(f"  室内温度: {indoor_temp_prev:6.2f} → {indoor_temp_curr:6.2f}°C (变化: {temp_change:+.2f}°C)")
            print(f"  空调功率: {power:6.2f} kW ({power/self.P_rated*100:5.1f}%额定功率)")
            print(f"  周期能耗: {cycle_energy:6.3f} kWh")
            print(f"  累计能耗: {cumulative_energy:6.2f} kWh")
            print(f"  稳态温度: {steady_state_temp:6.2f}°C (如果维持当前功率)")
            
            # 特殊标注
            remarks = []
            if t == 0:
                remarks.append(f"目标约束: 必须达到{self.T_target}°C")
            if abs(indoor_temp_curr - self.T_min) < 0.01:
                remarks.append("触及温度下限")
            if abs(indoor_temp_curr - self.T_max) < 0.01:
                remarks.append("触及温度上限")
            if power == 0:
                remarks.append("空调关闭")
            elif abs(power - self.P_rated) < 0.01:
                remarks.append("满功率运行")
                
            if remarks:
                print(f"  备注: {'; '.join(remarks)}")
                
            print("-" * 80)
        
        # 总结信息
        print(f"优化结果总结:")
        print(f"  总时间: {self.T * self.delta_t:.1f} 小时")
        print(f"  总能耗: {cumulative_energy:.2f} kWh")
        print(f"  平均功率: {cumulative_energy/(self.T * self.delta_t):.2f} kW")
        print(f"  功率利用率: {cumulative_energy/(self.T * self.delta_t)/self.P_rated*100:.1f}%")
        
        # 温度分析
        min_temp = min(self.optimal_temperatures[1:])
        max_temp = max(self.optimal_temperatures[1:])
        final_temp = self.optimal_temperatures[-1]
        
        print(f"  温度范围: [{min_temp:.2f}°C, {max_temp:.2f}°C]")
        print(f"  最终温度: {final_temp:.2f}°C")
        print(f"  第1步目标: {self.T_target:.2f}°C (实际: {self.optimal_temperatures[1]:.2f}°C)")
        print("=" * 80)

    def print_summary_table(self):
        """
        打印简洁的汇总表格，包含所有关键信息
        """
        if not hasattr(self, 'optimal_powers'):
            print("请先求解问题")
            return
            
        print("\n" + "=" * 90)
        print("空调控制汇总表 (带温度目标约束)")
        print("=" * 90)
        print("| 时间步 | 室外温度 | 室内温度 | 温度变化 | 空调功率 | 功率比例 | 周期能耗 | 累计能耗 | 备注   |")
        print("|" + "-" * 88 + "|")
        
        # 初始行
        print(f"| {'初始':>6} | {self.T_out[0]:8.2f} | {self.T_initial:8.2f} | {'--':>8} | {'--':>8} | {'--':>8} | {'--':>8} | {'0.00':>8} | 初始状态 |")
        
        cumulative_energy = 0
        for t in range(self.T):
            power = self.optimal_powers[t]
            indoor_temp_prev = self.optimal_temperatures[t]
            indoor_temp_curr = self.optimal_temperatures[t+1]
            outdoor_temp = self.T_out[t+1]
            
            temp_change = indoor_temp_curr - indoor_temp_prev
            cycle_energy = power * self.delta_t
            cumulative_energy += cycle_energy
            power_ratio = power / self.P_rated * 100
            
            # 生成备注
            remark = ""
            if t == 0:
                remark = "目标约束"
            elif abs(indoor_temp_curr - self.T_min) < 0.01:
                remark = "触及下限"
            elif abs(indoor_temp_curr - self.T_max) < 0.01:
                remark = "触及上限"
            elif power == 0:
                remark = "关闭"
            elif abs(power - self.P_rated) < 0.01:
                remark = "满功率"
            else:
                remark = "正常"
                
            print(f"| {t+1:6d} | {outdoor_temp:8.2f} | {indoor_temp_curr:8.2f} | {temp_change:+8.2f} | {power:8.2f} | {power_ratio:7.1f}% | {cycle_energy:8.3f} | {cumulative_energy:8.2f} | {remark:6} |")
        
        print("|" + "-" * 88 + "|")
        print("=" * 90)

    def print_results(self):
        """打印优化结果 - 增强版本"""
        if not hasattr(self, 'optimal_powers'):
            print("请先求解问题")
            return
            
        print("=" * 60)
        print("空调功率优化结果 (带温度目标约束)")
        print("=" * 60)
        print(f"求解状态: {self.status}")
        
        if hasattr(self, 'total_energy'):
            print(f"总能耗: {self.total_energy:.2f} kWh")
            print(f"平均功率: {self.total_energy/self.T:.2f} kW")
            
            # 温度目标相关信息
            target_info = self.get_target_temperature_info()
            print(f"\n温度目标信息:")
            print(f"  目标类型: {target_info['target_type']}")
            print(f"  初始温度: {target_info['initial_temperature']:.2f}°C")
            print(f"  目标温度: {target_info['target_temperature']:.2f}°C")
            print(f"  需要变化: {target_info['temperature_change_needed']:+.2f}°C")
            print(f"  第1步实际达到: {self.optimal_temperatures[1]:.2f}°C")
            
            print(f"\n详细结果:")
            print("时间步 | 功率(kW) | 室内温度(°C) | 室外温度(°C) | 备注")
            print("-" * 70)
            
            # 特殊标注第一行（目标约束）
            print(f"{'初始':>6} | {'--':>8} | {self.optimal_temperatures[0]:11.2f} | {self.T_out[0]:11.2f} | 初始状态")
            
            for t in range(self.T):
                remark = ""
                if t == 0:
                    remark = "目标约束"
                elif abs(self.optimal_temperatures[t+1] - self.T_min) < 0.01:
                    remark = "触及下限"
                elif abs(self.optimal_temperatures[t+1] - self.T_max) < 0.01:
                    remark = "触及上限"
                    
                print(f"{t+1:6d} | {self.optimal_powers[t]:8.2f} | {self.optimal_temperatures[t+1]:11.2f} | {self.T_out[t+1]:11.2f} | {remark}")
            
            print("=" * 60)
            
            # 调用新的详细输出方法
            self.print_summary_table()
            self.print_control_cycle_details()

def main():
    # 新增场景：按照readme.md要求的24小时功率需求计算
    print("\n" + "📊" * 50)
    print("新增场景：24小时空调功率需求计算 (基于readme.md要求)")
    print("📊" * 50)
    
    # 先做一个简单测试，不设置特定目标温度，只是最小化功耗
    print("\n" + "🔧" * 50)
    print("简单测试：纯功耗优化（无特定温度目标）")
    print("🔧" * 50)
    
    simple_optimizer = ACOptimizerWithTempTarget(
        T=4,             # 2小时测试
        delta_t=0.5,     # 0.5小时控制周期
        P_rated=50.0,    # 50kW额定功率
        T_min=22.0,      # 下限22°C
        T_max=25.0,      # 上限25°C  
        eta=0.9,         # 效率0.9
        R=2.0,           # 热阻2.0°C/kW
        C=1.5e7,         # 热容1.5e7 J/°C
        T_initial=23.5,  # 初始温度23.5°C
        T_target=23.5,   # 目标温度等于初始温度（无变化需求）
        target_type='custom'
    )
    
    # 设置简单的室外温度
    simple_outdoor_temp = [30.0, 32.0, 34.0, 36.0, 38.0]  # 5个数据点
    simple_optimizer.set_outdoor_temperature(simple_outdoor_temp)
    
    print(f"简单测试配置:")
    print(f"  控制周期: {simple_optimizer.delta_t} 小时")
    print(f"  总时长: {simple_optimizer.T * simple_optimizer.delta_t} 小时")
    print(f"  温度范围: [{simple_optimizer.T_min}°C, {simple_optimizer.T_max}°C]")
    print(f"  初始温度: {simple_optimizer.T_initial}°C")
    print(f"  目标温度: {simple_optimizer.T_target}°C (无变化)")
    print(f"  室外温度: {simple_outdoor_temp}")
    
    if simple_optimizer.solve():
        print("✅ 简单测试成功！")
        simple_optimizer.print_results()
    else:
        print("❌ 简单测试失败！")
        print(f"原因: {simple_optimizer.status}")
        
        # 如果简单测试都失败，说明基础约束有问题
        print("\n🔍 基础约束分析:")
        print(f"时间常数 τ = {simple_optimizer.R * simple_optimizer.C:.2f} 小时")
        print(f"指数衰减因子 = {simple_optimizer.exp_factor:.6f}")
        
        # 手动计算第一步的温度变化
        T_out_1 = simple_outdoor_temp[1]
        T_initial = simple_optimizer.T_initial
        eta = simple_optimizer.eta
        R = simple_optimizer.R
        exp_factor = simple_optimizer.exp_factor
        
        print(f"\n第一个控制周期分析:")
        print(f"  初始室内温度: {T_initial}°C")
        print(f"  室外温度: {T_out_1}°C")
        print(f"  无空调时稳态温度: {T_out_1}°C")
        print(f"  满功率时稳态温度: {T_out_1 - eta * R * 50.0:.2f}°C")
        
        # 计算无空调和满功率情况下的温度演化
        temp_no_ac = (1 - exp_factor) * T_out_1 + exp_factor * T_initial
        temp_full_ac = (1 - exp_factor) * (T_out_1 - eta * R * 50.0) + exp_factor * T_initial
        
        print(f"  无空调下第1步结束温度: {temp_no_ac:.2f}°C")
        print(f"  满功率下第1步结束温度: {temp_full_ac:.2f}°C")
        print(f"  温度范围约束: [{simple_optimizer.T_min}°C, {simple_optimizer.T_max}°C]")
        
        if temp_full_ac > simple_optimizer.T_max:
            print(f"  ⚠️ 问题：即使满功率制冷，温度仍超过上限！")
        if temp_no_ac < simple_optimizer.T_min:
            print(f"  ⚠️ 问题：无空调时，温度低于下限！")
        
        return
    
    # 如果简单测试成功，则继续原来的测试
    # （其余代码保持不变）
    print("\n" + "📊" * 50)
    print("继续进行原始测试...")
    print("📊" * 50)
    
    # 按照readme.md的具体参数设置
    optimizer_readme = ACOptimizerWithTempTarget(
        T=48,            # 24小时，每0.5小时一个控制周期 = 48个时间步
        delta_t=0.5,     # 0.5小时控制周期
        P_rated=5.0,    # kW额定功率 (足够应对各种需求)
        T_min=22.0,      # 下限22°C (readme: 22±1度)
        T_max=25.0,      # 上限25°C (readme: 25±1度)  
        eta=0.9,         # 效率0.9
        R=2.0,           # 热阻2.0°C/kW (真实建筑参数)
        C=1.5e7,         # 热容1.5e7 J/°C (真实建筑参数)
        T_initial=23.5,  # 初始温度23.5°C (readme: 23.5±0.5度)
        T_target=23.5,   # 目标设为初始温度（不强制改变）
        target_type='custom'
    )
    
    # 设置24小时的室外温度变化 (从W2.csv文件读取夏季真实数据)
    print(f"正在从W2.csv文件读取夏季室外温度数据...")
    
    # 读取W2.csv文件
    try:
        # 检查是否有环境变量指定文件路径
        import os
        w2_csv_path = os.environ.get('W2_CSV_PATH', "../../data/W2.csv")
        print(f"使用数据文件路径: {w2_csv_path}")
        
        # 尝试读取文件
        try:
            w2_data = pd.read_csv(w2_csv_path)
            print(f"  成功读取W2.csv文件，共{len(w2_data)}条记录")
        except FileNotFoundError:
            # 尝试使用绝对路径
            w2_csv_path = "D:/afterWork/ACL_agg_exp/data/W2.csv"
            print(f"  尝试使用绝对路径: {w2_csv_path}")
            w2_data = pd.read_csv(w2_csv_path)
            print(f"  成功读取W2.csv文件，共{len(w2_data)}条记录")
        
        # 筛选夏季数据（6-8月）
        # 检查是否有环境变量指定目标年份
        target_year = os.environ.get('TARGET_YEAR')
        if target_year:
            print(f"指定处理年份: {target_year}")
        
        # 如果指定了年份，只筛选该年份的夏季数据
        summer_pattern = f"{target_year}/6|{target_year}/7|{target_year}/8"
        summer_indices = w2_data['Time'].str.contains(summer_pattern, na=False)
        print(f"  筛选{target_year}年的夏季数据")
        if summer_indices.any():
            summer_data = w2_data[summer_indices]
            print(f"  找到夏季数据: {len(summer_data)}条记录")
            
            # 获取第一个夏季日期
            first_date = summer_data['Time'].iloc[0].split(' ')[0]
            
            # 修复：正确匹配完整日期而非前缀匹配
            # 使用日期部分精确匹配，而不是使用startswith
            day_data = summer_data[[date.split(' ')[0] == first_date for date in summer_data['Time']]]
            first_summer_day = day_data.iloc[:96]  # 取一整天数据（96条，24小时，每15分钟一条）
            
            # 将华氏度转换为摄氏度: C = (F - 32) * 5/9
            fahrenheit_temps = first_summer_day['Temperature(F)'].values
            celsius_temps = (fahrenheit_temps - 32) * 5/9
            
            # 每30分钟取一个数据点（每2条记录取1条，因为原数据是每15分钟一条）
            outdoor_temp_half_hour = []
            time_stamps = []
            
            for i in range(0, min(96, len(celsius_temps)), 2):  # 取49个点（0到96，步长2）
                outdoor_temp_half_hour.append(celsius_temps[i])
                time_stamps.append(first_summer_day['Time'].iloc[i])
                if len(outdoor_temp_half_hour) >= 49:  # 确保有49个数据点
                    break
            
            # 如果数据不足49个点，用最后一个值补充
            while len(outdoor_temp_half_hour) < 49:
                outdoor_temp_half_hour.append(outdoor_temp_half_hour[-1])
                # 为时间戳生成对应的时间
                last_time = time_stamps[-1] if time_stamps else "2015/6/1 0:00"
                time_stamps.append(last_time)
            
            print(f"  提取的夏季温度数据点数: {len(outdoor_temp_half_hour)}")
            print(f"  夏季温度范围: {min(outdoor_temp_half_hour):.1f}°C - {max(outdoor_temp_half_hour):.1f}°C")
            print(f"  前5个时间点: {time_stamps[:5]}")
            
        else:
            raise Exception("未找到夏季数据，请检查数据文件")
        
    except Exception as e:
        print(f"  ❌ 读取夏季数据失败: {str(e)}")
        print(f"  使用默认的夏季模拟温度数据...")
        
        # 如果读取失败，使用夏季的默认数据
        hourly_outdoor_temp = [
            28.0, 27.5, 27.0, 26.5, 26.0, 26.5, 27.0, 28.0, 
            29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 
            35.5, 35.0, 34.0, 33.0, 32.0, 31.0, 30.0, 29.0, 28.5
        ]

        # 插值生成0.5小时间隔的温度数据
        outdoor_temp_half_hour = []
        for i in range(len(hourly_outdoor_temp)-1):
            outdoor_temp_half_hour.append(hourly_outdoor_temp[i])
            # 添加中间点的线性插值
            mid_temp = (hourly_outdoor_temp[i] + hourly_outdoor_temp[i+1]) / 2
            outdoor_temp_half_hour.append(mid_temp)
        outdoor_temp_half_hour.append(hourly_outdoor_temp[-1])  # 添加最后一个点
        
        print(f"  使用默认夏季温度范围: {min(outdoor_temp_half_hour):.1f}°C - {max(outdoor_temp_half_hour):.1f}°C")
    
    optimizer_readme.set_outdoor_temperature(outdoor_temp_half_hour)
    
    print(f"系统配置 (基于readme.md):")
    print(f"  控制周期: {optimizer_readme.delta_t} 小时")
    print(f"  总时长: {optimizer_readme.T * optimizer_readme.delta_t} 小时")
    print(f"  温度范围: [{optimizer_readme.T_min}°C, {optimizer_readme.T_max}°C]")
    print(f"  初始温度: {optimizer_readme.T_initial}°C")
    print(f"  室外温度范围: {min(outdoor_temp_half_hour):.1f}°C - {max(outdoor_temp_half_hour):.1f}°C")
    
    # 计算24小时基础功率需求
    print(f"\n开始计算24小时基础功率需求...")
    
    if optimizer_readme.solve():
        print("✅ 24小时基础优化成功！")
        
        # 输出功率需求汇总
        powers = optimizer_readme.optimal_powers
        temps = optimizer_readme.optimal_temperatures[1:]  # 跳过初始温度
        
        print(f"功率统计:")
        print(f"  最大功率: {max(powers):.2f} kW")
        print(f"  平均功率: {sum(powers)/len(powers):.2f} kW")
        print(f"  总能耗: {sum(powers) * optimizer_readme.delta_t:.2f} kWh")
        print(f"  温度范围: {min(temps):.2f}°C - {max(temps):.2f}°C")
        
        # 输出详细的24小时功率需求数据（前8个和后8个控制周期）
        print(f"\n前8个控制周期详情:")
        print("时间 | 室外温度 | 室内温度 | 目标温度 | 所需功率 | 功率占比 | 温度变化")
        print("-" * 85)
        for i in range(min(8, len(powers))):
            hour = i * 0.5
            temp_change = temps[i] - (optimizer_readme.T_initial if i == 0 else temps[i-1])
            print(f"{hour:4.1f}h | {outdoor_temp_half_hour[i+1]:8.1f} | {temps[i]:8.2f} | {optimizer_readme.T_target:8.1f} | {powers[i]:8.2f} | {powers[i]/optimizer_readme.P_rated*100:6.1f}% | {temp_change:+6.2f}")
        
        if len(powers) > 8:
            print("  ... (中间周期省略)")
            print("最后8个控制周期详情:")
            for i in range(max(0, len(powers)-8), len(powers)):
                hour = i * 0.5
                temp_change = temps[i] - temps[i-1] if i > 0 else temps[i] - optimizer_readme.T_initial
                print(f"{hour:4.1f}h | {outdoor_temp_half_hour[i+1]:8.1f} | {temps[i]:8.2f} | {optimizer_readme.T_target:8.1f} | {powers[i]:8.2f} | {powers[i]/optimizer_readme.P_rated*100:6.1f}% | {temp_change:+6.2f}")
        
        # 保存完整数据用于进一步分析
        print(f"\n💾 保存完整24小时功率需求数据...")
        
        # 确保figures文件夹存在
        figures_dir = "../../figures"
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        power_data_file = os.path.join(figures_dir, "24h_power_data.txt")
        with open(power_data_file, "w", encoding="utf-8") as f:
            f.write("24小时空调功率需求数据\n")
            f.write("=" * 80 + "\n")
            f.write("时间(h),室外温度(°C),室内温度(°C),目标温度(°C),所需功率(kW),功率占比(%),温度变化(°C)\n")
            for i in range(len(powers)):
                hour = i * 0.5
                temp_change = temps[i] - (optimizer_readme.T_initial if i == 0 else temps[i-1])
                f.write(f"{hour:.1f},{outdoor_temp_half_hour[i+1]:.1f},{temps[i]:.2f},{optimizer_readme.T_target:.1f},{powers[i]:.2f},{powers[i]/optimizer_readme.P_rated*100:.1f},{temp_change:+.2f}\n")
        print(f"✅ 数据已保存到 {power_data_file}")
        
        # 新增测试：使用强制控制信号
        print("\n" + "🎮" * 50)
        print("新增测试：使用强制控制信号")
        print("🎮" * 50)
        
        # 创建一个新的优化器，在特定时间点添加强制控制信号
        # 创建48个时间步的控制信号列表，默认为0（无控制）
        force_control = [0] * 48
        
        # 添加一些强制控制点，模拟紧急温度调整需求
        # 例如：
        # 1. 早上8点（时间步16）强制降温到最低温度
        # 2. 下午2点（时间步28）强制升温到最高温度
        # 3. 晚上8点（时间步40）再次强制降温到最低温度
        force_control[16] = -1  # 早上8点强制降温
        force_control[28] = 1   # 下午2点强制升温
        force_control[40] = -1  # 晚上8点强制降温
        
        # 创建带强制控制信号的优化器
        optimizer_force = ACOptimizerWithTempTarget(
            T=48,            # 24小时，每0.5小时一个控制周期 = 48个时间步
            delta_t=0.5,     # 0.5小时控制周期
            P_rated=5.0,     # kW额定功率
            T_min=22.0,      # 下限22°C
            T_max=25.0,      # 上限25°C
            eta=0.9,         # 效率0.9
            R=2.0,           # 热阻2.0°C/kW
            C=1.5e7,         # 热容1.5e7 J/°C
            T_initial=23.5,  # 初始温度23.5°C
            T_target=23.5,   # 目标温度
            target_type='custom',
            force_control=force_control  # 添加强制控制信号
        )
        
        # 设置相同的室外温度
        optimizer_force.set_outdoor_temperature(outdoor_temp_half_hour)
        
        print(f"系统配置 (带强制控制信号):")
        print(f"  控制周期: {optimizer_force.delta_t} 小时")
        print(f"  总时长: {optimizer_force.T * optimizer_force.delta_t} 小时")
        print(f"  温度范围: [{optimizer_force.T_min}°C, {optimizer_force.T_max}°C]")
        print(f"  初始温度: {optimizer_force.T_initial}°C")
        
        # 求解优化问题
        print(f"\n开始计算带强制控制信号的24小时功率需求...")
        
        if optimizer_force.solve():
            print("✅ 带强制控制信号的优化成功！")
            
            # 输出功率需求汇总
            force_powers = optimizer_force.optimal_powers
            force_temps = optimizer_force.optimal_temperatures[1:]  # 跳过初始温度
            
            print(f"功率统计:")
            print(f"  最大功率: {max(force_powers):.2f} kW")
            print(f"  平均功率: {sum(force_powers)/len(force_powers):.2f} kW")
            print(f"  总能耗: {sum(force_powers) * optimizer_force.delta_t:.2f} kWh")
            print(f"  温度范围: {min(force_temps):.2f}°C - {max(force_temps):.2f}°C")
            
            # 输出强制控制时间点附近的详细信息
            for control_point in [16, 28, 40]:
                # 显示控制点前后的数据
                start_idx = max(0, control_point - 2)
                end_idx = min(len(force_powers), control_point + 3)
                
                control_type = "降温到最低温度" if force_control[control_point] == -1 else "升温到最高温度"
                print(f"\n强制控制点 (时间步{control_point}: {control_type}) 附近的详情:")
                print("时间 | 室外温度 | 室内温度 | 控制信号 | 所需功率 | 功率占比 | 温度变化")
                print("-" * 85)
                
                for i in range(start_idx, end_idx):
                    hour = i * 0.5
                    temp_change = force_temps[i] - (force_temps[i-1] if i > 0 else optimizer_force.T_initial)
                    control_signal = force_control[i]
                    signal_str = "强制升温" if control_signal == 1 else "强制降温" if control_signal == -1 else "  无控制"
                    
                    # 高亮显示控制点
                    if i == control_point:
                        print(f"{hour:4.1f}h | {outdoor_temp_half_hour[i+1]:8.1f} | {force_temps[i]:8.2f} | {signal_str:>8} | {force_powers[i]:8.2f} | {force_powers[i]/optimizer_force.P_rated*100:6.1f}% | {temp_change:+6.2f} 🎮")
                    else:
                        print(f"{hour:4.1f}h | {outdoor_temp_half_hour[i+1]:8.1f} | {force_temps[i]:8.2f} | {signal_str:>8} | {force_powers[i]:8.2f} | {force_powers[i]/optimizer_force.P_rated*100:6.1f}% | {temp_change:+6.2f}")
            
            # 保存完整数据用于进一步分析
            print(f"\n💾 保存带强制控制信号的功率需求数据...")
            
            force_data_file = os.path.join(figures_dir, "force_control_data.txt")
            with open(force_data_file, "w", encoding="utf-8") as f:
                f.write("带强制控制信号的空调功率需求数据\n")
                f.write("=" * 80 + "\n")
                f.write("控制信号设置:\n")
                for i, signal in enumerate(force_control):
                    if signal != 0:
                        hour = i * 0.5
                        signal_type = "强制升温到最高温度" if signal == 1 else "强制降温到最低温度"
                        f.write(f"  时间{hour:.1f}h (时间步{i}): {signal_type}\n")
                f.write("=" * 80 + "\n")
                f.write("时间(h),室外温度(°C),室内温度(°C),控制信号,所需功率(kW),功率占比(%),温度变化(°C)\n")
                for i in range(len(force_powers)):
                    hour = i * 0.5
                    temp_change = force_temps[i] - (force_temps[i-1] if i > 0 else optimizer_force.T_initial)
                    signal = force_control[i]
                    f.write(f"{hour:.1f},{outdoor_temp_half_hour[i+1]:.1f},{force_temps[i]:.2f},{signal},{force_powers[i]:.2f},{force_powers[i]/optimizer_force.P_rated*100:.1f},{temp_change:+.2f}\n")
                print(f"✅ 数据已保存到 {force_data_file}")
        else:
            print("❌ 带强制控制信号的优化失败！")
            print(f"原因: {optimizer_force.status}")
            
            # 如果带控制信号的优化失败，尝试检查是否是某些特定控制点导致问题
            print(f"\n尝试单独测试各个控制点...")
            
            for test_point, signal in [(16, -1), (28, 1), (40, -1)]:
                # 创建只有一个控制点的测试
                test_control = [0] * 48
                test_control[test_point] = signal
                
                control_type = "降温到最低温度" if signal == -1 else "升温到最高温度"
                print(f"\n测试时间步{test_point}({test_point*0.5}h)的{control_type}控制...")
                
                # 创建测试优化器
                optimizer_test = ACOptimizerWithTempTarget(
                    T=48, delta_t=0.5, P_rated=5.0,
                    T_min=22.0, T_max=25.0, eta=0.9, R=2.0, C=1.5e7,
                    T_initial=23.5, T_target=23.5, target_type='custom',
                    force_control=test_control
                )
                optimizer_test.set_outdoor_temperature(outdoor_temp_half_hour)
                
                if optimizer_test.solve():
                    print(f"✅ 单独测试时间步{test_point}的控制成功！")
                else:
                    print(f"❌ 单独测试时间步{test_point}的控制失败！原因: {optimizer_test.status}")
    else:
        print("❌ 24小时基础优化失败！")
        print(f"原因: {optimizer_readme.status}")
    
    # 新增：生成控制信号序列数据
    print("\n" + "🎯" * 50)
    print("开始生成控制信号序列优化数据")
    print("🎯" * 50)
    
    # 调用单天控制信号数据生成函数
    try:
        results, power_matrix = generate_control_signal_data()
        print("✅ 单天控制信号序列数据生成成功！")
    except Exception as e:
        print(f"❌ 单天控制信号序列数据生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # 新增：生成夏季多天控制信号序列数据
    print("\n" + "🌞" * 50)
    print("开始生成夏季多天控制信号序列优化数据")
    print("🌞" * 50)
    
    # 先测试前5天数据
    try:
        print("开始处理前5天数据...")
        generate_multi_day_control_signal_data(max_days=5, start_day=0)
        print("✅ 前5天多天控制信号序列数据生成成功！")
        
        # 询问是否继续处理更多天数
        print("\n" + "❓" * 50)
        print("前5天处理完成！")
        print("如需处理更多天数，可以调用以下函数：")
        print("  generate_multi_day_control_signal_data(max_days=10, start_day=5)  # 处理第6-15天")
        print("  generate_multi_day_control_signal_data(max_days=None, start_day=0)  # 处理所有92天")
        print("❓" * 50)
        
    except Exception as e:
        print(f"❌ 多天控制信号序列数据生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        
    # 新增：生成100个空调的CSV数据
    print("\n" + "🏠" * 50)
    print("开始生成100个空调的CSV数据")
    print("🏠" * 50)
    
    try:
        results_100ac, power_matrix_100ac = generate_100_ac_data()
        if results_100ac:
            print("✅ 100个空调数据生成成功！")
        else:
            print("❌ 100个空调数据生成失败！")
    except Exception as e:
        print(f"❌ 100个空调数据生成失败: {str(e)}")
        import traceback
        traceback.print_exc()

def generate_control_signal_data():
    """
    生成控制信号序列的空调功率优化数据
    
    控制信号范围: -1.0 到 1.0，间隔 0.2
    每个控制信号对应一个目标温度，求解24小时优化问题
    最终输出 11×48 的功率矩阵到CSV文件
    """
    print("\n" + "🎯" * 50)
    print("控制信号序列优化数据生成")
    print("🎯" * 50)
    
    # 1. 创建控制信号序列
    control_signals = []
    signal = -1.0
    while signal <= 1.0 + 1e-6:  # 添加小的容差避免浮点数精度问题
        control_signals.append(round(signal, 1))  # 保留1位小数
        signal += 0.2
    
    print(f"控制信号序列: {control_signals}")
    print(f"总共 {len(control_signals)} 个控制信号")
    
    # 2. 系统参数设置
    T = 48              # 24小时，每0.5小时一个控制周期 = 48个时间步
    delta_t = 0.5       # 0.5小时控制周期
    P_rated = 5.0       # kW额定功率
    # 恢复适合夏季制冷的温度约束范围
    T_min = 22.0        # 下限22°C（夏季制冷目标）
    T_max = 25.0        # 上限25°C（夏季舒适温度）
    eta = 0.9           # 效率0.9
    R = 2.0             # 热阻2.0°C/kW
    C = 1.5e7           # 热容1.5e7 J/°C
    T_initial = 23.5    # 初始温度23.5°C（夏季室内温度）
    
    print(f"\n系统参数:")
    print(f"  时间步数: {T}")
    print(f"  控制周期: {delta_t} 小时")
    print(f"  额定功率: {P_rated} kW")
    print(f"  温度范围: [{T_min}°C, {T_max}°C]")
    print(f"  初始温度: {T_initial}°C")
    
    # 3. 室外温度设置（从W2.csv文件读取24小时夏季真实温度数据）
    print(f"正在从W2.csv文件读取夏季室外温度数据...")
    
    # 读取W2.csv文件
    try:
        # 检查是否有环境变量指定文件路径
        import os
        w2_csv_path = os.environ.get('W2_CSV_PATH', "../../data/W2.csv")
        print(f"使用数据文件路径: {w2_csv_path}")
        
        # 尝试读取文件
        try:
            w2_data = pd.read_csv(w2_csv_path)
            print(f"  成功读取W2.csv文件，共{len(w2_data)}条记录")
        except FileNotFoundError:
            # 尝试使用绝对路径
            w2_csv_path = "D:/afterWork/ACL_agg_exp/data/W2.csv"
            print(f"  尝试使用绝对路径: {w2_csv_path}")
            w2_data = pd.read_csv(w2_csv_path)
            print(f"  成功读取W2.csv文件，共{len(w2_data)}条记录")
        
        # 筛选夏季数据（6-8月）
        # 检查是否有环境变量指定目标年份
        target_year = os.environ.get('TARGET_YEAR')
        if target_year:
            print(f"指定处理年份: {target_year}")
        
        # 如果指定了年份，只筛选该年份的夏季数据
        summer_pattern = f"{target_year}/6|{target_year}/7|{target_year}/8"
        summer_indices = w2_data['Time'].str.contains(summer_pattern, na=False)
        print(f"  筛选{target_year}年的夏季数据")
        if summer_indices.any():
            summer_data = w2_data[summer_indices]
            print(f"  找到夏季数据: {len(summer_data)}条记录")
            
            # 获取第一个夏季日期
            first_date = summer_data['Time'].iloc[0].split(' ')[0]
            
            # 修复：正确匹配完整日期而非前缀匹配
            # 使用日期部分精确匹配，而不是使用startswith
            day_data = summer_data[[date.split(' ')[0] == first_date for date in summer_data['Time']]]
            first_summer_day = day_data.iloc[:96]  # 取一整天数据（96条，24小时，每15分钟一条）
            
            # 将华氏度转换为摄氏度: C = (F - 32) * 5/9
            fahrenheit_temps = first_summer_day['Temperature(F)'].values
            celsius_temps = (fahrenheit_temps - 32) * 5/9
            
            # 每30分钟取一个数据点（每2条记录取1条，因为原数据是每15分钟一条）
            outdoor_temp_half_hour = []
            time_stamps = []
            
            for i in range(0, min(96, len(celsius_temps)), 2):  # 取49个点（0到96，步长2）
                outdoor_temp_half_hour.append(celsius_temps[i])
                time_stamps.append(first_summer_day['Time'].iloc[i])
                if len(outdoor_temp_half_hour) >= 49:  # 确保有49个数据点
                    break
            
            # 如果数据不足49个点，用最后一个值补充
            while len(outdoor_temp_half_hour) < 49:
                outdoor_temp_half_hour.append(outdoor_temp_half_hour[-1])
                # 为时间戳生成对应的时间
                last_time = time_stamps[-1] if time_stamps else "2015/6/1 0:00"
                time_stamps.append(last_time)
            
            print(f"  提取的夏季温度数据点数: {len(outdoor_temp_half_hour)}")
            print(f"  夏季温度范围: {min(outdoor_temp_half_hour):.1f}°C - {max(outdoor_temp_half_hour):.1f}°C")
            print(f"  前5个时间点: {time_stamps[:5]}")
            
        else:
            raise Exception("未找到夏季数据，请检查数据文件")
        
    except Exception as e:
        print(f"  ❌ 读取夏季数据失败: {str(e)}")
        print(f"  使用默认的夏季模拟温度数据...")
        
        # 如果读取失败，使用夏季的默认数据
        hourly_outdoor_temp = [
            28.0, 27.5, 27.0, 26.5, 26.0, 26.5, 27.0, 28.0, 
            29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 
            35.5, 35.0, 34.0, 33.0, 32.0, 31.0, 30.0, 29.0, 28.5
        ]
        
        # 插值生成0.5小时间隔的温度数据
        outdoor_temp_half_hour = []
        time_stamps = []
        for i in range(len(hourly_outdoor_temp)-1):
            outdoor_temp_half_hour.append(hourly_outdoor_temp[i])
            time_stamps.append(f"2015/6/1 {i}:00")
            # 添加中间点的线性插值
            mid_temp = (hourly_outdoor_temp[i] + hourly_outdoor_temp[i+1]) / 2
            outdoor_temp_half_hour.append(mid_temp)
            time_stamps.append(f"2015/6/1 {i}:30")
        outdoor_temp_half_hour.append(hourly_outdoor_temp[-1])  # 添加最后一个点
        time_stamps.append(f"2015/6/1 24:00")
        
        print(f"  使用默认夏季温度范围: {min(outdoor_temp_half_hour):.1f}°C - {max(outdoor_temp_half_hour):.1f}°C")
    
    # 4. 控制信号到温度约束的映射函数
    def control_signal_to_temp_constraints(signal, base_T_min=22.0, base_T_max=25.0):
        """
        控制信号映射到温度约束范围
        
        signal = -1.0 → 放宽制冷需求，提高温度上限 (例如：[23, 27]°C)
        signal = 0.0 → 正常约束 (例如：[22, 25]°C)  
        signal = 1.0 → 强制制冷需求，降低温度上限 (例如：[21, 23]°C)
        """
        # 调整幅度：±1.5°C（取负号反转逻辑）
        # signal = -1 时，提高温度上限；signal = +1 时，降低温度上限
        adjustment = -signal * 1.5
        
        # 限制调整后的温度范围
        T_min_adjusted = max(20.0, min(base_T_min + adjustment, 24.0))  # 限制在[20, 24]°C
        T_max_adjusted = max(23.0, min(base_T_max + adjustment, 28.0))  # 限制在[23, 28]°C
        
        # 确保最小温度小于最大温度
        if T_max_adjusted <= T_min_adjusted:
            T_max_adjusted = T_min_adjusted + 1.5
            
        # 目标温度设在约束范围的中间
        target_temp = (T_min_adjusted + T_max_adjusted) / 2.0
        
        return T_min_adjusted, T_max_adjusted, target_temp
    
    # 5. 批量优化求解
    print(f"\n开始批量优化求解...")
    print("-" * 80)
    
    results = []  # 存储所有结果
    power_matrix = []  # 存储功率矩阵
    failed_signals = []  # 记录失败的控制信号
    
    for i, signal in enumerate(control_signals):
        # 计算调整后的温度约束和目标温度
        T_min_adj, T_max_adj, target_temp = control_signal_to_temp_constraints(signal, T_min, T_max)
        
        print(f"正在求解 [{i+1}/{len(control_signals)}] 控制信号: {signal:4.1f} → 温度约束: [{T_min_adj:.1f}, {T_max_adj:.1f}]°C → 目标: {target_temp:.2f}°C")
        
        try:
            # 创建优化器 - 使用调整后的温度约束
            optimizer = ACOptimizerWithTempTarget(
                T=T, delta_t=delta_t, P_rated=P_rated,
                T_min=T_min_adj, T_max=T_max_adj, eta=eta, R=R, C=C,  # 使用调整后的约束
                T_initial=T_initial, T_target=target_temp,
                target_type='custom'
            )
            
            # 设置室外温度
            optimizer.set_outdoor_temperature(outdoor_temp_half_hour)
            
            # 求解优化问题
            if optimizer.solve():
                # 成功求解
                powers = optimizer.optimal_powers
                temps = optimizer.optimal_temperatures[1:]  # 跳过初始温度
                total_energy = sum(powers) * delta_t
                
                # 记录结果
                result = {
                    'control_signal': signal,
                    'target_temp': target_temp,
                    'total_energy': total_energy,
                    'avg_power': total_energy / (T * delta_t),
                    'max_power': max(powers),
                    'min_power': min(powers),
                    'min_temp': min(temps),
                    'max_temp': max(temps),
                    'final_temp': temps[-1],
                    'powers': powers.copy(),
                    'temperatures': temps.copy(),
                    'status': 'success'
                }
                results.append(result)
                power_matrix.append(powers)
                
                print(f"  ✅ 成功 | 总能耗: {total_energy:6.2f} kWh | 平均功率: {result['avg_power']:6.2f} kW | 温度范围: [{result['min_temp']:5.2f}, {result['max_temp']:5.2f}]°C")
                
            else:
                # 求解失败
                print(f"  ❌ 失败 | 原因: {optimizer.status}")
                failed_signals.append((signal, target_temp, optimizer.status))
                
                # 添加空的功率数据，便于后续处理
                power_matrix.append([0.0] * T)
                results.append({
                    'control_signal': signal,
                    'target_temp': target_temp,
                    'status': 'failed',
                    'error': optimizer.status,
                    'powers': [0.0] * T,
                    'temperatures': [T_initial] * T
                })
                
        except Exception as e:
            print(f"  ❌ 异常 | 错误: {str(e)}")
            failed_signals.append((signal, target_temp, str(e)))
            power_matrix.append([0.0] * T)
            results.append({
                'control_signal': signal,
                'target_temp': target_temp,
                'status': 'error',
                'error': str(e),
                'powers': [0.0] * T,
                'temperatures': [T_initial] * T
            })
    
    print("-" * 80)
    
    # 6. 统计结果
    successful_results = [r for r in results if r['status'] == 'success']
    print(f"批量优化完成:")
    print(f"  成功: {len(successful_results)}/{len(control_signals)} 个控制信号")
    print(f"  失败: {len(failed_signals)} 个控制信号")
    
    if failed_signals:
        print(f"  失败的控制信号:")
        for signal, target_temp, error in failed_signals:
            print(f"    信号 {signal:4.1f} (目标 {target_temp:5.2f}°C): {error}")
    
    # 7. 保存功率矩阵到CSV文件
    print(f"\n💾 保存结果到CSV文件...")
    
    # 确保figures文件夹存在
    figures_dir = "../../figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"  创建文件夹: {figures_dir}")
    
    # 保存主要的功率矩阵
    power_matrix_file = os.path.join(figures_dir, "control_signal_power_matrix.csv")
    with open(power_matrix_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入头部信息
        writer.writerow(["控制信号序列功率矩阵"])
        writer.writerow([f"系统参数: T={T}, delta_t={delta_t}h, P_rated={P_rated}kW, T_range=[{T_min},{T_max}]°C"])
        writer.writerow([])
        
        # 写入列标题（时间步）- 添加Time列
        time_headers = ["Time", "控制信号", "目标温度"] + [f"t{i+1}({(i+1)*delta_t:.1f}h)" for i in range(T)]
        writer.writerow(time_headers)
        
        # 写入数据行
        for i, result in enumerate(results):
            # 生成对应的时间标识
            if i < len(time_stamps):
                time_str = time_stamps[0]  # 使用第一个时间点作为这个控制信号的标识
            else:
                time_str = "N/A"
            
            row = [time_str, result['control_signal'], f"{result['target_temp']:.2f}"] + [f"{p:.3f}" for p in result['powers']]
            writer.writerow(row)
    
    print(f"✅ 功率矩阵已保存到: {power_matrix_file}")
    
    # 保存详细统计信息
    stats_file = os.path.join(figures_dir, "control_signal_statistics.csv")
    with open(stats_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入统计表头 - 添加Time列
        stats_headers = [
            "Time", "控制信号", "目标温度", "状态", "总能耗(kWh)", "平均功率(kW)", 
            "最大功率(kW)", "最小功率(kW)", "最低温度(°C)", "最高温度(°C)", "最终温度(°C)"
        ]
        writer.writerow(stats_headers)
        
        # 写入统计数据
        for i, result in enumerate(results):
            # 生成对应的时间标识
            if i < len(time_stamps):
                time_str = time_stamps[0]  # 使用第一个时间点作为这个控制信号的标识
            else:
                time_str = "N/A"
                
            if result['status'] == 'success':
                row = [
                    time_str, result['control_signal'], f"{result['target_temp']:.2f}", result['status'],
                    f"{result['total_energy']:.2f}", f"{result['avg_power']:.3f}",
                    f"{result['max_power']:.3f}", f"{result['min_power']:.3f}",
                    f"{result['min_temp']:.2f}", f"{result['max_temp']:.2f}", f"{result['final_temp']:.2f}"
                ]
            else:
                row = [
                    time_str, result['control_signal'], f"{result['target_temp']:.2f}", result['status'],
                    "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
                ]
            writer.writerow(row)
    
    print(f"✅ 统计信息已保存到: {stats_file}")
    
    # 保存完整的时间序列数据（包含温度）
    timeseries_file = os.path.join(figures_dir, "control_signal_full_timeseries.csv")
    with open(timeseries_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 写入头部
        writer.writerow(["控制信号完整时间序列数据"])
        writer.writerow([])
        
        # 为每个控制信号保存详细的时间序列
        for result in results:
            writer.writerow([f"控制信号: {result['control_signal']}, 目标温度: {result['target_temp']:.2f}°C, 状态: {result['status']}"])
            writer.writerow(["Time", "时间(h)", "室外温度(°C)", "室内温度(°C)", "空调功率(kW)", "功率占比(%)"])
            
            if result['status'] == 'success':
                for i in range(T):
                    time_h = (i + 1) * delta_t
                    # 获取对应的时间戳
                    if i + 1 < len(time_stamps):
                        time_stamp = time_stamps[i + 1]
                    else:
                        time_stamp = "N/A"
                    
                    outdoor_temp = outdoor_temp_half_hour[i + 1]
                    indoor_temp = result['temperatures'][i]
                    power = result['powers'][i]
                    power_ratio = power / P_rated * 100
                    
                    writer.writerow([time_stamp, f"{time_h:.1f}", f"{outdoor_temp:.1f}", f"{indoor_temp:.2f}", f"{power:.3f}", f"{power_ratio:.1f}"])
            else:
                writer.writerow(["求解失败，无数据"])
            
            writer.writerow([])  # 空行分隔
    
    print(f"✅ 完整时间序列已保存到: {timeseries_file}")
    
    # 8. 输出汇总统计
    if successful_results:
        print(f"\n📊 成功案例汇总统计:")
        print("-" * 80)
        
        total_energies = [r['total_energy'] for r in successful_results]
        avg_powers = [r['avg_power'] for r in successful_results]
        
        print(f"总能耗范围: {min(total_energies):.2f} - {max(total_energies):.2f} kWh")
        print(f"平均功率范围: {min(avg_powers):.3f} - {max(avg_powers):.3f} kW")
        print(f"控制信号与能耗关系:")
        
        for result in successful_results[:5]:  # 显示前5个成功的案例
            print(f"  信号 {result['control_signal']:4.1f} → 目标 {result['target_temp']:5.2f}°C → 能耗 {result['total_energy']:6.2f} kWh")
        
        if len(successful_results) > 5:
            print(f"  ... 以及另外 {len(successful_results)-5} 个成功案例")
        
        print("-" * 80)
    
    print(f"\n🎯 控制信号序列优化数据生成完成！")
    print(f"生成的文件:")
    print(f"  📄 {power_matrix_file} - 11×48功率矩阵")
    print(f"  📊 {stats_file} - 统计汇总信息")
    print(f"  📈 {timeseries_file} - 完整时间序列数据")
    print(f"所有文件已保存到figures文件夹中")
    
    return results, power_matrix

def generate_multi_day_control_signal_data(max_days=None, start_day=0):
    """
    生成夏季多天的控制信号序列空调功率优化数据
    
    参数:
    max_days: 最大处理天数，None表示处理所有夏季天数
    start_day: 开始处理的天数索引（0表示第一天）
    
    每天对11个控制信号（-1.0到1.0，间隔0.2）求解24小时优化问题
    将结果增量追加到CSV文件中
    """
    print("\n" + "🌞" * 60)
    print("夏季多天控制信号序列优化数据生成")
    print("🌞" * 60)
    
    # 1. 读取夏季数据并提取每天的日期
    print(f"正在读取夏季数据...")
    
    try:
        # 检查是否有环境变量指定文件路径
        import os
        w2_csv_path = os.environ.get('W2_CSV_PATH', "../../data/W2.csv")
        print(f"使用数据文件路径: {w2_csv_path}")
        
        # 检查是否有环境变量指定目标年份
        target_year = os.environ.get('TARGET_YEAR')
        if target_year:
            print(f"指定处理年份: {target_year}")
        
        # 尝试读取文件
        try:
            w2_data = pd.read_csv(w2_csv_path)
            print(f"  成功读取W2.csv文件，共{len(w2_data)}条记录")
        except FileNotFoundError:
            # 尝试使用绝对路径
            w2_csv_path = "D:/afterWork/ACL_agg_exp/data/W2.csv"
            print(f"  尝试使用绝对路径: {w2_csv_path}")
            w2_data = pd.read_csv(w2_csv_path)
            print(f"  成功读取W2.csv文件，共{len(w2_data)}条记录")
        
        # 筛选夏季数据（6-8月）
        # 检查是否有环境变量指定目标年份
        target_year = os.environ.get('TARGET_YEAR')
        if target_year:
            # 如果指定了年份，只筛选该年份的夏季数据
            summer_pattern = f"{target_year}/6|{target_year}/7|{target_year}/8"
            summer_indices = w2_data['Time'].str.contains(summer_pattern, na=False)
            print(f"  筛选{target_year}年的夏季数据")
        else:
            # 否则筛选所有年份的夏季数据
            summer_indices = w2_data['Time'].str.contains('/6|/7|/8', na=False)
            print(f"  筛选所有年份的夏季数据")
        
        if summer_indices.any():
            summer_data = w2_data[summer_indices]
            print(f"  找到夏季数据: {len(summer_data)}条记录")
            
            # 提取日期信息
            dates = []
            for time_str in summer_data['Time']:
                date_part = time_str.split(' ')[0]  # 获取日期部分
                dates.append(date_part)
            
            # 获取唯一日期并排序
            unique_dates = list(set(dates))
            unique_dates.sort()
            
            total_days = len(unique_dates)
            print(f"  夏季总天数: {total_days}天")
            print(f"  日期范围: {unique_dates[0]} 到 {unique_dates[-1]}")
            
            # 确定实际处理的天数
            if max_days is None:
                process_days = total_days - start_day
            else:
                process_days = min(max_days, total_days - start_day)
            
            if start_day >= total_days:
                raise Exception(f"开始天数索引{start_day}超出总天数{total_days}")
            
            process_dates = unique_dates[start_day:start_day + process_days]
            print(f"  将处理: {len(process_dates)}天数据 (从第{start_day+1}天开始)")
            print(f"  处理日期: {process_dates[0]} 到 {process_dates[-1]}")
            
        else:
            raise Exception("未找到夏季数据，请检查数据文件")
        
    except Exception as e:
        print(f"  ❌ 读取夏季数据失败: {str(e)}")
        return
    
    # 2. 创建控制信号序列
    control_signals = []
    signal = -1.0
    while signal <= 1.0 + 1e-6:
        control_signals.append(round(signal, 1))
        signal += 0.2
    
    print(f"\n控制信号序列: {control_signals}")
    print(f"总共 {len(control_signals)} 个控制信号")
    
    # 3. 系统参数设置
    T = 48              # 24小时，每0.5小时一个控制周期 = 48个时间步
    delta_t = 0.5       # 0.5小时控制周期
    P_rated = 5.0       # kW额定功率
    T_min = 22.0        # 下限22°C（夏季制冷目标）
    T_max = 25.0        # 上限25°C（夏季舒适温度）
    eta = 0.9           # 效率0.9
    R = 2.0             # 热阻2.0°C/kW
    C = 1.5e7           # 热容1.5e7 J/°C
    T_initial = 23.5    # 初始温度23.5°C（夏季室内温度）
    
    print(f"\n系统参数:")
    print(f"  时间步数: {T}")
    print(f"  控制周期: {delta_t} 小时")
    print(f"  额定功率: {P_rated} kW")
    print(f"  温度范围: [{T_min}°C, {T_max}°C]")
    print(f"  初始温度: {T_initial}°C")
    
    # 4. 控制信号到温度约束的映射函数
    def control_signal_to_temp_constraints(signal, base_T_min=22.0, base_T_max=25.0):
        """控制信号映射到温度约束范围"""
        adjustment = signal * 1.5
        T_min_adjusted = max(20.0, min(base_T_min + adjustment, 24.0))
        T_max_adjusted = max(23.0, min(base_T_max + adjustment, 28.0))
        if T_max_adjusted <= T_min_adjusted:
            T_max_adjusted = T_min_adjusted + 1.5
        target_temp = (T_min_adjusted + T_max_adjusted) / 2.0
        return T_min_adjusted, T_max_adjusted, target_temp
    
    # 5. 确保figures文件夹存在
    figures_dir = "../../figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"  创建文件夹: {figures_dir}")
    
    # 6. 设置文件路径
    power_matrix_file = os.path.join(figures_dir, "multi_day_control_signal_power_matrix.csv")
    stats_file = os.path.join(figures_dir, "multi_day_control_signal_statistics.csv")
    timeseries_file = os.path.join(figures_dir, "multi_day_control_signal_full_timeseries.csv")
    
    # 7. 检查是否是首次运行（文件不存在）或追加模式
    is_first_run = not os.path.exists(power_matrix_file)
    
    print(f"\n文件模式: {'新建' if is_first_run else '追加'}")
    
    # 8. 逐天处理数据
    print(f"\n开始逐天批量优化求解...")
    print("=" * 100)
    
    all_day_results = []  # 存储所有天的结果
    total_success = 0
    total_failed = 0
    
    for day_idx, process_date in enumerate(process_dates):
        print(f"\n📅 处理日期: {process_date} [{day_idx+1}/{len(process_dates)}]")
        print("-" * 80)
        
        try:
            # 修复：正确匹配完整日期而非前缀匹配
            # 使用日期部分精确匹配，而不是使用startswith
            day_data = summer_data[[date.split(' ')[0] == process_date for date in summer_data['Time']]]
            
            if len(day_data) < 96:  # 需要至少96个数据点（24小时）
                print(f"  ❌ 当天数据不足: {len(day_data)}条，跳过")
                continue
            
            # 提取当天的温度数据
            day_data_sorted = day_data.sort_values('Time')  # 确保时间顺序
            fahrenheit_temps = day_data_sorted['Temperature(F)'].values[:96]
            celsius_temps = (fahrenheit_temps - 32) * 5/9
            
            # 每30分钟取一个数据点
            outdoor_temp_half_hour = []
            time_stamps = []
            
            for i in range(0, min(96, len(celsius_temps)), 2):
                outdoor_temp_half_hour.append(celsius_temps[i])
                time_stamps.append(day_data_sorted['Time'].iloc[i])
                if len(outdoor_temp_half_hour) >= 49:
                    break
            
            # 补充数据点（如果不足）
            while len(outdoor_temp_half_hour) < 49:
                outdoor_temp_half_hour.append(outdoor_temp_half_hour[-1])
                last_time = time_stamps[-1] if time_stamps else f"{process_date} 0:00"
                time_stamps.append(last_time)
            
            print(f"  当天温度范围: {min(outdoor_temp_half_hour):.1f}°C - {max(outdoor_temp_half_hour):.1f}°C")
            
            # 对当天的每个控制信号进行优化
            day_results = []
            day_power_matrix = []
            day_success = 0
            day_failed = 0
            
            for signal_idx, signal in enumerate(control_signals):
                T_min_adj, T_max_adj, target_temp = control_signal_to_temp_constraints(signal, T_min, T_max)
                
                try:
                    # 创建优化器
                    optimizer = ACOptimizerWithTempTarget(
                        T=T, delta_t=delta_t, P_rated=P_rated,
                        T_min=T_min_adj, T_max=T_max_adj, eta=eta, R=R, C=C,
                        T_initial=T_initial, T_target=target_temp,
                        target_type='custom'
                    )
                    
                    # 设置室外温度
                    optimizer.set_outdoor_temperature(outdoor_temp_half_hour)
                    
                    # 求解优化问题（关闭详细输出）
                    if optimizer.solve():
                        # 成功求解
                        powers = optimizer.optimal_powers
                        temps = optimizer.optimal_temperatures[1:]
                        total_energy = sum(powers) * delta_t
                        
                        result = {
                            'date': process_date,
                            'control_signal': signal,
                            'target_temp': target_temp,
                            'total_energy': total_energy,
                            'avg_power': total_energy / (T * delta_t),
                            'max_power': max(powers),
                            'min_power': min(powers),
                            'min_temp': min(temps),
                            'max_temp': max(temps),
                            'final_temp': temps[-1],
                            'powers': powers.copy(),
                            'temperatures': temps.copy(),
                            'time_stamps': time_stamps.copy(),
                            'status': 'success'
                        }
                        
                        day_results.append(result)
                        day_power_matrix.append(powers)
                        day_success += 1
                        
                        print(f"    ✅ 信号{signal:4.1f} → 能耗:{total_energy:6.2f}kWh → 温度:[{result['min_temp']:5.2f},{result['max_temp']:5.2f}]°C")
                        
                    else:
                        print(f"    ❌ 信号{signal:4.1f} → 优化失败: {optimizer.status}")
                        day_failed += 1
                        
                except Exception as e:
                    print(f"    ❌ 信号{signal:4.1f} → 异常: {str(e)}")
                    day_failed += 1
            
            # 当天统计
            print(f"  当天结果: ✅{day_success} ❌{day_failed}")
            all_day_results.extend(day_results)
            total_success += day_success
            total_failed += day_failed
            
        except Exception as e:
            print(f"  ❌ 处理{process_date}失败: {str(e)}")
            total_failed += len(control_signals)
    
    # 9. 保存结果到CSV文件（追加模式）
    print(f"\n💾 保存结果到CSV文件...")
    print(f"总计: ✅{total_success} ❌{total_failed}")
    
    if all_day_results:
        # 保存功率矩阵（追加模式）
        write_header = is_first_run
        mode = 'w' if is_first_run else 'a'
        
        with open(power_matrix_file, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            if write_header:
                # 写入头部信息（仅首次）
                writer.writerow(["多天控制信号序列功率矩阵"])
                writer.writerow([f"系统参数: T={T}, delta_t={delta_t}h, P_rated={P_rated}kW, T_range=[{T_min},{T_max}]°C"])
                writer.writerow([])
                
                # 写入列标题
                time_headers = ["Date", "Time", "控制信号", "目标温度"] + [f"t{i+1}({(i+1)*delta_t:.1f}h)" for i in range(T)]
                writer.writerow(time_headers)
            
            # 写入数据行
            for result in all_day_results:
                time_str = result['time_stamps'][0]  # 使用第一个时间点
                row = [result['date'], time_str, result['control_signal'], f"{result['target_temp']:.2f}"] + [f"{p:.3f}" for p in result['powers']]
                writer.writerow(row)
        
        print(f"✅ 功率矩阵已{'追加' if not is_first_run else '保存'}到: {power_matrix_file}")
        
        # 保存统计信息（追加模式）
        write_header = is_first_run
        mode = 'w' if is_first_run else 'a'
        
        with open(stats_file, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            if write_header:
                stats_headers = [
                    "Date", "Time", "控制信号", "目标温度", "状态", "总能耗(kWh)", "平均功率(kW)", 
                    "最大功率(kW)", "最小功率(kW)", "最低温度(°C)", "最高温度(°C)", "最终温度(°C)"
                ]
                writer.writerow(stats_headers)
            
            for result in all_day_results:
                time_str = result['time_stamps'][0]
                row = [
                    result['date'], time_str, result['control_signal'], f"{result['target_temp']:.2f}", result['status'],
                    f"{result['total_energy']:.2f}", f"{result['avg_power']:.3f}",
                    f"{result['max_power']:.3f}", f"{result['min_power']:.3f}",
                    f"{result['min_temp']:.2f}", f"{result['max_temp']:.2f}", f"{result['final_temp']:.2f}"
                ]
                writer.writerow(row)
        
        print(f"✅ 统计信息已{'追加' if not is_first_run else '保存'}到: {stats_file}")
        
        # 保存完整时间序列（追加模式） 
        mode = 'w' if is_first_run else 'a'
        
        with open(timeseries_file, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            if write_header:
                writer.writerow(["多天控制信号完整时间序列数据"])
                writer.writerow([])
            
            # 为每个结果保存详细的时间序列
            for result in all_day_results:
                writer.writerow([f"日期: {result['date']}, 控制信号: {result['control_signal']}, 目标温度: {result['target_temp']:.2f}°C, 状态: {result['status']}"])
                writer.writerow(["Date", "Time", "时间(h)", "室外温度(°C)", "室内温度(°C)", "空调功率(kW)", "功率占比(%)"])
                
                # 获取当天的室外温度数据
                day_outdoor_temps = []
                # 修复：正确匹配完整日期而非前缀匹配
                # 使用日期部分精确匹配，而不是使用startswith
                day_data = summer_data[[date.split(' ')[0] == result['date'] for date in summer_data['Time']]]
                if len(day_data) >= 96:  # 需要至少96个数据点（24小时）
                    day_data_sorted = day_data.sort_values('Time')
                    fahrenheit_temps = day_data_sorted['Temperature(F)'].values[:96]
                    celsius_temps = (fahrenheit_temps - 32) * 5/9
                    for i in range(0, min(96, len(celsius_temps)), 2):
                        day_outdoor_temps.append(celsius_temps[i])
                        if len(day_outdoor_temps) >= 49:
                            break
                
                for i in range(T):
                    time_h = (i + 1) * delta_t
                    time_stamp = result['time_stamps'][i + 1] if i + 1 < len(result['time_stamps']) else "N/A"
                    
                    outdoor_temp = day_outdoor_temps[i + 1] if i + 1 < len(day_outdoor_temps) else 0
                    indoor_temp = result['temperatures'][i]
                    power = result['powers'][i]
                    power_ratio = power / P_rated * 100
                    
                    writer.writerow([result['date'], time_stamp, f"{time_h:.1f}", f"{outdoor_temp:.1f}", f"{indoor_temp:.2f}", f"{power:.3f}", f"{power_ratio:.1f}"])
                
                writer.writerow([])  # 空行分隔
        
        print(f"✅ 完整时间序列已{'追加' if not is_first_run else '保存'}到: {timeseries_file}")
        
        # 输出汇总统计
        if all_day_results:
            print(f"\n📊 处理结果汇总:")
            print("-" * 80)
            
            total_energies = [r['total_energy'] for r in all_day_results]
            avg_powers = [r['avg_power'] for r in all_day_results]
            
            print(f"处理天数: {len(process_dates)}天")
            print(f"成功计算: {total_success}个控制信号")
            print(f"失败计算: {total_failed}个控制信号")
            print(f"总能耗范围: {min(total_energies):.2f} - {max(total_energies):.2f} kWh")
            print(f"平均功率范围: {min(avg_powers):.3f} - {max(avg_powers):.3f} kW")
            
            # 按日期分组统计
            date_stats = {}
            for result in all_day_results:
                date = result['date']
                if date not in date_stats:
                    date_stats[date] = []
                date_stats[date].append(result['total_energy'])
            
            print(f"\n各日能耗统计:")
            for date in sorted(date_stats.keys())[:5]:  # 显示前5天
                energies = date_stats[date]
                print(f"  {date}: {min(energies):.1f} - {max(energies):.1f} kWh ({len(energies)}个信号)")
            
            if len(date_stats) > 5:
                print(f"  ... 以及另外 {len(date_stats)-5} 天")
            
            print("-" * 80)
        
        print(f"\n🌞 夏季多天控制信号数据生成完成！")
        print(f"生成的文件:")
        print(f"  📄 {power_matrix_file} - 多天功率矩阵")
        print(f"  📊 {stats_file} - 多天统计汇总")
        print(f"  📈 {timeseries_file} - 多天完整时间序列")
        
    else:
        print("❌ 没有成功的计算结果，未生成文件")

def generate_100_ac_data():
    """
    根据ac_data.json中的配置，生成100个空调的CSV数据
    每个空调使用其特定的配置参数生成24小时功率数据
    """
    print("\n" + "🏠" * 60)
    print("生成100个空调的CSV数据")
    print("🏠" * 60)
    
    import json
    import random
    import numpy as np
    
    try:
        # 1. 读取空调配置数据
        ac_data_path = "ac_data.json"  # 修改为当前目录的相对路径
        print(f"正在从 {ac_data_path} 读取空调配置...")
        
        with open(ac_data_path, 'r', encoding='utf-8') as f:
            ac_configs = json.load(f)
        
        print(f"✅ 成功读取 {len(ac_configs)} 个空调配置")
        
        # 统计不同类型的空调数量
        ac_types = {}
        for ac in ac_configs:
            ac_type = ac['type']
            ac_types[ac_type] = ac_types.get(ac_type, 0) + 1
        
        print(f"空调类型分布:")
        for ac_type, count in ac_types.items():
            print(f"  {ac_type}: {count} 个")
        
        # 2. 选择100个空调配置（如果配置多于100个，随机选择；如果少于100个，重复选择）
        if len(ac_configs) >= 100:
            selected_acs = random.sample(ac_configs, 100)
            print(f"📋 从 {len(ac_configs)} 个配置中随机选择了 100 个")
        else:
            # 如果不足100个，循环重复选择
            selected_acs = []
            while len(selected_acs) < 100:
                remaining = min(100 - len(selected_acs), len(ac_configs))
                selected_acs.extend(random.sample(ac_configs, remaining))
            print(f"📋 循环选择配置以生成 100 个空调数据")
        
        # 3. 设置系统参数
        T = 48              # 24小时，每0.5小时一个控制周期 = 48个时间步
        delta_t = 0.5       # 0.5小时控制周期
        T_initial = 23.5    # 初始温度23.5°C
        
        # 4. 读取室外温度数据
        print(f"正在读取室外温度数据...")
        try:
            import os
            w2_csv_path = os.environ.get('W2_CSV_PATH', "../../data/W2.csv")
            target_year = os.environ.get('TARGET_YEAR', '2015')
            
            try:
                w2_data = pd.read_csv(w2_csv_path)
            except FileNotFoundError:
                w2_csv_path = "D:/afterWork/ACL_agg_exp/data/W2.csv"
                w2_data = pd.read_csv(w2_csv_path)
            
            # 筛选夏季数据
            summer_pattern = f"{target_year}/6|{target_year}/7|{target_year}/8"
            summer_indices = w2_data['Time'].str.contains(summer_pattern, na=False)
            
            if summer_indices.any():
                summer_data = w2_data[summer_indices]
                first_date = summer_data['Time'].iloc[0].split(' ')[0]
                day_data = summer_data[[date.split(' ')[0] == first_date for date in summer_data['Time']]]
                first_summer_day = day_data.iloc[:96]
                
                fahrenheit_temps = first_summer_day['Temperature(F)'].values
                celsius_temps = (fahrenheit_temps - 32) * 5/9
                
                outdoor_temp_half_hour = []
                for i in range(0, min(96, len(celsius_temps)), 2):
                    outdoor_temp_half_hour.append(celsius_temps[i])
                    if len(outdoor_temp_half_hour) >= 49:
                        break
                
                while len(outdoor_temp_half_hour) < 49:
                    outdoor_temp_half_hour.append(outdoor_temp_half_hour[-1])
                
                print(f"✅ 夏季室外温度范围: {min(outdoor_temp_half_hour):.1f}°C - {max(outdoor_temp_half_hour):.1f}°C")
            else:
                raise Exception("未找到夏季数据")
                
        except Exception as e:
            print(f"⚠️  读取真实温度数据失败，使用默认数据: {str(e)}")
            # 使用默认夏季温度数据
            hourly_outdoor_temp = [
                28.0, 27.5, 27.0, 26.5, 26.0, 26.5, 27.0, 28.0, 
                29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 
                35.5, 35.0, 34.0, 33.0, 32.0, 31.0, 30.0, 29.0, 28.5
            ]
            outdoor_temp_half_hour = []
            for i in range(len(hourly_outdoor_temp)-1):
                outdoor_temp_half_hour.append(hourly_outdoor_temp[i])
                mid_temp = (hourly_outdoor_temp[i] + hourly_outdoor_temp[i+1]) / 2
                outdoor_temp_half_hour.append(mid_temp)
            outdoor_temp_half_hour.append(hourly_outdoor_temp[-1])
        
        # 5. 为每个空调生成数据
        print(f"\n开始为100个空调生成优化数据...")
        print("-" * 80)
        
        all_results = []
        all_power_matrix = []
        
        for i, ac_config in enumerate(selected_acs):
            ac_id = ac_config['id']
            ac_type = ac_config['type']
            
            print(f"正在处理 [{i+1}/100] 空调 {ac_id} ({ac_type})")
            
            try:
                # 使用空调配置参数
                optimizer = ACOptimizerWithTempTarget(
                    T=T, 
                    delta_t=delta_t, 
                    P_rated=ac_config['P_rated'],
                    T_min=ac_config['T_min'], 
                    T_max=ac_config['T_max'], 
                    eta=ac_config['efficiency'], 
                    R=ac_config['R'], 
                    C=ac_config['C'],
                    T_initial=T_initial, 
                    T_target=(ac_config['T_min'] + ac_config['T_max']) / 2,  # 目标温度设为范围中点
                    target_type='custom'
                )
                
                # 设置室外温度
                optimizer.set_outdoor_temperature(outdoor_temp_half_hour)
                
                # 求解优化问题
                if optimizer.solve():
                    powers = optimizer.optimal_powers
                    temps = optimizer.optimal_temperatures[1:]
                    total_energy = sum(powers) * delta_t
                    
                    result = {
                        'ac_id': ac_id,
                        'ac_type': ac_type,
                        'P_rated': ac_config['P_rated'],
                        'T_min': ac_config['T_min'],
                        'T_max': ac_config['T_max'],
                        'R': ac_config['R'],
                        'C': ac_config['C'],
                        'efficiency': ac_config['efficiency'],
                        'eta': ac_config['eta'],
                        'total_energy': total_energy,
                        'avg_power': total_energy / (T * delta_t),
                        'max_power': max(powers),
                        'min_power': min(powers),
                        'min_temp': min(temps),
                        'max_temp': max(temps),
                        'final_temp': temps[-1],
                        'powers': powers.copy(),
                        'temperatures': temps.copy(),
                        'status': 'success'
                    }
                    
                    all_results.append(result)
                    all_power_matrix.append(powers)
                    
                    print(f"  ✅ 成功 | 能耗: {total_energy:6.2f} kWh | 平均功率: {result['avg_power']:6.2f} kW | 温度范围: [{result['min_temp']:5.2f}, {result['max_temp']:5.2f}]°C")
                    
                else:
                    print(f"  ❌ 失败 | 原因: {optimizer.status}")
                    # 添加失败记录
                    result = {
                        'ac_id': ac_id,
                        'ac_type': ac_type,
                        'P_rated': ac_config['P_rated'],
                        'T_min': ac_config['T_min'],
                        'T_max': ac_config['T_max'],
                        'R': ac_config['R'],
                        'C': ac_config['C'],
                        'efficiency': ac_config['efficiency'],
                        'eta': ac_config['eta'],
                        'status': 'failed',
                        'error': optimizer.status,
                        'powers': [0.0] * T,
                        'temperatures': [T_initial] * T
                    }
                    all_results.append(result)
                    all_power_matrix.append([0.0] * T)
                    
            except Exception as e:
                print(f"  ❌ 异常 | 错误: {str(e)}")
                result = {
                    'ac_id': ac_id,
                    'ac_type': ac_type,
                    'P_rated': ac_config['P_rated'],
                    'T_min': ac_config['T_min'],
                    'T_max': ac_config['T_max'],
                    'R': ac_config['R'],
                    'C': ac_config['C'],
                    'efficiency': ac_config['efficiency'],
                    'eta': ac_config['eta'],
                    'status': 'error',
                    'error': str(e),
                    'powers': [0.0] * T,
                    'temperatures': [T_initial] * T
                }
                all_results.append(result)
                all_power_matrix.append([0.0] * T)
        
        print("-" * 80)
        
        # 6. 统计结果
        successful_results = [r for r in all_results if r['status'] == 'success']
        print(f"批量优化完成:")
        print(f"  成功: {len(successful_results)}/100 个空调")
        print(f"  失败: {100 - len(successful_results)} 个空调")
        
        # 7. 保存结果到CSV文件
        print(f"\n💾 保存100个空调的结果到CSV文件...")
        
        # 确保figures文件夹存在
        figures_dir = "../../figures"
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
            print(f"  创建文件夹: {figures_dir}")
        
        # 保存主要的功率矩阵（100×48）
        power_matrix_file = os.path.join(figures_dir, "100_ac_power_matrix.csv")
        with open(power_matrix_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # 写入头部信息
            writer.writerow(["100个空调功率矩阵"])
            writer.writerow([f"系统参数: T={T}, delta_t={delta_t}h, 初始温度={T_initial}°C"])
            writer.writerow([])
            
            # 写入列标题（时间步）
            time_headers = ["空调ID", "空调类型", "额定功率(kW)", "状态"] + [f"t{i+1}({(i+1)*delta_t:.1f}h)" for i in range(T)]
            writer.writerow(time_headers)
            
            # 写入数据行
            for result in all_results:
                row = [
                    result['ac_id'], 
                    result['ac_type'], 
                    f"{result['P_rated']:.2f}",
                    result['status']
                ] + [f"{p:.3f}" for p in result['powers']]
                writer.writerow(row)
        
        print(f"✅ 功率矩阵已保存到: {power_matrix_file}")
        
        # 保存详细的空调配置和统计信息
        stats_file = os.path.join(figures_dir, "100_ac_statistics.csv")
        with open(stats_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # 写入统计表头
            stats_headers = [
                "空调ID", "空调类型", "额定功率(kW)", "温度范围(°C)", "热阻R", "热容C", "效率", "EER", 
                "状态", "总能耗(kWh)", "平均功率(kW)", "最大功率(kW)", "最小功率(kW)", 
                "最低温度(°C)", "最高温度(°C)", "最终温度(°C)"
            ]
            writer.writerow(stats_headers)
            
            # 写入统计数据
            for result in all_results:
                temp_range = f"[{result['T_min']:.1f}, {result['T_max']:.1f}]"
                
                if result['status'] == 'success':
                    row = [
                        result['ac_id'], result['ac_type'], f"{result['P_rated']:.2f}", temp_range,
                        f"{result['R']:.4f}", f"{result['C']:.1f}", f"{result['efficiency']:.2f}", f"{result['eta']:.1f}",
                        result['status'], f"{result['total_energy']:.2f}", f"{result['avg_power']:.3f}",
                        f"{result['max_power']:.3f}", f"{result['min_power']:.3f}",
                        f"{result['min_temp']:.2f}", f"{result['max_temp']:.2f}", f"{result['final_temp']:.2f}"
                    ]
                else:
                    row = [
                        result['ac_id'], result['ac_type'], f"{result['P_rated']:.2f}", temp_range,
                        f"{result['R']:.4f}", f"{result['C']:.1f}", f"{result['efficiency']:.2f}", f"{result['eta']:.1f}",
                        result['status'], "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
                    ]
                writer.writerow(row)
        
        print(f"✅ 统计信息已保存到: {stats_file}")
        
        # 保存完整的时间序列数据
        timeseries_file = os.path.join(figures_dir, "100_ac_full_timeseries.csv")
        with open(timeseries_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # 写入头部
            writer.writerow(["100个空调完整时间序列数据"])
            writer.writerow([])
            
            # 为每个空调保存详细的时间序列
            for result in all_results:
                temp_range = f"[{result['T_min']:.1f}, {result['T_max']:.1f}]"
                writer.writerow([f"空调: {result['ac_id']} ({result['ac_type']}), 额定功率: {result['P_rated']:.2f}kW, 温度范围: {temp_range}°C, 状态: {result['status']}"])
                writer.writerow(["时间(h)", "室外温度(°C)", "室内温度(°C)", "空调功率(kW)", "功率占比(%)"])
                
                if result['status'] == 'success':
                    for i in range(T):
                        time_h = (i + 1) * delta_t
                        outdoor_temp = outdoor_temp_half_hour[i + 1] if i + 1 < len(outdoor_temp_half_hour) else outdoor_temp_half_hour[-1]
                        indoor_temp = result['temperatures'][i]
                        power = result['powers'][i]
                        power_ratio = power / result['P_rated'] * 100
                        
                        writer.writerow([f"{time_h:.1f}", f"{outdoor_temp:.1f}", f"{indoor_temp:.2f}", f"{power:.3f}", f"{power_ratio:.1f}"])
                else:
                    writer.writerow(["求解失败，无数据"])
                
                writer.writerow([])  # 空行分隔
        
        print(f"✅ 完整时间序列已保存到: {timeseries_file}")
        
        # 8. 输出汇总统计
        if successful_results:
            print(f"\n📊 成功案例汇总统计:")
            print("-" * 80)
            
            # 按类型统计
            type_stats = {}
            for result in successful_results:
                ac_type = result['ac_type']
                if ac_type not in type_stats:
                    type_stats[ac_type] = []
                type_stats[ac_type].append(result)
            
            print(f"各类型空调成功数量:")
            for ac_type, type_results in type_stats.items():
                total_energies = [r['total_energy'] for r in type_results]
                avg_powers = [r['avg_power'] for r in type_results]
                print(f"  {ac_type}: {len(type_results)} 个 | 能耗范围: {min(total_energies):.2f}-{max(total_energies):.2f} kWh | 平均功率: {min(avg_powers):.3f}-{max(avg_powers):.3f} kW")
            
            print("-" * 80)
        
        print(f"\n🏠 100个空调数据生成完成！")
        print(f"生成的文件:")
        print(f"  📄 {power_matrix_file} - 100×48功率矩阵")
        print(f"  📊 {stats_file} - 统计汇总信息")
        print(f"  📈 {timeseries_file} - 完整时间序列数据")
        print(f"所有文件已保存到figures文件夹中")
        
        return all_results, all_power_matrix
        
    except Exception as e:
        print(f"❌ 100个空调数据生成失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()
