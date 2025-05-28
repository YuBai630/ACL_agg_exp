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
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class ACOptimizerWithTempTarget:
    def __init__(self, T=24, delta_t=1.0, P_rated=3.0, T_min=20.0, T_max=26.0,
                 eta=0.8, R=2.0, C=5.0, T_initial=22.0, T_target=24.0, 
                 target_type='custom'):
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
        
        # 计算指数衰减因子: exp(-Δt/(R*C))
        # 这里 delta_t 是小时，R 是°C/kW，C 是 kWh/°C
        # 所以 R*C 的单位是 (°C/kW) * (kWh/°C) = h
        self.exp_factor = np.exp(-delta_t / (R * self.C))
        
        print(f"热容转换: {C:.1e} J/°C = {self.C:.1e} kWh/°C")
        print(f"时间常数 τ = R*C = {R:.1f} * {self.C:.1e} = {R * self.C:.2f} 小时")
        print(f"指数衰减因子: exp(-Δt/τ) = {self.exp_factor:.6f}")
        print(f"目标温度设置: {self.T_target}°C (类型: {target_type})")
        
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
        T_i = [pulp.LpVariable(f"T_i_{t}", lowBound=self.T_min, upBound=self.T_max) 
               for t in range(1, self.T + 1)]
        
        # 目标函数：最小化总功耗
        prob += pulp.lpSum([P[t-1] * self.delta_t for t in range(1, self.T + 1)]), "总功耗最小化"
        
        # 约束条件
        
        # 1. 功率约束（已在变量定义中包含）
        # 0 ≤ P_t ≤ P_rated for all t
        
        # 2. 温度约束（已在变量定义中包含）
        # T_min ≤ T_t ≤ T_max for all t
        
        # 3. 温度目标约束：第一个时间步结束时必须达到目标温度
        prob += T_i[0] == self.T_target, "第一时间步温度目标约束"
        
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
            
        else:
            self.status = f"求解失败: {pulp.LpStatus[prob.status]}"
            print(f"线性规划求解状态: {self.status}")
            
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
            print(f"  室内温度: {indoor_temp_prev:6.2f}°C → {indoor_temp_curr:6.2f}°C (变化: {temp_change:+.2f}°C)")
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
    """
    主函数：测试带温度目标约束的空调功率优化器
    包含多种不同的测试场景
    """
    print("=" * 100)
    print("空调功率优化器测试 - 带温度目标约束版本")
    print("=" * 100)
    
    # 测试场景1: 目标温度为最低温度 (降温场景)
    print("\n" + "🔥" * 50)
    print("测试场景1: 目标温度设为最低温度 (快速降温)")
    print("🔥" * 50)
    
    optimizer1 = ACOptimizerWithTempTarget(
        T=12,            # 12小时
        delta_t=1.0,     # 1小时时间步
        P_rated=5.0,     # 5kW额定功率
        T_min=20.0,      # 最低温度20°C
        T_max=26.0,      # 最高温度26°C
        eta=0.9,         # 效率0.9
        R=2.5,           # 热阻2.5°C/kW
        C=1.5e7,         # 热容1.5e7 J/°C
        T_initial=25.0,  # 初始温度25°C (较高)
        target_type='min' # 目标为最低温度
    )
    
    # 设置夏季高温室外温度
    summer_temp = [32.0, 33.0, 34.0, 35.0, 36.0, 35.0, 34.0, 32.0, 30.0, 29.0, 30.0, 31.0, 32.0]
    optimizer1.set_outdoor_temperature(summer_temp)
    
    if optimizer1.solve():
        optimizer1.print_results()
    else:
        print(f"场景1求解失败: {optimizer1.status}")
    
    # 测试场景2: 目标温度为最高温度 (升温场景)
    print("\n" + "❄️" * 50)
    print("测试场景2: 目标温度设为最高温度 (快速升温)")
    print("❄️" * 50)
    
    optimizer2 = ACOptimizerWithTempTarget(
        T=8,             # 8小时
        delta_t=0.5,     # 0.5小时时间步
        P_rated=3.5,     # 3.5kW额定功率
        T_min=18.0,      # 最低温度18°C
        T_max=24.0,      # 最高温度24°C
        eta=0.85,        # 效率0.85
        R=3.0,           # 热阻3.0°C/kW
        C=2.0e7,         # 热容2.0e7 J/°C
        T_initial=19.0,  # 初始温度19°C (较低)
        target_type='max' # 目标为最高温度
    )
    
    # 设置冬季低温室外温度 (需要16+1=17个数据点，因为T=8, delta_t=0.5)
    winter_temp = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 15.0, 14.0, 12.0, 10.0, 8.0]
    optimizer2.set_outdoor_temperature(winter_temp)
    
    if optimizer2.solve():
        optimizer2.print_results()
    else:
        print(f"场景2求解失败: {optimizer2.status}")
    
    # 测试场景3: 自定义目标温度 (舒适温度调节)
    print("\n" + "🌡️" * 50)
    print("测试场景3: 自定义目标温度 (精确温度控制)")
    print("🌡️" * 50)
    
    optimizer3 = ACOptimizerWithTempTarget(
        T=24,            # 24小时
        delta_t=1.0,     # 1小时时间步
        P_rated=4.0,     # 4kW额定功率
        T_min=21.0,      # 最低温度21°C
        T_max=25.0,      # 最高温度25°C
        eta=0.95,        # 高效率0.95
        R=2.0,           # 热阻2.0°C/kW
        C=1.8e7,         # 热容1.8e7 J/°C
        T_initial=22.5,  # 初始温度22.5°C
        T_target=23.5,   # 自定义目标温度23.5°C
        target_type='custom'
    )
    
    # 设置一天的室外温度变化 (模拟真实环境)
    daily_temp = [
        # 夜间: 0-6时
        26.0, 25.5, 25.0, 24.5, 24.0, 24.5,
        # 早晨: 6-12时  
        25.0, 26.0, 27.5, 29.0, 31.0, 33.0,
        # 下午: 12-18时
        35.0, 36.0, 35.5, 34.0, 32.0, 30.0,
        # 晚上: 18-24时
        28.5, 27.5, 27.0, 26.5, 26.0, 25.5,
        # 第二天开始
        25.0
    ]
    optimizer3.set_outdoor_temperature(daily_temp)
    
    if optimizer3.solve():
        optimizer3.print_results()
    else:
        print(f"场景3求解失败: {optimizer3.status}")
    
    # 测试场景4: 极端挑战场景 (大功率快速调节)
    print("\n" + "⚡" * 50)
    print("测试场景4: 极端挑战场景 (大功率快速温度调节)")
    print("⚡" * 50)
    
    optimizer4 = ACOptimizerWithTempTarget(
        T=6,             # 6小时短时间
        delta_t=0.25,    # 15分钟时间步 (精细控制)
        P_rated=8.0,     # 8kW大功率
        T_min=16.0,      # 宽温度范围
        T_max=28.0,      
        eta=0.92,        # 效率0.92
        R=1.5,           # 低热阻1.5°C/kW (快速响应)
        C=8.0e6,         # 小热容8.0e6 J/°C (快速变化)
        T_initial=16.5,  # 低初始温度
        T_target=27.0,   # 高目标温度 (大温差)
        target_type='custom'
    )
    
    # 设置变化剧烈的室外温度 (需要6/0.25+1=25个数据点)
    extreme_temp = [
        10.0, 12.0, 15.0, 18.0, 22.0, 25.0, 28.0, 32.0, 35.0, 38.0,
        40.0, 38.0, 35.0, 32.0, 28.0, 25.0, 22.0, 18.0, 15.0, 12.0,
        10.0, 8.0, 6.0, 5.0, 4.0
    ]
    optimizer4.set_outdoor_temperature(extreme_temp)
    
    if optimizer4.solve():
        optimizer4.print_results()
    else:
        print(f"场景4求解失败: {optimizer4.status}")
    
    # 测试场景5: 边界条件测试 (目标温度等于初始温度)
    print("\n" + "🎯" * 50)
    print("测试场景5: 边界条件测试 (目标温度等于初始温度)")
    print("🎯" * 50)
    
    optimizer5 = ACOptimizerWithTempTarget(
        T=10,            # 10小时
        delta_t=1.0,     # 1小时时间步
        P_rated=3.0,     # 3kW额定功率
        T_min=20.0,      # 温度范围
        T_max=26.0,      
        eta=0.88,        # 效率0.88
        R=2.2,           # 热阻2.2°C/kW
        C=1.2e7,         # 热容1.2e7 J/°C
        T_initial=23.0,  # 初始温度
        T_target=23.0,   # 目标温度等于初始温度
        target_type='custom'
    )
    
    # 设置稳定的室外温度
    stable_temp = [30.0] * 11  # 恒定30°C
    optimizer5.set_outdoor_temperature(stable_temp)
    
    if optimizer5.solve():
        optimizer5.print_results()
    else:
        print(f"场景5求解失败: {optimizer5.status}")
    
    print("\n" + "✅" * 50)
    print("所有测试场景完成！")
    print("✅" * 50)
    print(f"""
测试总结:
1. 场景1: 降温场景 - 验证从高温快速降至最低温度的功率调度
2. 场景2: 升温场景 - 验证从低温快速升至最高温度的功率调度  
3. 场景3: 日常调节 - 验证自定义目标温度的精确控制
4. 场景4: 极端挑战 - 验证大功率、大温差的快速调节能力
5. 场景5: 边界条件 - 验证目标温度等于初始温度的特殊情况

每个场景都展示了:
- 详细的控制周期信息
- 功率调度策略
- 温度目标约束的实现
- 能耗优化结果
""")

if __name__ == "__main__":
    main()
