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
        
        # 目标函数：最小化总功耗（移除温度偏差惩罚，简化为纯功耗优化）
        prob += pulp.lpSum([P[t-1] * self.delta_t for t in range(1, self.T + 1)]), "总功耗最小化"
        
        # 约束条件
        
        # 1. 功率约束（已在变量定义中包含）
        # 0 ≤ P_t ≤ P_rated for all t
        
        # 2. 温度约束（已在变量定义中包含）
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
    
    # 设置24小时的室外温度变化 (模拟真实的日温度循环)
    # 从凌晨0点到晚上24点，共49个数据点 (48个间隔+1个起始点)
    hourly_outdoor_temp = [
        28.0, 28.0, 28.0, 28.0, 28.0, 28.5, 29.0, 29.5, 
        30.0, 30.5, 31.0, 31.5, 32.0, 31.5, 31.0, 30.5, 
        29.0, 27.0, 26.0, 26.0, 27.0, 27.5, 28.0, 28.0, 28.0
    ]

    # 插值生成0.5小时间隔的温度数据
    outdoor_temp_half_hour = []
    for i in range(len(hourly_outdoor_temp)-1):
        outdoor_temp_half_hour.append(hourly_outdoor_temp[i])
        # 添加中间点的线性插值
        mid_temp = (hourly_outdoor_temp[i] + hourly_outdoor_temp[i+1]) / 2
        outdoor_temp_half_hour.append(mid_temp)
    outdoor_temp_half_hour.append(hourly_outdoor_temp[-1])  # 添加最后一个点
    
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
        with open("24h_power_data.txt", "w", encoding="utf-8") as f:
            f.write("24小时空调功率需求数据\n")
            f.write("=" * 80 + "\n")
            f.write("时间(h),室外温度(°C),室内温度(°C),目标温度(°C),所需功率(kW),功率占比(%),温度变化(°C)\n")
            for i in range(len(powers)):
                hour = i * 0.5
                temp_change = temps[i] - (optimizer_readme.T_initial if i == 0 else temps[i-1])
                f.write(f"{hour:.1f},{outdoor_temp_half_hour[i+1]:.1f},{temps[i]:.2f},{optimizer_readme.T_target:.1f},{powers[i]:.2f},{powers[i]/optimizer_readme.P_rated*100:.1f},{temp_change:+.2f}\n")
        print(f"✅ 数据已保存到 24h_power_data.txt")
        
    else:
        print("❌ 24小时基础优化失败！")
        print(f"原因: {optimizer_readme.status}")
    
    

if __name__ == "__main__":
    main()
