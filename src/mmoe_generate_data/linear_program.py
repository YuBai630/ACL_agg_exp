"""
空调功率优化线性规划问题求解

M_{AC}: \min_{\left \{P_{1},...,P_{T},T_{2}^{i},...,T_{T}^{i}\right \} }\sum_{t=1}^{T}P_{t}\Delta t,
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
            
    def solve(self):
        """求解线性规划问题"""
        # 创建线性规划问题
        prob = pulp.LpProblem("AC_Power_Optimization", pulp.LpMinimize)
        
        # 决策变量
        # P_t: 每个时间步的功率
        P = [pulp.LpVariable(f"P_{t}", lowBound=0, upBound=self.P_rated) 
             for t in range(1, self.T + 1)]
        
        # T_i_t: 每个时间步的室内温度
        T_i = [pulp.LpVariable(f"T_i_{t}", lowBound=self.T_min, upBound=self.T_max) 
               for t in range(2, self.T + 2)]
        
        # 目标函数：最小化总功耗
        prob += pulp.lpSum([P[t-1] * self.delta_t for t in range(1, self.T + 1)])
        
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
            self.optimal_powers = [P[t-1].varValue for t in range(1, self.T + 1)]
            self.optimal_temperatures = [self.T_initial] + [T_i[t-2].varValue for t in range(2, self.T + 2)]
            self.total_energy = sum(self.optimal_powers) * self.delta_t
            self.status = "最优解"
        else:
            self.status = f"求解失败: {pulp.LpStatus[prob.status]}"
            
        return prob.status == pulp.LpStatusOptimal
    
    def plot_results(self):
        """绘制结果图表"""
        if not hasattr(self, 'optimal_powers'):
            print("请先求解问题")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
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
            print(f"平均功率: {self.total_energy/self.T:.2f} kW")
            print("\n时间步 | 功率(kW) | 室内温度(°C) | 室外温度(°C)")
            print("-" * 50)
            for t in range(self.T):
                print(f"{t+1:6d} | {self.optimal_powers[t]:8.2f} | {self.optimal_temperatures[t+1]:11.2f} | {self.T_out[t+1]:11.2f}")

def main():
    """主函数示例"""
    # 创建优化器实例
    optimizer = ACOptimizer(
        T=48,           # 24小时，每0.5小时一个控制周期，共48个时间步
        delta_t=0.5,    # 0.5小时时间步
        P_rated=12.0,    # 额定功率
        T_min=21.0,     # 最低温度21°C
        T_max=25.0,     # 最高温度24°C
        eta=0.98,       # 效率0.98
        R=3.0,          # 热阻3°C/kW
        C=1.8e7,        # 热容J∕°C
        T_initial=23.0  # 初始温度23°C
    )
    
    # 设置室外温度（模拟一天的温度变化，每0.5小时一个数据点）
    # 将原有的24小时温度插值扩展为48个时间步
    original_temp = [
        28.0, 28.0, 28.0, 28.0, 28.0, 28.5, 29.0, 29.5, 
        30.0, 30.5, 31.0, 31.5, 32.0, 31.5, 31.0, 30.5, 
        29.0, 27.0, 26.0, 26.0, 27.0, 27.5, 28.0, 28.0, 28.0
    ]
    
    # 通过线性插值将原始24小时数据扩展为48个时间步
    outdoor_temp = []
    for i in range(len(original_temp)-1):
        outdoor_temp.append(original_temp[i])
        outdoor_temp.append((original_temp[i] + original_temp[i+1]) / 2)
    outdoor_temp.append(original_temp[-1])  # 添加最后一个值
    
    optimizer.set_outdoor_temperature(outdoor_temp)
    
    # 求解问题
    if optimizer.solve():
        optimizer.print_results()
        optimizer.plot_results()
    else:
        print(f"求解失败: {optimizer.status}")

if __name__ == "__main__":
    main() 