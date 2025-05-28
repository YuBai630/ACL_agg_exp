"""
稳态概率验证
1个负荷聚合商
参数及其分布定义
parameters = {
    "房屋面积": {"distribution": "uniform", "params": [88, 176]},
    "换气次数": {"distribution": "normal", "params": [0.5, 0.06]},
    "窗墙比": {"distribution": "normal", "params": [0.15, 0.01]},
    "窗户太阳得热系数": {"distribution": "uniform", "params": [0.22, 0.5]},
    "空调负荷能效比": {"distribution": "uniform", "params": [3, 4]},
    "屋顶热阻": {"distribution": "normal", "params": [5.28, 0.70]},
    "墙体热阻": {"distribution": "normal", "params": [2.99, 0.35]},
    "地面热阻": {"distribution": "normal", "params": [3.35, 0.35]},
    "窗体热阻": {"distribution": "normal", "params": [0.38, 0.03]},
    "外门热阻": {"distribution": "normal", "params": [0.88, 0.07]}
}

"""

import random
import numpy as np

# 定义空调数量
N_AC = 1000

# 控制周期 (0.5小时，单位：秒)
CONTROL_CYCLE_SEC = 0.5 * 3600

# 仿真时间步长(秒)，根据FSM.py中的SIM_DT
SIM_DT = 2.0  # 仿真时间步长 (秒)

# Pa,ij*/Prate,ij = 0.2 (给定的条件)
# 这意味着功率比率 power_ratio = 0.2
# 根据 FSM.py 中的 calculate_migration_probabilities 逻辑:
# 当 power_ratio <= 0.5 时, u1 = 0.0, 且
# u0 = random.uniform(0.5 * Δt / tlock_sec, 1.5 * Δt / tlock_sec)
# 其中 Δt = CONTROL_CYCLE_SEC，tlock_sec 是 OFFLOCK 时间 tofflock_sec

# 关闭后闭锁时间统一设定为3分钟 (180秒)
FIXED_TOFFLOCK_SEC = 3 * 60.0

print(f"为 {N_AC} 台空调计算迁移概率 u0:")
print(f"条件: Pa,ij*/Prate,ij = 0.2, 控制周期 = {CONTROL_CYCLE_SEC} 秒.")
print(f"每台空调的 tofflock_sec 统一设定为 {FIXED_TOFFLOCK_SEC} 秒 (3分钟)。")
print("-" * 50)

u0_values = []
individual_tofflocks_sec = []

for i in range(N_AC):
    # 为当前空调随机生成 tofflock_sec (范围: 1分钟到3分钟)
    # current_tofflock_sec = random.uniform(60.0, 180.0) # 旧的随机逻辑
    current_tofflock_sec = FIXED_TOFFLOCK_SEC # 新的固定值
    individual_tofflocks_sec.append(current_tofflock_sec)
    
    # 计算 u0
    # 修正: u0应该是纯速率 (1/s)，不应乘以SIM_DT
    min_u0_val = 0.5 * SIM_DT / current_tofflock_sec 
    max_u0_val = 1.5 * SIM_DT / current_tofflock_sec 
    u0 = random.uniform(min_u0_val, max_u0_val)
    u0_values.append(u0)

print(f"\n为 {N_AC} 台空调计算得到的 u0 值示例如下 (前10台):")
for i in range(min(10, N_AC)):
    tofflock_val = individual_tofflocks_sec[i]
    u0_val = u0_values[i]
    expected_min_u0 = 0.5 * SIM_DT / tofflock_val
    expected_max_u0 = 1.5 * SIM_DT / tofflock_val
    print(f"  空调 {i+1:03d}: tofflock={tofflock_val:6.2f}s, u0={u0_val:8.4f} (理论范围: [{expected_min_u0:8.4f}, {expected_max_u0:8.4f}])")

if N_AC > 10:
    print("  ...")

# 可以进一步分析 u0_values 列表，例如计算平均值、标准差等
if N_AC > 0:
    avg_u0 = sum(u0_values) / N_AC
    min_observed_u0 = min(u0_values)
    max_observed_u0 = max(u0_values)
    print(f"\n对 {N_AC} 台空调计算的 u0 统计:")
    print(f"  平均 u0: {avg_u0:.4f}")
    print(f"  最小 u0: {min_observed_u0:.4f}")
    print(f"  最大 u0: {max_observed_u0:.4f}")

# 对于 tofflock_sec = 60s, u0 范围 = [0.5*1800/60, 1.5*1800/60] = [15, 45]
# 对于 tofflock_sec = 180s, u0 范围 = [0.5*1800/180, 1.5*1800/180] = [5, 15]
# 所以 u0 的整体范围大致在 [5, 45]

print("\n计算完成。")

# 增加计算持续时间和概率的代码
print("\n" + "="*60)
print("计算四个状态的持续时间和稳态概率")
print("="*60)

# 由于 Pa_ij_star / Prate_ij = 0.2 < 0.5, 因此 u1 接近于 0
# 但为了公式完整性，我们计算 u1
# 对于每个空调，计算 T1, T2, T3, T4 以及 P1, P2, P3, P4

T1_values = []  # ON状态持续时间：T1 = Δt/u0
T2_values = []  # OFF状态持续时间：T2 = Δt/u1
T3_values = []  # ONLOCK状态持续时间：T3 = tonlock
T4_values = []  # OFFLOCK状态持续时间：T4 = tofflock

P1_values = []  # ON状态概率：P1 = T1/(T1+T2+T3+T4)
P2_values = []  # OFF状态概率：P2 = T2/(T1+T2+T3+T4)
P3_values = []  # ONLOCK状态概率：P3 = T3/(T1+T2+T3+T4)
P4_values = []  # OFFLOCK状态概率：P4 = T4/(T1+T2+T3+T4)

# 为低功率比例情况计算 u1
# 根据 FSM.py 中的计算公式
# 当 power_ratio = 0.2 < 0.5 时, u1 接近 0，但我们也计算其确切值
power_ratio = 0.2  # Pa_ij_star / Prate_ij
TONLOCK_SEC = 3 * 60.0  # 开启后闭锁时间也设为3分钟（与关闭后闭锁时间相同）

# 计算每台空调的稳态参数
for i in range(N_AC):
    u0 = u0_values[i] # u0已经是纯速率 u0_rate
    
    # 修正 u1 的计算公式，基于纯速率 u0_rate
    u1 = 0.0 # 默认值
    if power_ratio > 1e-9: # 避免 power_ratio 为0时除零
        # denominator_for_u1_value = (SIM_DT/u0 + TONLOCK_SEC)*(1-power_ratio) - power_ratio*FIXED_TOFFLOCK_SEC # 旧公式
        # 新公式基于 T1=1/u0, T2=1/u1
        # u1_rate = power_ratio / ( (1.0/u0_rate + TONLOCK_SEC)*(1-power_ratio) - power_ratio*FIXED_TOFFLOCK_SEC )
        actual_denominator_for_u1 = (1.0/u0 + TONLOCK_SEC)*(1-power_ratio) - power_ratio*FIXED_TOFFLOCK_SEC
            
        if abs(actual_denominator_for_u1) > 1e-10 and actual_denominator_for_u1 > 0: #分母需为正，保证u1为正
             u1 = power_ratio / actual_denominator_for_u1
             if u1 < 0: #额外保护，理论上若power_ratio和参数合理，不应发生
                 u1 = 1e-10 
        else:
             u1 = 1e-10 # 分母接近0或为负，设为极小正值，意味着T2极大
    else: 
        u1 = 0 
    
    # 计算四个状态的持续时间，基于纯速率
    T1 = 1.0 / u0 if u0 > 1e-10 else float('inf')
    T2 = 1.0 / u1 if u1 > 1e-10 else float('inf')
    T3 = TONLOCK_SEC
    T4 = FIXED_TOFFLOCK_SEC
    
    # 处理可能的无限大值
    if T2 > 1e10:  # 如果T2太大，视为有限但很大的值
        T2 = 1e10
    
    # 计算稳态概率
    T_sum = T1 + T2 + T3 + T4
    P1 = T1 / T_sum
    P2 = T2 / T_sum
    P3 = T3 / T_sum
    P4 = T4 / T_sum
    
    # 存储结果
    T1_values.append(T1)
    T2_values.append(T2)
    T3_values.append(T3)
    T4_values.append(T4)
    
    P1_values.append(P1)
    P2_values.append(P2)
    P3_values.append(P3)
    P4_values.append(P4)

# 输出部分结果
print("\n前10台空调的持续时间和稳态概率:")
print("  索引    T1(ON)   T2(OFF)  T3(ONLOCK) T4(OFFLOCK)    P1(ON)    P2(OFF)  P3(ONLOCK) P4(OFFLOCK)")
print("-" * 100)

for i in range(min(10, N_AC)):
    print(f"{i+1:5d} {T1_values[i]:10.2f} {T2_values[i]:10.2f} {T3_values[i]:10.2f} {T4_values[i]:11.2f} "
          f"{P1_values[i]:10.6f} {P2_values[i]:10.6f} {P3_values[i]:10.6f} {P4_values[i]:10.6f}")

# 计算并显示平均值
if N_AC > 0:
    avg_T1 = sum(T1_values) / N_AC
    avg_T2 = sum(T2_values) / N_AC
    avg_T3 = sum(T3_values) / N_AC
    avg_T4 = sum(T4_values) / N_AC
    
    avg_P1 = sum(P1_values) / N_AC
    avg_P2 = sum(P2_values) / N_AC
    avg_P3 = sum(P3_values) / N_AC
    avg_P4 = sum(P4_values) / N_AC
    
    print("\n所有空调的平均持续时间和稳态概率:")
    print(f"平均 T1(ON): {avg_T1:.2f} 秒")
    print(f"平均 T2(OFF): {avg_T2:.2f} 秒")
    print(f"平均 T3(ONLOCK): {avg_T3:.2f} 秒")
    print(f"平均 T4(OFFLOCK): {avg_T4:.2f} 秒")
    print()
    print(f"平均 P1(ON): {avg_P1:.6f}")
    print(f"平均 P2(OFF): {avg_P2:.6f}")
    print(f"平均 P3(ONLOCK): {avg_P3:.6f}")
    print(f"平均 P4(OFFLOCK): {avg_P4:.6f}")
    
    # 验证概率之和是否为1
    avg_P_sum = avg_P1 + avg_P2 + avg_P3 + avg_P4
    print(f"\n概率总和检验: {avg_P_sum:.6f} (应接近1)")

# 进一步统计：当前控制周期内的物理开启时间和物理关闭时间的比例
avg_on_time_ratio = (avg_P1 + avg_P3) if 'avg_P1' in locals() and 'avg_P3' in locals() else 0
avg_off_time_ratio = (avg_P2 + avg_P4) if 'avg_P2' in locals() and 'avg_P4' in locals() else 0

print(f"\n物理开启状态时间比例 (P1+P3): {avg_on_time_ratio:.6f} (这应接近原始功率比率 {power_ratio:.2f})")
print(f"物理关闭状态时间比例 (P2+P4): {avg_off_time_ratio:.6f}")

print("\n统计计算完成。")

# ==============================================================================
# 时域仿真：P1 (ON 状态) 随时间的变化
# ==============================================================================
print("\n" + "="*60)
print("时域仿真：P1 (ON 状态) 随时间的变化")
print("="*60)

try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
    from FSM import ACL_State, ACL_FSM # 假设 FSM.py 在同一目录或PYTHONPATH中
except ImportError:
    print("错误：无法导入 matplotlib 或 FSM模块。请确保已安装 matplotlib 并且 FSM.py 可访问。")
    print("pip install matplotlib")
    # 如果 FSM.py 不在 src 目录，或者运行时路径问题，需要调整导入
    # 例如: from .FSM import ACL_State, ACL_FSM (如果作为包的一部分运行)
    # 或者确保 FSM.py 所在的目录在 sys.path 中
    exit()

# 仿真参数
TOTAL_SIM_TIME_SEC = 1 * 3600  # 总仿真时间 (1小时，对应图表)
# SIM_DT 已经定义为 2.0 秒
PLOT_INTERVAL_SEC = 30         # 每隔多少秒记录一次数据用于绘图 (例如30秒)
                               # 图表横坐标到1h，如果步长太小，点会过多

# 初始化 N_AC 台空调的 FSM
# 假设初始状态全部为 ON，以匹配图表从 P1=1 开始
initial_fsm_state = ACL_State.ON
all_fsms = []
for i in range(N_AC):
    # 使用之前为稳态计算的 u0_values[i] 和 u1_values[i]
    # TONLOCK_SEC 和 FIXED_TOFFLOCK_SEC 也是之前定义的 (3分钟)
    fsm = ACL_FSM(initial_state=initial_fsm_state, 
                  tonlock_sec=TONLOCK_SEC, 
                  tofflock_sec=FIXED_TOFFLOCK_SEC,
                  sim_dt_sec=SIM_DT)  # 添加sim_dt_sec参数
    
    # 为每个FSM设置其特定的u0, u1迁移率
    # u0_values 和 u1_values 是之前脚本部分计算得到的列表
    current_u0 = u0_values[i]
    
    # 计算对应的u1 (与之前稳态部分逻辑一致)
    # 修正 u1 的计算公式 (与稳态部分保持一致)
    current_u1 = 0.0 # 默认值
    if power_ratio > 1e-9: 
        # denominator_for_u1_value_sim = (SIM_DT/current_u0 + TONLOCK_SEC)*(1-power_ratio) - power_ratio*FIXED_TOFFLOCK_SEC #旧公式
        actual_denominator_for_u1_sim = (1.0/current_u0 + TONLOCK_SEC)*(1-power_ratio) - power_ratio*FIXED_TOFFLOCK_SEC
        
        if abs(actual_denominator_for_u1_sim) > 1e-10 and actual_denominator_for_u1_sim > 0:
            current_u1 = power_ratio / actual_denominator_for_u1_sim
            if current_u1 < 0: 
                current_u1 = 1e-10 
        else:
            current_u1 = 1e-10 
    else: 
        current_u1 = 0 
            
    fsm.update_migration_probabilities(u0=current_u0, u1=current_u1)
    all_fsms.append(fsm)

print(f"初始化 {N_AC} 台空调FSM，初始状态: {initial_fsm_state}")
print(f"总仿真时间: {TOTAL_SIM_TIME_SEC / 3600:.1f} 小时, 仿真步长: {SIM_DT} 秒")
print(f"绘图数据记录间隔: {PLOT_INTERVAL_SEC} 秒")

# 仿真循环
time_points_for_plot_hr = []      # 时间点 (小时)
p1_proportion_over_time = []  # P1 (ON状态) 空调的比例
p2_proportion_over_time = []  # P2 (OFF状态) 空调的比例
p3_proportion_over_time = []  # P3 (ONLOCK状态) 空调的比例
p4_proportion_over_time = []  # P4 (OFFLOCK状态) 空调的比例

current_sim_time_sec = 0.0
next_plot_record_time_sec = 0.0

num_steps = int(TOTAL_SIM_TIME_SEC / SIM_DT)
print(f"总仿真步数: {num_steps}")

for step in range(num_steps + 1):
    current_sim_time_sec = step * SIM_DT
    
    # 记录数据用于绘图
    if current_sim_time_sec >= next_plot_record_time_sec or step == num_steps :
        on_state_count = 0
        off_state_count = 0
        onlock_state_count = 0
        offlock_state_count = 0

        for fsm in all_fsms:
            # 直接获取当前状态，而不是历史状态
            current_fsm_state = fsm.current_state 
            if current_fsm_state == ACL_State.ON:
                on_state_count += 1
            elif current_fsm_state == ACL_State.OFF:
                off_state_count += 1
            elif current_fsm_state == ACL_State.ONLOCK:
                onlock_state_count += 1
            elif current_fsm_state == ACL_State.OFFLOCK:
                offlock_state_count += 1
        
        p1_proportion = on_state_count / N_AC
        p2_proportion = off_state_count / N_AC
        p3_proportion = onlock_state_count / N_AC
        p4_proportion = offlock_state_count / N_AC
        
        time_points_for_plot_hr.append(current_sim_time_sec / 3600.0)
        p1_proportion_over_time.append(p1_proportion)
        p2_proportion_over_time.append(p2_proportion)
        p3_proportion_over_time.append(p3_proportion)
        p4_proportion_over_time.append(p4_proportion)
        
        # print(f"Time: {current_sim_time_sec/3600.0:.3f}h, P1: {p1_proportion:.4f}") # 调试信息
        next_plot_record_time_sec += PLOT_INTERVAL_SEC
        if step == num_steps and current_sim_time_sec < next_plot_record_time_sec : #确保最后一点被记录
             pass # 已在循环条件 or step == num_steps 中处理


    # 更新所有FSM的状态 (除了最后一次记录数据之后)
    if step < num_steps:
      for fsm in all_fsms:
          fsm.step(SIM_DT)

print("仿真完成。开始绘图...")

# 绘图 - 创建2x2的子图布局
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('空调状态概率随时间的变化', fontsize=16)

# 计算各状态的比例
time_points_for_plot_hr = np.array(time_points_for_plot_hr)
p1_proportion_over_time = np.array(p1_proportion_over_time)

# 统计其他状态的比例
p2_proportion_over_time = np.array(p2_proportion_over_time)
p3_proportion_over_time = np.array(p3_proportion_over_time)
p4_proportion_over_time = np.array(p4_proportion_over_time)

# 绘制P1 (ON状态)
axs[0, 0].plot(time_points_for_plot_hr, p1_proportion_over_time, label=f'暂态概率 (n={N_AC})', color='red')
if 'avg_P1' in locals() or 'avg_P1' in globals():
    axs[0, 0].axhline(y=avg_P1, color='black', linestyle='--', label=f'稳态概率 (理论均值: {avg_P1:.4f})')
axs[0, 0].set_title('P1 (ON状态)')
axs[0, 0].set_xlabel('t/h (时间/小时)')
axs[0, 0].set_ylabel('概率')
axs[0, 0].set_ylim(0, 1.05)
axs[0, 0].set_xlim(0, TOTAL_SIM_TIME_SEC / 3600.0)
axs[0, 0].grid(True)
axs[0, 0].legend()

# 绘制P2 (OFF状态)
axs[0, 1].plot(time_points_for_plot_hr, p2_proportion_over_time, label=f'暂态概率 (n={N_AC})', color='blue')
if 'avg_P2' in locals() or 'avg_P2' in globals():
    axs[0, 1].axhline(y=avg_P2, color='black', linestyle='--', label=f'稳态概率 (理论均值: {avg_P2:.4f})')
axs[0, 1].set_title('P2 (OFF状态)')
axs[0, 1].set_xlabel('t/h (时间/小时)')
axs[0, 1].set_ylabel('概率')
axs[0, 1].set_ylim(0, 1.05)
axs[0, 1].set_xlim(0, TOTAL_SIM_TIME_SEC / 3600.0)
axs[0, 1].grid(True)
axs[0, 1].legend()

# 绘制P3 (ONLOCK状态)
axs[1, 0].plot(time_points_for_plot_hr, p3_proportion_over_time, label=f'暂态概率 (n={N_AC})', color='green')
if 'avg_P3' in locals() or 'avg_P3' in globals():
    axs[1, 0].axhline(y=avg_P3, color='black', linestyle='--', label=f'稳态概率 (理论均值: {avg_P3:.4f})')
axs[1, 0].set_title('P3 (ONLOCK状态)')
axs[1, 0].set_xlabel('t/h (时间/小时)')
axs[1, 0].set_ylabel('概率')
axs[1, 0].set_ylim(0, 1.05)
axs[1, 0].set_xlim(0, TOTAL_SIM_TIME_SEC / 3600.0)
axs[1, 0].grid(True)
axs[1, 0].legend()

# 绘制P4 (OFFLOCK状态)
axs[1, 1].plot(time_points_for_plot_hr, p4_proportion_over_time, label=f'暂态概率 (n={N_AC})', color='orange')
if 'avg_P4' in locals() or 'avg_P4' in globals():
    axs[1, 1].axhline(y=avg_P4, color='black', linestyle='--', label=f'稳态概率 (理论均值: {avg_P4:.4f})')
axs[1, 1].set_title('P4 (OFFLOCK状态)')
axs[1, 1].set_xlabel('t/h (时间/小时)')
axs[1, 1].set_ylabel('概率')
axs[1, 1].set_ylim(0, 1.05)
axs[1, 1].set_xlim(0, TOTAL_SIM_TIME_SEC / 3600.0)
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

print("绘图完成。")