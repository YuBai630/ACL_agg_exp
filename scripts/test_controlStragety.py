import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import pandas as pd
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

# 尝试找到一个可用的中文字体
font_path = None
if os.name == 'nt': # Windows
    font_path = 'C:/Windows/Fonts/simhei.ttf' # 黑体
    if not os.path.exists(font_path):
        font_path = 'C:/Windows/Fonts/msyh.ttf' # 微软雅黑
elif os.name == 'posix': # Linux or macOS
    # 尝试常见的Linux路径
    common_linux_paths = [
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    ]
    for p in common_linux_paths:
        if os.path.exists(p):
            font_path = p
            break
    if not font_path and hasattr(os, 'uname') and os.uname().sysname == "Darwin": # macOS
        font_path = '/System/Library/Fonts/PingFang.ttc'
        if not os.path.exists(font_path):
             font_path = '/System/Library/Fonts/Supplemental/Songti.ttc' # 宋体

if font_path and os.path.exists(font_path):
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    print("警告：未找到指定的中文字体，图形中的中文可能无法正常显示。")
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 假设src目录与scripts目录在同一父目录下
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.oneControl_stragety import SingleAirConditioner, Aggregator, Grid
from src.ETP import SecondOrderETPModel
from src.FSM import ACL_FSM, ACL_State

# --- 新增：读取室外温度数据的函数 (来自 test_controlByItself.py) ---
def load_outdoor_temperature(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    return df
# --- 函数结束 ---

# --- 模拟参数 ---
NUM_AGGREGATORS = 4
ACS_PER_AGGREGATOR = 250 # 每个聚合商管理250台空调，总共1000台 -> Changed for potentially faster testing
CONTROL_CYCLE_DURATION_SEC = 30 * 60  # 30分钟控制周期
SIM_DT_SEC = 60  # FSM和ETP的内部仿真步长 (秒)

# 时间参数调整
START_HOUR_PRE_DR = 9    # 常规运行开始小时
START_HOUR_DR = 11       # 需求响应开始小时 (新: 11:00)
END_HOUR_DR = 14         # 需求响应结束小时 (新: 14:00)
END_HOUR_SIM = 21        # 模拟在21:00前结束 (覆盖到20:30-21:00的周期)

TOTAL_SIM_DURATION_SEC = (END_HOUR_SIM - START_HOUR_PRE_DR) * 3600
NUM_CONTROL_CYCLES = int(TOTAL_SIM_DURATION_SEC / CONTROL_CYCLE_DURATION_SEC)

# --- 室外温度设置 (采用 test_controlByItself.py 的方式) ---
outdoor_temp_file = "D:\\\\experiments\\\\ACL_agg_exp\\\\data\\\\outdoorTemperature.csv" # 使用双反斜杠以避免转义问题
all_outdoor_data = load_outdoor_temperature(outdoor_temp_file)

sim_date_str = "8/29/2007" # 假设与 test_controlByItself.py 中的日期一致

# 定义插值所需的时间窗口起点 (基于新的最早开始时间)
interp_window_start_dt = datetime.strptime(f"{sim_date_str} {START_HOUR_PRE_DR}:00:00", "%m/%d/%Y %H:%M:%S")

# 为了插值的鲁棒性，使用一个比模拟时间稍宽的数据窗口
interp_data_filter_start_dt = interp_window_start_dt - timedelta(hours=1)
# 确保结束时间覆盖整个模拟周期
interp_data_filter_end_dt = interp_window_start_dt + timedelta(seconds=TOTAL_SIM_DURATION_SEC) + timedelta(hours=1)

data_for_interpolation = all_outdoor_data[
    (all_outdoor_data['Time'] >= interp_data_filter_start_dt) &
    (all_outdoor_data['Time'] <= interp_data_filter_end_dt)
].copy()

if data_for_interpolation.empty:
    raise ValueError(
        f"在 {outdoor_temp_file} 中找不到从 {interp_data_filter_start_dt} 到 {interp_data_filter_end_dt} 的室外温度数据用于插值。"
    )

interp_data_timestamps_relative_sec = [
    (t - interp_window_start_dt).total_seconds() for t in data_for_interpolation['Time']
]
interp_data_temperatures = data_for_interpolation['Temperature'].values
outdoor_temp_interpolator = interp1d(
    interp_data_timestamps_relative_sec,
    interp_data_temperatures,
    bounds_error=False,
    fill_value="extrapolate"
)
# --- 室外温度设置结束 ---

# --- 从图像估算的功率数据 (单位: MW) ---
# 时间点: 18:00, 18:30, 19:00, 19:30, 20:00, 20:30 (对应每个控制周期的开始)
time_labels_plot = [f"{START_HOUR_PRE_DR + i*0.5:.1f}".replace(".5", ":30").replace(".0",":00") for i in range(NUM_CONTROL_CYCLES + 1)]
time_labels_data = time_labels_plot[:-1] # 用于数据索引

# 总基准功率 (MW) - 粉色点状线
image_total_baseline_mw = np.array([52, 44, 38, 35, 32, 31])
# 总目标响应功率 (MW) - 粉色阶梯线
image_total_target_response_mw = np.array([44, 38, 32, 29, 27, 26])

# 电网需要的总调峰功率 (MW) = 基准 - 目标响应
image_grid_total_peak_shaving_target_mw = image_total_baseline_mw - image_total_target_response_mw


# --- 模型参数 ---
ETP_PARAMS = { "Ca": 1.2e5, "Cm": 2.5e6, "Ua": 50, "Um": 500 }
FSM_PARAMS = {"tonlock_sec": 3 * 60, "tofflock_sec": 3 * 60}
AC_CONFIG_BASE = {
    "Tset": 24.0, "Tmin": 22.0, "Tmax": 26.0, "Prate": 2000, "COP": 3.0,
    "Ua_baseline": 50, "c_param": 1e5, "r1_param": -0.0005,
    "r2_param": -0.0025, "eta_param": 0.001, "Qm_ij_val": 100
}
# GRID_OPT_PARAMS is no longer needed as dispatch is simplified or maximal.


# --- 初始化函数 ---
def create_ac_system():
    aggregators = []
    all_acs_flat = []
    for i in range(NUM_AGGREGATORS):
        acs_in_agg = []
        for j in range(ACS_PER_AGGREGATOR):
            ac_id = f"agg{i+1}_ac{j+1}"
            ac_config = AC_CONFIG_BASE.copy()
            ac_config["Tset"] = np.random.uniform(23.5, 24.5)
            ac_config["Prate"] = np.random.uniform(1800, 2200)
            etp_hm = -ac_config["Prate"] * ac_config["COP"]
            current_etp_params = ETP_PARAMS.copy()
            current_etp_params["Hm"] = etp_hm
            current_etp_params["Qgain"] = ac_config["Qm_ij_val"]
            etp_model = SecondOrderETPModel(**current_etp_params)
            initial_state = ACL_State.ON if np.random.rand() > 0.5 else ACL_State.OFF # Random initial state
            fsm_model = ACL_FSM(initial_state=initial_state, **FSM_PARAMS, sim_dt_sec=SIM_DT_SEC)
            ac = SingleAirConditioner(
                id=ac_id, etp_model=etp_model, fsm_model=fsm_model,
                Tset=ac_config["Tset"], Tmin=ac_config["Tmin"], Tmax=ac_config["Tmax"],
                Prate=ac_config["Prate"], COP=ac_config["COP"], Ua=ac_config["Ua_baseline"],
                c_param=ac_config["c_param"], r1_param=ac_config["r1_param"],
                r2_param=ac_config["r2_param"], eta_param=ac_config["eta_param"],
                Qm_ij_val=ac_config["Qm_ij_val"]
            )
            # Initialize temperatures more realistically
            if initial_state == ACL_State.ON:
                ac.current_Ta_ij = np.random.uniform(ac_config["Tset"] - 0.5, ac_config["Tset"] + 0.5) # Closer to Tset if ON
            else: # OFF
                ac.current_Ta_ij = np.random.uniform(ac_config["Tset"] + 0.5, ac_config["Tset"] + 2.0) # Warmer if OFF
            ac.current_Tm_ij = ac.current_Ta_ij + np.random.uniform(0.1, 0.5) # Tm slightly higher than Ta
            
            acs_in_agg.append(ac)
            all_acs_flat.append(ac)
        agg = Aggregator(id=f"agg{i+1}", air_conditioners_list=acs_in_agg)
        aggregators.append(agg)
    grid = Grid(aggregators_list=aggregators)
    return grid, aggregators, all_acs_flat

# --- 主模拟循环 ---
print("开始模拟...")
grid_model, aggregator_list, ac_list = create_ac_system()

# Results storage
results_time_labels = []
results_sim_total_baseline_power_W = []
results_calculated_target_response_power_W = [] # New: For dynamic target based on phase
results_sim_total_actual_response_power_W = []
results_sim_agg_actual_response_power_W = [[] for _ in range(NUM_AGGREGATORS)]
results_outdoor_temp_C = []
results_avg_indoor_temp_C = []


current_sim_time_sec = 0
for cycle_idx in range(NUM_CONTROL_CYCLES):
    current_hour_of_day = START_HOUR_PRE_DR + (current_sim_time_sec / 3600)
    time_label_for_plot = f"{int(current_hour_of_day):02d}:{int((current_hour_of_day % 1) * 60):02d}"
    results_time_labels.append(time_label_for_plot)
    
    print(f"\n--- 控制周期 {cycle_idx+1}/{NUM_CONTROL_CYCLES} (时刻: {time_label_for_plot}, 模拟秒: {current_sim_time_sec}) ---")

    To_current = outdoor_temp_interpolator(current_sim_time_sec)
    results_outdoor_temp_C.append(To_current)
    print(f"当前室外温度: {To_current:.1f}°C")

    print("步骤1&2: 空调形成需求曲线, 聚合商聚合计算...")
    current_cycle_total_sim_baseline_W = 0
    current_total_sim_peak_shaving_capacity_W = 0
    for agg in aggregator_list:
        for ac in agg.acs:
            ac.form_demand_curve(CONTROL_CYCLE_DURATION_SEC, To_current)
        agg_baseline_W = agg.calculate_total_baseline_power(To_current)
        current_cycle_total_sim_baseline_W += agg_baseline_W
        agg.aggregate_demand_curves()
        agg_capacity_W = agg.calculate_peak_shaving_capacity()
        current_total_sim_peak_shaving_capacity_W += agg_capacity_W
    results_sim_total_baseline_power_W.append(current_cycle_total_sim_baseline_W)
    print(f"  模拟总基准功率: {current_cycle_total_sim_baseline_W / 1e6:.2f} MW")
    print(f"  模拟总调峰容量: {current_total_sim_peak_shaving_capacity_W / 1e6:.2f} MW")

    optimal_dispatch_results_W = {}
    calculated_target_for_this_cycle_W = 0

    if START_HOUR_DR <= current_hour_of_day < END_HOUR_DR:
        print(f"处于需求响应阶段 ({START_HOUR_DR}:00-{END_HOUR_DR}:00) - 目标削减基准的20%")
        # 步骤3 (DR): 计算目标总削减量 (基准的20%)
        grid_target_total_reduction_W = current_cycle_total_sim_baseline_W * 0.20
        print(f"  电网目标总调峰量 (基准的20%): {grid_target_total_reduction_W / 1e6:.2f} MW")

        # 使用 solve_optimal_dispatch 分配这个调峰任务
        # 该方法已被简化为按 peak_shaving_capacity 比例分配 P_target_total
        optimal_dispatch_results_W = grid_model.solve_optimal_dispatch(grid_target_total_reduction_W)
        
        actual_dispatched_sum_W = sum(optimal_dispatch_results_W.values())
        print(f"  聚合商实际被分配的总调峰任务量: {actual_dispatched_sum_W / 1e6:.2f} MW")
        
        calculated_target_for_this_cycle_W = current_cycle_total_sim_baseline_W - actual_dispatched_sum_W
    else:
        print(f"处于常规运行阶段 (非 {START_HOUR_DR}:00-{END_HOUR_DR}:00) - 无强制调峰")
        # 步骤3 (Pre-DR/Post-DR): 电网优化调度 - 目标调峰量为0
        for agg in grid_model.aggregators:
            optimal_dispatch_results_W[agg.id] = 0.0 # No peak shaving
        print(f"  电网目标总调峰量: 0.00 MW")
        calculated_target_for_this_cycle_W = current_cycle_total_sim_baseline_W # Target is baseline

    results_calculated_target_response_power_W.append(calculated_target_for_this_cycle_W)
    print(f"  本周期计算目标总响应功率: {calculated_target_for_this_cycle_W / 1e6:.2f} MW")
    
    print("步骤4: 电网下发指令, 聚合商反解虚拟价格并广播...")
    virtual_prices = grid_model.dispatch_power_commands_to_aggregators(
        optimal_dispatch_results_W, To_current, CONTROL_CYCLE_DURATION_SEC
    )
    # (Optional: print virtual prices for debugging)
    # for agg_id_vp, vp in virtual_prices.items():
    #     print(f"  Agg {agg_id_vp} virtual price: {vp:.3f}")

    print("步骤5: 空调本地控制模拟...")
    current_cycle_agg_actual_power_sum_W = [0.0] * NUM_AGGREGATORS
    num_small_steps = 0
    total_indoor_temp_sum_for_avg = 0
    total_ac_count_for_avg = 0

    sub_loop_time_sec = 0
    while sub_loop_time_sec < CONTROL_CYCLE_DURATION_SEC:
        for agg_idx, agg in enumerate(aggregator_list):
            agg_power_at_this_small_step_W = 0
            for ac in agg.acs:
                _, _, mode = ac.run_local_control_step(SIM_DT_SEC, To_current)
                ac_power = ac.Prate_ij * mode
                agg_power_at_this_small_step_W += ac_power
                total_indoor_temp_sum_for_avg += ac.current_Ta_ij
                total_ac_count_for_avg +=1
            current_cycle_agg_actual_power_sum_W[agg_idx] += agg_power_at_this_small_step_W
        sub_loop_time_sec += SIM_DT_SEC
        num_small_steps += 1
    
    avg_indoor_temp_this_cycle = total_indoor_temp_sum_for_avg / total_ac_count_for_avg if total_ac_count_for_avg > 0 else To_current
    results_avg_indoor_temp_C.append(avg_indoor_temp_this_cycle)
    print(f"  本周期结束时平均室内温度: {avg_indoor_temp_this_cycle:.2f}°C")

    current_cycle_total_actual_response_W = 0
    for agg_idx in range(NUM_AGGREGATORS):
        avg_agg_power_W = current_cycle_agg_actual_power_sum_W[agg_idx] / num_small_steps if num_small_steps > 0 else 0
        results_sim_agg_actual_response_power_W[agg_idx].append(avg_agg_power_W)
        current_cycle_total_actual_response_W += avg_agg_power_W
    results_sim_total_actual_response_power_W.append(current_cycle_total_actual_response_W)
    print(f"  模拟总实际响应功率: {current_cycle_total_actual_response_W / 1e6:.2f} MW")
    
    current_sim_time_sec += CONTROL_CYCLE_DURATION_SEC

print("\n模拟结束.")

# --- 结果绘图 ---
print("生成结果图表...")

results_sim_total_baseline_power_MW = np.array(results_sim_total_baseline_power_W) / 1e6
results_calculated_target_response_power_MW = np.array(results_calculated_target_response_power_W) / 1e6
results_sim_total_actual_response_power_MW = np.array(results_sim_total_actual_response_power_W) / 1e6
results_sim_agg_actual_response_power_MW = np.array(results_sim_agg_actual_response_power_W) / 1e6

fig, ax1 = plt.subplots(figsize=(16, 9)) # Wider figure for more data points

x_ticks_positions = np.arange(NUM_CONTROL_CYCLES)

# Plot powers on ax1
ax1.plot(x_ticks_positions, results_sim_total_baseline_power_MW,
        linestyle=':', color='magenta', linewidth=2, marker='.', label='总基准功率')
ax1.plot(x_ticks_positions, results_calculated_target_response_power_MW,
        linestyle='--', color='purple', linewidth=2, marker='x', label='目标响应功率')
ax1.plot(x_ticks_positions, results_sim_total_actual_response_power_MW,
        linestyle='-', color='black', linewidth=2, marker='o', markersize=5, label='总实际响应功率')

bar_width = 0.8
bottom_values = np.zeros(NUM_CONTROL_CYCLES)
colors = ['#FF1E1E', '#FF852D', '#1E78FF', '#1EFF2F']
for i in range(NUM_AGGREGATORS):
    agg_powers_MW = results_sim_agg_actual_response_power_MW[i, :]
    ax1.bar(x_ticks_positions, agg_powers_MW, width=bar_width,
           label=f'聚合商{i+1}实际响应', bottom=bottom_values, color=colors[i % len(colors)], edgecolor='grey', alpha=0.7)
    bottom_values += agg_powers_MW

ax1.set_xlabel('时刻 (控制周期)')
ax1.set_ylabel('功率 (MW)')
ax1.set_title(f'空调负荷分时调峰模拟 ({START_HOUR_PRE_DR}:00 - {END_HOUR_SIM}:00)')
ax1.set_xticks(x_ticks_positions)
ax1.set_xticklabels(results_time_labels, rotation=45, ha="right")
ax1.grid(True, linestyle='--', alpha=0.7)

# Create a second y-axis for temperatures
ax2 = ax1.twinx()
ax2.plot(x_ticks_positions, results_outdoor_temp_C, linestyle='-', color='darkorange', marker='s', markersize=4, label='室外温度')
ax2.plot(x_ticks_positions, results_avg_indoor_temp_C, linestyle='-', color='dodgerblue', marker='^', markersize=4, label='平均室内温度')
ax2.set_ylabel('温度 (°C)')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right', ncol=2)

plt.tight_layout()
plt.savefig("simulated_time_phased_peak_shaving.png")
print("图表已保存为 simulated_time_phased_peak_shaving.png")
plt.show()

# --- 新增：生成第二张仅包含总功率曲线的图表 ---
print("\n生成第二张图表 (仅总功率曲线)...")
fig2, ax_simple = plt.subplots(figsize=(14, 7))

ax_simple.plot(x_ticks_positions, results_sim_total_baseline_power_MW,
        linestyle=':', color='magenta', linewidth=2, marker='.', label='模拟总基准功率')
ax_simple.plot(x_ticks_positions, results_calculated_target_response_power_MW,
        linestyle='--', color='purple', linewidth=2, marker='x', label='计算目标响应功率')
ax_simple.plot(x_ticks_positions, results_sim_total_actual_response_power_MW,
        linestyle='-', color='black', linewidth=2, marker='o', markersize=5, label='模拟总实际响应功率')

ax_simple.set_xlabel('时刻 (控制周期)')
ax_simple.set_ylabel('功率 (MW)')
ax_simple.set_title(f'空调负荷总功率曲线 ({START_HOUR_PRE_DR}:00 - {END_HOUR_SIM}:00)')
ax_simple.set_xticks(x_ticks_positions)
ax_simple.set_xticklabels(results_time_labels, rotation=45, ha="right")
ax_simple.grid(True, linestyle='--', alpha=0.7)
ax_simple.legend(loc='upper right')

plt.tight_layout()
plt.savefig("simulated_total_power_curves.png")
print("第二张图表已保存为 simulated_total_power_curves.png")
plt.show()
# --- 新增图表结束 ---

print("\n脚本执行完毕。")
