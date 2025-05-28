import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from scipy.interpolate import interp1d
import time

# 添加src目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# 导入FSM和ETP模块
from FSM import ACL_FSM, ACL_State
from ETP import SecondOrderETPModel

# 读取室外温度数据
def load_outdoor_temperature(file_path):
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

# 主函数
def main():
    # 1. 配置参数
    # outdoor_temp_file = "../data/outdoorTemperature.csv"  
    outdoor_temp_file = "D:\experiments\ACL_agg_exp\data\outdoorTemperature.csv"  
    sim_dt = 2*60  # 仿真步长，秒
    
    # 控制周期参数
    control_cycle_sec = 30*60  # 控制周期为30分钟
    
    # 计算从9:00到23:59的总秒数
    start_hour = 9
    end_hour = 23
    end_minute = 59
    total_time = (end_hour - start_hour) * 3600 + end_minute * 60  # 从9点到23:59的秒数
    
    target_temp = 24.0  # 目标温度，仅用于图表显示
    deadband = 1.0  # 死区，仅用于图表显示
    
    # FSM参数
    u0 = 0.5  # 从OFF到ON的迁移率
    u1 = 0.5  # 从ON到OFF的迁移率
    
    # 空调数量
    num_acs = 1000
    
    print(f"开始模拟 {num_acs} 台空调...")
    print(f"控制周期: {control_cycle_sec/60} 分钟")
    start_time_sim = time.time()
    
    # 2. 读取室外温度数据
    outdoor_data = load_outdoor_temperature(outdoor_temp_file)
    
    # 筛选9:00到23:59的数据
    start_time_str = "8/29/2007 9:00"
    end_time_str = "8/29/2007 23:59"
    start_time = pd.to_datetime(start_time_str)
    end_time = pd.to_datetime(end_time_str)
    
    # 筛选指定时间范围内的数据
    filtered_data = outdoor_data[(outdoor_data['Time'] >= start_time) & 
                                 (outdoor_data['Time'] <= end_time)].copy()
    
    # 确保数据存在
    if filtered_data.empty:
        print("错误：找不到指定时间范围内的数据！")
        return
    
    # 3. 处理时间和温度数据，创建插值函数
    timestamps = [(t - filtered_data['Time'].iloc[0]).total_seconds() for t in filtered_data['Time']]
    temperatures = filtered_data['Temperature'].values
    
    # 创建一个插值函数，用于获取任意时间点的室外温度
    outdoor_temp_func = interp1d(timestamps, temperatures, bounds_error=False, fill_value="extrapolate")
    
    # 4. 初始化FSM和ETP模型
    # FSM参数
    tonlock_sec = 3 * 60  # 3分钟开锁时间
    tofflock_sec = 3 * 60  # 3分钟关锁时间
    
    # ETP参数（示例值，可根据实际情况调整）
    Ca = 2000000  # 空气热容量 [J/°C]
    Cm = 10000000  # 建筑质量热容量 [J/°C]
    Ua = 500  # 空气与室外的热传导系数 [W/°C]
    Um = 1000  # 空气与建筑质量的热传导系数 [W/°C]
    Hm = -2000  # 空调制冷功率 [W]（负值表示制冷）
    Qgain = 200  # 内部热增益 [W]
    
    # 空调额定功率（用于计算实际功耗）
    ac_rated_power = abs(Hm)  # 空调额定功率为Hm的绝对值
    
    # 初始化多个FSM和ETP模型
    acl_fsms = []
    etp_models = []
    
    for _ in range(num_acs):
        # 随机初始状态，约一半开启，一半关闭
        initial_state = ACL_State.ON if np.random.random() < 0.5 else ACL_State.OFF
        fsm = ACL_FSM(initial_state, tonlock_sec, tofflock_sec, sim_dt)
        fsm.update_migration_probabilities(u0=u0, u1=u1)
        acl_fsms.append(fsm)
        
        # 每个空调的ETP模型参数略有不同，模拟实际情况
        ca_var = Ca * (0.9 + 0.2 * np.random.random())  # 变化范围为原值的90%-110%
        cm_var = Cm * (0.9 + 0.2 * np.random.random())
        ua_var = Ua * (0.9 + 0.2 * np.random.random())
        um_var = Um * (0.9 + 0.2 * np.random.random())
        hm_var = Hm * (0.9 + 0.2 * np.random.random())
        qgain_var = Qgain * (0.9 + 0.2 * np.random.random())
        
        etp = SecondOrderETPModel(ca_var, cm_var, ua_var, um_var, hm_var, qgain_var)
        etp_models.append(etp)
    
    # 5. 设置初始条件
    Ta0_base = outdoor_temp_func(0) - 2  # 基准初始室内温度比室外低2度
    
    # 为每个空调设置略有不同的初始温度
    Ta0_list = [Ta0_base + np.random.uniform(-1, 1) for _ in range(num_acs)]
    Tm0_list = Ta0_list.copy()  # 初始建筑质量温度与室内温度相同
    
    # 6. 模拟循环
    time_points = np.arange(0, total_time, sim_dt)
    
    # 存储所有空调的平均状态
    avg_Ta_history = np.zeros(len(time_points))
    avg_state_history = np.zeros(len(time_points))
    avg_power_history = np.zeros(len(time_points))
    
    # 存储室外温度历史
    Tout_history = []
    
    # 当前所有空调的状态
    current_Ta_list = Ta0_list.copy()
    current_Tm_list = Tm0_list.copy()
    
    # 控制周期计时器
    last_control_update_time = 0
    
    # 模拟每个时间步
    for t_idx, t in enumerate(time_points):
        # 获取当前室外温度
        Tout = outdoor_temp_func(t)
        Tout_history.append(Tout)
        
        # 检查是否到达新的控制周期
        if t - last_control_update_time >= control_cycle_sec:
            # 在新的控制周期开始时更新所有FSM的迁移概率
            print(f"时间 {t/3600:.2f}h: 更新控制周期，重新设置FSM迁移概率")
            
            # 可以根据当前温度或其他条件动态调整u0和u1
            # 这里简单示例，实际应用中可能基于更复杂的控制策略
            current_u0 = u0 * (0.8 + 0.4 * np.random.random())  # 在基础值附近随机波动
            current_u1 = u1 * (0.8 + 0.4 * np.random.random())
            
            # 更新所有空调的FSM迁移概率
            for ac_idx in range(num_acs):
                # 为每台空调设置略有不同的迁移概率，模拟个体差异
                individual_u0 = current_u0 * (0.95 + 0.1 * np.random.random())
                individual_u1 = current_u1 * (0.95 + 0.1 * np.random.random())
                acl_fsms[ac_idx].update_migration_probabilities(u0=individual_u0, u1=individual_u1)
            
            # 更新控制周期计时器
            last_control_update_time = t
        
        # 当前时间步的所有空调状态
        current_states = []
        current_powers = []
        current_temps = []
        
        # 更新每台空调的状态
        for ac_idx in range(num_acs):
            # 更新FSM状态
            acl_fsms[ac_idx].step(sim_dt)
            physical_state = acl_fsms[ac_idx].get_physical_state()
            current_states.append(physical_state)
            
            # 计算当前空调功率
            current_power = ac_rated_power * physical_state
            current_powers.append(current_power)
            
            # 使用ETP模型计算一个时间步长后的室内温度
            result = etp_models[ac_idx].simulate(
                current_Ta_list[ac_idx], 
                current_Tm_list[ac_idx], 
                lambda t_: Tout, 
                lambda t_, state_: physical_state, 
                [0, sim_dt], 
                [sim_dt]
            )
            
            # 更新当前温度
            current_Ta_list[ac_idx] = result.y[0][-1]
            current_Tm_list[ac_idx] = result.y[1][-1]
            current_temps.append(current_Ta_list[ac_idx])
        
        # 计算并存储平均值
        avg_Ta_history[t_idx] = np.mean(current_temps)
        avg_state_history[t_idx] = np.mean(current_states)
        avg_power_history[t_idx] = np.mean(current_powers)
        
        # 显示进度
        if t_idx % 100 == 0:
            progress = (t_idx + 1) / len(time_points) * 100
            print(f"模拟进度: {progress:.1f}%")
    
    # 计算总平均功率
    avg_power = np.mean(avg_power_history)
    
    end_time_sim = time.time()
    print(f"模拟完成，耗时: {end_time_sim - start_time_sim:.2f} 秒")
    
    # 7. 绘制结果
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # 创建时间轴标签，修复numpy.int32转换问题
    start_time = filtered_data['Time'].iloc[0]
    time_labels = [start_time + timedelta(seconds=int(t)) for t in time_points]
    
    # 标记控制周期的垂直线
    control_cycle_times = np.arange(0, total_time, control_cycle_sec)
    for ax in [ax1, ax2, ax3]:
        for cycle_time in control_cycle_times:
            if cycle_time > 0:  # 跳过起始点
                cycle_time_label = start_time + timedelta(seconds=int(cycle_time))
                ax.axvline(x=cycle_time_label, color='gray', linestyle='--', alpha=0.5)
    
    # 绘制温度变化图
    ax1.plot(time_labels, avg_Ta_history, 'b-', label='平均室内温度')
    ax1.plot(time_labels, Tout_history, 'r-', label='室外温度')
    ax1.axhline(y=target_temp, color='g', linestyle='--', label='目标温度')
    ax1.axhspan(target_temp-deadband/2, target_temp+deadband/2, alpha=0.2, color='g', label='温度死区')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title(f'{num_acs}台空调的平均室内温度 (控制周期: {control_cycle_sec/60}分钟)')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制空调状态图
    ax2.plot(time_labels, avg_state_history, 'k-')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel('平均开启率')
    ax2.set_title(f'{num_acs}台空调的平均开启率 (控制周期: {control_cycle_sec/60}分钟)')
    ax2.grid(True)
    
    # 绘制空调功率图
    ax3.plot(time_labels, avg_power_history, 'r-')
    ax3.axhline(y=avg_power, color='b', linestyle='--', label=f'平均功率: {avg_power:.2f} W/台')
    ax3.set_ylabel('平均功率 (W/台)')
    ax3.set_title(f'{num_acs}台空调的平均功率 (额定功率: {ac_rated_power} W/台, 平均功率: {avg_power:.2f} W/台)')
    ax3.set_xlabel('时间')
    ax3.legend()
    ax3.grid(True)
    
    # 设置x轴日期格式
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    plt.tight_layout()
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    plt.savefig(f'figures/空调群控模拟_{num_acs}台_控制周期{int(control_cycle_sec/60)}分钟.png', dpi=300)
    plt.show()
    
    print(f"模拟完成，图表已保存为 'figures/空调群控模拟_{num_acs}台_控制周期{int(control_cycle_sec/60)}分钟.png'")
    print(f"空调平均功率: {avg_power:.2f} W/台 (额定功率的 {avg_power/ac_rated_power*100:.2f}%)")
    print(f"总功率: {avg_power * num_acs / 1000:.2f} kW")

if __name__ == "__main__":
    main()
