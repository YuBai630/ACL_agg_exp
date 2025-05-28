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
    
    # 初始化FSM和ETP模型
    acl_fsm = ACL_FSM(ACL_State.OFF, tonlock_sec, tofflock_sec, sim_dt)
    etp_model = SecondOrderETPModel(Ca, Cm, Ua, Um, Hm, Qgain)
    
    # 直接设置FSM迁移率
    acl_fsm.update_migration_probabilities(u0=u0, u1=u1)
    
    # 5. 设置初始条件
    Ta0 = outdoor_temp_func(0) - 2  # 初始室内温度比室外低2度
    Tm0 = Ta0  # 初始建筑质量温度
    
    # 6. 模拟循环
    time_points = np.arange(0, total_time, sim_dt)
    Ta_history = []  # 室内空气温度历史
    Tm_history = []  # 建筑质量温度历史
    Tout_history = []  # 室外温度历史
    state_history = []  # 空调状态历史
    power_history = []  # 空调功率历史
    
    current_Ta = Ta0
    current_Tm = Tm0
    
    for t in time_points:
        # 获取当前室外温度
        Tout = outdoor_temp_func(t)
        Tout_history.append(Tout)
        
        # 更新FSM状态（只使用给定的迁移率u0和u1）
        acl_fsm.step(sim_dt)
        physical_state = acl_fsm.get_physical_state()
        state_history.append(physical_state)
        
        # 计算当前空调功率
        current_power = ac_rated_power * physical_state  # 如果开启则为额定功率，否则为0
        power_history.append(current_power)
        
        # 使用ETP模型计算一个时间步长后的室内温度
        result = etp_model.simulate(current_Ta, current_Tm, 
                                   lambda t_: Tout, 
                                   lambda t_, state_: physical_state, 
                                   [0, sim_dt], 
                                   [sim_dt])
        
        # 更新当前温度
        current_Ta = result.y[0][-1]
        current_Tm = result.y[1][-1]
        
        Ta_history.append(current_Ta)
        Tm_history.append(current_Tm)
    
    # 计算平均功率
    avg_power = sum(power_history) / len(power_history)
    
    # 7. 绘制结果
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # 创建时间轴标签，修复numpy.int32转换问题
    start_time = filtered_data['Time'].iloc[0]
    time_labels = [start_time + timedelta(seconds=int(t)) for t in time_points]
    
    # 绘制温度变化图
    ax1.plot(time_labels, Ta_history, 'b-', label='室内温度')
    ax1.plot(time_labels, Tout_history, 'r-', label='室外温度')
    ax1.axhline(y=target_temp, color='g', linestyle='--', label='目标温度')
    ax1.axhspan(target_temp-deadband/2, target_temp+deadband/2, alpha=0.2, color='g', label='温度死区')
    ax1.set_ylabel('温度 (°C)')
    ax1.set_title(f'室内外温度变化 (9:00-23:59，FSM参数：u0={u0}, u1={u1})')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制空调状态图
    ax2.step(time_labels, state_history, 'k-', where='post')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_ylabel('空调状态 (0=关, 1=开)')
    ax2.set_title('空调开关状态 (FSM控制)')
    ax2.grid(True)
    
    # 绘制空调功率图
    ax3.step(time_labels, power_history, 'r-', where='post')
    ax3.axhline(y=avg_power, color='b', linestyle='--', label=f'平均功率: {avg_power:.2f} W')
    ax3.set_ylabel('功率 (W)')
    ax3.set_title(f'空调功率 (额定功率: {ac_rated_power} W, 平均功率: {avg_power:.2f} W)')
    ax3.set_xlabel('时间')
    ax3.legend()
    ax3.grid(True)
    
    # 设置x轴日期格式
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    
    plt.tight_layout()
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    plt.savefig(f'figures/空调FSM控制模拟结果_u0_{u0}_u1_{u1}.png', dpi=300)
    plt.show()
    
    print(f"模拟完成，图表已保存为 '空调FSM控制模拟结果_u0_{u0}_u1_{u1}.png'")
    print(f"空调平均功率: {avg_power:.2f} W (额定功率的 {avg_power/ac_rated_power*100:.2f}%)")

if __name__ == "__main__":
    main()
