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

# 计算单台空调的基准功率
def calculate_base_power(cop, r_eq, t_set, qm, t_o):
    """
    计算单台空调的基准功率
    
    参数:
    cop: 能效比 (COP)
    r_eq: 等效热阻 [°C/W]
    t_set: 设定温度 [°C]
    qm: 内部热增益 [W]
    t_o: 室外温度 [°C]
    
    返回:
    p_base: 基准功率 [W]
    """
    # 当室外温度低于设定温度时，基准功率为0
    if t_o <= t_set:
        return 0
    
    # 使用等效热阻计算基准功率
    # 传热系数 = 1/热阻
    ua = 1/r_eq if r_eq > 0 else 0
    p_base = (1/cop) * (ua * (t_o - t_set) + qm)
    return p_base

# 计算单个空调在控制周期内达到目标温度所需的功率
def calculate_power_for_target_temp(etp_model, initial_ta, initial_tm, target_ta, t_outdoor, control_cycle_sec, cop=3.5):
    """
    计算单个空调在控制周期内以特定功率运行后达到目标温度所需的功率
    
    参数:
    etp_model: 二阶ETP模型实例
    initial_ta: 初始室内空气温度 [°C]
    initial_tm: 初始建筑质量温度 [°C]
    target_ta: 目标室内温度 [°C]
    t_outdoor: 室外温度 [°C]
    control_cycle_sec: 控制周期 [秒]
    cop: 能效比 (COP)，默认为3.5
    
    返回:
    required_power: 达到目标温度所需的功率 [W]
    final_ta: 控制周期结束时的室内温度 [°C]
    final_tm: 控制周期结束时的建筑质量温度 [°C]
    """
    # 复制ETP模型参数
    Ca = etp_model.Ca
    Cm = etp_model.Cm
    Ua = etp_model.Ua
    Um = etp_model.Um
    Qgain = etp_model.Qgain
    
    # 二分法查找所需功率
    power_min = 0  # 最小功率
    power_max = abs(etp_model.Hm) * 1.5  # 最大功率，设为额定功率的1.5倍
    tolerance = 0.01  # 温度误差容忍度
    max_iterations = 20  # 最大迭代次数
    
    # 创建室外温度函数
    def tout_func(t):
        return t_outdoor
    
    best_power = None
    best_ta = None
    best_tm = None
    min_diff = float('inf')
    
    for _ in range(max_iterations):
        current_power = (power_min + power_max) / 2
        
        # 创建一个临时ETP模型，使用当前功率
        temp_etp = SecondOrderETPModel(Ca, Cm, Ua, Um, -current_power, Qgain)
        
        # 模拟控制周期结束时的温度
        def mode_func(t, state):
            return 1  # 空调始终开启
        
        result = temp_etp.simulate(
            initial_ta, 
            initial_tm, 
            tout_func, 
            mode_func, 
            [0, control_cycle_sec], 
            [control_cycle_sec]
        )
        
        final_ta = result.y[0][-1]
        final_tm = result.y[1][-1]
        
        temp_diff = abs(final_ta - target_ta)
        
        # 记录最佳结果
        if temp_diff < min_diff:
            min_diff = temp_diff
            best_power = current_power
            best_ta = final_ta
            best_tm = final_tm
        
        # 如果温度足够接近目标温度，结束迭代
        if temp_diff < tolerance:
            break
        
        # 调整功率范围
        if final_ta > target_ta:  # 温度太高，需要更多制冷功率
            power_min = current_power
        else:  # 温度太低，需要减少制冷功率
            power_max = current_power
    
    # 考虑COP计算实际电功率
    required_power = best_power / cop
    
    return required_power, best_ta, best_tm

# 计算单个空调在给定功率下控制周期结束时的温度
def calculate_temperature_with_power(etp_model, initial_ta, initial_tm, power, t_outdoor, control_cycle_sec):
    """
    计算单个空调在给定功率下控制周期结束时的温度
    
    参数:
    etp_model: 二阶ETP模型实例
    initial_ta: 初始室内空气温度 [°C]
    initial_tm: 初始建筑质量温度 [°C]
    power: 空调制冷功率 [W]（正值表示制冷功率）
    t_outdoor: 室外温度 [°C]
    control_cycle_sec: 控制周期 [秒]
    
    返回:
    final_ta: 控制周期结束时的室内温度 [°C]
    final_tm: 控制周期结束时的建筑质量温度 [°C]
    """
    # 复制ETP模型参数
    Ca = etp_model.Ca
    Cm = etp_model.Cm
    Ua = etp_model.Ua
    Um = etp_model.Um
    Qgain = etp_model.Qgain
    
    # 创建一个临时ETP模型，使用给定功率（注意ETP模型中负值表示制冷）
    temp_etp = SecondOrderETPModel(Ca, Cm, Ua, Um, -abs(power), Qgain)
    
    # 创建室外温度函数
    def tout_func(t):
        return t_outdoor
    
    # 空调始终开启
    def mode_func(t, state):
        return 1
    
    # 模拟控制周期结束时的温度
    result = temp_etp.simulate(
        initial_ta, 
        initial_tm, 
        tout_func, 
        mode_func, 
        [0, control_cycle_sec], 
        [control_cycle_sec]
    )
    
    final_ta = result.y[0][-1]
    final_tm = result.y[1][-1]
    
    return final_ta, final_tm

# 计算需求响应曲线
def calculate_demand_response(rated_power, pi, cop, r_eq, t_current, qm, t_o, t_set, t_min, t_max):
    """
    计算单个空调的需求响应曲线
    
    参数:
    rated_power: 额定功率 [W]
    pi: 控制参数 (0-1之间)
    cop: 能效比 (COP)
    r_eq: 等效热阻 [°C/W]
    t_current: 当前室内温度 [°C]
    qm: 内部热增益 [W]
    t_o: 室外温度 [°C]
    t_set: 设定温度 [°C]
    t_min: 最低温度限制 [°C]
    t_max: 最高温度限制 [°C]
    
    返回:
    pa_ij_star: 响应功率 [W]
    p_max: 最大功率 [W]
    p_min: 最小功率 [W]
    p_set: 设定功率 [W]
    max_heat_power: 最大热功率 [W]
    min_heat_power: 最小热功率 [W]
    set_heat_power: 设定热功率 [W]
    """
    # 计算基准功率
    p_base = calculate_base_power(cop, r_eq, t_set, qm, t_o)
    
    # 计算最大功率和最小功率
    p_max = rated_power  # 最大功率为额定功率
    p_min = 0  # 最小功率为0
    
    # 计算设定功率
    p_set = p_base
    
    # 根据pi值计算响应功率
    pa_ij_star = p_min + pi * (p_max - p_min)
    
    # 计算热功率（用于调试）
    max_heat_power = p_max * cop
    min_heat_power = p_min * cop
    set_heat_power = p_set * cop
    
    return pa_ij_star, p_max, p_min, p_set, max_heat_power, min_heat_power, set_heat_power

# 生成聚合需求响应曲线
def generate_aggregate_demand_curve(rated_powers, cop_values, r_eq_values, t_currents, qm_values, t_o, t_sets, t_mins, t_maxs, pi_steps=21):
    """
    生成聚合需求响应曲线
    
    参数:
    rated_powers: 额定功率列表 [W]
    cop_values: 能效比列表 (COP)
    r_eq_values: 等效热阻列表 [°C/W]
    t_currents: 当前室内温度列表 [°C]
    qm_values: 内部热增益列表 [W]
    t_o: 室外温度 [°C]
    t_sets: 设定温度列表 [°C]
    t_mins: 最低温度限制列表 [°C]
    t_maxs: 最高温度限制列表 [°C]
    pi_steps: pi值的步数
    
    返回:
    pi_values: pi值列表
    total_powers: 对应的总功率列表 [W]
    individual_powers: 每个空调在每个pi值下的功率 [W]
    interp_func: 从总功率到pi值的插值函数
    """
    num_acs = len(rated_powers)
    pi_values = np.linspace(0, 1, pi_steps)
    total_powers = np.zeros(pi_steps)
    individual_powers = np.zeros((num_acs, pi_steps))
    
    for i, pi in enumerate(pi_values):
        total_power = 0
        for j in range(num_acs):
            pa_ij_star, _, _, _, _, _, _ = calculate_demand_response(
                rated_powers[j], pi, cop_values[j], r_eq_values[j], 
                t_currents[j], qm_values[j], t_o, t_sets[j], t_mins[j], t_maxs[j]
            )
            individual_powers[j, i] = pa_ij_star
            total_power += pa_ij_star
        total_powers[i] = total_power
    
    # 创建从总功率到pi值的插值函数
    interp_func = interp1d(total_powers, pi_values, bounds_error=False, fill_value="extrapolate")
    
    return pi_values, total_powers, individual_powers, interp_func

# 从总功率计算pi值
def calculate_pi_from_power(target_power, interp_func, pi_values, total_powers):
    """
    从目标总功率计算pi值
    
    参数:
    target_power: 目标总功率 [W]
    interp_func: 从总功率到pi值的插值函数
    pi_values: pi值列表
    total_powers: 对应的总功率列表 [W]
    
    返回:
    pi: 对应的pi值
    """
    # 使用插值函数计算pi值
    pi = float(interp_func(target_power))
    
    # 确保pi值在[0,1]范围内
    pi = max(0, min(1, pi))
    
    return pi

# 基于温度差值调整FSM迁移概率
def adjust_migration_probabilities_with_target(etp_model, t_indoor, t_indoor_mass, t_outdoor, 
                                            target_temp, deadband, control_cycle_sec, 
                                            cop, tonlock_sec, tofflock_sec, sim_dt):
    """
    基于目标温度计算所需功率，并使用FSM计算迁移概率u0和u1
    
    参数:
    etp_model: 空调的ETP模型实例
    t_indoor: 当前室内温度 [°C]
    t_indoor_mass: 当前建筑质量温度 [°C]
    t_outdoor: 当前室外温度 [°C]
    target_temp: 目标温度 [°C]
    deadband: 温度死区 [°C]
    control_cycle_sec: 控制周期 [秒]
    cop: 能效比 (COP)
    tonlock_sec: 开锁时间 [秒]
    tofflock_sec: 关锁时间 [秒]
    sim_dt: 仿真步长 [秒]
    
    返回:
    adjusted_u0: 调整后的u0值
    adjusted_u1: 调整后的u1值
    required_power: 达到目标温度所需的功率 [W]
    """
    # 计算目标温度区间
    t_min = target_temp - deadband/2
    t_max = target_temp + deadband/2
    
    # 计算空调额定功率
    rated_power = abs(etp_model.Hm) / cop  # 电功率 = 热功率/COP
    
    # 创建一个临时FSM实例用于调用calculate_migration_probabilities
    temp_fsm = ACL_FSM(ACL_State.ON, tonlock_sec, tofflock_sec, sim_dt)
    
    # 根据室内温度与目标温度区间的关系计算响应功率和迁移概率
    if t_indoor < t_min:  # 室内温度低于下限，需要减少制冷
        # 如果温度已经低于目标下限，则设置功率为最小值
        required_power = 0  # 不需要制冷
        # 关闭概率=1，开启概率=0
        u0, u1 = 1.0, 0.0  
        
    elif t_indoor > t_max:  # 室内温度高于上限，需要增加制冷
        # 计算达到目标温度下限所需的功率
        required_power, _, _ = calculate_power_for_target_temp(
            etp_model, t_indoor, t_indoor_mass, t_min, 
            t_outdoor, control_cycle_sec, cop
        )
        # 确保功率不超过额定功率
        required_power = min(required_power, rated_power)
        
        # 使用FSM计算迁移概率
        u0, u1 = temp_fsm.calculate_migration_probabilities(required_power, rated_power)
        
    else:  # 温度在目标区间内，微调
        # 温度已在目标区间，设置维持温度所需的功率
        base_required_power, _, _ = calculate_power_for_target_temp(
            etp_model, t_indoor, t_indoor_mass, target_temp, 
            t_outdoor, control_cycle_sec, cop
        )
        # 确保功率不超过额定功率
        required_power = min(base_required_power, rated_power)
        
        # 使用FSM计算迁移概率
        u0, u1 = temp_fsm.calculate_migration_probabilities(required_power, rated_power)
    
    # 确保u0和u1在合理范围内
    # adjusted_u0 = max(0.1, min(0.9, u0))
    # adjusted_u1 = max(0.1, min(0.9, u1))
    adjusted_u0 = u0
    adjusted_u1 = u1
    
    return adjusted_u0, adjusted_u1, required_power

# 主函数
def main():
    # 1. 配置参数
    # outdoor_temp_file = "../data/outdoorTemperature.csv"  
    outdoor_temp_file = "D:\experiments\ACL_agg_exp\data\outdoorTemperature.csv"  
    sim_dt = 2 * 60  # 仿真步长，秒
    
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
    num_acs = 100
    
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

    # 创建一个包装函数，确保室外温度不低于20摄氏度
    def get_outdoor_temp(t):
        temp = outdoor_temp_func(t)
        return max(temp, 35.0) 
    
    # 4. 初始化FSM和ETP模型
    # FSM参数
    tonlock_sec = 3 * 60  # 3分钟开锁时间
    tofflock_sec = 3 * 60  # 3分钟关锁时间
    
    # ETP参数（示例值，可根据实际情况调整）
    Ca = 477000  # 空气热容量 [J/°C]
    Cm = 6600000  # 建筑质量热容量 [J/°C]
    Ua = 160  # 空气与室外的热传导系数 [W/°C]
    R_eq = 1/Ua  # 等效热阻 [°C/W]
    Um = 1000  # 空气与建筑质量的热传导系数 [W/°C]
    # Hm = -6000  # 空调制冷功率 [W]（负值表示制冷）
    Hm = -7000  # 空调制冷功率 [W]（负值表示制冷）
    Qgain = 198  # 内部热增益 [W]
    
    # 空调额定功率（用于计算实际功耗）
    ac_rated_power = abs(Hm)  # 空调额定功率为Hm的绝对值
    
    # 初始化多个FSM和ETP模型
    acl_fsms = []
    etp_models = []
    cop_values = []  # 存储每台空调的COP值
    r_eq_values = []  # 存储每台空调的等效热阻值
    qm_values = []   # 存储每台空调的Qm值
    
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
        r_eq_var = 1/ua_var  # 计算等效热阻
        r_eq_values.append(r_eq_var)  # 保存等效热阻值用于基准功率计算
        um_var = Um * (0.9 + 0.2 * np.random.random())
        hm_var = Hm * (0.9 + 0.2 * np.random.random())
        qgain_var = Qgain * (0.9 + 0.2 * np.random.random())
        qm_values.append(qgain_var)  # 保存Qm值用于基准功率计算
        
        # 为每台空调生成COP值 (能效比，一般在3-5之间)
        cop = 3.5 + np.random.random()  # 3.5到4.5之间的随机值
        cop_values.append(cop)
        
        etp = SecondOrderETPModel(ca_var, cm_var, ua_var, um_var, hm_var, qgain_var)
        etp_models.append(etp)
    
    # 5. 设置初始条件
    Ta0_base = outdoor_temp_func(0) + 5  # 基准初始室内温度比室外低2度
    
    # 为每个空调设置略有不同的初始温度
    Ta0_list = [Ta0_base + np.random.uniform(-1, 1) for _ in range(num_acs)]
    Tm0_list = Ta0_list.copy()  # 初始建筑质量温度与室内温度相同
    
    # 6. 模拟循环
    time_points = np.arange(0, total_time, sim_dt)
    
    # 存储所有空调的平均状态
    avg_Ta_history = np.zeros(len(time_points))
    
    # 存储室外温度历史
    Tout_history = []
    
    # 存储基准功率历史
    base_power_history = np.zeros(len(time_points))
    
    # 存储迁移概率历史
    u0_history = np.zeros(len(time_points))
    u1_history = np.zeros(len(time_points))
    
    # 当前所有空调的状态
    current_Ta_list = Ta0_list.copy()
    current_Tm_list = Tm0_list.copy()
    
    # 控制周期计时器和当前基准功率
    last_control_update_time = 0
    current_base_power = 0
    
    # 基础迁移概率
    base_u0 = 0.5  # 基础关闭概率
    base_u1 = 0.5  # 基础开启概率
    
    # 基准温差和调整系数
    base_temp_diff = 5.0  # 基准温差 [°C]
    alpha = 0.5  # 高温调整系数
    beta = 0.3   # 低温调整系数
    
    # 初始迁移概率
    adjusted_u0 = base_u0
    adjusted_u1 = base_u1
    
    # 模拟每个时间步
    for t_idx, t in enumerate(time_points):
        # 获取当前室外温度
        Tout = get_outdoor_temp(t)
        Tout_history.append(Tout)
        
        # 检查是否到达新的控制周期
        if t - last_control_update_time >= control_cycle_sec or t_idx == 0:
            # 在新的控制周期开始时更新基准功率
            print(f"时间 {t/3600:.2f}h: 更新控制周期，重新计算基准功率")
            
            # 计算每台空调的基准功率
            total_base_power = 0
            for ac_idx in range(num_acs):
                # 使用等效热阻计算基准功率
                p_base_ij = calculate_base_power(
                    cop_values[ac_idx],
                    r_eq_values[ac_idx],
                    target_temp,  # 使用目标温度作为设定温度
                    qm_values[ac_idx],
                    Tout  # 当前室外温度
                )
                total_base_power += p_base_ij
            
            # 更新当前基准功率（单位：kW）
            current_base_power = total_base_power / 1000
            
            # 更新每台空调的FSM迁移概率
            print("更新空调迁移概率...")
            
            # 计算当前平均室内温度
            avg_indoor_temp = np.mean(current_Ta_list)
            print(f"当前平均室内温度: {avg_indoor_temp:.2f}°C, 室外温度: {Tout:.2f}°C")
            
            # 初始化总响应功率和平均迁移概率
            total_response_power = 0
            avg_u0 = 0
            avg_u1 = 0
            
            # 更新所有空调的迁移概率
            for ac_idx in range(num_acs):
                # 使用新的基于目标温度的方法计算迁移概率
                adjusted_u0, adjusted_u1, required_power = adjust_migration_probabilities_with_target(
                    etp_models[ac_idx], 
                    current_Ta_list[ac_idx],  # 当前室内温度
                    current_Tm_list[ac_idx],  # 当前建筑质量温度
                    Tout,                     # 室外温度
                    target_temp,              # 目标温度
                    deadband,                 # 温度死区
                    control_cycle_sec,        # 控制周期
                    cop_values[ac_idx],       # 能效比
                    tonlock_sec,              # 开锁时间
                    tofflock_sec,             # 关锁时间
                    sim_dt                    # 仿真步长
                )
                
                # 保存原始计算的迁移概率（用于显示和历史记录）
                original_u0 = adjusted_u0
                original_u1 = adjusted_u1
                
                # 强制设置迁移概率: u0=1（一定关闭）, u1=0（一定不开启）
                # adjusted_u0 = 1.0
                # adjusted_u1 = 0.0
                
                # 更新FSM迁移概率
                acl_fsms[ac_idx].update_migration_probabilities(u0=adjusted_u0, u1=adjusted_u1)
                if ac_idx == 0:
                    rated_power = abs(etp_models[ac_idx].Hm)
                    print(f"空调1的迁移概率: u0={original_u0:.4f}(关闭概率), u1={original_u1:.4f}(开启概率), 需求功率比例：{required_power/rated_power:.4f}")
                    # print(f"空调的强制迁移概率: u0={adjusted_u0:.4f}(关闭概率), u1={adjusted_u1:.4f}(开启概率)")
                
                # 累加响应功率和平均迁移概率 (使用强制的概率)
                total_response_power += required_power
                avg_u0 += adjusted_u0
                avg_u1 += adjusted_u1
            
            # 计算平均值
            avg_u0 /= num_acs
            avg_u1 /= num_acs
            
            print(f"平均调整迁移概率: u0={avg_u0:.4f}(关闭概率), u1={avg_u1:.4f}(开启概率)")

            print(f"总响应功率: {total_response_power:.2f}W ({total_response_power/1000:.2f}kW)")
            
            # 更新控制周期计时器
            last_control_update_time = t
        
        # 存储当前基准功率
        base_power_history[t_idx] = current_base_power
        
        # 存储迁移概率
        u0_history[t_idx] = avg_u0
        u1_history[t_idx] = avg_u1
        
        # 当前时间步的所有空调状态
        current_states = []
        current_temps = []
        
        # 更新每台空调的状态
        for ac_idx in range(num_acs):
            # 更新FSM状态
            acl_fsms[ac_idx].step(sim_dt)
            physical_state = acl_fsms[ac_idx].get_physical_state()
            current_states.append(physical_state)
            
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
        
        # 显示进度
        if t_idx % 100 == 0:
            progress = (t_idx + 1) / len(time_points) * 100
            print(f"模拟进度: {progress:.1f}%")
    
    end_time_sim = time.time()
    print(f"模拟完成，耗时: {end_time_sim - start_time_sim:.2f} 秒")
    
    # 7. 绘制结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 创建时间轴标签，修复numpy.int32转换问题
    start_time = filtered_data['Time'].iloc[0]
    time_labels = [start_time + timedelta(seconds=int(t)) for t in time_points]
    
    # 绘制温度和基准功率变化图（上图）
    # 左Y轴 - 温度
    ax1.plot(time_labels, avg_Ta_history, 'b-', label='平均室内温度')
    ax1.plot(time_labels, Tout_history, 'r-', label='室外温度')
    ax1.axhline(y=target_temp, color='g', linestyle='--', label='目标温度')
    ax1.axhspan(target_temp-deadband/2, target_temp+deadband/2, alpha=0.2, color='g', label='温度死区')
    ax1.set_ylabel('温度 (°C)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 右Y轴 - 基准功率
    ax1b = ax1.twinx()
    ax1b.plot(time_labels, base_power_history, 'g-', label='总基准功率')
    ax1b.set_ylabel('总基准功率 (kW)', color='g')
    ax1b.tick_params(axis='y', labelcolor='g')
    
    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='upper right')
    
    ax1.set_title(f'{num_acs}台空调的平均室内温度和总基准功率随室外温度变化')
    ax1.grid(True)
    
    # 绘制迁移概率变化图（下图）
    ax2.plot(time_labels, u0_history, 'b-', label='u0 (关闭概率)')
    ax2.plot(time_labels, u1_history, 'r-', label='u1 (开启概率)')
    ax2.set_ylabel('迁移概率')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.set_title('温差调节的迁移概率变化')
    ax2.grid(True)
    
    # 设置x轴日期格式
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax2.set_xlabel('时间')
    
    plt.tight_layout()
    
    # 确保figures目录存在
    os.makedirs('figures', exist_ok=True)
    
    plt.savefig(f'figures/空调温差调节_{num_acs}台_35摄氏度_{Hm}W.png', dpi=300)
    plt.show()
    
    print(f"模拟完成，图表已保存为 'figures/空调温差调节_{num_acs}台_35摄氏度_{Hm}W.png'")

    

if __name__ == "__main__":
    main()
