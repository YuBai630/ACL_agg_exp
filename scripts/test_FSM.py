import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
import time
import random
from collections import defaultdict
from FSM import ACL_FSM, ACL_State


def test_fixed_probability_simulation():
    """
    固定概率仿真：给定固定的迁移概率率(u0, u1)，从某个初始状态开始仿真FSM较长时间
    """
    print("开始进行固定概率仿真测试...")
    
    # 设置FSM参数
    tonlock_sec = 3 * 60  # 开启后闭锁时间 3分钟
    tofflock_sec = 3 * 60  # 关闭后闭锁时间 3分钟
    u0 = 0.005  # 关闭概率率
    u1 = 0.002  # 开启概率率
    
    # 创建FSM实例
    initial_state = ACL_State.OFF
    fsm = ACL_FSM(initial_state, tonlock_sec, tofflock_sec)
    fsm.update_migration_probabilities(u0, u1)
    
    # 仿真参数
    sim_dt = 2.0  # 仿真时间步长 (秒)
    total_sim_time = 2 * 3600  # 总仿真时间 (2小时)
    
    # 记录状态历史
    time_points = []
    states = []
    physical_states = []
    time_in_states = []
    
    # 执行仿真
    current_time = 0.0
    while current_time < total_sim_time:
        # 记录当前状态
        time_points.append(current_time)
        states.append(fsm.current_state)
        physical_states.append(fsm.get_physical_state())
        time_in_states.append(fsm.time_in_current_state)
        
        # 推进仿真
        fsm.step(sim_dt)
        current_time += sim_dt
    
    # 绘制结果
    plt.figure(figsize=(14, 10))
    
    # 状态变化图
    plt.subplot(3, 1, 1)
    state_numeric = []
    for state in states:
        if state == ACL_State.ON:
            state_numeric.append(1)
        elif state == ACL_State.OFF:
            state_numeric.append(0)
        elif state == ACL_State.ONLOCK:
            state_numeric.append(1.5)
        elif state == ACL_State.OFFLOCK:
            state_numeric.append(0.5)
    
    minutes = np.array(time_points) / 60  # 转换为分钟
    plt.plot(minutes, state_numeric, 'b-')
    plt.yticks([0, 0.5, 1, 1.5], ['OFF', 'OFFLOCK', 'ON', 'ONLOCK'])
    plt.title('FSM状态变化 (固定概率仿真)')
    plt.xlabel('时间 (分钟)')
    plt.ylabel('状态')
    plt.grid(True)
    
    # 物理状态变化图 (0/1)
    plt.subplot(3, 1, 2)
    plt.plot(minutes, physical_states, 'r-', linewidth=2)
    plt.yticks([0, 1], ['关闭', '开启'])
    plt.title('空调物理状态变化')
    plt.xlabel('时间 (分钟)')
    plt.ylabel('物理状态')
    plt.grid(True)
    
    # 当前状态停留时间
    plt.subplot(3, 1, 3)
    plt.plot(minutes, np.array(time_in_states)/60, 'g-')
    plt.axhline(y=tonlock_sec/60, color='r', linestyle='--', label=f'闭锁时间 ({tonlock_sec/60}分钟)')
    plt.title('当前状态停留时间')
    plt.xlabel('时间 (分钟)')
    plt.ylabel('停留时间 (分钟)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fixed_probability_simulation_results.png')
    print("固定概率仿真测试完成，结果已保存为'fixed_probability_simulation_results.png'")
    
    return time_points, states, physical_states, time_in_states


def test_state_transition_logic():
    """
    验证状态转移逻辑是否严格遵守转移条件
    """
    print("开始验证状态转移逻辑...")
    
    # 设置FSM参数
    tonlock_sec = 3 * 60  # 开启后闭锁时间 3分钟
    tofflock_sec = 3 * 60  # 关闭后闭锁时间 3分钟
    u0 = 0.01  # 关闭概率率 (设置较大以便于观察状态转移)
    u1 = 0.01  # 开启概率率 (设置较大以便于观察状态转移)
    
    # 仿真参数
    sim_dt = 2.0  # 仿真时间步长 (秒)
    total_sim_time = 1 * 3600  # 总仿真时间 (1小时)
    
    # 用于记录转移事件的列表
    transitions = {
        "ON->OFFLOCK": [],
        "OFF->ONLOCK": [],
        "ONLOCK->ON": [],
        "OFFLOCK->OFF": []
    }
    
    # 创建并运行FSM实例，从每个状态开始
    for initial_state in [ACL_State.ON, ACL_State.OFF, ACL_State.ONLOCK, ACL_State.OFFLOCK]:
        print(f"  测试从{initial_state}状态开始的转移...")
        fsm = ACL_FSM(initial_state, tonlock_sec, tofflock_sec)
        fsm.update_migration_probabilities(u0, u1)
        
        # 如果从锁定状态开始，设置一个初始的停留时间
        if initial_state in [ACL_State.ONLOCK, ACL_State.OFFLOCK]:
            fsm.time_in_current_state = 0.0  # 刚进入锁定状态
        
        current_time = 0.0
        last_state = initial_state
        
        while current_time < total_sim_time:
            # 推进仿真前记录状态
            current_state = fsm.current_state
            time_in_state = fsm.time_in_current_state
            
            # 推进仿真
            fsm.step(sim_dt)
            current_time += sim_dt
            
            # 检测状态转移
            if fsm.current_state != current_state:
                transition_key = f"{current_state}->{fsm.current_state}"
                if transition_key == "ON->OFFLOCK":
                    transitions["ON->OFFLOCK"].append((current_time, time_in_state))
                elif transition_key == "OFF->ONLOCK":
                    transitions["OFF->ONLOCK"].append((current_time, time_in_state))
                elif transition_key == "ONLOCK->ON":
                    transitions["ONLOCK->ON"].append((current_time, time_in_state))
                elif transition_key == "OFFLOCK->OFF":
                    transitions["OFFLOCK->OFF"].append((current_time, time_in_state))
    
    # 分析转移数据
    print("\n状态转移逻辑分析结果:")
    
    # 检查从锁定状态到非锁定状态的转移是否在满足闭锁时间后发生
    onlock_to_on_times = [t[1] for t in transitions["ONLOCK->ON"]]
    offlock_to_off_times = [t[1] for t in transitions["OFFLOCK->OFF"]]
    
    if onlock_to_on_times:
        min_onlock_time = min(onlock_to_on_times)
        avg_onlock_time = sum(onlock_to_on_times) / len(onlock_to_on_times)
        print(f"  ONLOCK->ON 转移的最小停留时间: {min_onlock_time:.2f}秒 (闭锁时间: {tonlock_sec}秒)")
        print(f"  ONLOCK->ON 转移的平均停留时间: {avg_onlock_time:.2f}秒")
        if min_onlock_time < tonlock_sec:
            print("  ⚠️ 错误: ONLOCK->ON 转移在闭锁时间前发生!")
        else:
            print("  ✓ ONLOCK->ON 转移符合闭锁时间要求")
    
    if offlock_to_off_times:
        min_offlock_time = min(offlock_to_off_times)
        avg_offlock_time = sum(offlock_to_off_times) / len(offlock_to_off_times)
        print(f"  OFFLOCK->OFF 转移的最小停留时间: {min_offlock_time:.2f}秒 (闭锁时间: {tofflock_sec}秒)")
        print(f"  OFFLOCK->OFF 转移的平均停留时间: {avg_offlock_time:.2f}秒")
        if min_offlock_time < tofflock_sec:
            print("  ⚠️ 错误: OFFLOCK->OFF 转移在闭锁时间前发生!")
        else:
            print("  ✓ OFFLOCK->OFF 转移符合闭锁时间要求")
    
    # 创建状态转移验证图
    plt.figure(figsize=(12, 8))
    
    # ONLOCK->ON 转移时间分布
    plt.subplot(2, 2, 1)
    if onlock_to_on_times:
        plt.hist(onlock_to_on_times, bins=20, alpha=0.7)
        plt.axvline(x=tonlock_sec, color='r', linestyle='--', label=f'闭锁时间 ({tonlock_sec}秒)')
        plt.title('ONLOCK->ON 转移时间分布')
        plt.xlabel('ONLOCK状态停留时间 (秒)')
        plt.ylabel('频次')
        plt.legend()
    else:
        plt.text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center')
        plt.title('ONLOCK->ON 转移时间分布')
    
    # OFFLOCK->OFF 转移时间分布
    plt.subplot(2, 2, 2)
    if offlock_to_off_times:
        plt.hist(offlock_to_off_times, bins=20, alpha=0.7)
        plt.axvline(x=tofflock_sec, color='r', linestyle='--', label=f'闭锁时间 ({tofflock_sec}秒)')
        plt.title('OFFLOCK->OFF 转移时间分布')
        plt.xlabel('OFFLOCK状态停留时间 (秒)')
        plt.ylabel('频次')
        plt.legend()
    else:
        plt.text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center')
        plt.title('OFFLOCK->OFF 转移时间分布')
    
    # ON->OFFLOCK 和 OFF->ONLOCK 是随机转移，应该遵循指数分布
    # 绘制转移事件的时间间隔分布
    on_to_offlock_intervals = []
    if len(transitions["ON->OFFLOCK"]) > 1:
        for i in range(1, len(transitions["ON->OFFLOCK"])):
            on_to_offlock_intervals.append(transitions["ON->OFFLOCK"][i][0] - transitions["ON->OFFLOCK"][i-1][0])
    
    plt.subplot(2, 2, 3)
    if on_to_offlock_intervals:
        plt.hist(on_to_offlock_intervals, bins=20, alpha=0.7)
        plt.title('ON->OFFLOCK 转移时间间隔分布')
        plt.xlabel('时间间隔 (秒)')
        plt.ylabel('频次')
    else:
        plt.text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center')
        plt.title('ON->OFFLOCK 转移时间间隔分布')
    
    off_to_onlock_intervals = []
    if len(transitions["OFF->ONLOCK"]) > 1:
        for i in range(1, len(transitions["OFF->ONLOCK"])):
            off_to_onlock_intervals.append(transitions["OFF->ONLOCK"][i][0] - transitions["OFF->ONLOCK"][i-1][0])
    
    plt.subplot(2, 2, 4)
    if off_to_onlock_intervals:
        plt.hist(off_to_onlock_intervals, bins=20, alpha=0.7)
        plt.title('OFF->ONLOCK 转移时间间隔分布')
        plt.xlabel('时间间隔 (秒)')
        plt.ylabel('频次')
    else:
        plt.text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center')
        plt.title('OFF->ONLOCK 转移时间间隔分布')
    
    plt.tight_layout()
    plt.savefig('state_transition_logic_results.png')
    print("状态转移逻辑验证完成，结果已保存为'state_transition_logic_results.png'")
    
    return transitions


def calculate_theoretical_steady_state(u0, u1, tonlock, tofflock):
    """
    计算理论稳态概率
    根据论文中的公式12计算
    """
    # 计算平均ON和OFF持续时间
    # tON = 1/u0, tOFF = 1/u1 (指数分布的平均值)
    t_on_avg = 1 / u0
    t_off_avg = 1 / u1
    
    # 计算总循环时间 (ON->OFFLOCK->OFF->ONLOCK->ON)
    t_cycle = t_on_avg + tonlock + t_off_avg + tofflock
    
    # 计算各状态的稳态概率
    p_on = t_on_avg / t_cycle
    p_offlock = tonlock / t_cycle
    p_off = t_off_avg / t_cycle
    p_onlock = tofflock / t_cycle
    
    # 物理开启状态的稳态概率 (ON + ONLOCK)
    p_m = p_on + p_onlock
    
    return {
        ACL_State.ON: p_on,
        ACL_State.OFFLOCK: p_offlock,
        ACL_State.OFF: p_off,
        ACL_State.ONLOCK: p_onlock,
        "p_m": p_m
    }


def test_steady_state_probability():
    """
    验证稳态概率：仿真足够长的时间后，统计各个状态的占比，与理论值比较
    """
    print("开始验证稳态概率...")
    
    # 设置FSM参数
    tonlock_sec = 3 * 60  # 开启后闭锁时间 3分钟
    tofflock_sec = 3 * 60  # 关闭后闭锁时间 3分钟
    u0 = 0.005  # 关闭概率率
    u1 = 0.002  # 开启概率率
    
    # 仿真参数
    sim_dt = 2.0  # 仿真时间步长 (秒)
    total_sim_time = 10 * 3600  # 总仿真时间 (10小时，足够长以达到稳态)
    num_simulations = 100  # 仿真的FSM数量
    
    # 理论稳态概率
    theoretical_probs = calculate_theoretical_steady_state(
        u0, u1, tonlock_sec, tofflock_sec
    )
    print("\n理论稳态概率:")
    for state, prob in theoretical_probs.items():
        if state != "p_m":
            print(f"  {state}: {prob:.4f}")
    print(f"  物理开启状态概率 p_m: {theoretical_probs['p_m']:.4f}")
    
    # 并行仿真多个FSM
    state_counts = defaultdict(int)
    physical_state_counts = defaultdict(int)
    total_samples = 0
    
    print(f"\n并行仿真{num_simulations}个FSM，每个仿真{total_sim_time/3600:.1f}小时...")
    
    # 为每个FSM记录状态历史
    all_fsm_states = []
    all_physical_states = []
    
    for sim_idx in range(num_simulations):
        if sim_idx % 10 == 0:
            print(f"  正在仿真第{sim_idx}个FSM...")
        
        # 随机选择初始状态
        initial_states = [ACL_State.ON, ACL_State.OFF, ACL_State.ONLOCK, ACL_State.OFFLOCK]
        initial_state = random.choice(initial_states)
        
        fsm = ACL_FSM(initial_state, tonlock_sec, tofflock_sec)
        fsm.update_migration_probabilities(u0, u1)
        
        # 如果初始状态是锁定状态，随机设置一个初始停留时间
        if initial_state == ACL_State.ONLOCK:
            fsm.time_in_current_state = random.uniform(0, tonlock_sec)
        elif initial_state == ACL_State.OFFLOCK:
            fsm.time_in_current_state = random.uniform(0, tofflock_sec)
        
        fsm_states = []
        physical_states = []
        
        # 仿真
        current_time = 0.0
        while current_time < total_sim_time:
            # 记录当前状态
            fsm_states.append(fsm.current_state)
            physical_states.append(fsm.get_physical_state())
            
            # 统计状态
            state_counts[fsm.current_state] += 1
            physical_state_counts[fsm.get_physical_state()] += 1
            total_samples += 1
            
            # 推进仿真
            fsm.step(sim_dt)
            current_time += sim_dt
        
        all_fsm_states.append(fsm_states)
        all_physical_states.append(physical_states)
    
    # 计算仿真稳态概率
    simulated_probs = {}
    for state in [ACL_State.ON, ACL_State.OFFLOCK, ACL_State.OFF, ACL_State.ONLOCK]:
        simulated_probs[state] = state_counts[state] / total_samples
    
    # 物理开启状态概率
    simulated_pm = physical_state_counts[1] / total_samples
    simulated_probs["p_m"] = simulated_pm
    
    # 打印仿真结果
    print("\n仿真稳态概率:")
    for state, prob in simulated_probs.items():
        if state != "p_m":
            print(f"  {state}: {prob:.4f}")
    print(f"  物理开启状态概率 p_m: {simulated_probs['p_m']:.4f}")
    
    # 比较理论与仿真结果
    print("\n理论与仿真结果比较:")
    for state in [ACL_State.ON, ACL_State.OFFLOCK, ACL_State.OFF, ACL_State.ONLOCK, "p_m"]:
        theoretical = theoretical_probs[state]
        simulated = simulated_probs[state]
        error = abs(theoretical - simulated) / theoretical * 100
        print(f"  {state}: 理论={theoretical:.4f}, 仿真={simulated:.4f}, 误差={error:.2f}%")
    
    # 绘制稳态概率比较图
    plt.figure(figsize=(10, 8))
    
    # 条形图比较理论与仿真稳态概率
    states = [ACL_State.ON, ACL_State.OFFLOCK, ACL_State.OFF, ACL_State.ONLOCK, "p_m"]
    state_labels = ["ON", "OFFLOCK", "OFF", "ONLOCK", "物理开启(p_m)"]
    theory_probs = [theoretical_probs[state] for state in states]
    sim_probs = [simulated_probs[state] for state in states]
    
    x = np.arange(len(states))
    width = 0.35
    
    plt.bar(x - width/2, theory_probs, width, label='理论值')
    plt.bar(x + width/2, sim_probs, width, label='仿真值')
    
    plt.xlabel('状态')
    plt.ylabel('概率')
    plt.title('稳态概率比较 (理论 vs 仿真)')
    plt.xticks(x, state_labels)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('steady_state_probability_results.png')
    print("稳态概率验证完成，结果已保存为'steady_state_probability_results.png'")
    
    return theoretical_probs, simulated_probs, all_fsm_states, all_physical_states


def test_migration_probability_formula():
    """
    测试基于响应功率比例的迁移概率计算公式
    """
    print("开始测试基于响应功率比例的迁移概率计算...")
    
    # 设置FSM参数
    tonlock_sec = 3 * 60  # 开启后闭锁时间 3分钟
    tofflock_sec = 3 * 60  # 关闭后闭锁时间 3分钟
    
    # 创建FSM实例
    fsm = ACL_FSM(ACL_State.OFF, tonlock_sec, tofflock_sec)
    
    # 设置额定功率
    Prate_ij = 1000.0  # 空调额定功率，单位W
    
    # 测试不同的响应功率与额定功率比例对u0和u1的影响
    power_ratios = [0.0, 0.1, 0.3, 0.5, 0.6, 0.8, 1.0]
    
    # 为每个比例重复计算多次，以观察随机范围
    test_counts = 1000
    
    # 记录结果
    results = {ratio: {"u0": [], "u1": []} for ratio in power_ratios}
    
    print("\n测试不同功率比例下的迁移概率计算:")
    
    for ratio in power_ratios:
        print(f"\n功率比例 Pa,ij*/Prate,ij = {ratio}:")
        Pa_ij_star = ratio * Prate_ij
        
        for _ in range(test_counts):
            u0, u1 = fsm.calculate_migration_probabilities(Pa_ij_star, Prate_ij)
            results[ratio]["u0"].append(u0)
            results[ratio]["u1"].append(u1)
        
        # 计算统计数据
        avg_u0 = sum(results[ratio]["u0"]) / test_counts if results[ratio]["u0"] else 0
        avg_u1 = sum(results[ratio]["u1"]) / test_counts if results[ratio]["u1"] else 0
        
        min_u0 = min(results[ratio]["u0"]) if results[ratio]["u0"] else 0
        max_u0 = max(results[ratio]["u0"]) if results[ratio]["u0"] else 0
        
        min_u1 = min(results[ratio]["u1"]) if results[ratio]["u1"] else 0
        max_u1 = max(results[ratio]["u1"]) if results[ratio]["u1"] else 0
        
        # 打印结果
        if ratio <= 0.5:
            expected_range_u0 = f"[{0.5 * fsm.control_cycle_sec / tofflock_sec:.6f}, {1.5 * fsm.control_cycle_sec / tofflock_sec:.6f}]"
            print(f"  u0: 平均={avg_u0:.6f}, 范围=[{min_u0:.6f}, {max_u0:.6f}], 预期范围={expected_range_u0}")
            print(f"  u1: 平均={avg_u1:.6f}, 范围=[{min_u1:.6f}, {max_u1:.6f}], 预期=0")
        else:
            expected_range_u1 = f"[{0.5 * fsm.control_cycle_sec / tonlock_sec:.6f}, {1.5 * fsm.control_cycle_sec / tonlock_sec:.6f}]"
            print(f"  u0: 平均={avg_u0:.6f}, 范围=[{min_u0:.6f}, {max_u0:.6f}], 预期=0")
            print(f"  u1: 平均={avg_u1:.6f}, 范围=[{min_u1:.6f}, {max_u1:.6f}], 预期范围={expected_range_u1}")
    
    # 绘制结果
    plt.figure(figsize=(12, 10))
    
    # 绘制u0随功率比例的变化
    plt.subplot(2, 1, 1)
    box_data_u0 = [results[ratio]["u0"] for ratio in power_ratios]
    plt.boxplot(box_data_u0, labels=[f"{ratio}" for ratio in power_ratios])
    plt.title('u0 随功率比例 Pa,ij*/Prate,ij 的变化')
    plt.xlabel('功率比例 Pa,ij*/Prate,ij')
    plt.ylabel('u0 (关闭概率率)')
    plt.grid(True)
    
    # 标记 0.5 的分界线
    plt.axvline(x=4, color='r', linestyle='--', label='功率比例 = 0.5')
    plt.legend()
    
    # 绘制理论边界
    x_low = np.arange(1, 5)  # 对应比例 <= 0.5
    low_bound = 0.5 * fsm.control_cycle_sec / tofflock_sec
    high_bound = 1.5 * fsm.control_cycle_sec / tofflock_sec
    plt.plot(x_low, [low_bound] * len(x_low), 'g--', label='理论下界')
    plt.plot(x_low, [high_bound] * len(x_low), 'g--', label='理论上界')
    
    # 绘制u1随功率比例的变化
    plt.subplot(2, 1, 2)
    box_data_u1 = [results[ratio]["u1"] for ratio in power_ratios]
    plt.boxplot(box_data_u1, labels=[f"{ratio}" for ratio in power_ratios])
    plt.title('u1 随功率比例 Pa,ij*/Prate,ij 的变化')
    plt.xlabel('功率比例 Pa,ij*/Prate,ij')
    plt.ylabel('u1 (开启概率率)')
    plt.grid(True)
    
    # 标记 0.5 的分界线
    plt.axvline(x=4, color='r', linestyle='--', label='功率比例 = 0.5')
    plt.legend()
    
    # 绘制理论边界
    x_high = np.arange(5, 8)  # 对应比例 > 0.5
    low_bound = 0.5 * fsm.control_cycle_sec / tonlock_sec
    high_bound = 1.5 * fsm.control_cycle_sec / tonlock_sec
    plt.plot(x_high, [low_bound] * len(x_high), 'g--', label='理论下界')
    plt.plot(x_high, [high_bound] * len(x_high), 'g--', label='理论上界')
    
    plt.tight_layout()
    plt.savefig('migration_probability_formula_results.png')
    print("迁移概率计算公式测试完成，结果已保存为'migration_probability_formula_results.png'")
    
    # 测试update_migration_probabilities方法
    print("\n测试update_migration_probabilities方法:")
    
    # 直接提供u0和u1
    fsm.update_migration_probabilities(u0=0.01, u1=0.02)
    print(f"  直接提供: u0={fsm.u0}, u1={fsm.u1} (预期: u0=0.01, u1=0.02)")
    
    # 通过响应功率和额定功率计算
    fsm.update_migration_probabilities(Pa_ij_star=800, Prate_ij=1000)
    print(f"  通过功率计算 (Pa,ij*/Prate,ij=0.8): u0={fsm.u0}, u1={fsm.u1}")
    
    fsm.update_migration_probabilities(Pa_ij_star=300, Prate_ij=1000)
    print(f"  通过功率计算 (Pa,ij*/Prate,ij=0.3): u0={fsm.u0}, u1={fsm.u1}")
    
    return results


def main():
    """主函数：运行所有测试"""
    print("开始 FSM 半马尔科夫过程验证...")
    start_time = time.time()
    
    # 执行固定概率仿真测试
    time_points, states, physical_states, time_in_states = test_fixed_probability_simulation()
    
    # 执行状态转移逻辑验证
    transitions = test_state_transition_logic()
    
    # 执行稳态概率验证
    theoretical_probs, simulated_probs, all_fsm_states, all_physical_states = test_steady_state_probability()
    
    # 测试迁移概率计算公式
    migration_prob_results = test_migration_probability_formula()
    
    elapsed_time = time.time() - start_time
    print(f"所有测试完成！总耗时: {elapsed_time:.2f} 秒")
    print("请查看生成的图像文件，以分析 FSM 模型的控制过程。")


if __name__ == "__main__":
    main() 