import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
import time
from ETP import SecondOrderETPModel
from FSM import ACL_FSM, ACL_State

def simulate_coupled_system(etp_model, fsm_model, initial_etp_state, global_tout_func, total_sim_time_sec, dt_sec):
    """
    Simulates the coupled ETP and FSM system.

    Args:
        etp_model: An instance of SecondOrderETPModel.
        fsm_model: An instance of ACL_FSM.
        initial_etp_state: List [Ta0, Tm0], initial ETP temperatures.
        global_tout_func: Function global_tout_func(t_sec) -> temperature.
        total_sim_time_sec: Total simulation time in seconds.
        dt_sec: Simulation time step in seconds.

    Returns:
        Tuple of lists: (time_points, Ta_history, Tm_history, Tout_history, fsm_state_history, m_t_history)
    """
    print("开始进行ETP-FSM耦合仿真...")

    current_time_sec = 0.0
    etp_state = list(initial_etp_state) # Ensure it's a mutable list

    time_points = []
    Ta_history = []
    Tm_history = []
    Tout_history = []
    fsm_state_history = []
    m_t_history = []

    num_steps = int(total_sim_time_sec / dt_sec)

    # Log initial FSM state and m(t)
    last_fsm_state = fsm_model.current_state
    last_m_t = fsm_model.get_physical_state()
    print(f"  [t={current_time_sec:.2f}s] Initial FSM State: {last_fsm_state}, m(t): {last_m_t}")

    for step in range(num_steps):
        # 1. Get m(t) from FSM's current physical state
        current_m_t = fsm_model.get_physical_state() # Get m(t) *before* FSM step for current ETP step

        # Log m(t) change if it occurred (due to FSM state change in *previous* FSM step)
        if current_m_t != last_m_t:
            print(f"  [t={current_time_sec:.2f}s] m(t) changed from {last_m_t} to {current_m_t} (FSM state: {fsm_model.current_state})")
        
        # 2. Define Tout_func and mode_func for this ETP step
        #    The ETP's simulate function expects t_span relative to the start of the small step (0 to dt_sec)
        time_for_this_etp_step_start = current_time_sec
        def tout_for_current_etp_step(t_local_solve_ivp):
            return global_tout_func(time_for_this_etp_step_start + t_local_solve_ivp)
        
        def mode_for_current_etp_step(t_local_solve_ivp, etp_s): # etp_s is [Ta, Tm]
            return current_m_t # m(t) is constant for the duration of this dt_sec ETP step

        # 3. Simulate ETP for one dt_sec
        #    The ETP model's simulate method integrates from t_span[0] to t_span[1]
        #    We are taking discrete steps, so we simulate for a duration of dt_sec
        solution_step = etp_model.simulate(
            T0=etp_state[0], 
            Tm0=etp_state[1],
            Tout_func=tout_for_current_etp_step, 
            mode_func=mode_for_current_etp_step,
            t_span=[0, dt_sec],       # Simulate for the duration dt_sec
            t_eval=[dt_sec]          # Evaluate only at the end of the step
        )
        etp_state = [solution_step.y[0][-1], solution_step.y[1][-1]]

        # 4. Step FSM
        previous_fsm_state_for_logging = fsm_model.current_state # State before FSM step
        fsm_model.step(dt_sec)
        current_fsm_state_for_logging = fsm_model.current_state # State after FSM step

        # Log FSM state change if it occurred
        if current_fsm_state_for_logging != previous_fsm_state_for_logging:
            print(f"  [t={current_time_sec:.2f}s] FSM State changed from {previous_fsm_state_for_logging} to {current_fsm_state_for_logging}")

        # 5. Record history
        time_points.append(current_time_sec)
        Ta_history.append(etp_state[0])
        Tm_history.append(etp_state[1])
        Tout_history.append(global_tout_func(current_time_sec)) # Store Tout at the beginning of the interval
        fsm_state_history.append(current_fsm_state_for_logging) # Store the state *after* FSM step for next iteration's m(t)
        m_t_history.append(current_m_t) # m(t) used for *this* ETP step
        
        # Update last known states for next iteration's logging
        last_m_t = current_m_t
        # last_fsm_state is implicitly handled by previous_fsm_state_for_logging in the next loop

        if step % (num_steps // 20) == 0 or step == num_steps -1 : # Print progress
             print(f"  仿真进度: {current_time_sec/3600:.2f} / {total_sim_time_sec/3600:.2f} 小时 ({(step+1)*100/num_steps:.1f}%)")

        # 6. Advance time
        current_time_sec += dt_sec

    print("ETP-FSM耦合仿真完成.")
    return time_points, Ta_history, Tm_history, Tout_history, fsm_state_history, m_t_history

def plot_coupling_results(time_points_sec, Ta_history, Tm_history, Tout_history, fsm_state_history, m_t_history, filename='etp_fsm_coupling_results.png'):
    """
    Plots the results of the coupled simulation.
    """
    print(f"开始绘制耦合仿真结果图表，将保存为 {filename}...")
    time_points_hours = np.array(time_points_sec) / 3600.0

    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Panel 1: Temperatures
    axs[0].plot(time_points_hours, Ta_history, label='室内空气温度 (Ta)', color='blue')
    axs[0].plot(time_points_hours, Tm_history, label='建筑质量温度 (Tm)', color='red', linestyle='--')
    axs[0].plot(time_points_hours, Tout_history, label='室外温度 (Tout)', color='green', linestyle=':')
    axs[0].set_ylabel('温度 (°C)')
    axs[0].set_title('ETP-FSM 耦合仿真: 温度变化')
    axs[0].legend()
    axs[0].grid(True)

    # Panel 2: FSM State
    # Map FSM states to numerical values for plotting
    state_to_num = {ACL_State.OFF: 0, ACL_State.OFFLOCK: 0.5, ACL_State.ON: 1, ACL_State.ONLOCK: 1.5}
    fsm_numeric_history = [state_to_num[s] for s in fsm_state_history]
    axs[1].plot(time_points_hours, fsm_numeric_history, label='FSM 状态', color='purple', drawstyle='steps-post')
    axs[1].set_yticks([0, 0.5, 1, 1.5])
    axs[1].set_yticklabels(['OFF', 'OFFLOCK', 'ON', 'ONLOCK'])
    axs[1].set_ylabel('FSM 状态')
    axs[1].set_title('FSM 状态变化')
    axs[1].legend()
    axs[1].grid(True)

    # Panel 3: Physical state m(t)
    axs[2].plot(time_points_hours, m_t_history, label='空调物理状态 m(t)', color='orange', drawstyle='steps-post')
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(['关闭 (0)', '开启 (1)'])
    axs[2].set_xlabel('时间 (小时)')
    axs[2].set_ylabel('物理状态 m(t)')
    axs[2].set_title('空调物理运行状态 m(t)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"图表已保存至 {filename}")


def main():
    """Main function to run the coupled simulation test."""
    # ETP Model Parameters (example values)
    Ca = 1.0e4    # J/°C (reduced for faster response for visualization)
    Cm = 2.0e5    # J/°C (reduced for faster response)
    Ua = 150      # W/°C
    Um = 300      # W/°C
    Hm = -5000    # W (Cooling power, negative for cooling)
    Qgain = 200   # W (Internal heat gain)
    etp_model = SecondOrderETPModel(Ca, Cm, Ua, Um, Hm, Qgain)

    # FSM Model Parameters
    tonlock_sec = 3 * 60  # 3 minutes
    tofflock_sec = 3 * 60 # 3 minutes
    initial_fsm_state = ACL_State.OFF
    fsm_model = ACL_FSM(initial_fsm_state, tonlock_sec, tofflock_sec)
    # Set fixed transition probabilities for testing
    u0 = 0.001  # Probability rate from ON to OFFLOCK
    u1 = 0.002  # Probability rate from OFF to ONLOCK
    fsm_model.update_migration_probabilities(u0, u1)

    # Simulation Configuration
    initial_Ta = 28.0  # °C
    initial_Tm = 27.5  # °C
    initial_etp_state = [initial_Ta, initial_Tm]
    
    # Outdoor temperature profile (constant for simplicity)
    def constant_tout(t_sec):
        return 32.0  # °C

    total_sim_duration_hours = 3
    total_sim_time_sec = total_sim_duration_hours * 3600
    dt_sec = 10.0  # Simulation time step in seconds (larger for faster overall test simulation)

    # Run simulation
    start_time = time.time()
    time_points, Ta_hist, Tm_hist, Tout_hist, fsm_state_hist, m_t_hist = simulate_coupled_system(
        etp_model, fsm_model, initial_etp_state, constant_tout, total_sim_time_sec, dt_sec
    )
    end_time = time.time()
    print(f"总仿真耗时: {end_time - start_time:.2f} 秒")

    # Plot results
    plot_coupling_results(time_points, Ta_hist, Tm_hist, Tout_hist, fsm_state_hist, m_t_hist)

if __name__ == "__main__":
    main() 