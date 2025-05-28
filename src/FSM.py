import random
import time # 为了表示时间，或者使用一个内部计数器

# 定义状态 (可以使用Enum，这里为了简单用字符串或数字)
class ACL_State:
    ON = "ON"
    OFF = "OFF"
    ONLOCK = "ONLOCK"
    OFFLOCK = "OFFLOCK"

class ACL_FSM:
    def __init__(self, initial_state, tonlock_sec, tofflock_sec, sim_dt_sec):
        """
        初始化空调负荷的有限状态机
        :param initial_state: 初始状态 (ACL_State 枚举值)
        :param tonlock_sec: 开启后闭锁时间 (秒)
        :param tofflock_sec: 关闭后闭锁时间 (秒)
        :param sim_dt_sec: FSM仿真的时间步长 (秒)
        """
        self.current_state = initial_state
        self.time_in_current_state = 0.0
        self.tonlock = tonlock_sec
        self.tofflock = tofflock_sec
        self.sim_dt_sec = sim_dt_sec # FSM的仿真步长

        self.u0 = 0.0 # 从ON到OFFLOCK的转移概率 (在sim_dt_sec内)
        self.u1 = 0.0 # 从OFF到ONLOCK的转移概率 (在sim_dt_sec内)

    def calculate_migration_probabilities(self, Pa_ij_star, Prate_ij):
        """
        计算在 sim_dt_sec 时间步长内的状态转移概率 u0 和 u1。

        文献中的补充条件:
        Δt_exec / u1_prob = Rand(0.5*t_on_lock, 1.5*t_on_lock)  (当 P* > 0.5 Prate)
        => u1_prob = Δt_exec / Rand(0.5*t_on_lock, 1.5*t_on_lock)
        这里 Δt_exec 是 self.sim_dt_sec。 u1_prob 是在一个 Δt_exec 内的转移概率。

        稳态功率比率公式:
        P_ratio = (T_on_avg + self.tofflock) / (T_on_avg + T_off_avg + self.tonlock + self.tofflock)
        其中 T_on_avg = self.sim_dt_sec / u0_prob (平均ON时间，不含锁闭)
             T_off_avg = self.sim_dt_sec / u1_prob (平均OFF时间，不含锁闭)

        :param Pa_ij_star: 响应功率 Pa,ij* (W)
        :param Prate_ij: 额定功率 Prate,ij (W)
        """
        power_ratio = Pa_ij_star / Prate_ij if Prate_ij > 0 else 0.0
        power_ratio = max(0.0, min(1.0, power_ratio)) # 限制在 [0, 1]

        u0_prob, u1_prob = 0.0, 0.0
        delta_t_exec = self.sim_dt_sec

        # 处理T_lock为0或极小的情况，防止除以零
        min_lock_time = max(1.0, delta_t_exec) # 确保锁闭时间至少为一个步长，避免概率大于1

        current_tonlock = max(self.tonlock, min_lock_time)
        current_tofflock = max(self.tofflock, min_lock_time)

        if power_ratio > 0.5: # 倾向于开启 (计算 u1_prob, 然后推导 u0_prob)
            rand_time_off_intrinsic = random.uniform(0.5 / current_tonlock, 1.5 / current_tonlock)
            u1_prob = delta_t_exec * rand_time_off_intrinsic
            u1_prob = max(0.0, min(1.0, u1_prob)) # 确保概率在 [0,1]
            
            # P_ratio = (T0_avg + tofflock) / (T0_avg + T1_avg + tonlock + tofflock)
            # T1_avg = delta_t_exec / u1_prob (if u1_prob > 0)
            T1_avg = delta_t_exec / u1_prob if u1_prob > 1e-9 else float('inf')
            
            # Numerator for T0_avg: P_ratio * (T1_avg + tonlock + tofflock) - tofflock
            # Denominator for T0_avg: 1 - P_ratio
            num_T0_avg = power_ratio * (T1_avg + current_tonlock + current_tofflock) - current_tofflock
            den_T0_avg = 1.0 - power_ratio

            if den_T0_avg < 1e-9 : # power_ratio is 1, handled above
                T0_avg = float('inf')
            elif num_T0_avg <= 0: # 意味着应该长时间开启
                    T0_avg = float('inf') # u0_prob will be 0
            else:
                T0_avg = num_T0_avg / den_T0_avg
            
            u0_prob = delta_t_exec / T0_avg if T0_avg > 1e-9 and T0_avg != float('inf') else 0.0
            u0_prob = max(0.0, min(1.0, u0_prob))

        else: # 倾向于关闭 (计算 u0_prob, 然后推导 u1_prob)
            rand_time_on_intrinsic = random.uniform(0.5 / current_tofflock, 1.5 / current_tofflock)
            u0_prob = delta_t_exec * rand_time_on_intrinsic
            u0_prob = max(0.0, min(1.0, u0_prob))

            if abs(power_ratio - 0.0) < 1e-9: # 应该一直关
                u1_prob = 0.0
            else:
                # P_ratio * (T0_avg + T1_avg + tonlock + tofflock) = T0_avg + tofflock
                # T0_avg = delta_t_exec / u0_prob (if u0_prob > 0)
                T0_avg = delta_t_exec / u0_prob if u0_prob > 1e-9 else float('inf')

                # Numerator for T1_avg: (T0_avg + tofflock)*(1-P_ratio) - P_ratio*(tonlock)
                # Denominator for T1_avg: P_ratio
                num_T1_avg = (T0_avg + current_tofflock) * (1.0 - power_ratio) - power_ratio * current_tonlock
                den_T1_avg = power_ratio
                
                if den_T1_avg < 1e-9: # power_ratio is 0, handled above
                    T1_avg = float('inf')
                elif num_T1_avg <=0: # 意味着应该长时间关闭
                    T1_avg = float('inf') # u1_prob will be 0
                else:
                    T1_avg = num_T1_avg / den_T1_avg

                u1_prob = delta_t_exec / T1_avg if T1_avg > 1e-9 and T1_avg != float('inf') else 0.0
                u1_prob = max(0.0, min(1.0, u1_prob))
        
        # Fallback safety: if any prob is NaN due to unforeseen division issues, set to a neutral small value
        if not (u0_prob >= 0 and u0_prob <=1): u0_prob = 0.01
        if not (u1_prob >= 0 and u1_prob <=1): u1_prob = 0.01
            
        return u0_prob, u1_prob

    def update_migration_probabilities(self, u0=None, u1=None, Pa_ij_star=None, Prate_ij=None):
        """
        在新的控制周期开始时更新迁移概率
        :param u0: 新的关闭概率率 (如果直接提供)
        :param u1: 新的开启概率率 (如果直接提供)
        :param Pa_ij_star: 响应功率 Pa,ij* (W) (用于计算 u0 和 u1)
        :param Prate_ij: 额定功率 Prate,ij (W) (用于计算 u0 和 u1)
        """
        # 如果提供了响应功率和额定功率，则使用公式计算 u0 和 u1
        if Pa_ij_star is not None and Prate_ij is not None:
            self.u0, self.u1 = self.calculate_migration_probabilities(Pa_ij_star, Prate_ij)
        # 否则直接使用提供的 u0 和 u1 (如果有)
        elif u0 is not None and u1 is not None:
            self.u0 = u0
            self.u1 = u1
        # 如果既没有提供计算参数，也没有直接提供 u0 和 u1，则保持原值不变

    def step(self, dt):
        """
        仿真 FSM 经过一个小时间步长 dt。
        u0 和 u1 是在 self.sim_dt_sec 内发生转移的概率。
        因此，这里直接使用 self.u0 和 self.u1。
        传入的 dt 应等于 self.sim_dt_sec。
        :param dt: 仿真时间步长 (秒)，应等于 FSM 初始化时的 sim_dt_sec
        """
        # 确保传入的dt与FSM配置的sim_dt_sec一致
        if abs(dt - self.sim_dt_sec) > 1e-9:
            # 可以选择抛出错误或打印警告，这里打印警告并继续
            print(f"警告: FSM step() 方法传入的 dt ({dt}) 与 FSM 配置的 sim_dt_sec ({self.sim_dt_sec}) 不符。将使用 self.u0/u1 作为转移概率。")
            # 这种情况下，如果dt不同，u0/u1作为概率的含义就不准确了，除非它们是速率。
            # 但根据calculate_migration_probabilities的修改，u0/u1现在是针对self.sim_dt_sec的概率。

        next_state = self.current_state
        transitioned = False

        if self.current_state == ACL_State.ON:
            if random.random() < self.u0: # u0 是从ON到OFFLOCK的概率
                next_state = ACL_State.OFFLOCK
                transitioned = True

        elif self.current_state == ACL_State.OFF:
            if random.random() < self.u1: # u1 是从OFF到ONLOCK的概率
                next_state = ACL_State.ONLOCK
                transitioned = True

        elif self.current_state == ACL_State.ONLOCK:
            if self.time_in_current_state >= self.tonlock:
                next_state = ACL_State.ON
                transitioned = True

        elif self.current_state == ACL_State.OFFLOCK:
            if self.time_in_current_state >= self.tofflock:
                next_state = ACL_State.OFF
                transitioned = True

        if transitioned:
            self.current_state = next_state
            self.time_in_current_state = 0.0
        else:
            # 即使没有状态转移，时间步长 dt 仍然消耗在当前状态
            self.time_in_current_state += dt # 使用传入的实际步长 dt 来累加时间

    def get_physical_state(self):
        """
        获取当前的物理运行状态 m(t) [1]
        :return: 1 如果空调开启 (ON 或 ONLOCK)，0 如果空调关闭 (OFF 或 OFFLOCK)
        """
        if self.current_state in [ACL_State.ON, ACL_State.ONLOCK]:
            return 1
        else:
            return 0

# --- 示例使用 ---
if __name__ == "__main__":
    # 示例参数
    TONLOCK_SEC = 3 * 60  # 3分钟闭锁时间 [1]
    TOFFLOCK_SEC = 3 * 60 # 3分钟闭锁时间 [1]
    SIM_DT = 2.0         # 仿真时间步长 (秒) [2]
    CONTROL_CYCLE_SEC = 0.5 * 3600 # 控制周期 (0.5小时) [9]
    TOTAL_SIM_TIME_SEC = 1 * 3600 # 总仿真时间 (例如 1 小时)

    print("===== FSM示例：基于固定概率 =====")
    # 模拟一个空调的 FSM 运行
    # 假设某个控制周期内，通过本地控制逻辑计算得到的迁移概率率 [6, 8]
    # (注意：u0, u1 的具体值取决于响应功率 Pa,ij* 和求解过程，这里是示例值)
    EXAMPLE_U0 = 0.005 # 示例关闭概率率
    EXAMPLE_U1 = 0.002 # 示例开启概率率

    acl_fsm = ACL_FSM(ACL_State.OFF, TONLOCK_SEC, TOFFLOCK_SEC, SIM_DT)
    print(f"初始状态: {acl_fsm.current_state}, 物理状态 m(t): {acl_fsm.get_physical_state()}")

    # 在仿真开始或新的控制周期开始时更新迁移概率
    acl_fsm.update_migration_probabilities(u0=EXAMPLE_U0, u1=EXAMPLE_U1)
    print(f"更新迁移概率率: u0={acl_fsm.u0}, u1={acl_fsm.u1}")

    current_time_sec = 0.0
    state_history = []

    while current_time_sec < TOTAL_SIM_TIME_SEC:
        # 在实际的完整仿真中，可能每个 CONTROL_CYCLE_SEC 才更新 u0 和 u1
        # 但 FSM 的 step 方法在每个 SIM_DT 调用

        acl_fsm.step(SIM_DT)
        state_history.append((current_time_sec, acl_fsm.current_state, acl_fsm.get_physical_state()))

        current_time_sec += SIM_DT

    print("\n仿真结束，状态历史 (部分展示):")
    for i in range(min(10, len(state_history))):
        t, state, physical_state = state_history[i]
        print(f"Time: {t:.1f}s, State: {state}, Physical State m(t): {physical_state}")

    print("...")
    for i in range(max(0, len(state_history) - 10), len(state_history)):
         t, state, physical_state = state_history[i]
         print(f"Time: {t:.1f}s, State: {state}, Physical State m(t): {physical_state}")
         
    # 新增示例：基于响应功率比例的迁移概率计算
    print("\n\n===== FSM示例：基于响应功率比例 =====")
    
    # 重新创建实例
    acl_fsm_2 = ACL_FSM(ACL_State.ON, TONLOCK_SEC, TOFFLOCK_SEC, SIM_DT)
    
    # 设置额定功率和不同的响应功率
    PRATE_IJ = 1000.0  # 额定功率（例如1000W）
    
    # 测试不同响应功率下的迁移概率计算
    print("\n响应功率比例对迁移概率的影响:")
    
    # 1. 低响应功率（小于50%额定功率）- 倾向于关闭空调
    PA_IJ_STAR_LOW = 300.0  # 响应功率 (30% 额定功率)
    u0, u1 = acl_fsm_2.calculate_migration_probabilities(PA_IJ_STAR_LOW, PRATE_IJ)
    print(f"\n响应功率比例 Pa,ij*/Prate,ij = {PA_IJ_STAR_LOW/PRATE_IJ:.2f} (低响应功率，倾向关闭):")
    print(f"  计算得到: u0={u0:.6f}, u1={u1:.6f}")
    print(f"  预期范围: u0∈[{0.5 * CONTROL_CYCLE_SEC / TOFFLOCK_SEC:.6f}, {1.5 * CONTROL_CYCLE_SEC / TOFFLOCK_SEC:.6f}], u1=0")
    
    # 更新迁移概率并模拟状态转移
    acl_fsm_2.update_migration_probabilities(Pa_ij_star=PA_IJ_STAR_LOW, Prate_ij=PRATE_IJ)
    print(f"  当前状态: {acl_fsm_2.current_state}, 物理状态: {acl_fsm_2.get_physical_state()}")
    print(f"  FSM更新后: u0={acl_fsm_2.u0:.6f}, u1={acl_fsm_2.u1:.6f}")
    
    # 模拟状态转移
    print("  模拟低响应功率下的状态转移（10步）:")
    for i in range(10):
        old_state = acl_fsm_2.current_state
        acl_fsm_2.step(SIM_DT)
        print(f"    步骤 {i+1}: {old_state} -> {acl_fsm_2.current_state}, 物理状态: {acl_fsm_2.get_physical_state()}")
    
    # 2. 高响应功率（大于50%额定功率）- 倾向于开启空调
    PA_IJ_STAR_HIGH = 800.0  # 响应功率 (80% 额定功率)
    
    # 重置FSM以便于观察
    acl_fsm_2 = ACL_FSM(ACL_State.OFF, TONLOCK_SEC, TOFFLOCK_SEC, SIM_DT)
    
    u0, u1 = acl_fsm_2.calculate_migration_probabilities(PA_IJ_STAR_HIGH, PRATE_IJ)
    print(f"\n响应功率比例 Pa,ij*/Prate,ij = {PA_IJ_STAR_HIGH/PRATE_IJ:.2f} (高响应功率，倾向开启):")
    print(f"  计算得到: u0={u0:.6f}, u1={u1:.6f}")
    print(f"  预期范围: u0=0, u1∈[{0.5 * CONTROL_CYCLE_SEC / TONLOCK_SEC:.6f}, {1.5 * CONTROL_CYCLE_SEC / TONLOCK_SEC:.6f}]")
    
    # 更新迁移概率并模拟状态转移
    acl_fsm_2.update_migration_probabilities(Pa_ij_star=PA_IJ_STAR_HIGH, Prate_ij=PRATE_IJ)
    print(f"  当前状态: {acl_fsm_2.current_state}, 物理状态: {acl_fsm_2.get_physical_state()}")
    print(f"  FSM更新后: u0={acl_fsm_2.u0:.6f}, u1={acl_fsm_2.u1:.6f}")
    
    # 模拟状态转移
    print("  模拟高响应功率下的状态转移（10步）:")
    for i in range(10):
        old_state = acl_fsm_2.current_state
        acl_fsm_2.step(SIM_DT)
        print(f"    步骤 {i+1}: {old_state} -> {acl_fsm_2.current_state}, 物理状态: {acl_fsm_2.get_physical_state()}")
    
    print("\n示例结束。")