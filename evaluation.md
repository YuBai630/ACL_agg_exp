## FSM 与半马尔科夫过程的验证 (控制过程):

目标: 验证有限状态机能在设定的闭锁时间 (tonlock, tofflock) 和迁移概率率 (α0, α1 或等效的 u0, u1) 的驱动下正确转移状态，并能体现半马尔科夫过程的特性。

方法:

固定概率仿真: 给定固定的迁移概率率 (u0, u1)，从某个初始状态开始仿真 FSM 较长时间（远大于闭锁时间）。

检查状态转移逻辑: 验证状态转移是否严格遵守的转移条件：

ON -> OFFLOCK 仅在 t >= tonlock 且以概率 u0\*Δt 发生。

OFF -> ONLOCK 仅在 t >= tofflock 且以概率 u1\*Δt 发生。

ONLOCK -> ON 仅在满足 t >= tonlock 后确定发生。

OFFLOCK -> OFF 仅在满足 t >= tofflock 后确定发生。

检查闭锁时间 (tonlock, tofflock) 是否被正确强制执行，即在闭锁时间内不允许从 LOCK 状态转移出去，也不允许从 ON/OFF 状态转移到 LOCK 状态。

验证稳态概率: 仿真足够长的时间后，统计各个状态（ON, OFF, ONLOCK, OFFLOCK）在总时间内的占比。将这个实际占比与根据迁移概率率和闭锁时间计算出的稳态概率理论值 (pm，公式 12) 进行比较。如果仿真样本数足够大（例如论文中的 10000 个空调并行仿真

## 信息物理耦合的验证 (ETP 与 FSM 的连接)

目标: 验证 FSM 的离散状态如何正确驱动 ETP 模型的物理开关状态 m(t)

方法:
(1) 联立仿真: 在集成模型中进行仿真。在每个小的时间步长 (Δt)，首先根据 FSM 的当前状态确定 m(t) 的值（ONLOCK/ON 为 1，OFFLOCK/OFF 为 0）。然后将此 m(t) 值作为输入，更新 ETP 模型计算下一时刻的温度
(2) 观察对应关系: 观察仿真过程中 FSM 的状态序列和 ETP 模型中使用的 m(t) 序列，确认两者之间始终保持正确的对应关系。当 FSM 进入 ON 或 ONLOCK 状态时，m(t) 变为 1；进入 OFF 或 OFFLOCK 状态时，m(t) 变为 0。
