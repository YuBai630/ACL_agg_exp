"""
关于单个 ACL (空调负荷) 的需求曲线是如何建模的，可以概括如下：
1.目的与作用:

单个 ACL 的需求曲线 dij(π) 是其信息物理耦合模型中的重要组成部分。

它作为空调负荷 CPS (信息物理系统) 与外部（主要是聚合商）的联系接口。

通过需求曲线，空调负荷能够表达自身的用电迫切性和灵活性。

它使得空调能够响应从聚合商接收到的虚拟价格信号 π*，并确定其所需承担的响应功率 Pa,ij*。

使用需求曲线作为接口，也能更好地保护用户的数据隐私和信息安全。
2.建模形式:

需求曲线 dij(π) 是一个以虚拟价格 π 为横坐标（控制信号，范围限制在 [-1, 1]，不代表货币单位）、以动态功率为纵坐标的函数。

该需求曲线具有单调递减的特性。
3.关键点定义:

需求曲线由三个关键点确定，这三个点反映了空调在不同"虚拟价格"下的用电意愿：

点 A：(虚拟价格 π = 1, 对应动态功率 Pmin,ij)。虚拟价格为 1 代表电网最不希望用电（相当于电价很高），此时空调倾向于最低的用电功率 Pmin,ij。

点 B：(虚拟价格 π = 0, 对应动态功率 Pset,ij)。虚拟价格为 0 代表解除控制信号，此时空调倾向于消耗恰好能将室温维持或回到设定温度所需的功率 Pset,ij。

点 C：(虚拟价格 π = -1, 对应动态功率 Pmax,ij)。虚拟价格为 -1 代表电网最希望用电（相当于电价很低或需要削峰），此时空调倾向于最高的用电功率 Pmax,ij。
4.关键功率值的计算:

需求曲线中的纵坐标值，即 Pmin,ij、Pset,ij、Pmax,ij，是基于空调的动态功率 Pd,ij 概念计算得出的。

动态功率 Pd,ij 定义为在 1 个控制周期内，使当前室温变化到指定温度所需的期望功率。它由 ETP 模型 (等效热参数模型) 推导而来，计算公式见式 (18)。

基于动态功率，定义了几个特定的功率值：

Pmin0,ij：指定温度为 Tmax,ij (用户允许的最高温度) 时所需的动态功率。

Pmax0,ij：指定温度为 Tmin,ij (用户允许的最低温度) 时所需的动态功率。

Pset0,ij：指定温度为 Tset,ij (用户的设定温度) 时所需的动态功率，代表回到最适温度所需的功率。

最终用于需求曲线的 Pmin,ij、Pset,ij、Pmax,ij 会考虑功率约束：

Pmax,ij = min(Pmax0,ij, Prate,ij)：最高功率不能超过空调的额定功率 Prate,ij。

Pmin,ij = max(Pmin0,ij, 0)：最低功率在制冷场景下不能为负数。

Pset,ij = max(min(Pset0,ij, Prate,ij), 0)：设定功率需要在 0 和 Prate,ij 之间。
5.温度软边界:

为了应对概率控制和闭锁时间引入的室温控制误差，论文引入了温度"软边界"的概念。

这意味着在计算 Pmin0,ij (对应 Tmax,ij) 和 Pmax0,ij (对应 Tmin,ij) 时，实际使用的指定温度不是严格的 Tmin,ij 和 Tmax,ij，而是缩小的温度区间 [Tmin,ij + δ, Tmax,ij - δ] 内的边界温度。其中 δ 是一个根据温度范围确定的裕量。
6.
更新:

每个空调负荷会在每个控制周期（例如 0.5h）根据其当前状况（如当前室温）更新其需求曲线，以便准确反映其当前的灵活性。
总而言之，单个 ACL 的需求曲线建模是将其物理状态（当前室温、热参数）与控制策略（期望达到的温度、功率约束）相结合，通过动态功率的概念，将空调的"意愿"和"能力"映射到一个基于虚拟价格的灵活功率响应曲线上。这个曲线是其实现自治控制、参与聚合调峰的基础。

输入 (Input):
需求曲线 dij(π) 的输入是 虚拟价格 (虚拟价格 π)
这个虚拟价格 π 是一个控制信号，在本文中其范围被限制在 [-1, 1] 之间
输出 (Output):
需求曲线 dij(π) 的输出是 动态功率 (动态功率 P)
动态功率 Pd,ij 定义为在 1 个控制周期内，使当前室温变化到指定温度所需的期望功率
"""

# 需求响应曲线函数
def calculate_dynamic_power_from_demand_curve(virtual_price_pi, p_min_ij, p_set_ij, p_max_ij):
    """
    根据需求曲线 dij(π) 计算动态功率。

    需求曲线由三个关键点确定：
    - 点 A：(虚拟价格 π = 1, 对应动态功率 Pmin,ij)
    - 点 B：(虚拟价格 π = 0, 对应动态功率 Pset,ij)
    - 点 C：(虚拟价格 π = -1, 对应动态功率 Pmax,ij)
    该曲线是分段线性的，并且具有单调递减的特性。
    为了保证曲线单调递减，输入参数应满足 p_max_ij >= p_set_ij >= p_min_ij。

    参数:
        virtual_price_pi (float): 虚拟价格信号 π，取值范围在 [-1, 1] 之间。
        p_min_ij (float): 虚拟价格 π = 1 时的动态功率。
        p_set_ij (float): 虚拟价格 π = 0 时的动态功率。
        p_max_ij (float): 虚拟价格 π = -1 时的动态功率。

    返回:
        float: 计算得到的动态功率。

    异常:
        ValueError: 如果 virtual_price_pi 不在 [-1, 1] 范围内，
                    或者如果 p_max_ij < p_set_ij 或 p_set_ij < p_min_ij (导致曲线非单调递减)。
    """
    # 先确保虚拟价格在有效范围内
    if not (-1 <= virtual_price_pi <= 1):
        # 对于超出范围的虚拟价格，我们将其截断到 [-1, 1] 范围内
        virtual_price_pi = max(-1, min(1, virtual_price_pi))
        # 记录警告信息
        print(f"警告：虚拟价格 {virtual_price_pi} 已被截断到 [-1, 1] 范围内。")
    
    # 检查功率参数是否满足单调递减特性
    # 注意：为了实际中的容错，我们允许一定的误差，例如数值计算产生的微小差异
    # 我们也允许三个功率值相等的情况（p_max_ij = p_set_ij = p_min_ij）
    tolerance = 1e-6  # 微小误差容忍度
    
    # 允许功率值相等
    if abs(p_max_ij - p_set_ij) <= tolerance and abs(p_set_ij - p_min_ij) <= tolerance:
        # 所有功率值接近相等，直接返回p_set_ij
        return p_set_ij
    
    # 否则检查 p_max_ij >= p_set_ij >= p_min_ij (考虑微小误差)
    if p_max_ij < p_set_ij - tolerance or p_set_ij < p_min_ij - tolerance:
        # 如果不满足单调递减特性，抛出异常
        raise ValueError(
            f"输入功率值未能形成单调递减的需求曲线 (p_max_ij={p_max_ij}, p_set_ij={p_set_ij}, p_min_ij={p_min_ij})。"
        )
    
    # 对于特定的虚拟价格 π 计算动态功率
    if abs(virtual_price_pi) < tolerance:  # 接近于 0
        # 在 π = 0 时，返回 p_set_ij
        return p_set_ij
    elif virtual_price_pi > 0:  # π > 0
        # 在 π > 0 时，线性插值计算 p_set_ij 和 p_min_ij 之间的功率
        # (对应于虚拟价格在0和1之间)
        return p_set_ij + (p_min_ij - p_set_ij) * virtual_price_pi
    else:  # π < 0
        # 在 π < 0 时，线性插值计算 p_set_ij 和 p_max_ij 之间的功率
        # (对应于虚拟价格在-1和0之间)
        # 注意virtual_price_pi是负的，所以需要取负
        return p_set_ij + (p_max_ij - p_set_ij) * (-virtual_price_pi)


if __name__ == '__main__':
    # 示例用法
    print("--- 需求响应曲线函数测试 ---")

    # 1. 定义符合要求的参数
    p_min_test = 100.0  # 最低功率 (对应 π=1)
    p_set_test = 500.0  # 设定功率 (对应 π=0)
    p_max_test = 1000.0 # 最高功率 (对应 π=-1)
    print(f"\n测试参数: Pmax={p_max_test}, Pset={p_set_test}, Pmin={p_min_test}")
    print("预期: P(-1)=1000, P(0)=500, P(1)=100, 中间值为线性插值")

    test_pis = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for pi in test_pis:
        power = calculate_dynamic_power_from_demand_curve(pi, p_min_test, p_set_test, p_max_test)
        print(f"  π = {pi:4.1f} -> 动态功率 = {power:6.1f} W")

    # 2. 测试虚拟价格超出范围
    print("\n测试虚拟价格超出范围 (预期 ValueError):")
    try:
        calculate_dynamic_power_from_demand_curve(1.5, p_min_test, p_set_test, p_max_test)
    except ValueError as e:
        print(f"  π = 1.5 -> 捕获到错误: {e}")
    try:
        calculate_dynamic_power_from_demand_curve(-1.5, p_min_test, p_set_test, p_max_test)
    except ValueError as e:
        print(f"  π = -1.5 -> 捕获到错误: {e}")

    # 3. 测试功率值不符合单调递减 (预期 ValueError)
    print("\n测试功率值不符合单调递减 (预期 ValueError):")
    try:
        # p_set < p_min (不满足 p_set_ij >= p_min_ij)
        print(f"  测试 Pmax={p_max_test}, Pset=100.0, Pmin=500.0 (Pset < Pmin)")
        calculate_dynamic_power_from_demand_curve(0.5, p_min_ij=500.0, p_set_ij=100.0, p_max_ij=p_max_test)
    except ValueError as e:
        print(f"    -> 捕获到错误: {e}")
    try:
        # p_max < p_set (不满足 p_max_ij >= p_set_ij)
        print(f"  测试 Pmax=500.0, Pset=1000.0, Pmin={p_min_test} (Pmax < Pset)")
        calculate_dynamic_power_from_demand_curve(-0.5, p_min_ij=p_min_test, p_set_ij=1000.0, p_max_ij=500.0)
    except ValueError as e:
        print(f"    -> 捕获到错误: {e}")
    
    # 4. 测试平坦曲线 (允许，例如 Pmax=Pset=Pmin)
    p_flat = 500.0
    print(f"\n测试平坦曲线: Pmax={p_flat}, Pset={p_flat}, Pmin={p_flat}")
    pi = 0.5
    power = calculate_dynamic_power_from_demand_curve(pi, p_flat, p_flat, p_flat)
    print(f"  π = {pi:4.1f} -> 动态功率 = {power:6.1f} W (预期: {p_flat})")

    print("\n--- 测试结束 ---")
