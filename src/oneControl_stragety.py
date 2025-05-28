"""
1个负荷聚合商,假设下设1000台空调

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

# 导入依赖
import numpy as np
from scipy.optimize import minimize
import sys # Added
import os # Added

# 修正导入路径
# 将项目根目录（src的父目录）添加到sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.ETP import SecondOrderETPModel  # 如果从项目根目录运行
except ImportError:
    try:
        from ETP import SecondOrderETPModel  # 如果从src目录运行
    except ImportError:
        raise ImportError("无法导入SecondOrderETPModel，请确保ETP.py在正确路径")

try:
    from src.FSM import ACL_FSM, ACL_State  # 如果从项目根目录运行
except ImportError:
    try:
        from FSM import ACL_FSM, ACL_State  # 如果从src目录运行
    except ImportError:
        raise ImportError("无法导入ACL_FSM和ACL_State，请确保FSM.py在正确路径")

try:
    # 现在应该能直接从scripts导入
    from scripts.single_ACL_demand_curve import calculate_dynamic_power_from_demand_curve
except ImportError as e: # Capture the original error for better debugging
    raise ImportError(
        f"无法导入 calculate_dynamic_power_from_demand_curve。请确保 single_ACL_demand_curve.py 在 scripts 目录下。Original error: {e}"
    )

class SingleAirConditioner:
    def __init__(self, id, etp_model, fsm_model, Tset, Tmin, Tmax, Prate, COP, Ua, 
                 c_param, r1_param, r2_param, eta_param, Qm_ij_val=0): # Added new params
        """
        初始化单个空调对象

        参数:
        id (str): 空调标识符
        etp_model (SecondOrderETPModel): ETP模型实例
        fsm_model (ACL_FSM): FSM模型实例
        Tset (float): 设定温度 [°C]
        Tmin (float): 允许的最低温度 [°C]
        Tmax (float): 允许的最高温度 [°C]
        Prate (float): 额定功率 [W]
        COP (float): 空调能效比 [-]
        Ua (float): 等效热导系数 [W/°C]
        c_param (float): 公式A5-A10中的c参数，热容参数 [J/°C]
        r1_param (float): 公式A5-A10中的r1参数，与ETP系统特征值相关 [-]
        r2_param (float): 公式A5-A10中的r2参数，与ETP系统特征值相关，注意r1≠r2 [-]
        eta_param (float): 公式A5-A10中的η参数，系统时间常数参数 [-]
        Qm_ij_val (float, optional): 室内热增益 [W]，默认为0
        """
        self.id = id
        self.etp_model = etp_model  # ETP模型实例
        self.fsm_model = fsm_model  # FSM模型实例
        self.Tset_ij = Tset  # 设定温度
        self.Tmin_ij = Tmin  # 允许的最低温度
        self.Tmax_ij = Tmax  # 允许的最高温度
        self.Prate_ij = Prate  # 额定功率
        self.COP_ij = COP      # 空调能效比
        self.Ua_ij = Ua        # 等效热导系数，用于基准功率计算

        # Parameters for the new dynamic power formula (A5-A10)
        # 必须根据系统辨识或ETP模型分析提供这些参数
        self.c_param = c_param  # 热容参数
        self.r1_param = r1_param  # 系统特征值参数1
        self.r2_param = r2_param  # 系统特征值参数2 (不能等于r1)
        self.eta_param = eta_param  # 系统时间常数参数
        self.Qm_val_for_formula = Qm_ij_val  # 室内热增益，用于动态功率计算


        # 需求曲线的关键功率点 (electrical power)
        self.p_min_ij = 0
        self.p_set_ij = 0
        self.p_max_ij = 0

        # 当前状态
        self.current_Ta_ij = Tset # 初始室内温度假定为设定温度
        self.current_Tm_ij = Tset # 初始固体温度假定为设定温度
        
        # 基准功率
        self.p_base_ij = 0

        # 参数有效性检查
        if self.COP_ij <= 0:
            raise ValueError(f"空调 {id} 的COP必须大于0")
        if abs(self.r1_param - self.r2_param) < 1e-6:
            raise ValueError(f"空调 {id} 的r1和r2参数不能相等")
        if self.c_param <= 0:
            raise ValueError(f"空调 {id} 的c参数必须大于0")

    def calculate_baseline_power(self, To, Qm_ij=None):
        """
        计算基准功率 (Pbase,ij) - Electrical Power
        Pbase,ij = (Ua,ij * (Tset,ij - To) - Qm,ij) / COPij
        
        参数:
        To (float): 室外温度 [°C]
        Qm_ij (float, optional): 室内热增益 [W]，如果为None，则使用初始化时提供的值
        
        返回:
        float: 计算得到的基准功率 (电功率) [W]
        """
        # 使用传入的Qm_ij或默认值
        if Qm_ij is None:
            Qm_ij = self.Qm_val_for_formula
        
        # 注意：这里的 Ua,ij 是热导系数 [W/°C]
        # 使用传入的Ua_ij用于基准功率计算
        base_thermal_power = (self.Ua_ij * (self.Tset_ij - To) - Qm_ij)
        
        if self.COP_ij == 0:
            self.p_base_ij = 0
        else:
            # 在制冷场景下，当室外温度高于设定温度时，热功率为负，表示需要制冷
            # 此时应该取电功率的绝对值作为基准功率
            if base_thermal_power < 0:  # 制冷模式
                self.p_base_ij = abs(base_thermal_power) / self.COP_ij
            else:  # 制热模式或无需空调
                self.p_base_ij = 0
        
        return self.p_base_ij

    def _calculate_dynamic_power_pd_ij(self, target_Ta_k_plus_1, To_outdoor, time_step=2):
        """
        内部辅助函数，计算动态功率 Pd,ij(k)
        P_d(k) = A_1*T_a(k) + B_1 * T_a(k-1) + C_1 * T_m(k-1) + D_1(k)

        a = C_a

        b = H_m + UA + \frac{C_a H_m}{C_m}

        c = \frac{UA \cdot H_m}{C_m}

        d = \frac{H_m Q_m + H_m Q_a + UA \cdot H_m T_o}{C_m}

        r_1 = \frac{-b + \sqrt{b^2 - 4ac}}{2a}

        r_2 = \frac{-b - \sqrt{b^2 - 4ac}}{2a}

        \Delta = \text{COP}\left(\frac{1}{c} - e^{r_2 T} \left(\frac{1}{c} + \frac{\frac{1}{C_a}+\frac{r_2}{c}}{r_1 - r_2}\right) + e^{r_1 T} \left(\frac{\frac{1}{C_a}+\frac{r_2}{c}}{r_1 - r_2}\right)\right)

        A_1 = -\frac{1}{\Delta}

        B_1 = -\frac{1}{\Delta} \left\{ e^{r_2 T} \left[\frac{r_2}{r_1 - r_2} - \frac{H_m + UA}{C_a (r_1 - r_2)} - 1\right] - \frac{e^{r_1 T}}{r_1 - r_2} \left[-{r_2} - \frac{H_m + UA}{C_a}\right]\right\}
        
        C_1 = -\frac{1}{\Delta} \left[ e^{r_2 T} \frac{H_m}{C_a (r_1 - r_2)} - e^{r_1 T} \frac{H_m}{(r_1 - r_2) C_a}\right]

        D_1 = -\frac{1}{\Delta} \left\{- \frac{2Q_m + T_0 UA}{c} + e^{r_2 T} \left[\frac{Q_m}{C_a (r_1 - r_2)} + \frac{r_2 (2Q_m + T_0 UA)}{c(r_1 - r_2)} + \frac{T_0 UA}{C_a (r_1 - r_2)}\right] - \frac{e^{r_1 T}}{r_1 - r_2} \left[Q_m + \frac{r_2 (2Q_m + T_0 UA)}{c} + \frac{T_0 UA}{C_a}\right]\right\}
        
        Returns:
            float: 计算得到的动态功率 Pd,ij(k) [W]，制冷情况下为负值

        """
        # 获取ETP模型参数
        Ca = self.etp_model.Ca  # 空气热容 [J/°C]
        Cm = self.etp_model.Cm  # 建筑质量热容 [J/°C]
        UA = self.etp_model.Ua  # 空气与室外的热传导系数 [W/°C]
        Um = self.etp_model.Um  # 空气与建筑质量的热传导系数 [W/°C]
        Hm = self.etp_model.Hm  # 空调的额定热功率 [W]（负值代表制冷）
        Qm = self.Qm_val_for_formula  # 内部热增益 [W]
        COP = self.COP_ij  # 空调能效比
        
        # 当前状态温度
        Ta_k_minus_1 = self.current_Ta_ij  # 当前室内空气温度
        Tm_k_minus_1 = self.current_Tm_ij  # 当前室内固体温度
        T = time_step   # 控制周期时长
        
        # 参数有效性检查
        if abs(Ca) < 1e-6 or abs(Cm) < 1e-6:
            return 0  # 热容过小，无法进行计算
        
        # 计算中间参数 a, b, c, d
        a = Ca
        b = Hm + UA + (Ca * Hm) / Cm
        c = (UA * Hm) / Cm
        d = (Hm * Qm + 0 + UA * Hm * To_outdoor) / Cm  # 假设没有额外热源Qa=0
        
        # 防止除零错误
        if abs(c) < 1e-6:
            c = 1e-6  # 设置一个很小但非零的值
        if abs(a) < 1e-6:
            a = 1e-6
        
        # 计算二次方程特征根 r1, r2
        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            # 复数根情况，这种情况应该不会出现在物理上合理的系统中
            # 在制冷场景下，如果Ta_k_minus_1 > target_Ta_k_plus_1，则返回负值表示制冷
            if Ta_k_minus_1 > target_Ta_k_plus_1:
                return -abs(UA * (target_Ta_k_plus_1 - To_outdoor))  # 简化估计，确保为负值
            else:
                return 0  # 不需要制冷
            
        sqrt_discriminant = np.sqrt(discriminant)
        r1 = (-b + sqrt_discriminant) / (2 * a)
        r2 = (-b - sqrt_discriminant) / (2 * a)
        print(f"r1: {r1}, r2: {r2}")
        # 检查r1和r2是否相同（特殊情况）
        if abs(r1 - r2) < 1e-6:
            # 重根情况需要特殊处理
            r2 = r1 - 1e-5  # 微小扰动避免除零
        
        # 计算指数项
        exp_r1_T = np.exp(r1 * T)
        exp_r2_T = np.exp(r2 * T)
        print(f"exp_r1_T: {exp_r1_T}, exp_r2_T: {exp_r2_T}, r1 * T = {r1* T}")

        # 确保指数值不会溢出
        max_exp = 1e10
        exp_r1_T = min(exp_r1_T, max_exp)
        exp_r2_T = min(exp_r2_T, max_exp)
        
        # 计算Δ (Delta)
        term1 = 1.0 / c
        term2 = exp_r2_T * (1.0 / c + (1.0 / Ca + r2 / c) / (r1 - r2))
        term3 = exp_r1_T * ((1.0 / Ca + r2 / c) / (r1 - r2))
        Delta = COP * (term1 - term2 + term3)
        
        # 防止Delta为零或过小
        if abs(Delta) < 1e-6:
            # Delta接近零意味着无法求解或系统不稳定
            # 在制冷场景下，如果Ta_k_minus_1 > target_Ta_k_plus_1，则返回负值表示制冷
            if Ta_k_minus_1 > target_Ta_k_plus_1:
                return -abs(UA * (target_Ta_k_plus_1 - To_outdoor))  # 简化估计，确保为负值
            else:
                return 0  # 不需要制冷
        
        # 计算系数 A1
        A1 = -1.0 / Delta
        
        # 计算系数 B1
        B1_term1 = exp_r2_T * (r2 / (r1 - r2) - (Hm + UA) / (Ca * (r1 - r2)) - 1)
        B1_term2 = (exp_r1_T / (r1 - r2)) * (-r2 - (Hm + UA) / Ca)
        B1 = -1.0 / Delta * (B1_term1 - B1_term2)
        
        # 计算系数 C1
        C1_term1 = exp_r2_T * Hm / (Ca * (r1 - r2))
        C1_term2 = exp_r1_T * Hm / ((r1 - r2) * Ca)
        C1 = -1.0 / Delta * (C1_term1 - C1_term2)
        
        # 如果C1很小，可以视为0
        if abs(C1) < 1e-9:
            C1 = 0
        
        # 计算系数 D1
        common_term = 2 * Qm + To_outdoor * UA
        D1_term1 = -common_term / c
        D1_term2 = exp_r2_T * (
            Qm / (Ca * (r1 - r2)) + 
            r2 * common_term / (c * (r1 - r2)) + 
            To_outdoor * UA / (Ca * (r1 - r2))
        )
        D1_term3 = (exp_r1_T / (r1 - r2)) * (
            Qm / Ca + 
            r2 * common_term / c + 
            To_outdoor * UA / Ca
        )
        D1 = -1.0 / Delta * (D1_term1 + D1_term2 - D1_term3)
        
        # 计算最终的动态热功率 Pd,ij(k)
        Pd_k_thermal = A1 * target_Ta_k_plus_1 + B1 * Ta_k_minus_1 + C1 * Tm_k_minus_1 + D1
        
        # 对于制冷场景，确保热功率为负值
        # 在现实中，如果当前温度高于目标温度，表示需要制冷
        if Ta_k_minus_1 > target_Ta_k_plus_1:
            # 如果计算结果为正，则取反，确保制冷功率为负值
            if Pd_k_thermal > 0:
                Pd_k_thermal = -abs(Pd_k_thermal)
        else:
            # 当前温度已经低于目标温度，不需要制冷，功率应为0
            Pd_k_thermal = 0
        
        return Pd_k_thermal

    def form_demand_curve(self, control_period_seconds, To_outdoor):
        """
        形成需求曲线 (dij(π)) - Electrical Power
        关键功率 (Pmin,ij, Pset,ij, Pmax,ij) 的计算基于动态功率 (Pd,ij)
        Pd,ij是热功率，需要转换为电功率用于需求曲线。
        
        参数:
        control_period_seconds (float): 控制周期时长 [s]
        To_outdoor (float): 室外温度 [°C]
        
        返回:
        dict: 包含需求曲线关键点的字典 {p_min, p_set, p_max}
        """
        # 1. 计算基准功率 (Electrical Power) 作为参考
        self.calculate_baseline_power(To_outdoor, Qm_ij=self.Qm_val_for_formula)

        # 2. 计算对应 Tmin,ij, Tset,ij, Tmax,ij 的动态功率 (Thermal)
        #    Then convert to electrical power.
        try:
            # Pmax0_ij (electrical): corresponds to achieving Tmin_ij (max cooling)
            pd_thermal_for_t_min = self._calculate_dynamic_power_pd_ij(self.Tmin_ij, To_outdoor, control_period_seconds)
            # Pmin0_ij (electrical): corresponds to achieving Tmax_ij (min cooling)
            pd_thermal_for_t_max = self._calculate_dynamic_power_pd_ij(self.Tmax_ij, To_outdoor, control_period_seconds)
            # Pset0_ij (electrical): corresponds to achieving Tset_ij (setpoint cooling)
            pd_thermal_for_t_set = self._calculate_dynamic_power_pd_ij(self.Tset_ij, To_outdoor, control_period_seconds)
        except Exception as e:
            raise ValueError(f"空调 {self.id} 计算动态功率时出错: {str(e)}")

        if self.COP_ij == 0:
            # print(f"Warning: COP is 0 for AC {self.id}. Cannot convert thermal dynamic power to electrical.")
            p_max0_ij_elec = 0
            p_min0_ij_elec = 0
            p_set0_ij_elec = 0
        else:
            # 热功率转换为电功率，确保功率值合理
            # 在制冷模式下，热功率应该为负值，电功率为正值
            p_max0_ij_elec = abs(pd_thermal_for_t_min) / self.COP_ij  # 使用绝对值确保电功率为正
            p_min0_ij_elec = abs(pd_thermal_for_t_max) / self.COP_ij
            p_set0_ij_elec = abs(pd_thermal_for_t_set) / self.COP_ij

        # 3. 对动态功率 (Electrical) 进行约束
        # 确保有最小功率值，防止所有功率都为0
        MIN_POWER_VALUE = 100.0  # 最小功率值，确保空调运行时有一定功率消耗
        
        # 先根据物理含义约束：p_max >= p_set >= p_min
        raw_p_max = min(p_max0_ij_elec, self.Prate_ij)
        raw_p_min = max(p_min0_ij_elec, 0)
        raw_p_set = np.clip(p_set0_ij_elec, raw_p_min, raw_p_max)
        
        # 当制冷需求明确时，确保p_max有一个最小值
        if To_outdoor > self.Tset_ij + 1.0:  # 明显需要制冷
            self.p_max_ij = max(raw_p_max, MIN_POWER_VALUE)  # 确保最大功率不会太小
            self.p_set_ij = max(raw_p_set, MIN_POWER_VALUE * 0.5)  # 确保设定功率不会太小
            self.p_min_ij = raw_p_min  # 最小功率可以为0
            
            # 确保满足单调性：p_max_ij >= p_set_ij >= p_min_ij
            if self.p_set_ij > self.p_max_ij:
                self.p_set_ij = self.p_max_ij
            if self.p_min_ij > self.p_set_ij:
                self.p_min_ij = self.p_set_ij
        else:
            # 如果不需要明显制冷，使用通常的约束
            self.p_max_ij = raw_p_max
            self.p_set_ij = raw_p_set
            self.p_min_ij = raw_p_min
            
            # 确保满足单调性
            if self.p_set_ij > self.p_max_ij:
                self.p_set_ij = self.p_max_ij
            if self.p_min_ij > self.p_set_ij:
                self.p_min_ij = self.p_set_ij
        
        # 返回需求曲线的关键点 (或者一个可以被查询的函数/对象)
        return {"p_min": self.p_min_ij, "p_set": self.p_set_ij, "p_max": self.p_max_ij}

    def get_response_power(self, virtual_price_pi_star):
        """
        确定个体响应功率 (P_a_ij)* = dij(πi*)
        使用 single_ACL_demand_curve.py 中的函数
        """
        # from single_ACL_demand_curve import calculate_dynamic_power_from_demand_curve
        # 确保 p_max_ij, p_set_ij, p_min_ij 是最新的
        # 假设 form_demand_curve 已经被调用
        
        # 检查单调性，calculate_dynamic_power_from_demand_curve内部会检查
        # if not (self.p_max_ij >= self.p_set_ij >= self.p_min_ij):
        #     raise ValueError("Demand curve powers are not monotonic for response power calculation.")

        # 临时的 calculate_dynamic_power_from_demand_curve 实现，后续替换为导入
        # def temp_calculate_dynamic_power(pi, p_min, p_set, p_max):
        #     if not (-1 <= pi <= 1):
        #         raise ValueError("虚拟价格 (π) 必须在 [-1, 1] 之间。")
        #     if not (p_max >= p_set >= p_min):
        #          #允许等于的情况
        #         if not (abs(p_max - p_set) < 1e-6 and abs(p_set - p_min) < 1e-6 and abs(p_max-p_min) < 1e-6 ):
        #              # 如果不是全等，且不满足条件，则报错
        #             if not (p_max >= p_set and p_set >= p_min) :
        #                 raise ValueError(
        #                     f"输入功率值未能形成单调递减的需求曲线 (p_max={p_max}, p_set={p_set}, p_min={p_min})。"
        #                 )
        #     
        #     if pi == 0: return p_set
        #     elif pi > 0: # 在 (0, p_set) 和 (1, p_min) 之间插值
        #         return p_set + (p_min - p_set) * pi
        #     else: # pi < 0, 在 (-1, p_max) 和 (0, p_set) 之间插值
        #         return p_set + (p_max - p_set) * (-pi) # pi是负的，所以用-pi


        response_power = calculate_dynamic_power_from_demand_curve(
            virtual_price_pi_star, self.p_min_ij, self.p_set_ij, self.p_max_ij
        )
        return response_power

    def update_fsm_migration_probabilities(self, p_a_ij_star, control_cycle_duration_sec):
        """
        计算并更新FSM的迁移概率 u0 和 u1。
        FSM内部使用其初始化时传入的 sim_dt_sec 来计算这些概率。
        control_cycle_duration_sec 在此方法中不再直接用于设置FSM的内部参数。
        """
        # FSM的 sim_dt_sec 已在初始化时设置。
        # self.fsm_model.control_cycle_sec = control_cycle_duration_sec # 此行不再需要
        
        # 使用 FSM 实例的方法来更新迁移概率
        self.fsm_model.update_migration_probabilities(
            Pa_ij_star=p_a_ij_star, 
            Prate_ij=self.Prate_ij
        )
        
        # 返回更新后的迁移概率 (这些是针对 sim_dt_sec 步长的概率)
        return self.fsm_model.u0, self.fsm_model.u1

    def run_local_control_step(self, simulation_step_dt_sec, To_outdoor):
        """
        空调负荷按照FSM模型进行自主运行一步
        并更新当前的室内温度 Ta_ij 和固体温度 Tm_ij
        
        参数:
        simulation_step_dt_sec (float): 仿真步长 [s]
        To_outdoor (float): 当前室外温度 [°C]
        
        返回:
        tuple: (当前室内温度, 当前固体温度, 工作模式)
        """
        # 1. FSM 状态转移
        self.fsm_model.step(simulation_step_dt_sec)
        
        # 2. 获取物理状态 (开/关)
        mode = self.fsm_model.get_physical_state() # 1 for ON, 0 for OFF
        
        # 3. 更新ETP模型状态 (室内温度和固体温度)
        # 使用传入的室外温度，不再使用硬编码值
        
        # 使用 ETP 模型的 dynamics 方法计算温度变化率
        dTa_dt, dTm_dt = self.etp_model.dynamics(
            t=0, # 时间参数对于瞬时导数不重要
            state=[self.current_Ta_ij, self.current_Tm_ij],
            Tout=To_outdoor,
            mode=mode
        )
        
        self.current_Ta_ij += dTa_dt * simulation_step_dt_sec
        self.current_Tm_ij += dTm_dt * simulation_step_dt_sec
        
        return self.current_Ta_ij, self.current_Tm_ij, mode

    def get_current_SOA(self):
        """
        计算当前空调的舒适度指标 SOAij (针对削减负荷场景)
        SOAij = max(Ta,ij - Tset,ij, 0) / (Tmax,ij - Tset,ij)
        """
        if (self.Tmax_ij - self.Tset_ij) <= 0: # 防止除以零或负数
            return 0 # 如果温度范围无效，则舒适度无法评估或为0
        
        soa = max(self.current_Ta_ij - self.Tset_ij, 0) / (self.Tmax_ij - self.Tset_ij)
        return min(soa, 1.0) # SOA应该在0到1之间

    def update_indoor_temp_simplified(self, delta_t_seconds, T_o_current):
        """
        根据简化的单节点模型更新室内空气温度 T_i (self.current_Ta_ij)。
        公式: T_i(t + Δt) = T_i(t) + (Δt / C) * [Q_AC(t) + Q_ext(t) - (T_i(t) - T_o(t)) / R]
        空调仅能制冷。

        参数:
        delta_t_seconds (float): 时间步长 [s]
        T_o_current (float): 当前室外温度 [°C]
        """
        T_i_current = self.current_Ta_ij
        
        # 从ETP模型获取参数
        C = self.etp_model.Ca  # 空气热容 [J/°C]
        Ua = self.etp_model.Ua # 空气与室外的热传导系数 [W/°C]
        Hm = self.etp_model.Hm # 空调的额定制冷功率 [W] (通常为负值)
        Q_gain_internal = self.etp_model.Qgain # 内部热增益 [W]

        if C <= 1e-6: # 防止除以零
            # print(f"Warning: Air capacitance C for AC {self.id} is zero or too small. Temperature not updated.")
            return self.current_Ta_ij
        
        R_thermal_resistance = 0
        if abs(Ua) > 1e-6:
            R_thermal_resistance = 1.0 / Ua # 热阻 [°C/W]
        else:
            # 如果Ua为0，热阻视为无穷大，(T_i - T_o)/R 项为0，除非特殊处理
            # print(f"Warning: Ua for AC {self.id} is zero. Assuming no heat transfer through envelope via this term if R is not used carefully.")
            pass # (T_i - T_o)/R 会变成0如果R_thermal_resistance = 0 and T_i!=T_o

        # 获取空调运行状态 (1 for ON, 0 for OFF)
        mode = self.fsm_model.get_physical_state()
        
        q_ac = 0.0
        if mode == 1:
            q_ac = Hm  # Hm是负的代表制冷
        
        # 外部热源功率 Q_ext(t) 使用内部热增益
        q_ext = Q_gain_internal
        
        # 计算括号内的总热流率 [W]
        heat_flow_sum_W = q_ac + q_ext
        if R_thermal_resistance != 0: # 仅当热阻有效时才计算此项
             heat_flow_sum_W -= (T_i_current - T_o_current) / R_thermal_resistance
        elif abs(T_i_current - T_o_current) > 1e-6 and abs(Ua) < 1e-6 : # Ua=0 and temp diff exists
             # This case means infinite resistance, so (T_i - T_o)/R is 0. The subtraction is correctly skipped if R_thermal_resistance is 0.
             pass 

        # 计算温度变化率 dT_i/dt [°C/s]
        dT_i_dt = heat_flow_sum_W / C
        
        # 更新温度
        self.current_Ta_ij += dT_i_dt * delta_t_seconds
        
        return self.current_Ta_ij


class Aggregator:
    def __init__(self, id, air_conditioners_list):
        self.id = id
        self.acs = air_conditioners_list # 该聚合商下的空调列表
        self.num_acs = len(air_conditioners_list)
        self.aggregated_demand_curve_points = {} # Di(π) e.g. {pi_value: aggregated_power}
        self.peak_shaving_capacity = 0 # ΔPc_a_i
        self.current_aggregated_base_power = 0 # Pbase,i

    def aggregate_demand_curves(self, pi_values_to_sample=np.linspace(-1, 1, 21)):
        """
        聚合集群内空调负荷的需求曲线为总需求曲线 Di(π) = Σ dij(π)
        pi_values_to_sample: 用于采样和构建聚合需求曲线的虚拟价格点
        """
        self.aggregated_demand_curve_points = {}
        for pi in pi_values_to_sample:
            total_power_at_pi = 0
            for ac in self.acs:
                # 确保每个AC的需求曲线已更新 (p_min, p_set, p_max是最新的)
                # 实际上，AC的form_demand_curve应在每个控制周期初被调用
                # 这里假设AC的 p_min_ij, p_set_ij, p_max_ij 是最新的
                try:
                    total_power_at_pi += ac.get_response_power(pi)
                except ValueError as e:
                    # print(f"Error getting response power for AC {ac.id} at pi={pi}: {e}")
                    # 如果某个AC无法提供有效功率（例如其需求曲线参数无效），可以选择跳过或赋默认值
                    pass # 或者记录错误，或者如果单个AC曲线无效则聚合曲线也无效
            self.aggregated_demand_curve_points[pi] = total_power_at_pi
        return self.aggregated_demand_curve_points

    def get_aggregated_power_at_pi(self, pi_target):
        """
        从聚合需求曲线 Di(π) 获取在特定虚拟价格 π_target 下的总功率。
        如果 pi_target 不在采样点中，则进行线性插值。
        """
        if not self.aggregated_demand_curve_points:
            # print("Warning: Aggregated demand curve is empty. Call aggregate_demand_curves first.")
            # 尝试即时聚合一个点，但这通常不是期望行为
            # For now, return 0 or raise error
            return 0

        sorted_pis = sorted(self.aggregated_demand_curve_points.keys())
        
        if pi_target in self.aggregated_demand_curve_points:
            return self.aggregated_demand_curve_points[pi_target]
        
        # 线性插值
        if pi_target < sorted_pis[0]: # 小于最小采样pi，用最小pi的值
            return self.aggregated_demand_curve_points[sorted_pis[0]]
        if pi_target > sorted_pis[-1]: # 大于最大采样pi，用最大pi的值
            return self.aggregated_demand_curve_points[sorted_pis[-1]]

        # 找到pi_target所在的区间
        for i in range(len(sorted_pis) - 1):
            p_i, p_i1 = sorted_pis[i], sorted_pis[i+1]
            if p_i <= pi_target <= p_i1:
                # 插值: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                val_i = self.aggregated_demand_curve_points[p_i]
                val_i1 = self.aggregated_demand_curve_points[p_i1]
                if abs(p_i1 - p_i) < 1e-9: # 避免除以零
                    return val_i
                interpolated_power = val_i + (pi_target - p_i) * (val_i1 - val_i) / (p_i1 - p_i)
                return interpolated_power
        
        # Should not be reached if logic is correct
        return self.aggregated_demand_curve_points[sorted_pis[-1]] # Fallback

    def calculate_peak_shaving_capacity(self):
        """
        形成调峰容量 (ΔPc_a,i) = Di(0) - Di(1)
        Di(0): 集群 i 在虚拟价格为 0 时的总响应功率
        Di(1): 集群 i 在虚拟价格为 +1 时的总响应功率 (最小响应功率)
        """
        # 确保聚合需求曲线已计算
        if not self.aggregated_demand_curve_points:
            self.aggregate_demand_curves() # 如果为空则计算
            if not self.aggregated_demand_curve_points: # 再次检查
                 # print("Error: Cannot calculate peak shaving capacity, aggregated_demand_curve is still empty.")
                 self.peak_shaving_capacity = 0
                 return self.peak_shaving_capacity


        di_0 = self.get_aggregated_power_at_pi(0)
        di_1 = self.get_aggregated_power_at_pi(1)
        
        self.peak_shaving_capacity = di_0 - di_1
        # 调峰容量应为非负
        self.peak_shaving_capacity = max(0, self.peak_shaving_capacity)
        return self.peak_shaving_capacity

    def calculate_average_SOA(self):
        """
        计算平均舒适度 SOAavg,i
        SOAavg,i = (1 / Ni) * Σ(j=1 to Ni) SOAij
        """
        if self.num_acs == 0:
            return 0
        
        total_soa = 0
        for ac in self.acs:
            total_soa += ac.get_current_SOA()
        
        return total_soa / self.num_acs

    def form_bidding_strategy(self, delta_Pa_i, lambda_r, coeff_a, coeff_b):
        """
        形成报价策略 λi(ΔPa,i) = λr + ai * ΔPratio,i + bi * SOAavg,i
        ΔPratio,i = ΔPa,i / ΔPc_a,i
        """
        if self.peak_shaving_capacity == 0: # 避免除以零
            delta_Pratio_i = 0
        else:
            delta_Pratio_i = delta_Pa_i / self.peak_shaving_capacity
            # ΔPratio,i 应该在 [0, 1] 之间，如果 ΔPa,i > ΔPc_a,i，则比例大于1
            # 一般 ΔPa,i 不会超过 ΔPc_a,i
            delta_Pratio_i = np.clip(delta_Pratio_i, 0, 1) # 确保比例合理

        soa_avg_i = self.calculate_average_SOA()
        
        bid_price = lambda_r + coeff_a * delta_Pratio_i + coeff_b * soa_avg_i
        return bid_price

    def calculate_total_baseline_power(self, To_outdoor):
        """ 计算集群的总基准功率 Pbase,i = Σ Pbase,ij """
        self.current_aggregated_base_power = 0
        for ac in self.acs:
            self.current_aggregated_base_power += ac.calculate_baseline_power(To_outdoor)
        return self.current_aggregated_base_power


    def determine_virtual_price_from_target_power(self, target_aggregated_power_Pa_i_star):
        """
        确定虚拟价格信号 (πi)* = Di⁻¹(P*a,i)
        通过聚合需求曲线的反函数求得。这需要解方程 Di(π) = P*a,i
        这里我们使用数值方法，在已采样的聚合需求曲线上查找或插值。
        """
        if not self.aggregated_demand_curve_points:
            # print("Error: Aggregated demand curve is empty. Cannot determine virtual price.")
            return 0 # 返回默认价格，或抛出异常

        sorted_pis = sorted(self.aggregated_demand_curve_points.keys())
        powers = [self.aggregated_demand_curve_points[pi] for pi in sorted_pis]

        # 由于需求曲线是单调递减的，我们可以使用np.interp
        # np.interp(x_new, x_known, y_known)
        # 我们想找 pi (x_new) 使得 Di(pi) (y_known) = target_aggregated_power_Pa_i_star (x_known 的反函数)
        # 所以，我们要用 power 作为 x_known，pi 作为 y_known
        # 注意：np.interp 要求 x_known 是单调递增的。我们的 power 是单调递减的。
        # 所以，需要反转 powers 和 sorted_pis
        
        reversed_powers = powers[::-1]
        reversed_pis = sorted_pis[::-1]

        # 处理边界情况
        if target_aggregated_power_Pa_i_star >= reversed_powers[0]: # 如果目标功率比 Di(-1) 还大 (理论上不可能，除非曲线平坦)
            return -1 # 直接返回 pi = -1，确保在范围内
        if target_aggregated_power_Pa_i_star <= reversed_powers[-1]: # 如果目标功率比 Di(1) 还小
            return 1 # 直接返回 pi = 1，确保在范围内
            
        try:
            # 确保 reversed_powers 是严格单调递增的，对于插值
            # 如果有重复值，插值可能不唯一或行为不确定
            # 清理一下，确保单调性，同时保留对应关系
            unique_powers = []
            corresponding_pis = []
            last_power = None
            for i in range(len(reversed_powers)):
                if last_power is None or reversed_powers[i] > last_power: #严格递增
                    unique_powers.append(reversed_powers[i])
                    corresponding_pis.append(reversed_pis[i])
                    last_power = reversed_powers[i]
            
            if not unique_powers or len(unique_powers) < 2: # 如果点太少无法插值
                # print("Warning: Not enough unique points in aggregated demand curve for interpolation.")
                # 返回最近似的值，确保在[-1, 1]范围内
                if unique_powers:
                    idx = np.argmin(np.abs(np.array(unique_powers) - target_aggregated_power_Pa_i_star))
                    return np.clip(corresponding_pis[idx], -1, 1)
                else: # 如果连一个点都没有（例如所有空调都关闭且功率为0）
                    return 0 # 默认价格

            # 使用np.interp进行插值，并确保结果在[-1, 1]范围内
            virtual_price_star = np.interp(target_aggregated_power_Pa_i_star, unique_powers, corresponding_pis)
            virtual_price_star = np.clip(virtual_price_star, -1, 1) # 确保价格在[-1, 1]范围内
        except Exception as e:
            # print(f"Error during interpolation for virtual price: {e}")
            # 发生错误时，可以找最近的点，并确保结果在[-1, 1]范围内
            if len(powers) > 0:
                idx = np.argmin(np.abs(np.array(powers) - target_aggregated_power_Pa_i_star))
                virtual_price_star = np.clip(sorted_pis[idx], -1, 1)
            else:
                virtual_price_star = 0  # 如果无法计算，返回默认值0

        # 最后再次确保价格在[-1, 1]范围内
        return np.clip(virtual_price_star, -1, 1)

    def dispatch_virtual_price_to_acs(self, virtual_price_star, control_cycle_duration_sec):
        """
        将虚拟价格广播给集群内的空调负荷，并让它们更新状态。
        空调会：
        1. 根据虚拟价格确定自己的响应功率 P_a_ij_star
        2. 更新FSM的迁移概率 u0, u1
        """
        individual_response_powers = {}
        for ac in self.acs:
            p_a_ij_star = ac.get_response_power(virtual_price_star)
            individual_response_powers[ac.id] = p_a_ij_star
            ac.update_fsm_migration_probabilities(p_a_ij_star, control_cycle_duration_sec)
        return individual_response_powers


class Grid:
    def __init__(self, aggregators_list):
        self.aggregators = aggregators_list # 系统中的聚合商列表
        self.num_aggregators = len(aggregators_list)

    def solve_optimal_dispatch(self, P_target_total): # Removed lambda_r_base, coeff_a, coeff_b
        """
        简化版最优调度: 仅根据聚合商的调峰容量按比例分配总目标调峰功率。
        s.t. Σ ΔPa,i = P_target_total (如果总容量足够)
             0 <= ΔPa,i <= ΔPc_a,i (聚合商i的调峰容量)
        
        返回: 各聚合商应承担的最优调峰功率 ΔPa_i_star 字典
        """
        if not self.aggregators:
            return {}

        delta_pc_a_i_list = []
        aggregator_ids = []

        for agg in self.aggregators:
            # 假定 agg.peak_shaving_capacity 是最新的
            delta_pc_a_i_list.append(agg.peak_shaving_capacity)
            aggregator_ids.append(agg.id)
        
        delta_pc_a_i_arr = np.array(delta_pc_a_i_list)
        
        # 初始化分配结果
        optimal_delta_Pa_i_star_values = np.zeros(self.num_aggregators)

        total_available_capacity = np.sum(delta_pc_a_i_arr)

        if total_available_capacity <= 1e-6: # 总可用容量几乎为零
            # print("Warning: Total peak shaving capacity is zero or negligible. No dispatch possible.")
            # 所有聚合商分配为0，已在初始化时完成
            pass
        else:
            # 如果目标调峰量超过总可用容量，则让所有聚合商出全力
            actual_dispatch_total = min(P_target_total, total_available_capacity)
            
            # 按比例分配
            for i in range(self.num_aggregators):
                if delta_pc_a_i_arr[i] > 0: # 只对有容量的聚合商进行分配
                    proportion = delta_pc_a_i_arr[i] / total_available_capacity
                    allocated_power = proportion * actual_dispatch_total
                    # 确保分配不超过其自身容量且非负 (尽管比例分配理论上满足)
                    optimal_delta_Pa_i_star_values[i] = np.clip(allocated_power, 0, delta_pc_a_i_arr[i])
                else:
                    optimal_delta_Pa_i_star_values[i] = 0
            
            # 由于clip和浮点数精度，可能需要重新检查总和是否等于 actual_dispatch_total
            # 在这种简化模型中，如果 P_target_total > total_available_capacity, 那么 sum(optimal_delta_Pa_i_star_values) == total_available_capacity
            # 如果 P_target_total <= total_available_capacity, 那么 sum(optimal_delta_Pa_i_star_values) 应该接近 P_target_total
            # 为确保严格相等，可以进行一次微调，但这会增加复杂性，对于此简化模型可能非必需。
            # 例如，可以将差额按比例分配给仍有余量的聚合商。
            # 暂时接受当前按比例分配的结果。


        # 整理成字典返回
        dispatch_results = {}
        for i, agg_id in enumerate(aggregator_ids):
            dispatch_results[agg_id] = optimal_delta_Pa_i_star_values[i]
            
        return dispatch_results

    def dispatch_power_commands_to_aggregators(self, optimal_dispatch_results, To_outdoor, control_cycle_duration_sec):
        """
        将最优调峰功率指令 ΔPa,i* 下发给相应的聚合商。
        聚合商会：
        1. 计算集群的总目标响应功率 P*a,i = Pbase,i - ΔPa,i*
        2. 确定虚拟价格信号 πi* = Di⁻¹(P*a,i)
        3. 将虚拟价格广播给其下的空调
        """
        aggregator_virtual_prices = {}
        for agg in self.aggregators:
            if agg.id in optimal_dispatch_results:
                delta_Pa_i_star = optimal_dispatch_results[agg.id]
                
                # 1. 聚合商计算其总基准功率 (如果尚未更新)
                # agg.calculate_total_baseline_power(To_outdoor) # 假设已在之前步骤更新
                p_base_i = agg.current_aggregated_base_power # 使用当前聚合商的基准功率
                
                # 2. 计算集群的总目标响应功率
                p_a_i_star_target = p_base_i - delta_Pa_i_star
                
                # 3. 确定虚拟价格信号
                # 确保聚合商的聚合需求曲线是最新的
                # agg.aggregate_demand_curves() # 假设已在之前步骤更新
                virtual_price_i_star = agg.determine_virtual_price_from_target_power(p_a_i_star_target)
                aggregator_virtual_prices[agg.id] = virtual_price_i_star
                
                # 4. 聚合商将虚拟价格广播给其下的空调
                agg.dispatch_virtual_price_to_acs(virtual_price_i_star, control_cycle_duration_sec)
            else:
                # print(f"Warning: Aggregator {agg.id} not found in optimal dispatch results.")
                # 可以选择不操作，或者让它以pi=0运行
                agg.dispatch_virtual_price_to_acs(0, control_cycle_duration_sec)

        return aggregator_virtual_prices

if __name__ == "__main__":

    # --- 新增测试案例：特定条件下的动态功率 ---
    print("\n--- 测试案例：特定条件下的动态功率计算 ---")

    # 1. 定义 ETP 模型参数
    etp_params_test = {
        "Ca": 10000,  # 空气热容 [J/°C]
        "Cm": 100000, # 建筑质量热容 [J/°C]
        "Ua": 20,     # 空气与室外的热传导系数 [W/°C]
        "Um": 150,    # 空气与建筑质量的热传导系数 [W/°C]
        "Hm": -6000,  # 空调的额定制冷功率 [W] (负值代表制冷)
        "Qgain": 100  # 内部热增益 [W]
    }
    etp_model_test = SecondOrderETPModel(
        Ca=etp_params_test["Ca"],
        Cm=etp_params_test["Cm"],
        Ua=etp_params_test["Ua"],
        Um=etp_params_test["Um"],
        Hm=etp_params_test["Hm"],
        Qgain=etp_params_test["Qgain"]
    )

    # 2. 定义 FSM 模型参数
    Tset_test = 24.0  # 设定温度 [°C]
    fsm_model_test = ACL_FSM(
        ACL_State.OFF,  # 初始状态
        tonlock_sec=180,  # 最小开启时间 [s]
        tofflock_sec=180, # 最小关闭时间 [s]
        sim_dt_sec=2 * 60     # FSM内部仿真步长 [s]
    )


    # 3. 创建空调实例
    ac_test = SingleAirConditioner(
        id="AC_DynamicPowerTest",
        etp_model=etp_model_test,
        fsm_model=fsm_model_test,
        Tset=Tset_test,
        Tmin=Tset_test - 2.0, # 允许最低温度
        Tmax=Tset_test + 5.0, # 允许最高温度 (比当前室内温度39要低，确保测试有意义)
        Prate=2000,          # 额定功率 [W]
        COP=3.0,             # 能效比
        Ua=150,              # 等效热导系数 [W/°C] - 用于基准功率，非此测试核心
        # 下面是A5-A10公式参数，对于_calculate_dynamic_power_pd_ij不直接使用，但构造函数需要
        c_param=1.0,
        r1_param=0.1,
        r2_param=0.2, # r1 != r2
        eta_param=1.0,
        Qm_ij_val=etp_params_test["Qgain"] # 室内热增益，用于动态功率计算
    )

    # 4. 设置当前空调状态
    current_indoor_temp_test = 29.0  # 当前室内温度 [°C]
    ac_test.current_Ta_ij = current_indoor_temp_test
    ac_test.current_Tm_ij = current_indoor_temp_test # 假设固体温度与空气温度一致

    # 5. 定义测试参数
    target_temp_for_calc = Tset_test - 1.0  # 目标是将温度降至设定温度
    outdoor_temp_test = 30.0        # 室外温度 [°C]
    control_period_test_sec = 180    # 控制周期 [s] (e.g., 5 minutes)

    print(f"测试条件:")
    print(f"  空调ID: {ac_test.id}")
    print(f"  当前室内空气温度 (Ta_k-1): {ac_test.current_Ta_ij}°C")
    print(f"  当前室内固体温度 (Tm_k-1): {ac_test.current_Tm_ij}°C")
    print(f"  目标室内空气温度 (Ta_k+1): {target_temp_for_calc}°C")
    print(f"  室外温度 (To): {outdoor_temp_test}°C")
    print(f"  控制周期: {control_period_test_sec}s")
    print(f"  ETP模型 Hm: {ac_test.etp_model.Hm}W, Ua: {ac_test.etp_model.Ua} W/°C, Qm: {ac_test.Qm_val_for_formula}W, COP: {ac_test.COP_ij}")


    # 6. 调用动态功率计算方法
    try:
        dynamic_thermal_power = ac_test._calculate_dynamic_power_pd_ij(
            target_Ta_k_plus_1=target_temp_for_calc,
            To_outdoor=outdoor_temp_test,
            time_step=2
        )
        print(f"计算得到的动态热功率 Pd_ij(k): {dynamic_thermal_power:.2f} W")
        if dynamic_thermal_power < 0:
            print("  (负值表示制冷需求)")
        elif dynamic_thermal_power == 0:
            print("  (零表示不需要制冷/制热或已达目标)")
        else:
            print("  (正值表示制热需求 - 但此场景预期为制冷)")
            
    except ValueError as e:
        print(f"计算动态功率时发生错误: {e}")
    except Exception as e:
        print(f"计算动态功率时发生未知错误: {e}")
    # -----------------------------------------

