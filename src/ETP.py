import numpy as np
from scipy.integrate import solve_ivp
import random  # 添加random模块导入


class SecondOrderETPModel:
    """
    二阶等效热参数(ETP)模型
    参考论文：《聚合大规模空调负荷的信息物理建模与控制方法_王永权》
    """
    def __init__(self, Ca, Cm, Ua, Um, Hm, Qgain=0):
        """
        初始化二阶ETP模型参数
        
        参数:
        Ca: 空气热容量 [J/°C]
        Cm: 建筑质量热容量 [J/°C]
        Ua: 空气与室外的热传导系数 [W/°C]
        Um: 空气与建筑质量的热传导系数 [W/°C]
        Hm: 空调制冷/制热功率 [W]
        Qgain: 内部热增益 [W]
        """
        self.Ca = Ca
        self.Cm = Cm
        self.Ua = Ua
        self.Um = Um
        self.Hm = Hm
        self.Qgain = Qgain
        
    def dynamics(self, t, state, Tout, mode):
        """
        定义二阶ETP模型的动态方程
        
        参数:
        t: 时间
        state: 状态变量 [Ta, Tm]，Ta为室内空气温度，Tm为建筑质量温度
        Tout: 室外温度
        mode: 空调工作模式，1表示开启，0表示关闭
        
        返回:
        状态导数 [dTa/dt, dTm/dt]
        """
        Ta, Tm = state
        
        # 空调功率，根据模式决定是否开启
        Qhvac = self.Hm * mode
        
        # 室内空气温度变化率
        dTa_dt = (self.Ua * (Tout - Ta) + self.Um * (Tm - Ta) + Qhvac + self.Qgain) / self.Ca
        
        # 建筑质量温度变化率
        dTm_dt = (self.Um * (Ta - Tm)) / self.Cm
        
        return [dTa_dt, dTm_dt]
    
    def simulate(self, T0, Tm0, Tout_func, mode_func, t_span, t_eval=None):
        """
        模拟二阶ETP模型
        
        参数:
        T0: 初始室内空气温度 [°C]
        Tm0: 初始建筑质量温度 [°C]
        Tout_func: 室外温度函数，接受时间t作为输入
        mode_func: 空调模式函数，接受时间t和当前状态[Ta, Tm]作为输入
        t_span: 模拟时间范围 [t_start, t_end]
        t_eval: 评估时间点
        
        返回:
        模拟结果
        """
        def system(t, state):
            Tout = Tout_func(t)
            mode = mode_func(t, state)
            return self.dynamics(t, state, Tout, mode)
        
        initial_state = [T0, Tm0]
        solution = solve_ivp(system, t_span, initial_state, t_eval=t_eval, method='RK45')
        
        return solution


