# 复现逻辑，（1）ETP模型；（2）实现FSM和半马尔可夫过程；（3）实现CPS模型；（4）实现最优控制

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号
from scipy.integrate import solve_ivp
from models import SecondOrderETPModel, ACL_State, ACL_FSM, ACL_CyberPhysicalModel
import os

# 示例使用
if __name__ == "__main__":
    """
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
