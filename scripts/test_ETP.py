import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ETP import SecondOrderETPModel
import time


def main():
    # 设置ETP模型参数（这些参数可以根据实际情况调整）
    Ca = 2.5e6       # 空气热容量 [J/°C]
    Cm = 3.25e7      # 建筑质量热容量 [J/°C]
    Ua = 937.5       # 空气与室外的热传导系数 [W/°C]
    Um = 9862.5      # 空气与建筑质量的热传导系数 [W/°C]
    Hm = -9000       # 空调制冷功率 [W]，负值表示制冷
    Qgain = 1000     # 内部热增益 [W]
    
    # 初始化ETP模型
    model = SecondOrderETPModel(Ca, Cm, Ua, Um, Hm, Qgain)
    
    # 设置初始条件
    T0 = 30.0        # 初始室内温度 [°C]
    Tm0 = 29.0       # 初始墙体温度 [°C]
    # 创建24小时室外温度变化数组
    Tout = np.concatenate([
        np.full(12*3600, 35),          # 0-12小时保持35度
        np.linspace(35, 30, 3*3600),   # 12-15小时从35度线性降到30度
        np.linspace(30, 28, 5*3600),   # 15-20小时从30度线性降到28度
        np.linspace(28, 35, 4*3600),   # 20-24小时从28度线性升到35度
    ])
    T_set = 24.0     # 设定温度 [°C]
    
    # 模拟时间设置
    t_span = [0, 24*3600]  # 模拟24小时
    t_eval = np.linspace(0, 24*3600, 1440)  # 每分钟记录一次数据
    
    # 定义室外温度函数
    def Tout_func(t):
        return Tout[min(int(t), len(Tout) - 1)]
    
    # 定义空调控制策略函数
    def mode_func(t, state):
        Ta = state[0]
        # 简单的温控策略：温度高于设定值时开启空调
        if Ta > T_set + 0.5:  # 设置0.5°C的死区
            return 1  # 空调开启
        elif Ta < T_set - 0.5:
            return 0  # 空调关闭
        else:
            return 0  # 默认关闭
    
    # 运行模拟
    print(f"开始模拟：初始室温 {T0}°C，设定温度 {T_set}°C，室外温度 {Tout}°C")
    start_time = time.time()
    solution = model.simulate(T0, Tm0, Tout_func, mode_func, t_span, t_eval)
    end_time = time.time()
    print(f"模拟完成，耗时: {end_time - start_time:.2f}秒")
    
    # 转换时间为小时
    hours = solution.t / 3600
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    # 温度变化曲线
    plt.subplot(2, 1, 1)
    plt.plot(hours, solution.y[0], 'r-', label='室内温度')
    plt.plot(hours, [Tout_func(t) for t in solution.t], 'b-', label='室外温度')
    plt.axhline(y=T_set, color='g', linestyle='--', label=f'设定温度 ({T_set}°C)')
    plt.ylabel('温度 (°C)')
    plt.legend()
    plt.title('室外温度变化下的室内温度变化')
    
    # 计算空调开启状态
    ac_status = []
    for i in range(len(solution.t)):
        t = solution.t[i]
        state = [solution.y[0][i], solution.y[1][i]]
        ac_status.append(mode_func(t, state))
    
    # 绘制空调开启状态
    plt.subplot(2, 1, 2)
    plt.step(hours, ac_status, 'k-', where='post')
    plt.ylim(-0.1, 1.1)
    plt.ylabel('空调状态')
    plt.xlabel('时间 (小时)')
    plt.yticks([0, 0.5, 1], ['关闭', '半开', '开启'])
    
    plt.tight_layout()
    plt.savefig('恒定室外温度下的室内温度变化.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main() 