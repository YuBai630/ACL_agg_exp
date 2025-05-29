#!/usr/bin/env python
# 测试控制信号映射函数

import sys
# sys.path.append('../..')

# 从当前目录下的模块导入
from generate_data_V2 import control_signal_to_temp_constraints

def test_control_signals():
    # 测试不同控制信号的映射结果
    signals = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("控制信号 -> 温度约束范围 [最小温度, 最大温度] -> 目标温度")
    print("-" * 60)
    
    for signal in signals:
        t_min, t_max, t_target = control_signal_to_temp_constraints(signal)
        print(f"信号 {signal:4.1f} -> [{t_min:4.1f}, {t_max:4.1f}]°C -> {t_target:4.1f}°C")

if __name__ == "__main__":
    test_control_signals() 