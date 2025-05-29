#!/usr/bin/env python
# 修改控制信号逻辑的脚本
# 将控制信号的逻辑反转：当控制信号为1时降低最大温度上限，当控制信号为-1时提高最大温度上限

import re

def modify_control_signal_logic(file_path):
    print(f"正在修改文件: {file_path}")
    
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义修改函数的模式
    # 使用正则表达式匹配两个控制信号到温度约束的映射函数
    pattern1 = r'def control_signal_to_temp_constraints\(signal, base_T_min=22\.0, base_T_max=25\.0\):\s+""".*?adjustment = signal \* 1\.5'
    pattern2 = r'def control_signal_to_temp_constraints\(signal, base_T_min=22\.0, base_T_max=25\.0\):\s+"""控制信号映射到温度约束范围"""\s+adjustment = signal \* 1\.5'
    
    # 替换第一个函数
    content = re.sub(pattern1, 
                     'def control_signal_to_temp_constraints(signal, base_T_min=22.0, base_T_max=25.0):\n'
                     '        """\n'
                     '        将控制信号映射到温度约束范围\n'
                     '        控制信号影响温度约束的上下限，从而影响优化行为\n'
                     '        \n'
                     '        signal = -1.0 → 放宽制冷需求，提高温度上限 (例如：[23, 27]°C)\n'
                     '        signal = 0.0 → 正常约束 (例如：[22, 25]°C)  \n'
                     '        signal = 1.0 → 强制制冷需求，降低温度上限 (例如：[21, 23]°C)\n'
                     '        """\n'
                     '        # 根据控制信号调整温度约束\n'
                     '        # signal = -1: 更宽松的制冷要求，提高约束范围\n'
                     '        # signal = +1: 更严格的制冷要求，降低约束范围\n'
                     '        \n'
                     '        # 调整幅度：±1.5°C 的约束调整（取负号反转逻辑）\n'
                     '        adjustment = -signal * 1.5', 
                     content, flags=re.DOTALL)
    
    # 替换第二个函数
    content = re.sub(pattern2, 
                     'def control_signal_to_temp_constraints(signal, base_T_min=22.0, base_T_max=25.0):\n'
                     '        """\n'
                     '        控制信号映射到温度约束范围\n'
                     '        \n'
                     '        signal = -1.0 → 放宽制冷需求，提高温度上限 (例如：[23, 27]°C)\n'
                     '        signal = 0.0 → 正常约束 (例如：[22, 25]°C)  \n'
                     '        signal = 1.0 → 强制制冷需求，降低温度上限 (例如：[21, 23]°C)\n'
                     '        """\n'
                     '        # 调整幅度：±1.5°C（取负号反转逻辑）\n'
                     '        adjustment = -signal * 1.5', 
                     content, flags=re.DOTALL)
    
    # 将修改后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("文件修改完成！")

if __name__ == "__main__":
    file_path = "generate_data_V2.py"
    modify_control_signal_logic(file_path) 