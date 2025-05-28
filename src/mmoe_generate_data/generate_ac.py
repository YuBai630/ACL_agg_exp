"""
线性规划方程：
M_{AC}: \min_{\left \{P_{1},...,P_{T},T_{2}^{i},...,T_{T}^{i}\right \} }\sum_{t=1}^{T}P_{t}\Delta t,
subject to for \forall t \in \{1,2,..,T\}:
0 \le P_{t} \le P_{rated},
T_{min} \le T_{t} \le T_{max}

and
室内气温更新公式：
T_{t+1}^{i} = T_{t+1}^{out} - \eta P_{t} R_{t} - (T_{t+1}^{out} - \eta P_{t} R_{t} - T_{t}^{i}) e^{- \Delta t / R C}
"""

import numpy as np
import pandas as pd
import json
import os
import random

def generate_ac_data(num_acs=100, small_buildings=60, medium_buildings=30, large_buildings=10):
    """
    生成空调数据集，基于1阶ETP（等效热阻热容）模型
    
    参数:
    num_acs: 空调总数
    small_buildings: 小型建筑数量
    medium_buildings: 中型建筑数量
    large_buildings: 大型建筑数量
    
    返回:
    ac_data: 包含空调参数的字典列表
    """
    assert num_acs == small_buildings + medium_buildings + large_buildings, "建筑数量总和必须等于空调总数"
    
    ac_data = []
    
    # 为不同类型的建筑设置参数范围
    building_params = {
        "small": {
            "R_range": (1.8, 2.5),      # 等效热阻范围 (℃/KW)
            "C_range": (1.5e7, 2.5e7),  # 等效热容范围 (J/℃)
            "P_rated_range": (2.5, 3.5)  # 空调额定功率范围 (KW)
        },
        "medium": {
            "R_range": (1.2, 1.8),      # 等效热阻范围 (℃/KW)
            "C_range": (2.5e7, 4.0e7),  # 等效热容范围 (J/℃)
            "P_rated_range": (3.5, 5.0)  # 空调额定功率范围 (KW)
        },
        "large": {
            "R_range": (0.8, 1.2),      # 等效热阻范围 (℃/KW)
            "C_range": (4.0e7, 6.0e7),  # 等效热容范围 (J/℃)
            "P_rated_range": (5.0, 8.0)  # 空调额定功率范围 (KW)
        }
    }
    
    # 更新后的通用参数
    common_params = {
        "efficiency": 0.98,  # 空调效率统一为0.98
        "eta": 3.5           # 能效比 (COP)
    }
    
    # 生成小型建筑的空调数据
    for i in range(small_buildings):
        params = building_params["small"]
        # 随机设置温度范围
        t_min = round(random.uniform(20.0, 22.0), 1)
        t_max = round(random.uniform(24.0, 26.0), 1)
        
        ac = {
            "id": f"AC_S_{i+1:03d}",
            "type": "small",
            "R": round(random.uniform(*params["R_range"]), 4),  # 等效热阻 (℃/KW)
            "C": round(random.uniform(*params["C_range"]), 1),  # 等效热容 (J/℃)
            "P_rated": round(random.uniform(*params["P_rated_range"]), 2),  # 额定功率 (KW)
            "T_min": t_min,  # 最小允许室内温度 (℃)
            "T_max": t_max,  # 最大允许室内温度 (℃)
            "efficiency": common_params["efficiency"],  # 空调效率
            "eta": common_params["eta"]  # 能效比
        }
        ac_data.append(ac)
    
    # 生成中型建筑的空调数据
    for i in range(medium_buildings):
        params = building_params["medium"]
        # 随机设置温度范围
        t_min = round(random.uniform(20.0, 22.0), 1)
        t_max = round(random.uniform(24.0, 26.0), 1)
        
        ac = {
            "id": f"AC_M_{i+1:03d}",
            "type": "medium",
            "R": round(random.uniform(*params["R_range"]), 4),  # 等效热阻 (℃/KW)
            "C": round(random.uniform(*params["C_range"]), 1),  # 等效热容 (J/℃)
            "P_rated": round(random.uniform(*params["P_rated_range"]), 2),  # 额定功率 (KW)
            "T_min": t_min,  # 最小允许室内温度 (℃)
            "T_max": t_max,  # 最大允许室内温度 (℃)
            "efficiency": common_params["efficiency"],  # 空调效率
            "eta": common_params["eta"]  # 能效比
        }
        ac_data.append(ac)
    
    # 生成大型建筑的空调数据
    for i in range(large_buildings):
        params = building_params["large"]
        # 随机设置温度范围
        t_min = round(random.uniform(20.0, 22.0), 1)
        t_max = round(random.uniform(24.0, 26.0), 1)
        
        ac = {
            "id": f"AC_L_{i+1:03d}",
            "type": "large",
            "R": round(random.uniform(*params["R_range"]), 4),  # 等效热阻 (℃/KW)
            "C": round(random.uniform(*params["C_range"]), 1),  # 等效热容 (J/℃)
            "P_rated": round(random.uniform(*params["P_rated_range"]), 2),  # 额定功率 (KW)
            "T_min": t_min,  # 最小允许室内温度 (℃)
            "T_max": t_max,  # 最大允许室内温度 (℃)
            "efficiency": common_params["efficiency"],  # 空调效率
            "eta": common_params["eta"]  # 能效比
        }
        ac_data.append(ac)
    
    return ac_data

def save_data(data, filename="ac_data.json"):
    """保存数据到JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"数据已保存到 {filename}")

def main():
    # 生成100个空调数据
    ac_data = generate_ac_data(num_acs=100, small_buildings=60, medium_buildings=30, large_buildings=10)
    
    # 保存数据
    save_data(ac_data)
    
    # 打印统计信息
    print(f"总共生成了 {len(ac_data)} 个空调数据")
    print(f"小型建筑: {sum(1 for ac in ac_data if ac['type'] == 'small')} 个")
    print(f"中型建筑: {sum(1 for ac in ac_data if ac['type'] == 'medium')} 个")
    print(f"大型建筑: {sum(1 for ac in ac_data if ac['type'] == 'large')} 个")
    
    # 打印一些样例数据
    print("\n样例数据:")
    for building_type in ["small", "medium", "large"]:
        sample = next((ac for ac in ac_data if ac["type"] == building_type), None)
        if sample:
            print(f"\n{building_type.capitalize()}型建筑样例:")
            for key, value in sample.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()

