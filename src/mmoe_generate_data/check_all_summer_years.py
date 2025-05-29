import pandas as pd
import numpy as np
import os

# 添加NumPy向后兼容性修复
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

print("分析W2.csv中的年份和夏季数据")
print("=" * 50)

# 尝试不同的可能路径
possible_paths = [
    "../../data/W2.csv",
    "../../../data/W2.csv",
    "D:/afterWork/ACL_agg_exp/data/W2.csv"
]

w2_data = None
for path in possible_paths:
    try:
        print(f"尝试读取: {path}")
        w2_data = pd.read_csv(path)
        print(f"  成功读取W2.csv文件，共{len(w2_data)}条记录")
        break
    except FileNotFoundError:
        print(f"  文件不存在: {path}")
    except Exception as e:
        print(f"  读取失败: {str(e)}")

if w2_data is None:
    print("无法读取W2.csv文件，程序退出")
    exit(1)

# 分析时间格式
print("\n时间格式分析:")
sample_times = w2_data['Time'].head(5).tolist()
print(f"前5条时间记录: {sample_times}")

# 提取所有年份
print("\n提取所有年份:")
years = set()
for time_str in w2_data['Time']:
    try:
        # 假设格式为 YYYY/MM/DD 或 YYYY-MM-DD
        year = time_str.split('/')[0] if '/' in time_str else time_str.split('-')[0]
        # 确保提取到的是年份（4位数字）
        if len(year) == 4 and year.isdigit():
            years.add(year)
    except Exception:
        continue

print(f"发现{len(years)}个不同年份: {sorted(years)}")

# 为所有年份统计夏季数据
print("\n各年份夏季数据统计:")
all_summer_data_count = 0

for year in sorted(years):
    # 构建夏季过滤条件 (6-8月)
    summer_pattern = f"{year}/6|{year}/7|{year}/8"
    summer_indices = w2_data['Time'].str.contains(summer_pattern, na=False)
    summer_data = w2_data[summer_indices]
    
    if len(summer_data) > 0:
        # 获取这个年份夏季数据的日期范围
        dates = []
        for time_str in summer_data['Time']:
            date_part = time_str.split(' ')[0] if ' ' in time_str else time_str
            dates.append(date_part)
        
        unique_dates = sorted(set(dates))
        date_range = f"{unique_dates[0]} 到 {unique_dates[-1]}"
        
        all_summer_data_count += len(summer_data)
        print(f"  {year}年夏季: {len(summer_data)}条记录, {len(unique_dates)}天")
        print(f"    日期范围: {date_range}")
        print(f"    温度范围: {summer_data['Temperature(F)'].min():.1f}°F - {summer_data['Temperature(F)'].max():.1f}°F")
    else:
        print(f"  {year}年: 无夏季数据")

# 总结
print("\n总结:")
print(f"W2.csv共有{len(w2_data)}条记录")
print(f"所有年份的夏季数据共{all_summer_data_count}条记录")
if len(years) > 1:
    print("\n修改建议:")
    print("当前代码只筛选2015年的夏季数据，应该修改为筛选所有年份的夏季数据:")
    print("summer_indices = w2_data['Time'].str.contains('/6|/7|/8', na=False)  # 匹配所有年份的6、7、8月") 