import pandas as pd
from datetime import datetime

# 读取W2.csv文件
data = pd.read_csv('../../data/W2.csv')

print('夏季数据天数分析')
print('=' * 50)

# 筛选夏季数据（6-8月）
summer_indices = data['Time'].str.contains('2015/6|2015/7|2015/8', na=False)
summer_data = data[summer_indices]

print(f'夏季数据总条数: {len(summer_data)}')
print(f'数据频率: 每15分钟一条')
print(f'每天数据条数: 96条 (24小时 × 4)')
print(f'理论天数: {len(summer_data) // 96}天')

# 提取日期信息
dates = []
for time_str in summer_data['Time']:
    # 格式：2015/6/1 0:00
    date_part = time_str.split(' ')[0]  # 获取日期部分
    dates.append(date_part)

# 获取唯一日期
unique_dates = list(set(dates))
unique_dates.sort()

print(f'\n实际唯一日期数: {len(unique_dates)}天')
print(f'日期范围: {unique_dates[0]} 到 {unique_dates[-1]}')

# 显示前10天和后10天
print(f'\n前10天:')
for i, date in enumerate(unique_dates[:10]):
    print(f'  {i+1:2d}. {date}')

if len(unique_dates) > 20:
    print(f'  ...')
    print(f'后10天:')
    for i, date in enumerate(unique_dates[-10:], len(unique_dates)-9):
        print(f'  {i:2d}. {date}')

# 检查每天的数据完整性
print(f'\n数据完整性检查:')
incomplete_days = 0
for date in unique_dates[:5]:  # 检查前5天
    day_count = sum(1 for d in dates if d == date)
    if day_count != 96:
        print(f'  {date}: {day_count}条数据 (不完整)')
        incomplete_days += 1
    else:
        print(f'  {date}: {day_count}条数据 ✅')

if incomplete_days == 0:
    print(f'前5天数据完整性: ✅ 全部完整')
else:
    print(f'前5天数据完整性: ❌ {incomplete_days}天不完整')

print(f'\n建议处理方案:')
print(f'1. 使用前{min(10, len(unique_dates))}天作为测试')
print(f'2. 每天生成11×48的功率矩阵')
print(f'3. 增量追加到CSV文件中') 