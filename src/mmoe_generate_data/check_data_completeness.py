import pandas as pd
from collections import defaultdict

# 读取W2.csv文件
data = pd.read_csv('../../data/W2.csv')

print('夏季数据完整性详细分析')
print('=' * 60)

# 筛选夏季数据（6-8月）
summer_indices = data['Time'].str.contains('2015/6|2015/7|2015/8', na=False)
summer_data = data[summer_indices]

print(f'夏季数据总条数: {len(summer_data)}')

# 按日期分组统计
date_counts = defaultdict(int)
for time_str in summer_data['Time']:
    date_part = time_str.split(' ')[0]  # 获取日期部分
    date_counts[date_part] += 1

# 排序并分析
sorted_dates = sorted(date_counts.keys())

print(f'\n数据分布统计:')
print(f'日期范围: {sorted_dates[0]} 到 {sorted_dates[-1]}')

# 统计每天的数据量
complete_days = []
incomplete_days = []

for date in sorted_dates[:20]:  # 检查前20天
    count = date_counts[date]
    if count >= 96:  # 完整的一天至少96条记录
        complete_days.append((date, count))
    else:
        incomplete_days.append((date, count))

print(f'\n前20天数据完整性:')
print(f'完整天数: {len(complete_days)}天')
print(f'不完整天数: {len(incomplete_days)}天')

if complete_days:
    print(f'\n完整数据的天数:')
    for i, (date, count) in enumerate(complete_days[:10]):
        print(f'  {i+1:2d}. {date}: {count}条数据')
    if len(complete_days) > 10:
        print(f'  ... 以及另外{len(complete_days)-10}天')

if incomplete_days:
    print(f'\n不完整数据的天数:')
    for i, (date, count) in enumerate(incomplete_days[:10]):
        print(f'  {i+1:2d}. {date}: {count}条数据')
    if len(incomplete_days) > 10:
        print(f'  ... 以及另外{len(incomplete_days)-10}天')

# 分析数据间隔
print(f'\n时间间隔分析 (以前3个完整天为例):')
for date, count in complete_days[:3]:
    day_data = summer_data[summer_data['Time'].str.startswith(date)]
    times = day_data['Time'].values
    print(f'\n{date} ({count}条数据):')
    print(f'  开始时间: {times[0]}')
    print(f'  结束时间: {times[-1]}')
    print(f'  前5个时间点: {times[:5].tolist()}')
    
    # 检查时间间隔
    intervals = []
    for i in range(1, min(10, len(times))):
        t1 = pd.to_datetime(times[i-1])
        t2 = pd.to_datetime(times[i])
        interval = (t2 - t1).total_seconds() / 60  # 分钟
        intervals.append(interval)
    
    if intervals:
        print(f'  时间间隔: {intervals[:5]} 分钟')

# 建议处理策略
print(f'\n建议处理策略:')
if len(complete_days) >= 10:
    print(f'1. 使用完整数据的天数: {len(complete_days)}天')
    print(f'2. 推荐处理前{min(10, len(complete_days))}天进行测试')
    print(f'3. 完整天数列表: {[date for date, _ in complete_days[:10]]}')
else:
    print(f'1. 完整数据天数较少: {len(complete_days)}天')
    print(f'2. 需要调整数据处理逻辑')
    print(f'3. 考虑使用较少的时间点或插值处理')

# 重新计算理论天数
actual_complete_days = len([date for date, count in date_counts.items() if count >= 96])
print(f'\n实际完整天数: {actual_complete_days}天')
print(f'理论总天数: {len(sorted_dates)}天')
print(f'数据完整率: {actual_complete_days/len(sorted_dates)*100:.1f}%') 