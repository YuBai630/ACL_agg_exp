import pandas as pd

# 读取W2.csv文件
w2_data = pd.read_csv("../../data/W2.csv")
print('调试日期处理过程')
print('=' * 50)

# 筛选夏季数据（6-8月）
summer_indices = w2_data['Time'].str.contains('2015/6|2015/7|2015/8', na=False)
summer_data = w2_data[summer_indices]
print(f'夏季数据总条数: {len(summer_data)}')

# 提取日期信息
dates = []
for time_str in summer_data['Time']:
    date_part = time_str.split(' ')[0]  # 获取日期部分
    dates.append(date_part)

# 获取唯一日期并排序
unique_dates = list(set(dates))
unique_dates.sort()

print(f'唯一日期数: {len(unique_dates)}')
print(f'前10个日期: {unique_dates[:10]}')

# 测试前5天的数据提取
process_dates = unique_dates[:5]
print(f'\n处理日期: {process_dates}')

for i, process_date in enumerate(process_dates):
    print(f'\n=== 处理第{i+1}天: {process_date} ===')
    
    # 模拟原代码的数据提取过程
    day_data = summer_data[summer_data['Time'].str.startswith(process_date)]
    print(f'找到的数据条数: {len(day_data)}')
    
    if len(day_data) > 0:
        print(f'前5条时间: {day_data["Time"].head().tolist()}')
        print(f'后5条时间: {day_data["Time"].tail().tolist()}')
        
        # 检查是否有足够的数据
        print(f'数据是否足够 (>=98): {len(day_data) >= 98}')
        
        if len(day_data) >= 98:
            # 模拟温度数据提取
            day_data_sorted = day_data.sort_values('Time')
            fahrenheit_temps = day_data_sorted['Temperature(F)'].values[:98]
            celsius_temps = (fahrenheit_temps - 32) * 5/9
            
            print(f'温度数据前5个: {celsius_temps[:5]}')
            print(f'温度范围: {min(celsius_temps):.1f}°C - {max(celsius_temps):.1f}°C')
        else:
            print('❌ 数据不足，无法处理')
    else:
        print('❌ 没有找到任何数据')

# 深入分析第一天（应该有完整数据）
print(f'\n=== 深入分析第一天: {process_dates[0]} ===')
first_day = process_dates[0]
day_data = summer_data[summer_data['Time'].str.startswith(first_day)]

print(f'数据条数: {len(day_data)}')
print(f'数据时间范围:')
print(f'  最早: {day_data["Time"].min()}')
print(f'  最晚: {day_data["Time"].max()}')

# 检查时间分布
times_sample = day_data['Time'].head(20).tolist()
print(f'前20个时间点: {times_sample}')

# 检查是否时间跨度有问题
print(f'\n检查时间跨度问题:')
for time_str in day_data['Time'].head(10):
    if not time_str.startswith(first_day):
        print(f'❌ 发现异常时间: {time_str}')
    else:
        print(f'✅ 正常时间: {time_str}')

# 测试字符串匹配
print(f'\n测试字符串匹配:')
test_date = '2015/6/1'
sample_times = ['2015/6/1 0:00', '2015/6/1 23:45', '2015/6/10 0:00', '2015/6/19 23:45']
for time_str in sample_times:
    match = time_str.startswith(test_date)
    print(f'{time_str} 匹配 {test_date}: {match}') 