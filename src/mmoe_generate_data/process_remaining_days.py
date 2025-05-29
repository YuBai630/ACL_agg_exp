import os
import sys
import time
import warnings
import argparse
import json

# 忽略与 np.bool 相关的警告
warnings.filterwarnings('ignore', category=FutureWarning)

def check_environment():
    """检查运行环境，处理库版本兼容性问题"""
    try:
        # 先导入numpy并添加向后兼容性
        import numpy as np
        if not hasattr(np, 'bool'):
            np.bool = bool
        if not hasattr(np, 'object'):
            np.object = object
        if not hasattr(np, 'float'):
            np.float = float
        if not hasattr(np, 'int'):
            np.int = int
        
        # 再导入pandas
        import pandas as pd
        print(f"✅ 成功导入环境: NumPy {np.__version__}, Pandas {pd.__version__}")
        return True
    except Exception as e:
        print(f"❌ 环境检查失败: {str(e)}")
        print("请尝试更新 Pandas 或降级 NumPy 版本:")
        print("  pip install pandas --upgrade")
        print("  或")
        print("  pip install numpy==1.19.5")
        return False

def check_data_file():
    """检查数据文件是否存在，返回正确的文件路径"""
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 尝试不同的相对路径组合
    possible_paths = [
        os.path.join(current_dir, "../../data/W2.csv"),
        os.path.join(current_dir, "../../../data/W2.csv"),
        os.path.join(current_dir, "../../data/W2.csv"),
        os.path.normpath(os.path.join(current_dir, "../../data/W2.csv")),
        os.path.abspath(os.path.join(current_dir, "../../data/W2.csv")),
        "D:/afterWork/ACL_agg_exp/data/W2.csv"  # 直接使用绝对路径
    ]
    
    # 检查每个可能的路径
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ 找到数据文件: {path}")
            return path
    
    print("❌ 未找到数据文件 W2.csv")
    return None

def load_ac_configs():
    """加载空调配置数据"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ac_data_path = os.path.join(current_dir, "ac_data.json")
        
        # 检查文件是否存在
        if not os.path.exists(ac_data_path):
            print(f"❌ 未找到空调配置文件: {ac_data_path}")
            return None
            
        with open(ac_data_path, 'r', encoding='utf-8') as f:
            ac_configs = json.load(f)
        
        print(f"✅ 成功加载{len(ac_configs)}个空调配置")
        return ac_configs
    except Exception as e:
        print(f"❌ 加载空调配置失败: {str(e)}")
        return None

def get_available_years(data_file_path):
    """获取数据文件中可用的年份列表"""
    try:
        import pandas as pd
        import numpy as np
        
        # 添加NumPy兼容性修复
        if not hasattr(np, 'bool'):
            np.bool = bool
        if not hasattr(np, 'object'):
            np.object = object
        
        # 读取数据文件
        print(f"正在读取数据文件以获取可用年份...")
        w2_data = pd.read_csv(data_file_path)
        
        # 提取年份
        years = set()
        for time_str in w2_data['Time']:
            try:
                year = time_str.split('/')[0]
                if len(year) == 4 and year.isdigit():
                    years.add(year)
            except Exception:
                continue
        
        years_list = sorted(list(years))
        print(f"找到{len(years_list)}个可用年份: {years_list}")
        return years_list
    except Exception as e:
        print(f"获取可用年份时出错: {str(e)}")
        return []

def generate_single_ac_multi_day_data(ac_config, target_year=None, start_day=0, batch_size=10, data_file_path=None):
    """
    为单个空调生成多年夏季数据
    
    参数:
    ac_config: 空调配置字典
    target_year: 要处理的目标年份，None表示处理所有年份
    start_day: 开始处理的天数索引（0表示第一天）
    batch_size: 每批处理的天数
    data_file_path: 数据文件路径
    """
    from generate_data_V2 import ACOptimizerWithTempTarget
    import pandas as pd
    import numpy as np
    
    # 添加NumPy兼容性修复
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'object'):
        np.object = object
    
    ac_id = ac_config['id']
    ac_type = ac_config['type']
    
    print(f"\n🔧 开始处理空调: {ac_id} ({ac_type})")
    print(f"参数: P_rated={ac_config['P_rated']}kW, T_range=[{ac_config['T_min']}, {ac_config['T_max']}]°C")
    
    try:
        # 读取夏季数据
        w2_data = pd.read_csv(data_file_path)
        
        # 筛选夏季数据（6-8月）
        if target_year:
            summer_pattern = f"{target_year}/6|{target_year}/7|{target_year}/8"
            summer_indices = w2_data['Time'].str.contains(summer_pattern, na=False)
            print(f"  筛选{target_year}年的夏季数据")
        else:
            summer_indices = w2_data['Time'].str.contains('/6|/7|/8', na=False)
            print(f"  筛选所有年份的夏季数据")
        
        if not summer_indices.any():
            print(f"  ❌ 未找到夏季数据")
            return
        
        summer_data = w2_data[summer_indices]
        print(f"  找到夏季数据: {len(summer_data)}条记录")
        
        # 提取日期信息
        dates = []
        for time_str in summer_data['Time']:
            date_part = time_str.split(' ')[0]
            dates.append(date_part)
        
        # 获取唯一日期并排序
        unique_dates = list(set(dates))
        unique_dates.sort()
        
        total_days = len(unique_dates)
        print(f"  夏季总天数: {total_days}天")
        
        # 确定实际处理的天数
        remaining_days = total_days - start_day
        num_batches = (remaining_days + batch_size - 1) // batch_size
        
        # 创建带空调ID的输出文件
        figures_dir = "../../figures"
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        power_matrix_file = os.path.join(figures_dir, f"{ac_id}_multi_day_power_matrix.csv")
        stats_file = os.path.join(figures_dir, f"{ac_id}_multi_day_statistics.csv")
        timeseries_file = os.path.join(figures_dir, f"{ac_id}_multi_day_full_timeseries.csv")
        
        # 初始化结果文件
        all_results = []
        
        # 逐批处理
        for batch in range(num_batches):
            batch_start = start_day + batch * batch_size
            batch_end = min(batch_start + batch_size, total_days)
            
            print(f"  批次{batch+1}/{num_batches}: 第{batch_start+1}-{batch_end}天")
            
            process_dates = unique_dates[batch_start:batch_end]
            
            # 逐天处理
            for day_idx, process_date in enumerate(process_dates):
                try:
                    # 提取当天数据
                    day_data = summer_data[[date.split(' ')[0] == process_date for date in summer_data['Time']]]
                    
                    if len(day_data) < 96:
                        print(f"    日期{process_date}: 数据不足，跳过")
                        continue
                    
                    # 提取温度数据（每30分钟一个点）
                    day_data_sorted = day_data.sort_values('Time')
                    fahrenheit_temps = day_data_sorted['Temperature(F)'].values[:96]
                    celsius_temps = (fahrenheit_temps - 32) * 5/9
                    
                    outdoor_temp_half_hour = []
                    for i in range(0, min(96, len(celsius_temps)), 2):
                        outdoor_temp_half_hour.append(celsius_temps[i])
                        if len(outdoor_temp_half_hour) >= 49:
                            break
                    
                    while len(outdoor_temp_half_hour) < 49:
                        outdoor_temp_half_hour.append(outdoor_temp_half_hour[-1])
                    
                    # 创建空调优化器
                    optimizer = ACOptimizerWithTempTarget(
                        T=48,
                        delta_t=0.5,
                        P_rated=ac_config['P_rated'],
                        T_min=ac_config['T_min'],
                        T_max=ac_config['T_max'],
                        eta=ac_config['eta'],
                        R=ac_config['R'],
                        C=ac_config['C'],
                        T_initial=23.5,
                        T_target=(ac_config['T_min'] + ac_config['T_max']) / 2.0,
                        target_type='custom'
                    )
                    
                    # 设置室外温度
                    optimizer.set_outdoor_temperature(outdoor_temp_half_hour)
                    
                    # 求解优化问题
                    success = optimizer.solve()
                    
                    if success:
                        # 从优化器获取结果
                        powers = optimizer.optimal_powers
                        temperatures = optimizer.optimal_temperatures
                        total_energy = optimizer.total_energy
                        
                        # 保存结果
                        day_result = {
                            'ac_id': ac_id,
                            'ac_type': ac_type,
                            'date': process_date,
                            'status': 'success',
                            'total_energy': total_energy,
                            'avg_power': sum(powers) / len(powers),
                            'max_power': max(powers),
                            'powers': powers,
                            'temperatures': temperatures
                        }
                        all_results.append(day_result)
                        print(f"    日期{process_date}: 成功 (能耗: {total_energy:.2f}kWh)")
                        
                    else:
                        print(f"    日期{process_date}: 优化失败")
                        
                except Exception as e:
                    print(f"    日期{process_date}: 处理出错 - {str(e)}")
                    continue
            
            # 每批处理后暂停
            time.sleep(2)
        
        # 保存所有结果到CSV文件
        if all_results:
            save_ac_results_to_csv(all_results, ac_config, power_matrix_file, stats_file, timeseries_file)
            print(f"  ✅ {ac_id} 数据已保存，共处理{len(all_results)}天")
        else:
            print(f"  ❌ {ac_id} 无有效数据")
            
    except Exception as e:
        print(f"  ❌ {ac_id} 处理失败: {str(e)}")

def save_ac_results_to_csv(results, ac_config, power_matrix_file, stats_file, timeseries_file):
    """保存单个空调的结果到CSV文件"""
    import pandas as pd
    
    ac_id = ac_config['id']
    ac_type = ac_config['type']
    
    # 1. 保存功率矩阵
    with open(power_matrix_file, 'w', encoding='utf-8') as f:
        f.write(f"{ac_id} 多天功率矩阵\n")
        f.write(f"\"空调参数: P_rated={ac_config['P_rated']}kW, T_range=[{ac_config['T_min']}, {ac_config['T_max']}]°C\"\n\n")
        
        # 写入表头
        header = "日期,状态," + ",".join([f"t{i+1}({(i+1)*0.5}h)" for i in range(48)])
        f.write(header + "\n")
        
        # 写入数据
        for result in results:
            if result['status'] == 'success':
                powers_str = ",".join([f"{p:.3f}" for p in result['powers']])
                f.write(f"{result['date']},{result['status']},{powers_str}\n")
    
    # 2. 保存统计信息
    stats_data = []
    for result in results:
        if result['status'] == 'success':
            stats_data.append({
                '空调ID': ac_id,
                '空调类型': ac_type,
                '日期': result['date'],
                '状态': result['status'],
                '总能耗(kWh)': result['total_energy'],
                '平均功率(kW)': result['avg_power'],
                '最大功率(kW)': result['max_power'],
                '额定功率(kW)': ac_config['P_rated'],
                '温度范围(°C)': f"[{ac_config['T_min']}, {ac_config['T_max']}]"
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_file, index=False, encoding='utf-8')
    
    # 3. 保存完整时间序列
    with open(timeseries_file, 'w', encoding='utf-8') as f:
        f.write(f"{ac_id} 多天完整时间序列数据\n\n")
        
        for result in results:
            if result['status'] == 'success':
                f.write(f"\"日期: {result['date']}, 空调: {ac_id} ({ac_type}), 状态: {result['status']}\"\n")
                f.write("时间(h),室内温度(°C),空调功率(kW),功率占比(%)\n")
                
                for i in range(48):
                    time_h = (i + 1) * 0.5
                    temp = result['temperatures'][i] if i < len(result['temperatures']) else 0
                    power = result['powers'][i] if i < len(result['powers']) else 0
                    power_ratio = (power / ac_config['P_rated'] * 100) if power > 0 else 0
                    f.write(f"{time_h},{temp:.2f},{power:.3f},{power_ratio:.1f}\n")
                
                f.write("\n")

def process_remaining_days_for_all_acs(target_year=None, start_day=0, batch_size=10, selected_ac_ids=None, selected_ac_type=None):
    """
    为所有空调分别处理剩余的夏季天数
    
    参数:
    target_year: 要处理的目标年份，None表示处理所有年份
    start_day: 开始处理的天数索引（0表示第一天）
    batch_size: 每批处理的天数
    selected_ac_ids: 指定要处理的空调ID列表，None表示处理所有空调
    selected_ac_type: 按类型筛选空调
    """
    # 环境检查
    if not check_environment():
        print("❌ 环境检查未通过，终止执行")
        return
    
    # 检查数据文件
    data_file_path = check_data_file()
    if not data_file_path:
        print("❌ 数据文件不存在，终止执行")
        return
    
    # 加载空调配置
    ac_configs = load_ac_configs()
    if not ac_configs:
        print("❌ 无法加载空调配置，终止执行")
        return
    
    # 筛选要处理的空调
    if selected_ac_ids:
        ac_configs = [ac for ac in ac_configs if ac['id'] in selected_ac_ids]
        print(f"✅ 选择处理{len(ac_configs)}个指定空调: {selected_ac_ids}")
    elif selected_ac_type:
        ac_configs = [ac for ac in ac_configs if ac['type'] == selected_ac_type]
        print(f"✅ 选择处理{len(ac_configs)}个{selected_ac_type}型空调")
    else:
        print(f"✅ 将处理所有{len(ac_configs)}个空调")
    
    if not ac_configs:
        print("❌ 没有找到要处理的空调配置")
        return
    
    # 获取可用年份
    available_years = get_available_years(data_file_path)
    if not available_years:
        print("❌ 无法获取可用年份，终止执行")
        return
    
    # 验证目标年份
    if target_year is not None:
        if target_year not in available_years:
            print(f"❌ 指定的年份 {target_year} 不在可用年份列表中")
            print(f"可用年份: {available_years}")
            return
        years_to_process = [target_year]
        print(f"✅ 将处理 {target_year} 年的夏季数据")
    else:
        years_to_process = available_years
        print(f"✅ 将处理所有年份的夏季数据: {years_to_process}")
    
    # 为每个年份和每个空调生成数据
    total_acs = len(ac_configs)
    total_years = len(years_to_process)
    
    print(f"\n🚀 开始批量处理: {total_acs}个空调 × {total_years}个年份 = {total_acs * total_years}个任务")
    print("=" * 100)
    
    for year_idx, year in enumerate(years_to_process):
        print(f"\n🗓️  处理年份: {year} [{year_idx+1}/{total_years}]")
        print("🔄" * 50)
        
        for ac_idx, ac_config in enumerate(ac_configs):
            print(f"\n[{ac_idx+1}/{total_acs}] 空调: {ac_config['id']} - 年份: {year}")
            
            try:
                # 为每个空调生成多天数据
                generate_single_ac_multi_day_data(
                    ac_config=ac_config, 
                    target_year=year, 
                    start_day=start_day, 
                    batch_size=batch_size, 
                    data_file_path=data_file_path
                )
                
                # 每个空调处理完后短暂暂停
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ 处理空调 {ac_config['id']} 时出错: {str(e)}")
                continue
    
    print(f"\n🎉 所有空调的多年夏季数据处理完成!")
    print(f"生成的文件格式: [空调ID]_multi_day_[类型].csv")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='为每个空调生成多年夏季优化数据')
    parser.add_argument('--year', type=str, help='要处理的年份，如2015、2016等，不指定则处理所有年份')
    parser.add_argument('--start', type=int, default=0, help='开始处理的天数索引（0表示第一天），默认为0')
    parser.add_argument('--batch', type=int, default=10, help='每批处理的天数，默认为10')
    parser.add_argument('--ac-ids', type=str, nargs='*', help='指定要处理的空调ID，如 AC_S_001 AC_M_001，不指定则处理所有空调')
    parser.add_argument('--ac-type', type=str, choices=['small', 'medium', 'large'], help='按空调类型筛选，可选small/medium/large')
    parser.add_argument('--list-acs', action='store_true', help='仅列出所有可用的空调ID和类型，不进行处理')
    
    args = parser.parse_args()
    
    # 如果只是要列出空调
    if args.list_acs:
        ac_configs = load_ac_configs()
        if ac_configs:
            print(f"\n📋 可用空调列表 (共{len(ac_configs)}个):")
            print("-" * 60)
            
            # 按类型分组
            by_type = {'small': [], 'medium': [], 'large': []}
            for ac in ac_configs:
                by_type[ac['type']].append(ac)
            
            for ac_type_key, acs_list in by_type.items():
                if acs_list:
                    print(f"\n{ac_type_key.upper()}型空调 ({len(acs_list)}个):")
                    for ac_item in acs_list:
                        print(f"  {ac_item['id']}: P_rated={ac_item['P_rated']}kW, T_range=[{ac_item['T_min']}, {ac_item['T_max']}]°C")
        exit(0)
    
    # 执行数据处理
    fixed_processing_years = ["2015", "2017", "2018", "2019", "2020", "2021"]

    if args.year:
        print(f"ℹ️  注意: 检测到命令行参数 --year '{args.year}'。")
        print(f"脚本将按照预设的年份列表处理数据: {fixed_processing_years}。")
        if args.year not in fixed_processing_years:
             print(f"命令行指定的年份 '{args.year}' 不在预设列表中，但仍会按预设列表执行。")
        print("如果您只想处理命令行指定的单个年份，请考虑修改脚本或移除预设年份列表的逻辑。")
    
    print(f"\n🚀 即将按预设年份列表为所有(或选定)空调处理数据: {fixed_processing_years}")
    print("=" * 100)

    for year_to_process in fixed_processing_years:
        print(f"\n🔥🔥🔥 开始处理年份: {year_to_process} 🔥🔥🔥")
        process_remaining_days_for_all_acs(
            target_year=year_to_process,      # 使用循环中的当前年份
            start_day=args.start,             # 遵循命令行参数
            batch_size=args.batch,            # 遵循命令行参数
            selected_ac_ids=args.ac_ids,      # 遵循命令行参数 (用于筛选空调)
            selected_ac_type=args.ac_type     # 遵循命令行参数 (用于筛选空调类型)
        )
        print(f"🏁🏁🏁 完成处理年份: {year_to_process} 🏁🏁🏁")
        print("-" * 100)
        # 添加一个小的延时，使得输出更易读，特别是在处理多个年份时
        time.sleep(3)
    
    print("\n🎉🎉🎉 所有预设年份的数据处理完成! 🎉🎉🎉") 