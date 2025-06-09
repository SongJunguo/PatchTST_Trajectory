#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级数据清理器 - 多进程并行完整版 (集成多线程最终检查)

功能:
1. 去掉每个ID开头和结尾的缺失值
2. ID中间缺失时间<30s: 插值
3. ID中间缺失时间>30s: 分割ID (原ID00001, 原ID00002...)
4. 过滤长度<544个点的轨迹段
5. 检查并删除全空的轨迹 (采用多线程优化)
6. 统计所有轨迹的空值情况 (采用多线程优化)
7. 多进程并行处理，充分利用多核CPU进行数据分析和清理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import traceback

# 忽略pandas的性能警告，因为我们知道我们在做什么
warnings.filterwarnings('ignore')

# =============================================================================
# 1. 核心辅助函数 (在所有进程中共享)
# =============================================================================

def parse_time_string(time_str):
    """
    解析时间字符串，支持多种格式
    """
    if pd.isna(time_str) or time_str == '':
        return None
    try:
        if isinstance(time_str, str):
            time_str = time_str.strip()
            formats = [
                "%Y%m%d %H:%M:%S.%f", "%Y%m%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(time_str, fmt)
                except ValueError:
                    continue
        return pd.to_datetime(time_str)
    except Exception:
        # 在大规模并行处理中，减少打印，只返回None
        return None

def calculate_time_gap(time1, time2):
    """
    计算两个时间之间的间隔（秒）
    """
    if time1 is None or time2 is None:
        return None
    try:
        return abs((time2 - time1).total_seconds())
    except Exception:
        return None

def interpolate_missing_coordinates(data_segment):
    """
    对数据段中的缺失坐标进行线性插值
    增强版：更严格地检查插值结果
    """
    segment = data_segment.copy()
    for col in ['H', 'Lon', 'Lat']:
        if col in segment.columns:
            # 检查是否整列都是NaN
            if segment[col].isna().all():
                print(f"警告: 列 {col} 在段中全部为NaN，无法插值")
                return None  # 返回None表示无法处理

            # 如果有缺失值，进行插值
            if segment[col].isna().any():
                # 先尝试线性插值
                segment[col] = segment[col].interpolate(method='linear')
                # 对于开头和结尾的NaN，使用前向和后向填充
                segment[col] = segment[col].fillna(method='ffill').fillna(method='bfill')

                # 再次检查是否还有NaN
                if segment[col].isna().any():
                    print(f"警告: 列 {col} 插值后仍有NaN值")
                    return None  # 返回None表示插值失败

    return segment

def process_segment_missing_values(segment):
    """
    对单个段进行二次缺失值处理，移除开头结尾的NaN并处理中间的
    增强版：更严格地检查和处理NaN值
    """
    if len(segment) == 0:
        return []

    key_columns = ['H', 'Lon', 'Lat']
    missing_mask = pd.Series(False, index=segment.index)
    for col in key_columns:
        if col in segment.columns:
            missing_mask |= segment[col].isna()

    # 如果所有关键列的所有行都是NaN，直接返回空
    if missing_mask.all():
        return []

    valid_indices = segment.index[~missing_mask]
    if len(valid_indices) == 0:
        return []

    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]
    trimmed_segment = segment.loc[first_valid_idx:last_valid_idx].copy()

    if len(trimmed_segment) <= 1:
        return [trimmed_segment] if len(trimmed_segment) == 1 else []

    # 尝试插值
    interpolated = interpolate_missing_coordinates(trimmed_segment)

    # 检查插值是否成功
    if interpolated is None:
        # 插值失败，丢弃这个段
        return []

    # 再次严格检查是否还有NaN值
    has_nan = False
    for col in key_columns:
        if col in interpolated.columns and interpolated[col].isna().any():
            has_nan = True
            break

    if not has_nan:
        return [interpolated]
    else:
        # 如果插值后仍有缺失，则丢弃
        print(f"警告: 段插值后仍有NaN值，丢弃该段")
        return []

def process_single_id_data(id_data, original_id, max_gap_seconds=30, min_length=576):
    """
    处理单个ID的数据的核心逻辑（由工作函数调用）
    """
    if len(id_data) == 0:
        return []

    id_data = id_data.copy()
    id_data['parsed_time'] = id_data['Time'].apply(parse_time_string)
    id_data = id_data[id_data['parsed_time'].notna()]

    if len(id_data) < 2:
        return []
    
    id_data = id_data.sort_values('parsed_time').reset_index(drop=True)

    segments = []
    current_start = 0
    for i in range(1, len(id_data)):
        time_gap = calculate_time_gap(id_data.iloc[i-1]['parsed_time'], id_data.iloc[i]['parsed_time'])
        if time_gap is not None and time_gap > max_gap_seconds:
            segments.append(id_data.iloc[current_start:i])
            current_start = i
    segments.append(id_data.iloc[current_start:])
    
    final_segments = []
    segment_counter = 1
    is_split = len(segments) > 1

    for segment in segments:
        if len(segment) > 0:
            processed_sub_segments = process_segment_missing_values(segment)
            for sub_segment in processed_sub_segments:
                if len(sub_segment) >= min_length:
                    new_id = f"{original_id}{segment_counter:05d}" if is_split else original_id
                    segment_counter += 1
                    sub_segment['ID'] = new_id
                    final_segments.append(sub_segment.drop(columns=['parsed_time']))

    return final_segments


# =============================================================================
# 2. 并行工作函数 (Worker Functions)
# =============================================================================

def cleaning_worker(args):
    """
    (并行工作函数) 用于数据清理
    """
    flight_id, id_data, max_gap_seconds, min_length = args
    return process_single_id_data(id_data, flight_id, max_gap_seconds=max_gap_seconds, min_length=min_length)

def analyze_gaps_worker(id_data):
    """
    (并行工作函数) 用于分析时间间隔
    """
    gaps = []
    id_data = id_data.copy()
    id_data['parsed_time'] = id_data['Time'].apply(parse_time_string)
    id_data = id_data[id_data['parsed_time'].notna()].sort_values('parsed_time')
    
    if len(id_data) >= 2:
        gaps = id_data['parsed_time'].diff().dt.total_seconds().dropna().tolist()
            
    return gaps


# =============================================================================
# 3. 并行化的主功能函数
# =============================================================================

def analyze_time_gaps_parallel(input_file, num_workers=None):
    """
    分析数据中的时间间隔分布 - 多进程并行版
    """
    try:
        print(f"分析时间间隔 (并行模式): {input_file}")
        df = pd.read_csv(input_file, usecols=['ID', 'Time'])
        
        grouped = df.groupby('ID')
        tasks = [group for _, group in grouped]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(analyze_gaps_worker, tasks), total=len(tasks), desc="分析时间间隔"))

        time_gaps = [gap for sublist in results for gap in sublist]
        
        if time_gaps:
            time_gaps = np.array(time_gaps)
            print("\n时间间隔统计:")
            print(f"总间隔数: {len(time_gaps)}")
            print(f"平均间隔: {np.mean(time_gaps):.2f}秒")
            print(f"中位数间隔: {np.median(time_gaps):.2f}秒")
            print(f"最小间隔: {np.min(time_gaps):.2f}秒")
            print(f"最大间隔: {np.max(time_gaps):.2f}秒")
            print(f"标准差: {np.std(time_gaps):.2f}秒")
            
            thresholds = [10, 20, 30, 60, 120, 300]
            print("\n不同阈值下超过阈值的间隔数:")
            for threshold in thresholds:
                count = np.sum(time_gaps > threshold)
                pct = (count / len(time_gaps)) * 100 if len(time_gaps) > 0 else 0
                print(f">{threshold}秒: {count} ({pct:.2f}%)")
        else:
            print("未能计算出任何时间间隔。")
        
    except Exception as e:
        print(f"时间间隔分析失败: {e}")
        traceback.print_exc()

def clean_advanced_data_parallel(input_file, output_file, max_gap_seconds, min_length, num_workers=None):
    """
    高级数据清理主函数 - 多进程并行版
    """
    try:
        print(f"开始高级数据清理: {input_file}")
        
        print("正在读取数据...")
        df = pd.read_csv(input_file)
        
        print(f"原始数据形状: {df.shape}")
        
        grouped = df.groupby('ID')
        tasks = [(flight_id, id_data.copy(), max_gap_seconds, min_length) for flight_id, id_data in grouped]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(cleaning_worker, tasks), total=len(tasks), desc="清理轨迹数据"))

        print("所有子进程处理完成，开始整理结果...")
        
        all_processed_segments = [segment for sublist in results for segment in sublist]
        
        if all_processed_segments:
            final_data = pd.concat(all_processed_segments, ignore_index=True)
            final_data = final_data.reindex(columns=df.columns)
        else:
            final_data = pd.DataFrame(columns=df.columns)
        
        print("\n处理完成统计:")
        print(f"原始ID数: {len(tasks)}")
        print(f"最终ID数: {final_data['ID'].nunique()}")
        print(f"最终数据行数: {len(final_data)}")

        if len(final_data) > 0:
            trajectory_lengths = final_data.groupby('ID').size()
            print("\n轨迹长度统计:")
            print(f"平均长度: {trajectory_lengths.mean():.1f}个点")
            print(f"最短轨迹: {trajectory_lengths.min()}个点")
            print(f"最长轨迹: {trajectory_lengths.max()}个点")

            # ================================================================= #
            # 新增/修改部分：使用多线程进行最终检查
            # ================================================================= #

            key_columns = ['H', 'Lon', 'Lat']

            # 检查并删除全空的轨迹 (多线程版)
            print("\n检查全空轨迹 (多线程)...")

            # 内联工作函数，用于检查单个轨迹是否完全为空
            def check_if_trajectory_is_empty(group_tuple):
                trajectory_id, trajectory_data = group_tuple
                if trajectory_data[key_columns].isna().all().all():
                    return trajectory_id
                return None
            
            # 按ID分组，准备并行任务
            trajectory_groups = list(final_data.groupby('ID'))
            empty_trajectories = []
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(executor.map(check_if_trajectory_is_empty, trajectory_groups), total=len(trajectory_groups), desc="检查空轨迹"))
                empty_trajectories = [res for res in results if res is not None]

            if empty_trajectories:
                print(f"发现 {len(empty_trajectories)} 个全空轨迹，正在删除...")
                final_data = final_data[~final_data['ID'].isin(empty_trajectories)]
                print(f"删除后剩余ID数: {final_data['ID'].nunique()}")
                print(f"删除后剩余数据行数: {len(final_data)}")
            else:
                print("未发现全空轨迹")

            # 统计所有轨迹的空值情况 (多线程版)
            print("\n统计空值情况 (多线程)...")

            # 内联工作函数，用于计算单个轨迹的空值
            def calculate_trajectory_nulls(group_tuple):
                trajectory_id, trajectory_data = group_tuple
                trajectory_nulls = 0
                detailed_nulls = {}

                for col in key_columns:
                    if col in trajectory_data.columns:
                        col_nulls = trajectory_data[col].isna().sum()
                        trajectory_nulls += col_nulls
                        detailed_nulls[col] = col_nulls

                if trajectory_nulls > 0:
                    return {
                        'ID': trajectory_id,
                        'null_count': trajectory_nulls,
                        'total_points': len(trajectory_data),
                        'null_percentage': (trajectory_nulls / (len(trajectory_data) * len(key_columns))) * 100,
                        'detailed_nulls': detailed_nulls
                    }
                return {'null_count': 0}

            # 重新分组，因为空轨迹可能已被删除
            remaining_groups = list(final_data.groupby('ID'))
            total_null_count = 0
            trajectory_null_stats = []

            if remaining_groups:
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    results = list(tqdm(executor.map(calculate_trajectory_nulls, remaining_groups), total=len(remaining_groups), desc="统计空值"))
                    
                    for stat in results:
                        total_null_count += stat['null_count']
                        if 'ID' in stat: # 只有包含空值的轨迹才有 'ID' 键
                            trajectory_null_stats.append(stat)

            print(f"总空值数量: {total_null_count}")
            print(f"有空值的轨迹数量: {len(trajectory_null_stats)}")

            if trajectory_null_stats:
                print("\n空值轨迹详情 (前10个):")
                sorted_stats = sorted(trajectory_null_stats, key=lambda x: x['null_count'], reverse=True)
                for i, stat in enumerate(sorted_stats[:10]):
                    detailed_info = ""
                    if 'detailed_nulls' in stat:
                        details = [f"{col}:{count}" for col, count in stat['detailed_nulls'].items() if count > 0]
                        detailed_info = f" ({', '.join(details)})"

                    print(f"  {i+1}. ID: {stat['ID']}, 空值: {stat['null_count']}{detailed_info}, "
                          f"总点数: {stat['total_points']}, 空值比例: {stat['null_percentage']:.2f}%")

                if len(sorted_stats) > 10:
                    print(f"  ... 还有 {len(sorted_stats) - 10} 个轨迹有空值")

                # 检查是否有完全为空的轨迹（这种情况不应该存在，但我们要检查）
                completely_empty = [stat for stat in sorted_stats
                                  if stat['null_count'] == stat['total_points'] * len(key_columns)]
                if completely_empty:
                    print(f"\n⚠️  警告: 发现 {len(completely_empty)} 个完全为空的轨迹！")
                    for stat in completely_empty[:5]:  # 只显示前5个
                        print(f"    ID: {stat['ID']}, 总点数: {stat['total_points']}")

            elif total_null_count == 0:
                print("✅ 所有轨迹都没有空值！")

        print(f"\n正在保存到文件: {output_file}")
        final_data.to_csv(output_file, index=False, encoding='utf-8')

        print(f"高级数据清理完成! 文件已保存为: {output_file}")
        return True
        
    except Exception as e:
        print(f"高级数据清理过程中发生错误: {str(e)}")
        traceback.print_exc()
        return False


# =============================================================================
# 4. 主执行入口
# =============================================================================

def main():
    """主函数"""
    print("=" * 80)
    print("高级数据清理器 (多进程并行版)")
    print("功能:")
    print("1. 去掉每个ID开头和结尾的缺失值")
    print("2. ID中间缺失时间<30s: 插值")
    print("3. ID中间缺失时间>30s: 分割ID")
    print("4. 过滤长度<576个点的轨迹段")
    print("5. 检查并删除全空的轨迹 (多线程优化)")
    print("6. 统计所有轨迹的空值情况 (多线程优化)")
    print("7. 多进程并行处理，充分利用多核CPU")
    print("=" * 80)
    
    input_file = "./dataset/processed_2022-05-01.csv"
    output_file = "./dataset/advanced_cleaned_2022-05-01.csv"
    
    # --- 核心参数设置 ---
    # 根据您的硬件（如196核），设置一个合理的进程数。
    # 不建议一次性用满，以防系统不稳定或内存占用过高。
    # 可以从64, 96, 128开始尝试，并使用htop等工具监控系统负载。
    # 设置为None将使用所有可用的核心。
    NUM_CORES_TO_USE = 64

    # --- 步骤1: 并行分析时间间隔 ---
    print(f"步骤1: 分析时间间隔分布 (将使用最多 {NUM_CORES_TO_USE} 个核心)")
    analyze_time_gaps_parallel(input_file, num_workers=NUM_CORES_TO_USE)
    
    print("\n" + "="*50 + "\n")
    
    # --- 步骤2: 并行执行高级数据清理 ---
    print(f"步骤2: 执行高级数据清理 (将使用最多 {NUM_CORES_TO_USE} 个核心)")
    success = clean_advanced_data_parallel(input_file=input_file, 
                                         output_file=output_file, 
                                         max_gap_seconds=30, 
                                         min_length=576,
                                         num_workers=NUM_CORES_TO_USE)
    
    if success:
        print("\n✅ 高级数据清理完成!")
        print(f"输出文件: {output_file}")
    else:
        print("\n❌ 高级数据清理失败!")


if __name__ == "__main__":
    # 确保在Windows/macOS等系统上多进程能正常工作
    # 将主逻辑放在这个if块内是安全的多进程编程实践
    main()