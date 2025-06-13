import pandas as pd
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from functools import partial
from tqdm import tqdm

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两点之间的球面距离（Haversine公式）。
    """
    R = 6371  # 地球半径，单位为公里
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c
    return distance

def check_trajectory_worker(group_tuple, h_min, h_max, lon_min, lon_max, lat_min, lat_max, max_speed_kmh, max_vs_ms):
    """
    工作函数：对单个轨迹段执行一系列数据质量检查。
    """
    unique_id, df = group_tuple
    errors = []
    
    # 0. 复制以避免SettingWithCopyWarning
    df = df.copy()

    # 1. ID 和关键列检查
    if 'Unique_ID' not in df.columns:
        errors.append("缺少关键列: Unique_ID")
        # 如果没有ID，后续检查无意义，直接返回
        return unique_id, False, [f"[{unique_id}] " + e for e in errors]

    if df['Unique_ID'].isnull().any() or (df['Unique_ID'] == '').any():
        errors.append("发现空的或无效的 Unique_ID 值")

    if df['Unique_ID'].nunique() > 1:
        errors.append(f"在一个轨迹段内发现多个 Unique_ID: {df['Unique_ID'].unique()}")

    # 检查其他关键列是否存在
    required_cols = ['Lat', 'Lon', 'H', 'Time']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"缺少关键列: {', '.join(missing_cols)}")
    
    # 如果有任何列缺失或ID有问题，提前返回
    if errors:
        return unique_id, False, [f"[{unique_id}] " + e for e in errors]
        
    if df[required_cols].isnull().values.any():
        errors.append(f"在 Lat, Lon, H, Time 列中发现NaN值")

    # 2. 时间戳检查
    try:
        df['Time'] = pd.to_datetime(df['Time'])
        if not df['Time'].is_monotonic_increasing:
            errors.append("时间戳不具备单调递增性")
        
        time_diffs = df['Time'].diff().dt.total_seconds().dropna()
        if (time_diffs <= 0).any():
            errors.append("发现时间倒流或重复的时间点")
        
        # 检查大的时间跳跃（例如，超过15秒）
        if (time_diffs > 500).any():
            errors.append(f"发现大于15秒的时间跳跃，最大跳跃: {time_diffs.max():.2f}s")

    except Exception as e:
        errors.append(f"时间戳转换或处理失败: {e}")
        return unique_id, False, errors

    # 3. 数值范围检查
    if not df['Lat'].between(lat_min, lat_max).all():
        errors.append(f"纬度 (Lat) 超出范围 [{lat_min}, {lat_max}]")
    if not df['Lon'].between(lon_min, lon_max).all():
        errors.append(f"经度 (Lon) 超出范围 [{lon_min}, {lon_max}]")
    if not df['H'].between(h_min, h_max).all():
        errors.append(f"高度 (H) 超出范围 [{h_min}, {h_max}]")

    # 4. 飞行剖面异常检测 (速度和垂直速率)
    if len(df) > 1:
        distances = haversine_distance(df['Lon'].iloc[:-1].values, df['Lat'].iloc[:-1].values,
                                     df['Lon'].iloc[1:].values, df['Lat'].iloc[1:].values)
        time_deltas_s = df['Time'].diff().dt.total_seconds().iloc[1:].values
        
        # 避免除以零
        time_deltas_s[time_deltas_s == 0] = 1 
        
        speeds_kmh = (distances / (time_deltas_s / 3600))
        
        if (speeds_kmh > max_speed_kmh).any():
            errors.append(f"检测到瞬时速度超过 {max_speed_kmh} km/h，最大速度: {speeds_kmh.max():.2f} km/h")

        height_diff_m = df['H'].diff().iloc[1:].values
        vertical_speeds_ms = height_diff_m / time_deltas_s
        
        if (np.abs(vertical_speeds_ms) > max_vs_ms).any():
            errors.append(f"检测到垂直速率超过 {max_vs_ms} m/s，最大速率: {vertical_speeds_ms[np.argmax(np.abs(vertical_speeds_ms))]:.2f} m/s")

    if errors:
        return unique_id, False, [f"[{unique_id}] " + e for e in errors]
    else:
        return unique_id, True, []

def main():
    parser = argparse.ArgumentParser(description="检查已切分轨迹数据的质量")
    parser.add_argument('--input_file', type=str, required=True, help='输入的CSV文件路径 (_02_after_split_before_military_filter.csv)')
    parser.add_argument('--output_dir', type=str, required=True, help='存放输出文件的目录')
    parser.add_argument('--max_workers', type=int, default=16, help='用于并行处理的最大工作进程数')
    
    # 添加与主脚本一致的过滤参数，以便进行范围检查
    parser.add_argument('--h_min', type=float, default=0, help='最低高度 (米)')
    parser.add_argument('--h_max', type=float, default=20000, help='最高高度 (米)')
    parser.add_argument('--lon_min', type=float, default=-180, help='最小经度')
    parser.add_argument('--lon_max', type=float, default=180, help='最大经度')
    parser.add_argument('--lat_min', type=float, default=-90, help='最小纬度')
    parser.add_argument('--lat_max', type=float, default=42, help='最大纬度')

    # 飞行剖面检查参数
    parser.add_argument('--max_speed_kmh', type=float, default=2470, help='最大允许瞬时速度 (km/h), 约2马赫')
    parser.add_argument('--max_vs_ms', type=float, default=300, help='最大允许垂直速率 (m/s)')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"已创建输出目录: {args.output_dir}")

    log_filepath = os.path.join(args.output_dir, "_03_data_check_log.txt")
    cleaned_filepath = os.path.join(args.output_dir, "_03_checked_and_cleaned_data.csv")

    print(f"--- 开始读取输入文件: {args.input_file} ---")
    try:
        df = pd.read_csv(args.input_file, low_memory=False)
        print(f"--- 文件读取成功，共 {len(df)} 行，{df['Unique_ID'].nunique()} 个独立轨迹。 ---")
    except FileNotFoundError:
        print(f"--- 错误: 输入文件未找到: {args.input_file} ---")
        return

    # --- 新增功能: 统计每个ID的数据点数量并保存 ---
    if 'Unique_ID' in df.columns:
        print("--- 开始统计每个ID的数据点数量... ---")
        point_counts = df['Unique_ID'].value_counts().reset_index()
        point_counts.columns = ['Unique_ID', 'point_count']
        point_counts.sort_values(by='point_count', ascending=False, inplace=True)
        
        stats_filepath = os.path.join(args.output_dir, "_03_id_point_counts.csv")
        point_counts.to_csv(stats_filepath, index=False, encoding='utf-8-sig')
        print(f"--- ID数据点统计信息已保存到: {stats_filepath} ---")
    else:
        print("--- 警告: 输入文件中未找到 'Unique_ID' 列，跳过数据点统计。 ---")

    groups = list(df.groupby('Unique_ID'))
    
    print(f"--- 开始使用 {args.max_workers} 个进程并行检查 {len(groups)} 个轨迹... ---")

    # 使用 partial 绑定固定参数
    worker_func = partial(check_trajectory_worker, 
                          h_min=args.h_min, h_max=args.h_max,
                          lon_min=args.lon_min, lon_max=args.lon_max,
                          lat_min=args.lat_min, lat_max=args.lat_max,
                          max_speed_kmh=args.max_speed_kmh, max_vs_ms=args.max_vs_ms)

    valid_ids = []
    all_errors = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # 使用tqdm显示进度条
        results = list(tqdm(executor.map(worker_func, groups), total=len(groups), desc="检查轨迹"))

    for unique_id, is_valid, errors in results:
        if is_valid:
            valid_ids.append(unique_id)
        else:
            all_errors.extend(errors)

    print(f"--- 检查完成。 ---")
    print(f"--- 发现 {len(valid_ids)} 个有效轨迹。 ---")
    print(f"--- 发现 {len(groups) - len(valid_ids)} 个包含错误的轨迹。 ---")

    # 写入日志文件
    if all_errors:
        print(f"--- 正在将 {len(all_errors)} 条错误信息写入日志: {log_filepath} ---")
        with open(log_filepath, 'w', encoding='utf-8') as f:
            for error in all_errors:
                f.write(error + '\n')
    else:
        print("--- 未发现任何错误。 ---")
        # 如果没有错误，也创建一个空的日志文件
        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write("未检测到数据质量问题。\n")


    # 保存干净的数据
    if valid_ids:
        print(f"--- 正在保存有效的轨迹到: {cleaned_filepath} ---")
        cleaned_df = df[df['Unique_ID'].isin(valid_ids)].copy()
        cleaned_df.to_csv(cleaned_filepath, index=False, encoding='utf-8-sig')
        print(f"--- 清理后的数据保存成功，共 {len(cleaned_df)} 行。 ---")
    else:
        print("--- 没有有效的轨迹可以保存。 ---")

if __name__ == '__main__':
    main()