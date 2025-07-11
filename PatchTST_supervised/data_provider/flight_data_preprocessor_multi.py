import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from datetime import timedelta
from sklearn.neighbors import LocalOutlierFactor
import os
import glob
# --- 新增的库 ---
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
# from os import cpu_count  # 未使用，已注释

# ==============================================================================
# 用于并行处理的辅助函数与工作函数
# (这部分保持不变, 逻辑已经是最优的单任务处理模式)
# ==============================================================================

def dms_to_decimal(dms):
    """
    将度分秒 (DD.MMSSs) 格式的经纬度转换为十进制度。
    """
    if pd.isna(dms) or not isinstance(dms, (int, float)):
        return np.nan
    try:
        dms_str = f"{dms:.4f}"
        parts = dms_str.split('.')
        if len(parts) != 2 or len(parts[1]) < 4:
            return np.nan
        degrees = int(parts[0])
        minutes = int(parts[1][:2])
        seconds = int(parts[1][2:4])
        return degrees + minutes / 60 + seconds / 3600
    except (ValueError, IndexError):
        return np.nan

def _process_trajectory_group_worker(group_df, ground_altitude_threshold=300):
    """
    工作函数：处理单个飞行器的所有数据（一个原始ID）。
    执行排序、切分以及Unique_ID（唯一ID）的生成。
    """
    all_segments = []
    df_sorted = group_df.sort_values(by='Time').reset_index(drop=True)
    
    if len(df_sorted) < 2:
        if not df_sorted.empty:
            new_df = df_sorted.copy()
            first_point_time = pd.to_datetime(new_df.iloc[0]['Time'])
            time_str = first_point_time.strftime('%m%d%H%M')
            original_id = new_df.iloc[0]['ID']
            new_df['Unique_ID'] = f"{original_id}{time_str}"
            all_segments.append(new_df)
        return all_segments

    split_indices = []
    for i in range(1, len(df_sorted)):
        time_diff = pd.to_datetime(df_sorted.loc[i, 'Time']) - pd.to_datetime(df_sorted.loc[i-1, 'Time'])
        if time_diff > timedelta(minutes=15):
            # if df_sorted.loc[i-1, 'H'] < ground_altitude_threshold and df_sorted.loc[i, 'H'] < ground_altitude_threshold:
            #     split_indices.append(i)
            split_indices.append(i)

    if split_indices:
        original_id = df_sorted['ID'].iloc[0]
        print(f"--- 信息: 原始航班ID {original_id} 已被切分为 {len(split_indices) + 1} 个航段。 ---")

    sub_dfs = []
    start_idx = 0
    for end_idx in split_indices:
        sub_dfs.append(df_sorted.iloc[start_idx:end_idx])
        start_idx = end_idx
    sub_dfs.append(df_sorted.iloc[start_idx:])

    for sub_df in sub_dfs:
        if not sub_df.empty:
            new_df = sub_df.copy()
            first_point_time = pd.to_datetime(new_df.iloc[0]['Time'])
            time_str = first_point_time.strftime('%m%d%H%M')
            original_id = new_df.iloc[0]['ID']
            new_df['Unique_ID'] = f"{original_id}{time_str}"
            all_segments.append(new_df)
            
    return all_segments

def _filter_by_point_count_worker(df, min_point_count=792):
    """工作函数：根据数据点数量过滤单个轨迹（已添加异常处理）。

    Args:
        df: 轨迹数据DataFrame
        min_point_count: 最小点数阈值，默认792点（对应约13.2分钟的1秒采样数据）
    """
    try:
        if df is None or len(df) < min_point_count:
            if df is not None:
                unique_id = df['Unique_ID'].iloc[0]
                print(f"已过滤掉点数不足的轨迹: {unique_id} (点数: {len(df)}, 需要: {min_point_count})")
            return None

        return df

    except Exception as e:
        # 在多进程中，这个打印操作可以帮助我们定位到导致崩溃的具体数据块
        unique_id = df['Unique_ID'].iloc[0] if 'Unique_ID' in df.columns and not df.empty else "Unknown ID"
        print(f"--- 工作进程错误: 在过滤点数时处理轨迹 {unique_id} 失败。原因: {e} ---")
        return None

def _validate_sampling_rate_worker(df):
    """工作函数：验证轨迹数据的采样率是否为1秒。

    Args:
        df: 轨迹数据DataFrame

    Returns:
        df: 如果采样率正确则返回原数据，否则返回None
    """
    try:
        if df is None or len(df) < 2:
            return df

        # 将时间列转换为datetime类型
        df_copy = df.copy()
        df_copy['Time'] = pd.to_datetime(df_copy['Time'])

        # 计算时间间隔
        time_diffs = df_copy['Time'].diff().dropna()

        # 检查是否大部分时间间隔都是1秒（允许一些误差）
        one_second = pd.Timedelta(seconds=1)
        tolerance = pd.Timedelta(milliseconds=100)  # 100ms的容差

        correct_intervals = ((time_diffs >= one_second - tolerance) &
                           (time_diffs <= one_second + tolerance)).sum()
        total_intervals = len(time_diffs)

        # 如果超过95%的间隔都是1秒左右，认为采样率正确
        if correct_intervals / total_intervals >= 0.95:
            return df
        else:
            unique_id = df['Unique_ID'].iloc[0]
            print(f"--- 警告: 轨迹 {unique_id} 的采样率不是1秒 (正确间隔比例: {correct_intervals/total_intervals:.2%}) ---")
            return df  # 仍然返回数据，只是给出警告

    except Exception as e:
        unique_id = df['Unique_ID'].iloc[0] if 'Unique_ID' in df.columns and not df.empty else "Unknown ID"
        print(f"--- 工作进程错误: 验证轨迹 {unique_id} 采样率失败。原因: {e} ---")
        return df  # 出错时仍返回原数据

def _filter_military_worker(df, lon_threshold=2.0, lat_threshold=2.0):
    """工作函数：根据位移过滤单个轨迹（疑似军航）。"""
    if len(df) < 2: return None
    unique_id = df['Unique_ID'].iloc[0]
    print(f" Current Flight ID: {unique_id}")
    start_point = df.iloc[0]
    end_point = df.iloc[-1]
    delta_lon = abs(end_point['Lon'] - start_point['Lon'])
    delta_lat = abs(end_point['Lat'] - start_point['Lat'])
    if delta_lon <= lon_threshold and delta_lat <= lat_threshold: return df
    unique_id = df['Unique_ID'].iloc[0]
    print(f"已过滤掉军航轨迹: {unique_id} (经度差: {delta_lon:.2f}, 纬度差: {delta_lat:.2f})")
    return None

def _interpolate_and_smooth_worker(segment_df):
    """
    工作函数：对单个轨迹段执行插值和平滑处理。
    """
    try:
        proc_df = segment_df.copy()
        proc_df['Time'] = pd.to_datetime(proc_df['Time'])

        if proc_df.duplicated(subset='Time').any():
            numeric_cols = ['Lat', 'Lon', 'H']
            unique_id_val = proc_df['Unique_ID'].iloc[0]
            proc_df = proc_df.groupby('Time')[numeric_cols].mean().reset_index()
            proc_df['Unique_ID'] = unique_id_val

        if len(proc_df) < 2: return None

        proc_df.set_index('Time', inplace=True)
        resampled_df = proc_df.resample("1s").asfreq()
        resampled_df['Unique_ID'] = resampled_df['Unique_ID'].ffill()
        
        numeric_cols_to_interpolate = ['Lat', 'Lon', 'H']
        resampled_df[numeric_cols_to_interpolate] = resampled_df[numeric_cols_to_interpolate].interpolate(method='linear')
        resampled_df.reset_index(inplace=True)

        features = ['Lat', 'Lon', 'H']
        resampled_df.dropna(subset=features, inplace=True)
        if resampled_df.empty: return None

        X = resampled_df[features].values
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.05)
        outlier_mask = lof.fit_predict(X)
        resampled_df.loc[outlier_mask == -1, features] = np.nan
        resampled_df[features] = resampled_df[features].interpolate(method='linear')

        heights = resampled_df['H'].values
        times = resampled_df['Time'].values
        anomaly_results = detect_height_anomalies(heights, times)
        resampled_df.loc[~anomaly_results['mask'], features] = np.nan
        
        # --- 代码修改部分 ---
        # 1. 再次线性插值填充异常段产生的NaN
        resampled_df[features] = resampled_df[features].interpolate(method='linear')
        
        # 2. 使用向后和向前填充来处理边缘的NaN值，而不是删除行
        #    这样可以保证1秒的时间间隔不被破坏
        resampled_df[features] = resampled_df[features].bfill().ffill()
        
        # 之前的代码在这里有一行 resampled_df.dropna(inplace=True)，这是导致问题的根源。
        # 我们已经用上面的 bfill().ffill() 替代了它。

        if len(resampled_df) <= 51:
            flight_id = resampled_df['Unique_ID'].iloc[0] if not resampled_df.empty else "未知"
            print(f"--- 信息: 跳过航班 {flight_id}: 数据点不足 ({len(resampled_df)})，无法进行平滑处理。 ---")
            return None

        window_length, polyorder = 51, 2
        resampled_df['Lat'] = savgol_filter(resampled_df['Lat'], window_length, polyorder)
        resampled_df['Lon'] = savgol_filter(resampled_df['Lon'], window_length, polyorder)
        resampled_df['H'] = savgol_filter(resampled_df['H'], window_length, polyorder)
        
        return resampled_df
    except Exception as e:
        flight_id = segment_df['Unique_ID'].iloc[0] if not segment_df.empty else "未知"
        print(f"--- 工作进程错误: 处理航班 {flight_id} 失败。原因: {e} ---")
        return None

def detect_height_anomalies(heights, times, max_rate=100, max_duration=180):
    mask = np.ones(len(heights), dtype=bool)
    # ... (此函数无需修改)
    height_diff = np.diff(heights)
    rates = height_diff
    anomaly_starts = np.where(np.abs(rates) > max_rate)[0]

    if len(anomaly_starts) == 0:
        return {'is_valid': True, 'mask': mask, 'anomaly_info': []}

    i = 0
    while i < len(anomaly_starts):
        start_idx = anomaly_starts[i] - 1 if anomaly_starts[i] > 0 else 0
        start_height = heights[start_idx]
        start_time = pd.Timestamp(times[start_idx]).to_pydatetime()
        start_rate = rates[anomaly_starts[i]]
        start_direction = np.sign(start_rate)
        j = anomaly_starts[i] + 1
        search_count = 0
        end_idx = anomaly_starts[i]
        found_end = False

        while j < len(heights) - 1:
            if search_count >= max_duration:
                i = np.searchsorted(anomaly_starts, j+1)
                found_end = True
                break
            if abs(rates[j]) > max_rate:
                current_direction = np.sign(rates[j])
                if current_direction == -start_direction:
                    end_idx = j
                    current_height = heights[j]
                    current_time = pd.Timestamp(times[j]).to_pydatetime()
                    time_diff = (current_time - start_time).total_seconds()
                    slope = (current_height - start_height) / time_diff if time_diff > 0 else 0

                    if abs(slope) <= max_rate:
                        k = j + 1
                        while k < len(heights) - 1:
                            end_idx = k
                            if abs(rates[k]) > max_rate and np.sign(rates[k]) == current_direction:
                                j = k
                                k += 1
                            else:
                                break
                        mask[start_idx + 1:end_idx - 1] = False
                        found_end = True
                        break
            j += 1
            search_count += 1
        if not found_end:
            break
        i = np.searchsorted(anomaly_starts, end_idx + 1)
    return {'is_valid': True, 'mask': mask, 'anomaly_info': []}


# ==============================================================================
# 主数据处理流程
# (这部分负责管理和分发并行任务)
# ==============================================================================

def load_and_process_files(directory_path, output_directory):
    """第一阶段：数据聚合与初始解析（此部分无需更改）"""
    all_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not all_files:
        print(f"在目录中未找到CSV文件: {directory_path}")
        return pd.DataFrame()
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file, encoding='gbk', low_memory=False)
            df_list.append(df)
        except UnicodeDecodeError:
            try:
                print(f"文件 {os.path.basename(file)} 使用GBK解码失败，正在尝试UTF-8...")
                df = pd.read_csv(file, encoding='utf-8', low_memory=False)
                df_list.append(df)
            except Exception as e:
                print(f"无法使用UTF-8或GBK读取文件 {os.path.basename(file)}。错误: {e}")
    if not df_list:
        print("所有文件都加载失败。请检查文件编码。")
        return pd.DataFrame()
    full_df = pd.concat(df_list, ignore_index=True)
    if not full_df.empty:
        merged_filename = os.path.join(output_directory, "_00_merged_raw_data.csv")
        print(f"\n--- 正在保存合并后的原始数据到 {merged_filename} ---")
        full_df.to_csv(merged_filename, index=False, encoding='utf-8-sig')
        print("--- 合并后的原始数据保存成功。---\n")
    initial_rows = len(full_df)
    required_cols = ['JD', 'WD', 'H']
    if all(col in full_df.columns for col in required_cols):
        full_df = full_df[(full_df['JD'] != 0) & (full_df['WD'] != 0) & (full_df['H'] != 0)].copy()
        rows_after_zero_filter = len(full_df)
        print(f"移除了 {initial_rows - rows_after_zero_filter} 行在'JD', 'WD', 或 'H'列中为0的数据。")
    else:
        print(f"警告: 列 {required_cols} 中的一个或多个未找到。跳过零值过滤。")
    full_df['Time'] = pd.to_datetime(full_df['DTRQ'], format='%d-%b-%y %I.%M.%S.%f000 %p', errors='coerce')
    full_df['Lon'] = full_df['JD'].apply(dms_to_decimal)
    full_df['Lat'] = full_df['WD'].apply(dms_to_decimal)
    full_df.rename(columns={'P1': 'ID', 'H': 'H'}, inplace=True)
    full_df.dropna(subset=['ID', 'H', 'Lon', 'Lat', 'Time'], inplace=True)
    df = full_df[['ID', 'H', 'Lon', 'Lat', 'Time']].copy()
    df['Time'] = df['Time'].dt.strftime('%Y%m%d %H:%M:%S.%f').str[:-3]

    # --- 新增功能: 根据用户反馈，在此处保存包含5个核心列的中间文件 ---
    if not df.empty:
        intermediate_filename = os.path.join(output_directory, "_01_initial_parsed_data.csv")
        print(f"\n--- 正在保存初始解析和转换后的数据到 {intermediate_filename} ---")
        df.to_csv(intermediate_filename, index=False, encoding='utf-8-sig')
        print(f"--- 初始数据保存成功，共 {len(df)} 行。---\n")

    return df

def filter_by_geographic_and_altitude_range(df, h_min, h_max, lon_min, lon_max, lat_min, lat_max):
    """
    新增阶段：根据地理和高度范围过滤轨迹。
    只删除超出范围的数据行，而不是整个ID。
    """
    print("\n--- 新增阶段: 开始基于地理和高度范围的过滤 ---")
    
    # 使用从命令行传入的参数
    print(f"--- 过滤参数: H=[{h_min}, {h_max}], Lon=[{lon_min}, {lon_max}], Lat=[{lat_min}, {lat_max}] ---")
    
    initial_rows = len(df)

    # 保留在范围内的行
    filtered_df = df[
        (df['H'] >= h_min) & (df['H'] <= h_max) &
        (df['Lon'] >= lon_min) & (df['Lon'] <= lon_max) &
        (df['Lat'] >= lat_min) & (df['Lat'] <= lat_max)
    ].copy()
    
    final_rows = len(filtered_df)
    rows_removed = initial_rows - final_rows
    
    if rows_removed > 0:
        print(f"--- 移除了 {rows_removed} 行超出地理或高度范围的数据。 ---")
    else:
        print("--- 所有数据点均在指定的地理和高度范围内。无需删除。 ---")

    print(f"--- 新增阶段: 完成。过滤前共有 {initial_rows} 行，过滤后剩余 {final_rows} 行。 ---")
    
    return filtered_df

def _parallel_executor(worker_func, items_to_process, max_workers):
    """通用辅助函数，用于并行运行任何工作函数。"""
    results_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(worker_func, items_to_process)
        for result in results:
            if result is not None:
                if isinstance(result, list):
                    results_list.extend(result)
                else:
                    results_list.append(result)
    return results_list

def sort_and_split_trajectory(df, ground_altitude_threshold=300, max_workers=16):
    """
    第二阶段：轨迹排序与切分 (重点并行优化版)
    此函数将取代原有的 for 循环，将每个ID的数据分发到不同进程中处理。
    """
    # **微优化**: 在分组前排序可以提升groupby性能
    df = df.sort_values('ID')
    
    # 准备任务列表：每个任务是拥有相同ID的所有数据点
    # sort=False 告诉groupby不要再次排序，因为我们已经手动排过了
    groups = [group for _, group in df.groupby('ID', sort=False)]
    
    print(f"--- 准备将 {len(groups)} 个独立ID的轨迹数据分发到 {max_workers} 个进程中进行切分... ---")
    
    # 使用 functools.partial 创建一个“内嵌”了阈值参数的新函数
    worker = partial(_process_trajectory_group_worker, ground_altitude_threshold=ground_altitude_threshold)
    
    # 并行执行所有任务
    all_segments = _parallel_executor(worker, groups, max_workers)
                
    return all_segments

def filter_short_trajectories_by_points(segments, min_point_count=792, max_workers=16):
    """并行地按数据点数量过滤轨迹。"""
    worker = partial(_filter_by_point_count_worker, min_point_count=min_point_count)
    return _parallel_executor(worker, segments, max_workers)

def validate_sampling_rates(segments, max_workers=16):
    """并行地验证轨迹的采样率。"""
    return _parallel_executor(_validate_sampling_rate_worker, segments, max_workers)

def filter_military_flights(segments, lon_threshold=2.0, lat_threshold=2.0, max_workers=16):
    """并行地过滤疑似军航的轨迹。"""
    worker = partial(_filter_military_worker, lon_threshold=lon_threshold, lat_threshold=lat_threshold)
    return _parallel_executor(worker, segments, max_workers)

def interpolate_and_smooth(trajectory_segments, max_workers=16):
    """第三阶段：数据插值与平滑 (并行版)"""
    return _parallel_executor(_interpolate_and_smooth_worker, trajectory_segments, max_workers)

def main(input_directory, output_directory, max_workers, force_regenerate,
         h_min, h_max, lon_min, lon_max, lat_min, lat_max):
    """主函数，执行整个预处理流程。"""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"已创建输出目录: {output_directory}")

    intermediate_filepath = os.path.join(output_directory, "_01_initial_parsed_data.csv")

    # 根据参数和文件存在情况决定如何加载第一阶段数据
    if not force_regenerate and os.path.exists(intermediate_filepath):
        print(f"--- 检测到已存在的初始解析文件，将直接读取: {intermediate_filepath} ---")
        initial_df = pd.read_csv(intermediate_filepath, low_memory=False)
        print(f"--- 从缓存文件加载了 {len(initial_df)} 条记录。 ---")
    else:
        if force_regenerate:
            print("--- 用户指定了 --force-regenerate，将强制重新生成初始数据。 ---")
        else:
            print(f"--- 未找到初始解析文件，将从原始数据开始处理。 ---")
        
        print("\n--- 阶段一: 开始数据聚合与初始解析 ---")
        initial_df = load_and_process_files(input_directory, output_directory)
        if initial_df.empty:
            print("--- 阶段一: 失败。没有可处理的数据。 ---")
            return
        print(f"--- 阶段一: 完成。初始解析后共加载 {len(initial_df)} 条记录。 ---")

    # 在轨迹切分前进行地理和高度过滤
    filtered_df = filter_by_geographic_and_altitude_range(
        initial_df, h_min, h_max, lon_min, lon_max, lat_min, lat_max
    )
    if filtered_df.empty:
        print("--- 地理范围过滤后，没有可处理的数据。 ---")
        return

    print(f"\n--- 阶段二: 开始轨迹切分与初步过滤 (使用 {max_workers} 个进程) ---")
    trajectory_segments = sort_and_split_trajectory(filtered_df, max_workers=max_workers)
    print(f"--- 轨迹切分完成。共生成 {len(trajectory_segments)} 个潜在的轨迹段。 ---")

    # --- 新增功能: 根据用户请求，在军航过滤前保存轨迹段 ---
    if trajectory_segments:
        split_segments_df = pd.concat(trajectory_segments, ignore_index=True)
        # 确保时间列是字符串格式，以便于CSV存储
        if 'Time' in split_segments_df.columns and pd.api.types.is_datetime64_any_dtype(split_segments_df['Time']):
            split_segments_df['Time'] = split_segments_df['Time'].dt.strftime('%Y%m%d %H:%M:%S.%f').str[:-3]
        
        output_filename_before_military = os.path.join(output_directory, "_02_after_split_before_military_filter.csv")
        print(f"\n--- 正在保存在军航过滤前的轨迹段到 {output_filename_before_military} ---")
        split_segments_df.to_csv(output_filename_before_military, index=False, encoding='utf-8-sig')
        print(f"--- 保存成功，共 {len(split_segments_df)} 行，{split_segments_df['Unique_ID'].nunique()} 个轨迹段。 ---\n")

    # 先过滤军航，再进行插值平滑
    filtered_segments = filter_military_flights(trajectory_segments, max_workers=max_workers)
    print(f"--- 军航过滤后剩余 {len(filtered_segments)} 个航段。 ---")
    print(f"--- 阶段二: 完成。 ---")

    print(f"\n--- 阶段三: 开始插值与平滑处理 (使用 {max_workers} 个进程) ---")
    smoothed_trajectories = interpolate_and_smooth(filtered_segments, max_workers=max_workers)
    print(f"--- 插值与平滑处理完成。共处理了 {len(smoothed_trajectories)} 个航段。 ---")

    # 验证采样率
    print(f"--- 开始验证采样率 (使用 {max_workers} 个进程) ---")
    validated_trajectories = validate_sampling_rates(smoothed_trajectories, max_workers=max_workers)
    print(f"--- 采样率验证完成。 ---")

    # 最后根据点数过滤短轨迹（792点对应约13.2分钟）
    print(f"--- 开始根据点数过滤短轨迹 (最小792点) (使用 {max_workers} 个进程) ---")
    final_trajectories = filter_short_trajectories_by_points(validated_trajectories, min_point_count=792, max_workers=max_workers)
    print(f"--- 阶段三: 完成。点数过滤后剩余 {len(final_trajectories)} 个航段。 ---")

    print("\n--- 正在保存已处理的轨迹 ---")
    if final_trajectories:
        full_processed_df = pd.concat(final_trajectories, ignore_index=True)
        if 'ID' in full_processed_df.columns:
             full_processed_df.drop(columns=['ID'], inplace=True, errors='ignore')
        full_processed_df.rename(columns={'Unique_ID': 'ID'}, inplace=True)
        final_columns = ['ID', 'H', 'Lon', 'Lat', 'Time']
        full_processed_df = full_processed_df[final_columns]
        full_processed_df['Time'] = pd.to_datetime(full_processed_df['Time']).dt.strftime('%Y%m%d %H:%M:%S.%f').str[:-3]
        output_filename = os.path.join(output_directory, "processed_trajectories_multi.csv")
        full_processed_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"--- 所有阶段完成。已保存 {full_processed_df['ID'].nunique()} 条处理后的轨迹到 {output_filename} ---")
    else:
        print("--- 所有阶段完成。没有可保存的轨迹。 ---")

if __name__ == '__main__':
    # --- 配置路径与参数解析 ---
    parser = argparse.ArgumentParser(description="飞行数据预处理脚本")
    parser.add_argument('--input_dir', type=str, default='./dataset/raw/', help='包含原始CSV文件的输入目录路径')
    parser.add_argument('--output_dir', type=str, default='./dataset/processed_data/', help='用于存放处理后数据的输出目录路径')
    parser.add_argument('--max_workers', type=int, default=16, help='用于并行处理的最大工作进程数')
    parser.add_argument('--force_regenerate', action='store_true', help='如果设置此项，则强制重新生成_01_initial_parsed_data.csv文件，即使它已存在')
    
    # --- 新增：地理和高度过滤参数 ---
    parser.add_argument('--h_min', type=float, default=0, help='最低高度 (米)')
    parser.add_argument('--h_max', type=float, default=20000, help='最高高度 (米)')
    parser.add_argument('--lon_min', type=float, default=110, help='最小经度')
    parser.add_argument('--lon_max', type=float, default=120, help='最大经度')
    parser.add_argument('--lat_min', type=float, default=33, help='最小纬度')
    parser.add_argument('--lat_max', type=float, default=42, help='最大纬度')

    args = parser.parse_args()

    pd.set_option('future.no_silent_downcasting', True)

    if args.input_dir == 'path/to/your/raw_csv_directory' or args.output_dir == 'path/to/your/processed_data/':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 请检查 --input_dir 和 --output_dir 参数是否已正确设置 !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main(
            input_directory=args.input_dir,
            output_directory=args.output_dir,
            max_workers=args.max_workers,
            force_regenerate=args.force_regenerate,
            h_min=args.h_min,
            h_max=args.h_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            lat_min=args.lat_min,
            lat_max=args.lat_max
        )