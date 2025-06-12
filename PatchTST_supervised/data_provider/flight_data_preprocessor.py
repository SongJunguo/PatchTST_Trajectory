import pandas as pd
from scipy.signal import savgol_filter
import numpy as np
from datetime import datetime, timedelta
from sklearn.neighbors import LocalOutlierFactor
import os
import glob

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

def load_and_process_files(directory_path, output_directory):
    """
    第一阶段：数据聚合与初始解析
    """
    all_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not all_files:
        print(f"No CSV files found in directory: {directory_path}")
        return pd.DataFrame()

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file, encoding='utf-8', low_memory=False)
            df_list.append(df)
        except UnicodeDecodeError:
            try:
                print(f"UTF-8 decoding failed for {os.path.basename(file)}, trying GBK...")
                df = pd.read_csv(file, encoding='gbk', low_memory=False)
                df_list.append(df)
            except Exception as e:
                print(f"Could not read file {os.path.basename(file)} with UTF-8 or GBK. Error: {e}")

    if not df_list:
        print("All files failed to load. Please check file encodings.")
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)

    if not full_df.empty:
        merged_filename = os.path.join(output_directory, "_00_merged_raw_data.csv")
        print(f"\n--- Saving merged raw data to {merged_filename} ---")
        full_df.to_csv(merged_filename, index=False, encoding='utf-8-sig')
        print("--- Merged raw data saved successfully. ---\n")

    initial_rows = len(full_df)
    required_cols = ['JD', 'WD', 'H']
    if all(col in full_df.columns for col in required_cols):
        full_df = full_df[(full_df['JD'] != 0) & (full_df['WD'] != 0) & (full_df['H'] != 0)].copy()
        rows_after_zero_filter = len(full_df)
        print(f"Removed {initial_rows - rows_after_zero_filter} rows with 0 in JD, WD, or H.")
    else:
        print(f"Warning: One or more of the columns {required_cols} not found. Skipping zero-value filtering.")

    full_df['Time'] = pd.to_datetime(full_df['DTRO'], format='%d-%b-%y %I.%M.%S.%f000 %p', errors='coerce')
    full_df['Lon'] = full_df['JD'].apply(dms_to_decimal)
    full_df['Lat'] = full_df['WD'].apply(dms_to_decimal)
    full_df.rename(columns={'PI': 'ID', 'H': 'H'}, inplace=True)
    full_df.dropna(subset=['ID', 'H', 'Lon', 'Lat', 'Time'], inplace=True)
    df = full_df[['ID', 'H', 'Lon', 'Lat', 'Time']].copy()
    df['Time'] = df['Time'].dt.strftime('%Y%m%d %H:%M:%S.%f').str[:-3]
    
    return df

def sort_and_split_trajectory(df, ground_altitude_threshold=300):
    """
    第二阶段：轨迹排序与切分 (重构版)
    ---
    采用 groupby 的方式，对每个ID的轨迹独立进行切分。
    这确保了不同ID的轨迹不会相互干扰，从根本上解决所有数据被归为一个ID的问题。
    """
    all_segments = []
    # 按 'ID' 分组，对每个飞行器的轨迹独立处理
    for original_id, group in df.groupby('ID'):
        # 对当前飞行器的轨迹按时间排序
        df_sorted = group.sort_values(by='Time').reset_index(drop=True)
        
        if len(df_sorted) < 2:
            # 如果轨迹点少于2个，无法切分，直接生成Unique_ID并加入结果
            if not df_sorted.empty:
                new_df = df_sorted.copy()
                first_point_time = pd.to_datetime(new_df.iloc[0]['Time'])
                time_str = first_point_time.strftime('%m%d%H%M')
                # original_id 已经从 groupby 中获取
                new_df['Unique_ID'] = f"{original_id}{time_str}"
                all_segments.append(new_df)
            continue

        split_indices = []
        for i in range(1, len(df_sorted)):
            time_diff = pd.to_datetime(df_sorted.loc[i, 'Time']) - pd.to_datetime(df_sorted.loc[i-1, 'Time'])
            if time_diff > timedelta(minutes=10):
                if df_sorted.loc[i-1, 'H'] < ground_altitude_threshold and df_sorted.loc[i, 'H'] < ground_altitude_threshold:
                    split_indices.append(i)

        # 新增：如果轨迹被切分，打印原始ID
        if split_indices:
            print(f"--- INFO: Original flight ID {original_id} was split into {len(split_indices) + 1} segments. ---")

        # 根据找到的切分点，将当前ID的轨迹切分为子段
        sub_dfs = []
        start_idx = 0
        for end_idx in split_indices:
            sub_dfs.append(df_sorted.iloc[start_idx:end_idx])
            start_idx = end_idx
        sub_dfs.append(df_sorted.iloc[start_idx:])

        # 为每个子段生成新的 Unique_ID
        for sub_df in sub_dfs:
            if not sub_df.empty:
                new_df = sub_df.copy()
                first_point_time = pd.to_datetime(new_df.iloc[0]['Time'])
                time_str = first_point_time.strftime('%m%d%H%M')
                original_id = new_df.iloc[0]['ID']
                new_df['Unique_ID'] = f"{original_id}{time_str}"
                all_segments.append(new_df)
                
    return all_segments

def filter_military_flights(trajectory_segments, lon_threshold=200.0, lat_threshold=200.0):
    """
    军航轨迹过滤
    """
    filtered_segments = []
    for df in trajectory_segments:
        if len(df) < 2:
            continue
        start_point = df.iloc[0]
        end_point = df.iloc[-1]
        delta_lon = abs(end_point['Lon'] - start_point['Lon'])
        delta_lat = abs(end_point['Lat'] - start_point['Lat'])
        if delta_lon <= lon_threshold and delta_lat <= lat_threshold:
            filtered_segments.append(df)
        else:
            unique_id = df['Unique_ID'].iloc[0]
            print(f"Filtered out military flight: {unique_id} (dLon: {delta_lon:.2f}, dLat: {delta_lat:.2f})")
    return filtered_segments

def filter_short_trajectories(trajectory_segments, min_duration_minutes=15):
    """
    新增：根据持续时间过滤轨迹段。
    """
    long_enough_segments = []
    min_duration = timedelta(minutes=min_duration_minutes)
    for df in trajectory_segments:
        if len(df) < 2:
            continue
        
        start_time = pd.to_datetime(df.iloc[0]['Time'])
        end_time = pd.to_datetime(df.iloc[-1]['Time'])
        duration = end_time - start_time
        
        if duration >= min_duration:
            long_enough_segments.append(df)
        else:
            unique_id = df['Unique_ID'].iloc[0]
            print(f"Filtered out short trajectory: {unique_id} (Duration: {duration})")
    return long_enough_segments

def detect_height_anomalies(heights, times, max_rate=100, max_duration=180):
    """
    检测高度异常段（急速上升或下降）
    """
    mask = np.ones(len(heights), dtype=bool)
    anomaly_segments = []
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
                        anomaly_segments.append({
                            'start_idx': start_idx, 'end_idx': j, 'duration': time_diff,
                            'height_change': current_height - start_height, 'overall_rate': abs(slope)
                        })
                        found_end = True
                        break
            j += 1
            search_count += 1
        if not found_end:
            break
        i = np.searchsorted(anomaly_starts, end_idx + 1)
    return {'is_valid': True, 'mask': mask, 'anomaly_info': anomaly_segments}

def interpolate_and_smooth(trajectory_segments):
    """
    第三阶段：数据插值与平滑
    ---
    已更新：移除 'traffic' 库的依赖，并直接使用原始列名进行操作。
    ---
    """
    processed_flights = []

    for segment_df in trajectory_segments:
        # 复制以避免 SettingWithCopyWarning
        proc_df = segment_df.copy()

        # 1. 准备数据：确保时间格式正确
        proc_df['Time'] = pd.to_datetime(proc_df['Time'])

        # --- 新增：处理重复时间戳 ---
        # 在设置索引之前，通过对重复时间戳的数据取平均值来消除它们
        if proc_df.duplicated(subset='Time').any():
            numeric_cols = ['Lat', 'Lon', 'H']
            # 保存 Unique_ID，因为它不能被平均
            unique_id_val = proc_df['Unique_ID'].iloc[0]
            # 按时间分组并计算平均值
            proc_df = proc_df.groupby('Time')[numeric_cols].mean().reset_index()
            # 将 Unique_ID 添加回来
            proc_df['Unique_ID'] = unique_id_val

        if len(proc_df) < 2:
            flight_id = proc_df['Unique_ID'].iloc[0] if not proc_df.empty else "Unknown"
            print(f"--- INFO: Skipping flight {flight_id} in Stage 3: less than 2 data points. ---")
            continue

        # 2. 使用 pandas 进行重采样与基础插值
        try:
            proc_df.set_index('Time', inplace=True)
            resampled_df = proc_df.resample("1s").asfreq()
            # 按照 FutureWarning 的建议更新代码
            resampled_df['Unique_ID'] = resampled_df['Unique_ID'].ffill()
            
            # 只对数值列进行插值，避免对 'Unique_ID' (object类型) 插值产生警告
            numeric_cols_to_interpolate = ['Lat', 'Lon', 'H']
            resampled_df[numeric_cols_to_interpolate] = resampled_df[numeric_cols_to_interpolate].interpolate(method='linear')
            
            resampled_df.reset_index(inplace=True)
        except Exception as e:
            flight_id = segment_df['Unique_ID'].iloc[0]
            print(f"Could not resample or interpolate flight {flight_id}: {e}")
            continue


        # 3. 离群点检测 (LOF)
        features = ['Lat', 'Lon', 'H']
        resampled_df.dropna(subset=features, inplace=True)
        if resampled_df.empty:
            flight_id = proc_df.iloc[0]['Unique_ID']
            print(f"--- INFO: Skipping flight {flight_id} in Stage 3: no valid data after initial dropna. ---")
            continue

        X = resampled_df[features].values
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        outlier_mask = lof.fit_predict(X)
        resampled_df.loc[outlier_mask == -1, features] = np.nan

        # 4. 插值 (填充离群点)
        resampled_df[features] = resampled_df[features].interpolate(method='linear')

        # 5. 异常高度变化段去除
        heights = resampled_df['H'].values
        times = resampled_df['Time'].values
        anomaly_results = detect_height_anomalies(heights, times)
        resampled_df.loc[~anomaly_results['mask'], features] = np.nan

        # 6. 再次插值 (填充异常段)
        resampled_df[features] = resampled_df[features].interpolate(method='linear')
        resampled_df.dropna(inplace=True)

        if len(resampled_df) <= 51:
            flight_id = resampled_df['Unique_ID'].iloc[0] if not resampled_df.empty else "Unknown"
            print(f"--- INFO: Skipping flight {flight_id} in Stage 3: not enough data points ({len(resampled_df)}) for smoothing. ---")
            continue

        # 7. 轨迹平滑 (Savitzky-Golay)
        window_length = 51
        polyorder = 2
        resampled_df['Lat'] = savgol_filter(resampled_df['Lat'], window_length, polyorder)
        resampled_df['Lon'] = savgol_filter(resampled_df['Lon'], window_length, polyorder)
        resampled_df['H'] = savgol_filter(resampled_df['H'], window_length, polyorder)
        
        # 8. 无需改回列名
        processed_flights.append(resampled_df)

    return processed_flights

def main(input_directory, output_directory):
    """
    主函数，执行整个预处理流程。
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    print("--- Stage 1: Starting Data Aggregation and Initial Parsing ---")
    initial_df = load_and_process_files(input_directory, output_directory)
    if initial_df.empty:
        print("--- Stage 1: Failed. No data to process. ---")
        return
    print(f"--- Stage 1: Completed. Loaded {len(initial_df)} records after initial parsing. ---")

    print("\n--- Stage 2: Starting Trajectory Splitting and Filtering ---")
    # 修正：传入地面高度阈值参数
    trajectory_segments = sort_and_split_trajectory(initial_df)
    print(f"--- Found {len(trajectory_segments)} potential trajectory segments after splitting. ---")

    # 新增：根据持续时间过滤轨迹段
    duration_filtered_segments = filter_short_trajectories(trajectory_segments, min_duration_minutes=15)
    print(f"--- {len(duration_filtered_segments)} segments remaining after duration filtering (< 15 min). ---")
    
    filtered_segments = filter_military_flights(duration_filtered_segments)
    print(f"--- Stage 2: Completed. {len(filtered_segments)} segments remaining after military flight filtering. ---")

    print("\n--- Stage 3: Starting Interpolation and Smoothing ---")
    final_trajectories = interpolate_and_smooth(filtered_segments)
    print(f"--- Stage 3: Completed. {len(final_trajectories)} segments processed. ---")

    print("\n--- Saving processed trajectories ---")
    if final_trajectories:
        # 合并所有处理好的轨迹到一个 DataFrame
        full_processed_df = pd.concat(final_trajectories, ignore_index=True)

        # --- 按要求格式化最终输出 ---
        # 1. 明确删除原始的 'ID' 列
        full_processed_df.drop(columns=['ID'], inplace=True)
        
        # 2. 将 'Unique_ID' 重命名为 'ID'
        full_processed_df.rename(columns={'Unique_ID': 'ID'}, inplace=True)
        
        # 3. 定义并应用最终的列顺序，确保为5列
        final_columns = ['ID', 'H', 'Lon', 'Lat', 'Time']
        full_processed_df = full_processed_df[final_columns]
        
        # 4. 格式化时间列以匹配 'YYYYMMDD HH:MM:SS.ms'
        full_processed_df['Time'] = full_processed_df['Time'].dt.strftime('%Y%m%d %H:%M:%S.%f').str[:-3]
        # --- 结束 ---
 
        output_filename = os.path.join(output_directory, "processed_trajectories.csv")
        # 保存为单个文件
        full_processed_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"--- All stages complete. Saved {len(final_trajectories)} processed trajectories to {output_filename} ---")
    else:
        print("--- All stages complete. No trajectories to save. ---")

if __name__ == '__main__':
    # --- 配置路径 ---
    # !!! 用户需要修改这里的路径 !!!
    INPUT_DIR = './dataset/raw/'
    OUTPUT_DIR = './dataset/processed_data/'

    if INPUT_DIR == 'path/to/your/raw_csv_directory' or OUTPUT_DIR == 'path/to/your/processed_data/':
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 请先修改脚本中的 INPUT_DIR 和 OUTPUT_DIR 变量 !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main(input_directory=INPUT_DIR, output_directory=OUTPUT_DIR)