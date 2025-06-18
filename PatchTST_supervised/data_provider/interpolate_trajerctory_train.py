import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
from sklearn.neighbors import LocalOutlierFactor
from scipy.interpolate import CubicSpline, interp1d

def is_five_digit_id(filename):
    """
    检查文件名是否为5位数ID
    :param filename: 文件名（不包含扩展名）
    :return: 如果是5位数ID返回True，否则返回False
    """
    # 使用正则表达式匹配5位数字
    pattern = r'^\d{5}$'
    return bool(re.match(pattern, filename))

def detect_outliers(data, n_neighbors=50, distance_threshold=None):
    """
    使用基于距离阈值的方法检测离群点
    :param data: numpy array，包含[高度,经度,纬度]数据
    :param n_neighbors: 近邻数量
    :param distance_threshold: 距离阈值，超过此阈值的点被视为异常点，None表示自动计算
    :return: 布尔数组，True表示正常点，False表示离群点
    """
    if len(data) <= n_neighbors:
        # 数据点太少，返回全部为正常点
        return np.ones(len(data), dtype=bool)
    
    # 计算每个点到其k近邻的平均距离
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(data))).fit(data)
    distances, _ = nbrs.kneighbors(data)
    avg_distances = np.mean(distances, axis=1)
    
    if distance_threshold is None:
        # 如果没有提供距离阈值，使用数据的统计特性自动计算
        # 使用IQR(四分位距)方法设置阈值
        q1 = np.percentile(avg_distances, 25)
        q3 = np.percentile(avg_distances, 75)
        iqr = q3 - q1
        distance_threshold = q3 + 1.5 * iqr
        print(f"自动计算的距离阈值: {distance_threshold:.4f}")
    
    # 识别异常点
    mask = avg_distances <= distance_threshold
    print(f"检测到 {np.sum(~mask)} 个离群点 (阈值: {distance_threshold:.4f})")
    return mask

def calculate_ground_speed(lat1, lon1, lat2, lon2):
    """
    计算两点之间的地面速度（米/秒）
    由于是插值后的数据，时间间隔固定为1秒
    :param lat1, lon1: 第一个点的经纬度（度）
    :param lat2, lon2: 第二个点的经纬度（度）
    :return: 地面速度（米/秒）
    """
    R = 6371000  # 地球半径（米）
    
    # 将角度转换为弧度
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # 计算经纬度差
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine公式计算距离
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    
    # 由于时间间隔固定为1秒，所以速度就是距离
    return distance

def split_trajectory_by_speed(df, max_ground_speed=250, max_vertical_speed=50, min_points=10):
    """
    根据速度限制切分轨迹
    由于是插值后的数据，时间间隔固定为1秒
    :param df: 轨迹数据DataFrame
    :param max_ground_speed: 最大地面速度（米/秒）
    :param max_vertical_speed: 最大垂直速度（米/秒）
    :param min_points: 最小有效点数，小于此值的轨迹将被废弃
    :return: 切分后的轨迹列表
    """
    if len(df) < 2:
        return [df]
    
    # 计算地面速度和垂直速度
    speeds = []
    for i in range(1, len(df)):
        lat1, lon1 = df.iloc[i-1]['Lat'], df.iloc[i-1]['Lon']
        lat2, lon2 = df.iloc[i]['Lat'], df.iloc[i]['Lon']
        
        # 计算地面速度（水平距离）
        ground_speed = calculate_ground_speed(lat1, lon1, lat2, lon2)
        # 计算垂直速度（高度变化绝对值）
        vertical_speed = abs(df.iloc[i]['H'] - df.iloc[i-1]['H'])  # 时间间隔为1秒，所以速度就是高度差
        
        speeds.append((ground_speed, vertical_speed))
    
    # 找出需要切分的位置（超过速度限制的点）
    split_indices = []
    for i, (ground_speed, vertical_speed) in enumerate(speeds):
        if ground_speed > max_ground_speed or vertical_speed > max_vertical_speed:
            split_indices.append(i+1)  # +1是因为速度计算是基于i-1和i点
    
    # 如果没有需要切分的位置，返回原始轨迹
    if not split_indices:
        return [df]
    
    # 切分轨迹
    split_indices = [0] + split_indices + [len(df)]
    trajectories = []
    for i in range(len(split_indices)-1):
        start_idx = split_indices[i]
        end_idx = split_indices[i+1]
        
        # 检查轨迹点数是否足够
        if end_idx - start_idx < min_points:
            print(f"轨迹段点数不足{min_points}，将被废弃")
            continue
            
        traj = df.iloc[start_idx:end_idx].copy()
        trajectories.append(traj)
    
    return trajectories

def split_trajectory_by_time(df, time_gap_threshold=600):
    """
    根据时间间隔切分轨迹
    :param df: 轨迹数据DataFrame
    :param time_gap_threshold: 时间间隔阈值（秒）
    :return: 切分后的轨迹列表
    """
    if len(df) < 2:
        return [df]
    
    # 计算相邻点之间的时间差（秒）
    time_diff = df['Time'].diff().dt.total_seconds()
    
    # 找出需要切分的位置（时间间隔大于阈值的点）
    split_indices = time_diff[time_diff > time_gap_threshold].index.tolist()
    
    # 如果没有需要切分的位置，返回原始轨迹
    if not split_indices:
        return [df]
    
    # 切分轨迹
    split_indices = [0] + split_indices + [len(df)]
    trajectories = []
    for i in range(len(split_indices)-1):
        start_idx = split_indices[i]
        end_idx = split_indices[i+1]
        traj = df.iloc[start_idx:end_idx].copy()
        trajectories.append(traj)
    
    return trajectories

def generate_new_id(original_id, total_splits, current_index):
    """
    生成新的ID
    :param original_id: 原始ID
    :param total_splits: 总分割数
    :param current_index: 当前索引（从1开始）
    :return: 新的ID
    """
    # 格式化数字部分
    total_str = f"{total_splits:03d}"  # 总段数，3位数字，如025
    index_str = f"{current_index:03d}"  # 当前段序号，3位数字，如001
    
    # 组合新的ID: 原ID + 总段数(3位) + 当前段序号(3位)
    new_id = f"{original_id}{total_str}{index_str}"
    return new_id

def detect_height_anomalies(heights, times, max_rate=100, max_duration=180):
    """
    检测高度异常段（急速上升或下降）
    :param heights: 高度数据序列（已经是1s间隔的数据）
    :param times: 对应的时间序列
    :param max_rate: 最大允许的高度变化率（米/秒）
    :param max_duration: 最大允许的异常持续时间（秒）
    :return: 字典，包含：
        - 'is_valid': 布尔值，表示该轨迹是否有效
        - 'mask': 布尔数组，True表示正常点，False表示异常点
        - 'anomaly_info': 异常段信息列表
    """
    # 初始化掩码（所有点默认为正常）
    mask = np.ones(len(heights), dtype=bool)
    anomaly_segments = []

    # 计算高度变化率（米/秒）
    height_diff = np.diff(heights)
    rates = height_diff  # 因为已经是1s间隔的数据

    # 找出超过阈值的点
    anomaly_starts = np.where(np.abs(rates) > max_rate)[0]

    # 如果没有异常点，直接返回
    if len(anomaly_starts) == 0:
        return {'is_valid': True, 'mask': mask, 'anomaly_info': []}

    # 分析每个异常点
    i = 0
    while i < len(anomaly_starts):
        # 找到异常开始前的正常点
        start_idx = anomaly_starts[i] - 1 if anomaly_starts[i] > 0 else 0
        start_height = heights[start_idx]
        # 将numpy时间转换为datetime对象
        start_time = pd.Timestamp(times[start_idx]).to_pydatetime()

        # 获取开始跳变的方向
        start_rate = rates[anomaly_starts[i]]
        start_direction = np.sign(start_rate)

        # 寻找异常结束点
        j = anomaly_starts[i] + 1
        search_count = 0  # 添加搜索计数器
        jump_count = 0  # 跳变计数
        end_idx = anomaly_starts[i]  # 记录最后一次跳变的位置
        found_end = False

        # 在后续点中寻找异常结束的位置
        while j < len(heights) - 1:
            # 检查是否超过最大搜索时间
            if search_count >= max_duration:
                print(f"  - 发现长时间异常，判断为速度超极限")
                i = np.searchsorted(anomaly_starts, j+1)
                found_end = True
                break

            # 检查是否是新的跳变
            if abs(rates[j]) > max_rate:
                current_direction = np.sign(rates[j])
                # 如果是相反方向的跳变（高度先急升后急降，或先急降后急升）
                if current_direction == -start_direction:
                    jump_count += 1
                    end_idx = j

                    # 计算当前点与起始点之间的平均变化率
                    current_height = heights[j]
                    current_time = pd.Timestamp(times[j]).to_pydatetime()
                    time_diff = (current_time - start_time).total_seconds()
                    slope = (current_height - start_height) / time_diff

                    # 如果平均变化率在允许范围内，认为找到了异常结束点
                    if abs(slope) <= max_rate:
                        # 继续检查后续点，直到找到变化率小于阈值的位置
                        k = j + 1
                        while k < len(heights) - 1:
                            end_idx = k
                            if abs(rates[k]) > max_rate and np.sign(rates[k]) == current_direction:
                                # 更新结束点
                                j = k
                                k += 1
                            else:
                                break

                        # 标记异常段
                        mask[start_idx + 1:end_idx - 1] = False
                        anomaly_segments.append({
                            'start_idx': start_idx,
                            'end_idx': j,
                            'duration': time_diff,
                            'height_change': current_height - start_height,
                            'overall_rate': abs(slope)
                        })
                        found_end = True  # 标记找到终止点
                        break

            j += 1
            search_count += 1  # 增加搜索计数

        # 如果搜索到序列末尾还没找到终止点
        if not found_end:
            break

        # 更新索引，从当前异常段结束后的点开始寻找下一个异常段
        i = np.searchsorted(anomaly_starts, end_idx + 1)

    # 返回分析结果
    return {
        'is_valid': True,
        'mask': mask,
        'anomaly_info': anomaly_segments
    }

def process_single_file(file_path, interpolation_method='linear',
                        detect_outliers_params={'enabled': True, 'n_neighbors': 20, 'distance_threshold': None},
                        height_anomaly_params={'max_rate': 100, 'max_duration': 180},
                        speed_params={'max_ground_speed': 250, 'max_vertical_speed': 50},
                        time_gap_params={'threshold': 600},  # 10分钟
                        min_points=10,
                        military_threshold=1.0):
    """
    处理单个CSV文件的时间插值
    :param file_path: CSV文件路径
    :param interpolation_method: 插值方法，默认为'linear'
    :param detect_outliers_params: 离群点检测参数
    :param height_anomaly_params: 高度异常检测参数
    :param speed_params: 速度限制参数
    :param time_gap_params: 时间间隔参数
    :param min_points: 最小有效点数
    :param military_threshold: 军航轨迹判断阈值（度）
    :return: 处理后的DataFrame
    """
    print(f"\n开始处理文件：{os.path.basename(file_path)}")

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 将时间字符串转换为datetime对象（格式为 YYYYMMDD HH:MM:SS.fff）
        df['Time'] = pd.to_datetime(df['Time'])

        # 获取原始ID，用于后续所有处理
        original_id = df['ID'].iloc[0]
        
        # 临时保存所有处理过的轨迹
        all_processed_trajectories = []
        total_trajectory_count = 0

        # ===== 1. 首先根据时间间隔切分轨迹 =====
        time_split_trajectories = split_trajectory_by_time(df, time_gap_params['threshold'])
        print(f"根据时间间隔切分出 {len(time_split_trajectories)} 段轨迹")

        for i, time_traj in enumerate(time_split_trajectories):
            if len(time_traj) < min_points:
                print(f"时间切分轨迹段 {i+1} 点数不足 {min_points}，跳过")
                continue

            # ===== 2. 判断是否为军航轨迹 =====
            first_point = time_traj.iloc[0]
            last_point = time_traj.iloc[-1]
            
            # 计算首尾点的经度和纬度差值
            lon_diff = abs(last_point['Lon'] - first_point['Lon'])
            lat_diff = abs(last_point['Lat'] - first_point['Lat'])
            
            if lon_diff > military_threshold or lat_diff > military_threshold:
                print(f"时间切分轨迹段 {i+1} 被识别为军航轨迹，跳过处理")
                continue

            # ===== 3. 进行空间离群点检测 =====
            if detect_outliers_params['enabled']:
                data_for_outliers = time_traj[['H', 'Lon', 'Lat']].values
                mask = detect_outliers(
                    data_for_outliers,
                    n_neighbors=min(detect_outliers_params['n_neighbors'], len(time_traj) - 1),
                    distance_threshold=detect_outliers_params.get('distance_threshold', None)
                )
                time_traj = time_traj[mask]

            if len(time_traj) < min_points:
                print(f"时间切分轨迹段 {i+1} 剔除离群点后点数不足 {min_points}，跳过")
                continue

            # ===== 4. 进行第一次插值 =====
            numeric_df = time_traj[['Time', 'H', 'Lon', 'Lat']].copy()
            numeric_df = numeric_df.set_index('Time')
            resampled = numeric_df.resample('1s').mean()
            interpolated = resampled.interpolate(method='linear', limit_direction='both')
            
            # 恢复为DataFrame
            interpolated = interpolated.reset_index()
            
            # ===== 5. 检测高度异常段 =====
            height_result = detect_height_anomalies(
                interpolated['H'].values,
                interpolated['Time'].values,
                max_rate=height_anomaly_params['max_rate'],
                max_duration=height_anomaly_params['max_duration']
            )

            # 记录异常段信息
            if len(height_result['anomaly_info']) > 0:
                print(f"发现 {len(height_result['anomaly_info'])} 个高度异常段:")
                for segment in height_result['anomaly_info']:
                    print(f"  - 持续时间: {segment['duration']}秒, "
                          f"高度变化: {segment['height_change']:.2f}米, "
                          f"整体变化率: {segment['overall_rate']:.2f}米/秒")

            # 应用高度异常检测的掩码（移除异常点）
            interpolated = interpolated.iloc[height_result['mask']]

            # ===== 6. 进行最后一次插值 =====
            if len(interpolated) >= min_points:
                # 只对数值列进行重采样和插值
                numeric_df = interpolated[['Time', 'H', 'Lon', 'Lat']].copy()
                numeric_df = numeric_df.set_index('Time')
                resampled = numeric_df.resample('1s').mean()
                interpolated_numeric = resampled.interpolate(method='linear', limit_direction='both')
                
                # 重置索引
                interpolated = interpolated_numeric.reset_index()
            else:
                print(f"时间切分轨迹段 {i+1} 剔除异常段后点数过少，跳过处理")
                continue

            # ===== 7. 根据速度限制切分轨迹 =====
            speed_split_trajectories = split_trajectory_by_speed(
                interpolated,
                max_ground_speed=speed_params['max_ground_speed'],
                max_vertical_speed=speed_params['max_vertical_speed'],
                min_points=min_points
            )
            
            # 将速度切分后的轨迹添加到总队列中
            for speed_traj in speed_split_trajectories:
                if len(speed_traj) >= min_points:
                    all_processed_trajectories.append(speed_traj)
                    total_trajectory_count += 1
        
        # 如果没有可处理的轨迹，直接返回
        if not all_processed_trajectories:
            print("没有可处理的轨迹数据")
            return None
            
        # ===== 8. 为所有切分后的轨迹分配新的ID =====
        final_trajectories = []
        for idx, traj in enumerate(all_processed_trajectories):
            # 使用新的ID生成逻辑
            new_id = generate_new_id(original_id, total_trajectory_count, idx+1)
            traj['ID'] = new_id
            final_trajectories.append(traj)
            print(f"处理完成轨迹 {new_id}，包含 {len(traj)} 个点")

        # ===== 合并所有处理后的轨迹 =====
        result = pd.concat(final_trajectories, ignore_index=True)

        # 将时间转换为标准格式 YYYYMMDD HH:MM:SS.fff
        result['Time'] = result['Time'].apply(
            lambda x: f"{x.strftime('%Y%m%d')} {x.strftime('%H:%M:%S')}.{x.microsecond // 1000:03d}")

        # 调整列顺序
        result = result[['ID', 'H', 'Lon', 'Lat', 'Time']]

        # 按ID和时间排序
        result = result.sort_values(['ID', 'Time'])

        print(f"处理完成，最终数据形状：{result.shape}")
        return result

    except Exception as e:
        print(f"处理文件时出错：{str(e)}")
        raise

def process_all_files(input_dir, output_dir, 
                      detect_outliers_params={'enabled': True, 'n_neighbors': 20, 'distance_threshold': None},
                      height_anomaly_params={'max_rate': 100, 'max_duration': 180},
                      speed_params={'max_ground_speed': 250, 'max_vertical_speed': 50},
                      time_gap_params={'threshold': 600},  # 10分钟
                      min_points=10,
                      military_threshold=1.0):
    """
    处理目录下所有CSV文件
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param detect_outliers_params: 离群点检测参数
    :param height_anomaly_params: 高度异常检测参数
    :param speed_params: 速度限制参数
    :param time_gap_params: 时间间隔参数
    :param min_points: 最小有效点数，小于此值的轨迹将被废弃
    :param military_threshold: 军航轨迹判断阈值（度）
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录：{output_dir}")

    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    print(f"找到{len(csv_files)}个CSV文件")

    # 筛选出5位数ID的文件
    five_digit_files = []
    for filename in csv_files:
        file_id = os.path.splitext(filename)[0]  # 获取文件名（不含扩展名）
        if is_five_digit_id(file_id):
            five_digit_files.append(filename)
    
    print(f"其中{len(five_digit_files)}个文件是5位数ID")
    if not five_digit_files:
        print("没有找到5位数ID的文件，程序退出")
        return

    # 处理每个文件
    for filename in five_digit_files:
        input_path = os.path.join(input_dir, filename)
        
        try:
            # 调用单文件处理函数
            result = process_single_file(
                input_path,
                detect_outliers_params=detect_outliers_params,
                height_anomaly_params=height_anomaly_params,
                speed_params=speed_params,
                time_gap_params=time_gap_params,
                min_points=min_points,
                military_threshold=military_threshold
            )

            # 保存处理结果
            if result is not None:
                # 按ID分组保存为单独的CSV文件
                for id_value, group in result.groupby('ID'):
                    output_file = os.path.join(output_dir, f"{id_value}.csv")
                    group.to_csv(output_file, index=False, header=False)
                    print(f"ID {id_value} 的轨迹保存到：{output_file}，包含 {len(group)} 个点")
                
                print(f"文件 {filename} 处理完成，生成了 {len(result['ID'].unique())} 个轨迹文件")
            else:
                print(f"文件 {filename} 处理后没有有效数据，跳过保存")

        except Exception as e:
            print(f"处理文件 {filename} 失败：{str(e)}")
            continue

if __name__ == "__main__":
    # 设置输入输出路径
    input_dir = os.path.abspath('../../data/processed_data')
    output_dir = os.path.abspath('../../data/train')

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 参数配置
    params = {
        'detect_outliers_params': {
            'enabled': True,
            'n_neighbors': 40,
            'distance_threshold': None
        },
        'height_anomaly_params': {
            'max_rate': 300,
            'max_duration': 60
        },
        'speed_params': {
            'max_ground_speed': 680,  # 最大地面速度（米/秒）
            'max_vertical_speed': 680   # 最大垂直速度（米/秒）
        },
        'time_gap_params': {
            'threshold': 900  # 10分钟
        },
        'min_points': 80,  # 最小有效点数
        'military_threshold': 1.0  # 军航轨迹判断阈值（度）
    }

    # 开始处理所有文件
    process_all_files(
        input_dir,
        output_dir,
        **params
    )