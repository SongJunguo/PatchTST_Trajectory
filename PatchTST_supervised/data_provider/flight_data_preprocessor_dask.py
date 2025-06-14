# -*- coding: utf-8 -*-
# ==============================================================================
# ** 高性能飞行数据预处理器 (Dask内存版) **
#
# ** 版本: 1.2 (最终修复版) **
#
# ** 依赖项: **
#   pip install "polars[numpy,pandas,pyarrow]" scipy scikit-learn "dask[complete]"
#
# ** 运行说明: **
#   python flight_data_preprocessor_dask.py --input_dir [输入目录] --output_dir [输出目录]
#
# ** 核心改进点 (与 Polars版 对比): **
#   1.  **端到端并行:** 使用 Dask 从文件读取到数据转换，再到并行处理，全程利用多核优势。
#   2.  **纯内存计算:** 移除了将中间结果写入磁盘临时文件的步骤。通过 Dask 的
#       `client.persist()` 将数据直接持久化到所有工作进程的内存中，消除了I/O瓶颈。
#   3.  **高级调度:** 利用 Dask 的高级调度器替代手动的进程池，能更好地管理任务依赖、
#       数据局部性，并提供一个强大的诊断仪表盘。
#   4.  **代码复用:** 通过一个轻量级的适配器函数，复用了原版中经过优化的 Polars
#       核心处理算法 `_smooth_and_clean_udf`，兼顾了性能和开发效率。
# ==============================================================================

# 1. Imports
import polars as pl
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.neighbors import LocalOutlierFactor
import os
import logging
import argparse
from datetime import timedelta
import warnings
from typing import Optional, Dict, cast

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask.dataframe import DataFrame as DaskDataFrame

# 2. Logging Setup
def setup_logging(log_dir: str, log_level: str):
    """配置日志，使其同时输出到控制台和文件。"""
    log_filename = os.path.join(log_dir, "dask_preprocessing.log")
    os.makedirs(log_dir, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )

# 3. Constants
# Dask/Pandas 使用的DTypes
DTYPES = {
    'P1': str,
    'DTRQ': str,
    'JD': float,
    'WD': float,
    'H': float,
}

# 最终输出的元数据结构，用于Dask
META_OUTPUT = pd.DataFrame({
    "ID": pd.Series(dtype='str'),
    "Time": pd.Series(dtype='datetime64[ns]'),
    "Lon": pd.Series(dtype='float64'),
    "Lat": pd.Series(dtype='float64'),
    "H": pd.Series(dtype='float64'),
})

# 4. Helper Functions (部分从Polars版迁移并适配)

def detect_height_anomalies(heights, times, max_rate=100, max_duration=180):
    """
    从旧脚本中原样复制的函数，用于检测高度异常。
    输入为Numpy数组，与Pandas/Polars UDF可以良好集成。
    """
    mask = np.ones(len(heights), dtype=bool)
    if len(heights) < 2:
        return {'is_valid': True, 'mask': mask, 'anomaly_info': []}
    
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


def dms_to_decimal_pandas(dms_series: pd.Series) -> pd.Series:
    """
    将DMS格式(DD.MMSSff)的经纬度Pandas Series转换为十进制度。
    """
    degrees = np.floor(dms_series)
    minutes = np.floor((dms_series - degrees) * 100)
    seconds = (((dms_series - degrees) * 100) - minutes) * 100
    return pd.Series(degrees + minutes / 60 + seconds / 3600)


def _smooth_and_clean_udf(group_df: pl.DataFrame, min_len_for_clean: int) -> Optional[pl.DataFrame]:
    """
    (无需改动) 原版的Polars核心处理UDF，健壮且逻辑优化。
    遵循“先清洗、后插值、再平滑”的原则。
    """
    output_schema = {
        "Unique_ID": pl.Utf8, "Time": pl.Datetime, "Lon": pl.Float64,
        "Lat": pl.Float64, "H": pl.Float64,
    }
    empty_df = pl.DataFrame(schema=output_schema)
    if group_df.is_empty():
        return empty_df
        
    unique_id = group_df["Unique_ID"][0]
    features = ['Lat', 'Lon', 'H']

    try:
        proc_df = group_df.select("Unique_ID", "Time", *features).sort("Time").unique(subset=["Time"], keep="first")

        if len(proc_df) < min_len_for_clean:
            return None

        X = proc_df.select(features).to_numpy()
        n_neighbors = min(min_len_for_clean, len(X) - 1)
        if n_neighbors > 0:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            lof_outlier_mask = lof.fit_predict(X) == -1
        else:
            lof_outlier_mask = np.zeros(len(X), dtype=bool)

        heights = proc_df['H'].to_numpy()
        times = proc_df['Time'].to_numpy()
        height_anomaly_results = detect_height_anomalies(heights, times)
        height_anomaly_mask = height_anomaly_results['mask']

        is_good_point_mask = ~lof_outlier_mask & height_anomaly_mask
        
        cleaned_df = proc_df.with_columns(
            pl.when(pl.lit(pl.Series(is_good_point_mask)))
              .then(pl.col(c))
              .otherwise(None)
              .alias(c)
            for c in features
        )

        resampled_df = cleaned_df.upsample(time_column='Time', every='1s')

        resampled_df = resampled_df.with_columns(
            pl.col("Unique_ID").forward_fill(),
            pl.col(features).interpolate(method='linear').backward_fill().forward_fill()
        )

        if len(resampled_df) <= 51:
            logging.warning(f"航段 {unique_id} 插值后数据点不足 (<=51)，无法进行平滑处理。")
            return resampled_df.select(list(output_schema.keys()))
            
        window_length, polyorder = 51, 2
        
        final_df = resampled_df.with_columns([
            pl.col(c).map_batches(lambda s: pl.Series(values=savgol_filter(s.to_numpy(), window_length, polyorder), dtype=s.dtype))
            for c in features
        ])
        
        logging.debug(f"航段 {unique_id} 成功处理完成。")
        return final_df.select(list(output_schema.keys()))

    except Exception as e:
        logging.error(f"处理航段 {unique_id} 时发生严重错误: {e}", exc_info=True)
        return empty_df


def dask_polars_adapter_udf(partition_df: pd.DataFrame, min_len_for_clean: int) -> pd.DataFrame:
    """
    Dask-Polars适配器UDF。
    接收Dask传入的Pandas分区，将其转换为Polars DataFrame，
    按Unique_ID分组后，对每个组应用核心的Polars UDF，最后将结果合并转回Pandas。
    """
    if partition_df.empty:
        return META_OUTPUT.iloc[:0]

    # 1. 从Pandas零拷贝转换到Polars
    polars_df = pl.from_pandas(partition_df)
    
    processed_groups = []
    # 2. 按航段ID分组，并对每个航段应用核心处理函数
    for group_id, group_data in polars_df.group_by("Unique_ID"):
        result = _smooth_and_clean_udf(group_data, min_len_for_clean)
        if result is not None and not result.is_empty():
            processed_groups.append(result)
    
    if not processed_groups:
        return META_OUTPUT.iloc[:0]

    # 3. 合并处理后的结果，并零拷贝转回Pandas
    final_polars_df = pl.concat(processed_groups)
    return final_polars_df.to_pandas()


# 5. Main Processing Function
def process_flight_data_dask(client: Client, input_dir: str, output_dir: str, output_format: str,
                             h_min: float, h_max: float, lon_min: float, lon_max: float,
                             lat_min: float, lat_max: float,
                             segment_split_minutes: int, log_level: str, min_len_for_clean: int,
                             unique_id_strategy: str = 'numeric'):
    """
    主处理流程函数 (Dask版)。
    """
    setup_logging(output_dir, log_level)
    logging.info("===== 开始飞行数据预处理流程 (Dask内存版) =====")

    # --- 阶段一: Dask并行读取与预处理 ---
    logging.info(f"--- 阶段一: Dask正在从 {input_dir} 并行扫描CSV文件... ---")
    input_files = os.path.join(input_dir, "*.csv")
    ddf = dd.read_csv(
        input_files,
        dtype=DTYPES,
        sep=',',
        encoding='gbk',
        on_bad_lines='warn',
        assume_missing=True
    )

    # --- 阶段二: 核心转换 (Dask Lazy & Parallel) ---
    logging.info("--- 阶段二: 正在执行Dask惰性数据转换与轨迹切分... ---")
    
    transformed_ddf = ddf.rename(columns={'P1': 'ID'})
    # 修正了日期格式字符串
    transformed_ddf['Time'] = dd.to_datetime(transformed_ddf['DTRQ'], format="%d-%b-%y %I.%M.%S.%f %p", errors='coerce')
    transformed_ddf['Lon'] = dms_to_decimal_pandas(transformed_ddf['JD'])
    transformed_ddf['Lat'] = dms_to_decimal_pandas(transformed_ddf['WD'])

    transformed_ddf = transformed_ddf[['ID', 'Time', 'Lon', 'Lat', 'H']].dropna()
    transformed_ddf = transformed_ddf[
        (transformed_ddf['H'] != 0) & (transformed_ddf['Lon'] != 0) & (transformed_ddf['Lat'] != 0) &
        (transformed_ddf['H'] >= h_min) & (transformed_ddf['H'] <= h_max) &
        (transformed_ddf['Lon'] >= lon_min) & (transformed_ddf['Lon'] <= lon_max) &
        (transformed_ddf['Lat'] >= lat_min) & (transformed_ddf['Lat'] <= lat_max)
    ]
    
    # --- 轨迹切分与ID生成 (稳健版) ---
    # 在每个ID组内，按时间排序并计算差值，以识别新的航段
    transformed_ddf['time_diff'] = transformed_ddf.groupby('ID')['Time'].apply(
        lambda s: s.sort_values().diff(), 
        meta=pd.Series([], dtype='timedelta64[ns]', name='time_diff')
    )
    
    transformed_ddf['new_segment_marker'] = (transformed_ddf['time_diff'] > timedelta(minutes=segment_split_minutes)).fillna(False)
    
    # 按ID分组，对新航段标记进行累加，以生成唯一的航段ID
    transformed_ddf['segment_id'] = transformed_ddf.groupby('ID')['new_segment_marker'].cumsum()

    if unique_id_strategy == 'numeric':
        transformed_ddf['Unique_ID'] = transformed_ddf['ID'].astype(str) + '_' + transformed_ddf['segment_id'].astype(str)
    else: # timestamp strategy
        # .first() creates a Series, must convert to frame for merging
        trajectory_start_time = transformed_ddf.groupby(['ID', 'segment_id'])['Time'].first()
        start_time_df = trajectory_start_time.to_frame(name='start_time')
        
        # Merge on columns on the left and on the index of the right
        transformed_ddf = transformed_ddf.merge(start_time_df, left_on=['ID', 'segment_id'], right_index=True, how='left')
        
        time_str = transformed_ddf['start_time'].dt.strftime("%Y%m%dT%H%M%S%f").str.slice(0, -3).str.replace(".", "M", regex=False)
        transformed_ddf['Unique_ID'] = transformed_ddf['ID'].astype(str) + '_' + time_str
        
        # Clean up intermediate columns
        transformed_ddf = transformed_ddf.drop(columns=['start_time', 'time_diff'])

    # --- 阶段三: 纯内存并行化平滑处理 ---
    logging.info("--- 阶段三: 准备进行纯内存并行化平滑处理... ---")
    
    logging.info("正在执行预转换并持久化数据到集群内存...")
    segmented_ddf = cast(DaskDataFrame, client.persist(transformed_ddf))
    logging.info("数据已成功加载到内存中。")

    logging.info(f"准备将 {segmented_ddf.npartitions} 个数据分区应用核心平滑算法...")
    
    meta = META_OUTPUT.iloc[:0]
    
    processed_ddf = segmented_ddf.map_partitions(dask_polars_adapter_udf, min_len_for_clean=min_len_for_clean, meta=meta)

    # --- 阶段四: 保存结果 ---
    logging.info("--- 阶段四: 正在并行保存最终结果... ---")
    
    final_ddf = processed_ddf.rename(columns={"Unique_ID": "ID"})
    final_ddf = final_ddf[['ID', 'Time', 'Lon', 'Lat', 'H']]
    
    final_ddf['Time'] = final_ddf['Time'].dt.strftime("%Y%m%d %H:%M:%S.%f").str.slice(0, -3)
    
    output_path = os.path.join(output_dir, f"processed_trajectories_dask.{output_format}")
    
    try:
        if output_format == 'csv':
            final_ddf.to_csv(output_path, index=False, single_file=False)
        elif output_format == 'parquet':
            final_ddf.to_parquet(output_path, engine='pyarrow', write_index=False)
        
        logging.info(f"成功！已保存处理后的轨迹到目录: {output_path}")
    except Exception as e:
        logging.error(f"保存文件到 {output_path} 时失败: {e}")
            
    logging.info("===== 数据预处理流程全部完成 =====")


# 6. Main Execution Block
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="高性能飞行数据预处理器 (Dask内存版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input_dir', type=str, required=True, help='包含原始CSV文件的输入目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='用于存放处理后数据的输出目录路径')
    parser.add_argument('--output_format', type=str, default='csv', choices=['csv', 'parquet'], help='输出文件的格式 (parquet推荐)')
    parser.add_argument('--n_workers', type=int, default=os.cpu_count(), help='Dask使用的最大工作进程数')
    parser.add_argument('--segment_split_minutes', type=int, default=5, help='用于切分轨迹的时间间隔（分钟）')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='设置日志记录级别')
    parser.add_argument('--min_len_for_clean', type=int, default=792, help='航段进行清洗（异常检测等）所需的最小数据点数')
    parser.add_argument('--unique_id_strategy', type=str, default='numeric', choices=['numeric', 'timestamp'], help='生成轨迹唯一ID的策略')
    
    # 地理和高度过滤参数
    parser.add_argument('--h_min', type=float, default=0, help='最低高度 (米)')
    parser.add_argument('--h_max', type=float, default=20000, help='最高高度 (米)')
    parser.add_argument('--lon_min', type=float, default=110, help='最小经度')
    parser.add_argument('--lon_max', type=float, default=120, help='最大经度')
    parser.add_argument('--lat_min', type=float, default=33, help='最小纬度')
    parser.add_argument('--lat_max', type=float, default=42, help='最大纬度')

    args = parser.parse_args()

    with LocalCluster(n_workers=args.n_workers, threads_per_worker=1) as cluster, Client(cluster) as client:
        logging.info(f"Dask集群已启动，仪表盘地址: {client.dashboard_link}")
        
        process_flight_data_dask(
            client=client,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            output_format=args.output_format,
            h_min=args.h_min,
            h_max=args.h_max,
            lon_min=args.lon_min,
            lon_max=args.lon_max,
            lat_min=args.lat_min,
            lat_max=args.lat_max,
            segment_split_minutes=args.segment_split_minutes,
            log_level=args.log_level,
            min_len_for_clean=args.min_len_for_clean,
            unique_id_strategy=args.unique_id_strategy,
        )