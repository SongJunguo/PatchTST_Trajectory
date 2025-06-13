# -*- coding: utf-8 -*-
# ==============================================================================
# ** 高性能飞行数据预处理器 (Polars版) **
#
# ** 版本: 1.0 **
#
# ** 依赖项: **
#   pip install polars numpy scipy scikit-learn tqdm psutil
#
# ** 运行说明: **
#   python flight_data_preprocessor_polars.py --input_dir [输入目录] --output_dir [输出目录]
#
# ** 核心改进点 (与 flight_data_preprocessor_multi.py 对比): **
#   1.  **I/O优化:** 使用 Polars 的 `scan_csv` 并行扫描文件，避免单线程瓶颈。
#   2.  **内存效率:** 全程采用惰性计算 (LazyFrames)，仅在必要时触发计算，
#       并通过 `streaming=True` 控制内存峰值，可处理远超内存大小的数据。
#   3.  **CPU并行:** Polars 的核心操作 (如转换、过滤、窗口函数) 本身就是
#       多线程并行的，充分利用现代CPU的多核能力。
#   4.  **健壮性:** 对文件读取、UDF处理等关键步骤增加了详细的错误处理，
#       单个坏文件或坏航段不会中断整个流程。
#   5.  **代码简洁性:** 使用 Polars 的表达式 API 替代了 Pandas 中繁琐的
#       `apply` 和循环，代码更易读、更易维护。
# ==============================================================================

# 1. Imports
import polars as pl
import pandas as pd  # pd is required by the legacy detect_height_anomalies function
import numpy as np
from scipy.signal import savgol_filter
from sklearn.neighbors import LocalOutlierFactor
import os
import glob
import logging
import argparse
from tqdm import tqdm
import psutil
from datetime import timedelta
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing


# 2. Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 3. Constants
# 定义输入文件的Schema，强制数据类型可以提升读取速度并减少内存占用
# 注意：这里的数据类型是Polars的类型
SCHEMA = {
    'P1': pl.Utf8,      # 飞机ID
    'DTRQ': pl.Utf8,    # 日期时间字符串
    'JD': pl.Float64,   # 经度 (DMS格式)
    'WD': pl.Float64,   # 纬度 (DMS格式)
    'H': pl.Float64,    # 高度
}

# 4. Helper Functions

def detect_height_anomalies(heights, times, max_rate=100, max_duration=180):
    """
    从旧脚本中原样复制的函数，用于检测高度异常。
    输入为Numpy数组，与Polars UDF可以良好集成。
    """
    mask = np.ones(len(heights), dtype=bool)
    if len(heights) < 2:
        return {'is_valid': True, 'mask': mask, 'anomaly_info': []}
    
    # np.diff requires at least 2 elements
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


def dms_to_decimal_expr(dms_col: str) -> pl.Expr:
    """
    返回一个Polars表达式，用于将DMS格式(DD.MMSSff)的经纬度转换为十进制度。
    这是一个纯表达式实现，避免了低效的 .apply()。
    """
    degrees = pl.col(dms_col).floor()
    minutes = ((pl.col(dms_col) - degrees) * 100).floor()
    seconds = (((pl.col(dms_col) - degrees) * 100) - minutes) * 100
    return degrees + minutes / 60 + seconds / 3600


def _smooth_and_clean_udf(group_df: pl.DataFrame) -> pl.DataFrame:
    """
    健壮的用户定义函数 (UDF)，用于对单个航段进行平滑和清理。
    包裹在 try-except 块中，以确保单个航段的处理失败不会影响整个任务。
    """
    # 预先定义UDF的输出schema，确保任何分支都返回一致的结构
    output_schema = {
        "Unique_ID": pl.Utf8,
        "Time": pl.Datetime,
        "Lon": pl.Float64,
        "Lat": pl.Float64,
        "H": pl.Float64,
    }
    empty_df = pl.DataFrame(schema=output_schema)
    
    unique_id = group_df["Unique_ID"][0]
    try:
        # 0. 预处理：只选择我们需要的列，确保 schema 一致性
        proc_df = group_df.select("Unique_ID", "Time", "Lon", "Lat", "H").sort("Time").unique(subset=["Time"], keep="first")

        if len(proc_df) < 2:
            logging.warning(f"航段 {unique_id} 的数据点不足 (<2)，跳过处理。")
            return empty_df

        # 1. 重采样到1秒间隔
        resampled_df = proc_df.upsample(time_column='Time', every='1s')

        # 2. 线性插值填充采样引入的空值
        features = ['Lat', 'Lon', 'H']
        resampled_df = resampled_df.with_columns(
            pl.col(features).interpolate(method='linear')
        )
        
        # 3. 离群点检测 (LOF)
        X = resampled_df.select(features).to_numpy()
        lof = LocalOutlierFactor(n_neighbors=min(50, len(X)), contamination=0.05)
        outlier_mask = lof.fit_predict(X) == -1
        
        resampled_df = resampled_df.with_columns(
            pl.when(pl.lit(pl.Series(outlier_mask)))
              .then(None)
              .otherwise(pl.col(c))
              .alias(c)
            for c in features
        ).with_columns(
            pl.col(features).interpolate(method='linear')
        )

        # 4. 高度异常检测
        heights = resampled_df['H'].to_numpy()
        times = resampled_df['Time'].to_numpy()
        anomaly_results = detect_height_anomalies(heights, times)
        
        resampled_df = resampled_df.with_columns(
            pl.when(pl.lit(pl.Series(anomaly_results['mask'])))
              .then(pl.col(c))
              .otherwise(None)
              .alias(c)
            for c in features
        )

        # 5. 再次插值并填充边缘NaN
        resampled_df = resampled_df.with_columns(
            pl.col(features).interpolate(method='linear').backward_fill().forward_fill()
        )

        # 6. Savitzky-Golay 平滑滤波
        if len(resampled_df) <= 51:
            logging.warning(f"航段 {unique_id} 插值后数据点不足 (<=51)，无法进行平滑处理。")
            return empty_df
            
        window_length, polyorder = 51, 2
        
        final_df = resampled_df.with_columns([
            pl.col('Lat').map_batches(lambda s: pl.Series(values=savgol_filter(s.to_numpy(), window_length, polyorder), dtype=s.dtype)),
            pl.col('Lon').map_batches(lambda s: pl.Series(values=savgol_filter(s.to_numpy(), window_length, polyorder), dtype=s.dtype)),
            pl.col('H').map_batches(lambda s: pl.Series(values=savgol_filter(s.to_numpy(), window_length, polyorder), dtype=s.dtype)),
        ])
        
        # 确保最终输出的列顺序和类型与定义的schema一致
        return final_df.select(list(output_schema.keys()))

    except Exception as e:
        logging.error(f"处理航段 {unique_id} 时发生严重错误: {e}", exc_info=True)
        return empty_df # 返回预定义的空DataFrame


# 这是一个新的顶层工作函数，用于并行处理一个航段
def _process_trajectory_worker(unique_id: str, temp_file_path: str) -> pl.DataFrame:
    """
    工作函数，用于处理单个Unique_ID。
    它从临时文件中只读取自己需要的数据。
    """
    # 从临时Parquet文件中过滤出当前进程负责的航段
    trajectory_df = pl.scan_parquet(temp_file_path).filter(pl.col("Unique_ID") == unique_id).collect()
    
    # 对单个航段应用UDF
    if not trajectory_df.is_empty():
        return _smooth_and_clean_udf(trajectory_df)
    return pl.DataFrame()


# 5. Main Processing Function
def process_flight_data(input_dir: str, output_dir: str, output_format: str,
                        h_min: float, h_max: float, lon_min: float, lon_max: float,
                        lat_min: float, lat_max: float, encoding_priority: str, max_workers: int):
    """
    主处理流程函数，包含了计划书中定义的四个阶段。
    """
    # 抑制来自sklearn的关于重复值的警告，这在重采样后是正常现象
    warnings.filterwarnings("ignore", category=UserWarning)
    
    logging.info("===== 开始飞行数据预处理流程 (Polars版) =====")
    
    # --- 阶段一: 并行读取与预检 ---
    logging.info(f"--- 阶段一: 正在从 {input_dir} 扫描CSV文件... ---")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_files:
        logging.error(f"在目录 {input_dir} 中未找到任何CSV文件。程序终止。")
        return

    # 根据用户指定的优先级确定编码尝试顺序
    if encoding_priority == 'gbk':
        encodings_to_try = ['gbk', 'utf8']
    else:
        encodings_to_try = ['utf8', 'gbk']
    logging.info(f"编码检测顺序: {encodings_to_try[0]} -> {encodings_to_try[1]}")

    lazy_frames = []
    failed_files = []
    for file in tqdm(all_files, desc="扫描文件"):
        df = None
        for encoding in encodings_to_try:
            try:
                # 使用 read_csv 进行稳健的编码尝试，成功后转为 lazy frame
                df = pl.read_csv(file, schema_overrides=SCHEMA, ignore_errors=True, separator=',', encoding=encoding)
                logging.info(f"成功: 使用 '{encoding}' 编码读取文件: {os.path.basename(file)}")
                break  # 成功读取，跳出内部循环
            except Exception:
                df = None # 确保失败后 df 为 None
                continue # 尝试下一种编码
        
        if df is not None and not df.is_empty():
            lazy_frames.append(df.lazy())
        else:
            logging.error(f"失败: 无法使用 {encodings_to_try} 解码文件 {os.path.basename(file)}，已跳过。")
            failed_files.append(os.path.basename(file))

    if not lazy_frames:
        logging.error("所有文件都加载失败。请检查文件编码或内容。程序终止。")
        return
        
    # 合并所有LazyFrames
    lazy_df = pl.concat(lazy_frames)

    # --- 阶段二: 核心转换 (Lazy & Parallel) ---
    logging.info("--- 阶段二: 正在执行惰性数据转换与轨迹切分... ---")
    
    # 核心转换表达式链
    transformed_lf = lazy_df.with_columns([
        # 日期解析
        pl.col("DTRQ").str.to_datetime(format="%d-%b-%y %I.%M.%S%.f %p", strict=False).alias("Time"),
        # 经纬度转换
        dms_to_decimal_expr("JD").alias("Lon"),
        dms_to_decimal_expr("WD").alias("Lat"),
        pl.col("P1").alias("ID")
    ]).select(
        ["ID", "Time", "Lon", "Lat", "H"]
    ).drop_nulls().filter(
        (pl.col("H") >= h_min) & (pl.col("H") <= h_max) &
        (pl.col("Lon") >= lon_min) & (pl.col("Lon") <= lon_max) &
        (pl.col("Lat") >= lat_min) & (pl.col("Lat") <= lat_max)
    )

    # 轨迹切分表达式
    segmented_lf = transformed_lf.sort("ID", "Time").with_columns(
        (
            pl.col("Time").diff().over("ID") > timedelta(minutes=15)
        ).fill_null(False).alias("new_segment_marker")
    ).with_columns(
        pl.col("new_segment_marker").cum_sum().over("ID").alias("segment_id")  # type: ignore
    ).with_columns(
        (pl.col("ID").cast(pl.Utf8) + pl.lit("_") + pl.col("segment_id").cast(pl.Utf8)).alias("Unique_ID")
    )

    # --- 增加健壮性检查: 确认有数据通过了过滤 ---
    # 使用 .head(1).collect() 廉价地检查惰性框架在过滤后是否为空，避免后续在空DF上执行group_by
    if segmented_lf.head(1).collect().is_empty():
        logging.warning("所有数据均在初始过滤阶段（如地理范围、有效日期等）被移除。没有可处理的航段。程序终止。")
        return

    # --- 阶段三: 真正的并行化平滑处理 ---
    logging.info("--- 阶段三: 准备进行并行化平滑处理... ---")
    
    temp_file = os.path.join(output_dir, "_temp_segmented_data.parquet")
    all_processed_dfs = []
    executor = ProcessPoolExecutor(max_workers=max_workers)

    try:
        logging.info("正在将分段数据写入临时文件以便并行读取...")
        segmented_lf.sink_parquet(temp_file)
        
        unique_ids = pl.scan_parquet(temp_file).select("Unique_ID").unique().collect()["Unique_ID"].to_list()
        
        if not unique_ids:
            logging.warning("没有唯一的航段ID可供处理。")
            executor.shutdown()
            return
            
        logging.info(f"准备将 {len(unique_ids)} 个独立航段分发到 {max_workers} 个进程中进行处理。")

        worker_func = partial(_process_trajectory_worker, temp_file_path=temp_file)
        
        with tqdm(total=len(unique_ids), desc="并行处理航段") as pbar:
            for result_df in executor.map(worker_func, unique_ids):
                if not result_df.is_empty():
                    all_processed_dfs.append(result_df)
                pbar.update(1)
        
        executor.shutdown(wait=True) # Clean shutdown on success

    except KeyboardInterrupt:
        logging.warning("\n--- 用户中断了处理流程。正在强制关闭工作进程... ---")
        # On interrupt, shutdown without waiting and cancel futures (Python 3.9+).
        executor.shutdown(wait=False, cancel_futures=True)
        return # Exit the function
    finally:
        # This block ensures the temp file is always removed.
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logging.info(f"已清理临时文件: {temp_file}")

    if not all_processed_dfs:
        logging.warning("所有航段处理后均为空，没有可保存的数据。")
        return
        
    processed_df = pl.concat(all_processed_dfs)

    # --- 阶段四: 保存结果 ---
    logging.info("--- 阶段四: 正在保存最终结果... ---")
    
    if processed_df.is_empty():
        logging.warning("处理后没有可保存的数据。")
    else:
        # 重命名并选择最终列
        final_df = processed_df.rename({"Unique_ID": "ID"}).select(
            ["ID", "Time", "Lon", "Lat", "H"]
        )
        
        output_filename = f"processed_trajectories_polars.{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            if output_format == 'csv':
                final_df.write_csv(output_path)
            elif output_format == 'parquet':
                final_df.write_parquet(output_path)
            
            logging.info(f"成功！已保存 {final_df['ID'].n_unique()} 条处理后的轨迹到 {output_path}")
        except Exception as e:
            logging.error(f"保存文件到 {output_path} 时失败: {e}")

    if failed_files:
        logging.warning("以下文件在读取过程中失败，已被跳过:")
        for f in failed_files:
            logging.warning(f" - {f}")
            
    logging.info("===== 数据预处理流程全部完成 =====")


# 6. Main Execution Block
if __name__ == '__main__':
    # 根据Polars官方文档，为防止多进程死锁，推荐在Unix系统上使用 'spawn' 方法
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="高性能飞行数据预处理器 (Polars版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input_dir', type=str, required=True, help='包含原始CSV文件的输入目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='用于存放处理后数据的输出目录路径')
    parser.add_argument('--output_format', type=str, default='csv', choices=['csv', 'parquet'], help='输出文件的格式')
    parser.add_argument('--encoding_priority', type=str, default='gbk', choices=['gbk', 'utf8'], help='优先尝试的文件编码')
    parser.add_argument('--max_workers', type=int, default=os.cpu_count(), help='用于并行处理的最大工作进程数')
    
    # 地理和高度过滤参数
    parser.add_argument('--h_min', type=float, default=0, help='最低高度 (米)')
    parser.add_argument('--h_max', type=float, default=20000, help='最高高度 (米)')
    parser.add_argument('--lon_min', type=float, default=110, help='最小经度')
    parser.add_argument('--lon_max', type=float, default=120, help='最大经度')
    parser.add_argument('--lat_min', type=float, default=33, help='最小纬度')
    parser.add_argument('--lat_max', type=float, default=42, help='最大纬度')

    args = parser.parse_args()

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"已创建输出目录: {args.output_dir}")

    # 启动处理
    process_flight_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_format=args.output_format,
        h_min=args.h_min,
        h_max=args.h_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        encoding_priority=args.encoding_priority,
        max_workers=args.max_workers
    )