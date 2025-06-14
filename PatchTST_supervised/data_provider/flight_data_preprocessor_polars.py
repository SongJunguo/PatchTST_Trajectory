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
from typing import Optional


# 2. Logging Setup
def setup_logging(log_dir: str, log_level: str):
    """配置日志，使其同时输出到控制台和文件。"""
    log_filename = os.path.join(log_dir, "polars_preprocessing.log")
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename, mode='w'), # 写入文件，每次覆盖
            logging.StreamHandler()                     # 输出到控制台
        ]
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


def _smooth_and_clean_udf(group_df: pl.DataFrame, min_len_for_clean: int) -> Optional[pl.DataFrame]:
    """
    健壮且逻辑优化的用户定义函数 (UDF)，用于对单个航段进行平滑和清理。
    遵循“先清洗、后插值、再平滑”的原则。
    """
    output_schema = {
        "Unique_ID": pl.Utf8, "Time": pl.Datetime, "Lon": pl.Float64,
        "Lat": pl.Float64, "H": pl.Float64,
    }
    empty_df = pl.DataFrame(schema=output_schema)
    unique_id = group_df["Unique_ID"][0]
    features = ['Lat', 'Lon', 'H']

    try:
        # 步骤 0: 预处理 - 排序并确保时间戳唯一
        proc_df = group_df.select("Unique_ID", "Time", *features).sort("Time").unique(subset=["Time"], keep="first")

        if len(proc_df) < min_len_for_clean: # 增加阈值以进行更可靠的异常检测
            # 对于过短的航段，直接返回None，由主进程统一计数和报告
            return None

        # 步骤 1: 在原始数据上进行离群点检测 (LOF)
        X = proc_df.select(features).to_numpy()
        # n_neighbors不能超过样本数
        n_neighbors = min(min_len_for_clean, len(X) - 1)
        if n_neighbors > 0:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            lof_outlier_mask = lof.fit_predict(X) == -1 # True表示异常
        else:
            lof_outlier_mask = np.zeros(len(X), dtype=bool) # 点太少，不认为有异常

        # 步骤 2: 在原始数据上进行高度异常检测
        heights = proc_df['H'].to_numpy()
        times = proc_df['Time'].to_numpy()
        height_anomaly_results = detect_height_anomalies(heights, times)
        # mask中True表示正常，False表示异常
        height_anomaly_mask = height_anomaly_results['mask']

        # 步骤 3: 合并异常标记，并将所有异常点设为None
        # 一个点是“好”的，当且仅当它不是LOF离群点 AND 不是高度异常点
        is_good_point_mask = ~lof_outlier_mask & height_anomaly_mask
        
        cleaned_df = proc_df.with_columns(
            pl.when(pl.lit(pl.Series(is_good_point_mask)))
              .then(pl.col(c))
              .otherwise(None)
              .alias(c)
            for c in features
        )

        # 步骤 4: 重采样到1秒间隔
        resampled_df = cleaned_df.upsample(time_column='Time', every='1s')

        # 步骤 5: 对所有空值（来自异常点和重采样）进行一次性填充
        # - ID列使用前向填充
        # - 特征列使用线性插值
        resampled_df = resampled_df.with_columns(
            pl.col("Unique_ID").forward_fill(),
            pl.col(features).interpolate(method='linear').backward_fill().forward_fill()
        )

        # 步骤 6: Savitzky-Golay 平滑滤波
        if len(resampled_df) <= 51:
            logging.warning(f"航段 {unique_id} 插值后数据点不足 (<=51)，无法进行平滑处理。")
            # 即使不能平滑，也返回插值后的规整数据
            return resampled_df.select(list(output_schema.keys()))
            
        window_length, polyorder = 51, 2
        
        final_df = resampled_df.with_columns([
            pl.col(c).map_batches(lambda s: pl.Series(values=savgol_filter(s.to_numpy(), window_length, polyorder), dtype=s.dtype))
            for c in features
        ])
        
        # 确保最终输出的列顺序和类型与定义的schema一致
        logging.debug(f"航段 {unique_id} 成功处理完成。")
        return final_df.select(list(output_schema.keys()))

    except Exception as e:
        logging.error(f"处理航段 {unique_id} 时发生严重错误: {e}", exc_info=True)
        return empty_df


# 这是一个新的顶层工作函数，用于并行处理一个航段
def _process_trajectory_worker(unique_id: str, temp_file_path: str, min_len_for_clean: int) -> Optional[pl.DataFrame]:
    """
    工作函数，用于处理单个Unique_ID。
    它从临时文件中只读取自己需要的数据。
    """
    # 从临时Parquet文件中过滤出当前进程负责的航段
    trajectory_df = pl.scan_parquet(temp_file_path).filter(pl.col("Unique_ID") == unique_id).collect()
    
    # 对单个航段应用UDF
    if not trajectory_df.is_empty():
        return _smooth_and_clean_udf(trajectory_df, min_len_for_clean=min_len_for_clean)
    return pl.DataFrame()


# 5. Main Processing Function
def process_flight_data(input_dir: str, output_dir: str, output_format: str,
                        h_min: float, h_max: float, lon_min: float, lon_max: float,
                        lat_min: float, lat_max: float, encoding_priority: str, max_workers: int,
                        segment_split_minutes: int, log_level: str, min_len_for_clean: int,
                        unique_id_strategy: str = 'numeric', save_debug_segmented_file: bool = False):
    """
    主处理流程函数，包含了计划书中定义的四个阶段。
    """
    # 配置日志记录
    setup_logging(output_dir, log_level)

    # 抑制来自sklearn的关于重复值的警告，这在重采样后是正常现象
    # warnings.filterwarnings("ignore", category=UserWarning)
    
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
    logging.info("正在统计原始数据点总数...")
    total_raw_rows = lazy_df.select(pl.len()).collect().item()
    logging.info(f"所有CSV文件共包含 {total_raw_rows:,} 个原始数据点。")

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
        # 增加健壮性：明确排除经纬高为0的无效填充数据
        (pl.col("H") != 0) & (pl.col("Lon") != 0) & (pl.col("Lat") != 0) &
        
        # 保留原有的地理和高度范围过滤
        (pl.col("H") >= h_min) & (pl.col("H") <= h_max) &
        (pl.col("Lon") >= lon_min) & (pl.col("Lon") <= lon_max) &
        (pl.col("Lat") >= lat_min) & (pl.col("Lat") <= lat_max)
    )

    # --- 轨迹切分与ID生成 ---
    # 步骤1: 标记时间间隔超限的点，作为新航段的起点
    segmented_lf = transformed_lf.sort("ID", "Time").with_columns(
        (
            pl.col("Time").diff().over("ID") > timedelta(minutes=segment_split_minutes)
        ).fill_null(False).alias("new_segment_marker")
    )

    # 步骤2: 使用累积和为每个航段生成一个数字ID
    segmented_lf = segmented_lf.with_columns(
        pl.col("new_segment_marker").cum_sum().over("ID").alias("segment_id")  # type: ignore
    )

    # 步骤3: 根据用户选择的策略生成最终的Unique_ID
    if unique_id_strategy == 'numeric':
        logging.info("使用 'numeric' 策略生成轨迹ID (例如: PlaneA_0)")
        segmented_lf = segmented_lf.with_columns(
            (pl.col("ID").cast(pl.Utf8) + pl.lit("_") + pl.col("segment_id").cast(pl.Utf8)).alias("Unique_ID")
        )
    elif unique_id_strategy == 'timestamp':
        logging.info("使用 'timestamp' 策略生成轨迹ID (例如: PlaneA_20230101T100000M123)")
        # 获取每个航段的第一个时间点
        trajectory_start_time = pl.col("Time").first().over("ID", "segment_id")
        # 将起始时间格式化为 YYYYMMDDTHHMMSS.fff 的格式，然后用'M'替换'.'
        time_str_with_dot = trajectory_start_time.dt.strftime("%Y%m%dT%H%M%S%.3f")
        time_str_with_M = time_str_with_dot.str.replace(r".", "M", literal=True)
        
        segmented_lf = segmented_lf.with_columns(
            (pl.col("ID").cast(pl.Utf8) + pl.lit("_") + time_str_with_M).alias("Unique_ID")
        )

    # --- 增加健壮性检查: 确认有数据通过了过滤 ---
    logging.info("正在统计过滤和切分后的有效数据点总数...")
    # 注意：这次 collect 会触发完整的计算计划，但只为了获取行数，开销可控
    total_processed_rows = segmented_lf.select(pl.len()).collect(engine='streaming').item()
    logging.info(f"经过过滤和轨迹切分后，剩余 {total_processed_rows:,} 个有效数据点。")
    if total_raw_rows > 0:
        logging.info(f"数据保留率: {total_processed_rows / total_raw_rows:.2%}")

    if total_processed_rows == 0:
        logging.warning("所有数据均在初始过滤阶段（如地理范围、有效日期等）被移除。没有可处理的航段。程序终止。")
        return

    # --- (可选) 保存用于调试的中间文件 ---
    if save_debug_segmented_file:
        logging.info("正在以流式方式生成用于调试的航段分割文件 (内存安全)...")
        
        # 准备最终的 LazyFrame，包含所有需要的列选择和格式化逻辑
        debug_lf_to_save = segmented_lf.select(
            pl.col("Unique_ID").alias("ID"),
            pl.col("Time").dt.strftime("%Y-%m-%d %H:%M:%S%.3f").alias("Time"), # 使用人类可读的时间格式
            "Lon", "Lat", "H",
            pl.col("ID").alias("Original_Plane_ID") # 保留原始飞机ID以供参考
        )
        
        # 定义输出路径并以流式方式写入，避免高内存占用
        debug_output_path = os.path.join(output_dir, "segmented_for_visualization.csv")
        try:
            debug_lf_to_save.sink_csv(debug_output_path)
            logging.info(f"成功！已保存调试文件到: {debug_output_path}")
        except Exception as e:
            logging.error(f"保存调试文件失败: {e}", exc_info=True)

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

        worker_func = partial(_process_trajectory_worker, temp_file_path=temp_file, min_len_for_clean=min_len_for_clean)
        
        skipped_short_count = 0
        with tqdm(total=len(unique_ids), desc="并行处理航段") as pbar:
            for result_df in executor.map(worker_func, unique_ids):
                if result_df is None:
                    skipped_short_count += 1
                elif not result_df.is_empty():
                    all_processed_dfs.append(result_df)
                # is_empty() 且 not None 的情况意味着UDF内部发生错误并返回了empty_df
                pbar.update(1)
        
        executor.shutdown(wait=True) # Clean shutdown on success

        if skipped_short_count > 0:
            logging.info(f"共跳过 {skipped_short_count} 条因数据点过少(<{min_len_for_clean})的航段。")

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
        
        # 按照用户指定的格式 'YYYYMMDD HH:MM:SS.fff' 格式化时间列
        final_df = final_df.with_columns(
            pl.col("Time").dt.strftime("%Y%m%d %H:%M:%S%.3f").alias("Time")
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
    parser.add_argument('--segment_split_minutes', type=int, default=5, help='用于切分轨迹的时间间隔（分钟）')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='设置日志记录级别')
    parser.add_argument('--min_len_for_clean', type=int, default=792, help='航段进行清洗（异常检测等）所需的最小数据点数')
    parser.add_argument('--unique_id_strategy', type=str, default='numeric', choices=['numeric', 'timestamp'], help='生成轨迹唯一ID的策略: "numeric" (例如: PlaneA_0) 或 "timestamp" (例如: PlaneA_20230101T100000M123)')
    parser.add_argument('--save_debug_segmented_file', action='store_true', help='如果设置，将在航段切分后保存一个用于调试的CSV文件')
    
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
        max_workers=args.max_workers,
        segment_split_minutes=args.segment_split_minutes,
        log_level=args.log_level,
        min_len_for_clean=args.min_len_for_clean,
        unique_id_strategy=args.unique_id_strategy,
        save_debug_segmented_file=args.save_debug_segmented_file
    )