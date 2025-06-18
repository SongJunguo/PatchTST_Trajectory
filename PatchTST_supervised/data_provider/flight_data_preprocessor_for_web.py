# -*- coding: utf-8 -*-
# ==============================================================================
# ** 飞行数据预处理器 (Web可视化专用版) **
#
# ** 版本: 1.0 **
#
# ** 功能: **
#   - 读取原始飞行数据 (GBK/UTF-8编码)。
#   - 将度分秒格式的经纬度转换为十进制度。
#   - 根据P1和时间间隔切分航段。
#   - 为每个航段生成唯一的ID (P1_起始时间)。
#   - 对每个航段内的P1, TASK, PLANETYPE列进行众数填充。
#   - 对GP列进行前后填充。
#   - 按指定频率重采样数据。
#   - 将清洗后的数据保存为Parquet或CSV格式，编码为UTF-8。
#
# ** 运行说明: **
#   python flight_data_preprocessor_for_web.py --input_dir [输入目录] --output_dir [输出目录]
# ==============================================================================

import argparse
import glob
import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from datetime import timedelta

import polars as pl
from tqdm import tqdm

def setup_logging(log_dir: str, log_level: str):
    """配置日志，使其同时输出到控制台和文件。"""
    log_filename = os.path.join(log_dir, "web_preprocessing.log")
    os.makedirs(log_dir, exist_ok=True)
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, mode="w"),
            logging.StreamHandler(),
        ],
    )

# 定义输入文件的Schema
SCHEMA = {
    "PARTNO": pl.Utf8,
    "P1": pl.Utf8,
    "GP": pl.Utf8,
    "H": pl.Float64,
    "JD": pl.Float64,
    "WD": pl.Float64,
    "TASK": pl.Utf8,
    "PLANETYPE": pl.Utf8,
    "DTRQ": pl.Utf8,
}

def dms_to_decimal_expr(dms_col: str) -> pl.Expr:
    """返回一个Polars表达式，用于将DMS格式(DD.MMSSff)的经纬度转换为十进制度。"""
    degrees = pl.col(dms_col).floor()
    minutes = ((pl.col(dms_col) - degrees) * 100).floor()
    seconds = (((pl.col(dms_col) - degrees) * 100) - minutes) * 100
    return degrees + minutes / 60 + seconds / 3600

def process_single_trajectory(group_df: pl.DataFrame, resample_freq: str) -> pl.DataFrame | None:
    """处理单个轨迹的数据清洗和格式化"""
    if group_df.is_empty():
        return None

    # 1. 生成唯一ID
    first_row = group_df.head(1)
    p1_val = first_row["P1"][0]
    start_time = first_row["Time"][0]
    
    if p1_val is None or start_time is None:
        return None

    # 格式化时间为 YYYYMMDD_HHMMSS
    try:
        time_str = start_time.strftime("%Y%m%d_%H%M%S")
        unique_id = f"{p1_val}_{time_str}"
    except (TypeError, ValueError):
         # 如果时间格式化失败，跳过这个轨迹
        return None

    # 2. 众数填充
    p1_mode = group_df["P1"].mode()[0]
    task_mode = group_df["TASK"].mode()[0]
    planetype_mode = group_df["PLANETYPE"].mode()[0]

    # 3. 应用填充和转换
    processed_df = group_df.with_columns([
        pl.lit(unique_id).alias("ID"),
        pl.lit(p1_mode).alias("P1"),
        pl.lit(task_mode).alias("TASK"),
        pl.lit(planetype_mode).alias("PLANETYPE"),
        pl.col("GP").forward_fill().backward_fill()
    ])

    # 4. 重采样
    # Polars的resample需要一个有序的时间索引
    resampled_df = processed_df.sort("Time").group_by_dynamic(
        "Time", every=resample_freq
    ).agg([
        pl.first("ID"),
        pl.first("PARTNO"),
        pl.first("P1"),
        pl.first("GP"),
        pl.col("H").interpolate().forward_fill().backward_fill(),
        pl.col("JD").interpolate().forward_fill().backward_fill(),
        pl.col("WD").interpolate().forward_fill().backward_fill(),
        pl.first("TASK"),
        pl.first("PLANETYPE"),
    ])
    
    # 向前向后填充重采样后可能产生的空值
    resampled_df = resampled_df.with_columns(
        pl.col(pl.Utf8).forward_fill().backward_fill()
    )

    return resampled_df

def process_flight_data_for_web(
    input_dir: str,
    output_dir: str,
    output_format: str,
    encoding_priority: str,
    max_workers: int,
    segment_split_seconds: int,
    resample_freq: str,
    log_level: str,
):
    """主处理流程函数"""
    setup_logging(output_dir, log_level)
    logging.info("===== 开始Web可视化数据预处理流程 =====")
    
    args_dict = locals()
    logging.info("--- 运行参数配置 ---")
    for key, value in args_dict.items():
        logging.info(f"  - {key}: {value}")
    logging.info("--------------------")

    # --- 阶段一: 读取与初步转换 ---
    logging.info(f"--- 阶段一: 正在从 {input_dir} 扫描CSV文件... ---")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_files:
        logging.error(f"在目录 {input_dir} 中未找到任何CSV文件。程序终止。")
        return

    encodings_to_try = ["gbk", "utf8"] if encoding_priority == "gbk" else ["utf8", "gbk"]
    
    lazy_frames = []
    for file in tqdm(all_files, desc="扫描并读取文件"):
        df = None
        for encoding in encodings_to_try:
            try:
                df = pl.read_csv(
                    file,
                    schema_overrides=SCHEMA,
                    ignore_errors=True,
                    encoding=encoding,
                )
                break
            except Exception:
                continue
        if df is not None and not df.is_empty():
            lazy_frames.append(df.lazy())

    if not lazy_frames:
        logging.error("所有文件都加载失败。程序终止。")
        return

    lazy_df = pl.concat(lazy_frames)

    # --- 阶段二: 核心转换与轨迹切分 ---
    logging.info("--- 阶段二: 正在执行数据转换与轨迹切分... ---")
    transformed_lf = lazy_df.with_columns([
        pl.col("DTRQ").str.to_datetime(format="%d-%b-%y %I.%M.%S%.f %p", strict=False).alias("Time"),
        dms_to_decimal_expr("JD").alias("JD"),
        dms_to_decimal_expr("WD").alias("WD"),
    ]).drop_nulls(subset=["Time", "P1"]).sort("P1", "Time")

    # 标记时间间隔超限的点，作为新航段的起点
    segmented_lf = transformed_lf.with_columns(
        (pl.col("Time").diff().over("P1") > timedelta(seconds=segment_split_seconds))
        .fill_null(True) # 第一个点总是新航段的开始
        .alias("new_segment_marker")
    )

    # 使用累积和为每个航段生成一个数字ID
    segmented_lf = segmented_lf.with_columns(
        pl.col("new_segment_marker").cum_sum().over("P1").alias("segment_id")
    )
    
    # 创建轨迹的唯一标识符
    segmented_lf = segmented_lf.with_columns(
        (pl.col("P1").cast(pl.Utf8) + "_" + pl.col("segment_id").cast(pl.Utf8)).alias("trajectory_group_id")
    )
    
    logging.info("正在将数据收集到内存中以进行分组处理...")
    # .collect() 是必须的，因为后续的 group_by().apply() 不支持流式处理
    df_collected = segmented_lf.collect()

    # --- 阶段三: 并行处理每个轨迹 ---
    logging.info("--- 阶段三: 准备并行处理所有轨迹... ---")
    
    # 按轨迹ID分组
    trajectory_groups = df_collected.group_by("trajectory_group_id")
    
    all_processed_dfs = []
    
    # 使用 partial 来固定不变的参数
    worker_func = partial(process_single_trajectory, resample_freq=resample_freq)

    # 使用 ProcessPoolExecutor 进行并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 使用 executor.map 来处理每个分组
        # 注意：group_by返回的是一个元组(key, dataframe)，我们只需要dataframe
        results = list(tqdm(
            executor.map(worker_func, [group for _, group in trajectory_groups]),
            total=trajectory_groups.n_groups,
            desc="并行处理轨迹"
        ))

    # 过滤掉处理失败的 None 结果
    all_processed_dfs = [df for df in results if df is not None and not df.is_empty()]

    if not all_processed_dfs:
        logging.warning("所有航段处理后均为空，没有可保存的数据。")
        return

    # --- 阶段四: 合并与保存结果 ---
    logging.info("--- 阶段四: 正在合并结果并保存... ---")
    final_df = pl.concat(all_processed_dfs)
    
    # 格式化Time列以便输出
    final_df = final_df.with_columns(
        pl.col("Time").dt.strftime("%Y%m%d %H:%M:%S.%3f").str.slice(0, -3).alias("Time")
    )
    
    # 调整列顺序
    final_df = final_df.select([
        "ID", "PARTNO", "P1", "GP", "H", "JD", "WD", "TASK", "PLANETYPE", "Time"
    ])

    output_filename = f"history_data.{output_format}"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        if output_format == "csv":
            final_df.write_csv(output_path)
        elif output_format == "parquet":
            final_df.write_parquet(output_path, compression="zstd")
        logging.info(f"成功！已保存 {final_df.height} 行数据到 {output_path}")
    except Exception as e:
        logging.error(f"保存最终文件到 {output_path} 时失败: {e}")

    logging.info("===== Web可视化数据预处理流程全部完成 =====")


if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="飞行数据预处理器 (Web可视化专用版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input_dir", type=str, required=True, help="包含原始CSV文件的输入目录路径")
    parser.add_argument("--output_dir", type=str, required=True, help="用于存放处理后数据的输出目录路径")
    parser.add_argument("--output_format", type=str, default="parquet", choices=["csv", "parquet"], help="输出文件的格式")
    parser.add_argument("--encoding_priority", type=str, default="gbk", choices=["gbk", "utf8"], help="优先尝试的文件编码")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count(), help="用于并行处理的最大工作进程数")
    parser.add_argument("--segment_split_seconds", type=int, default=300, help="用于切分轨迹的时间间隔（秒）")
    parser.add_argument("--resample_freq", type=str, default="1s", help="重采样的时间频率 (例如: '1s', '5s')")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="设置日志记录级别")
    
    args = parser.parse_args()

    process_flight_data_for_web(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_format=args.output_format,
        encoding_priority=args.encoding_priority,
        max_workers=args.max_workers,
        segment_split_seconds=args.segment_split_seconds,
        resample_freq=args.resample_freq,
        log_level=args.log_level,
    )