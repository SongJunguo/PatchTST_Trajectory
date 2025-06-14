# -*- coding: utf-8 -*-
# ==============================================================================
# ** 高性能飞行数据预处理器 (Polars/Traffic版) **
#
# ** 版本: 2.2 (Final) **
#
# ** 依赖项: **
#   pip install polars numpy tqdm psutil traffic pandas
#
# ** 运行说明: **
#   python flight_data_preprocessor_traffic.py --input_dir [输入目录] --output_dir [输出目录]
#
# ** 核心改进点 (与原版对比): **
#   - 使用 traffic 库替换了自定义的平滑和清理函数。
#   - 采用多阶段滤波策略（去离群点 -> 预平滑 -> 特征工程 -> 全局平滑），
#     以稳健地处理高噪声数据。
#   - 增加保存预处理中间文件的功能，便于对比分析。
# ==============================================================================

# 1. Imports
import argparse
import glob
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from functools import partial
from typing import Optional

import polars as pl
from tqdm import tqdm

import numpy as np
from traffic.algorithms.filters import FilterAboveSigmaMedian, FilterMedian
from traffic.algorithms.filters.kalman import KalmanSmoother6D
from traffic.core import Flight


# 2. Logging Setup
def setup_logging(log_dir: str, log_level: str):
    """配置日志，使其同时输出到控制台和文件。"""
    log_filename = os.path.join(log_dir, "traffic_preprocessing.log")

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_filename, mode="w"),  # 写入文件，每次覆盖
            logging.StreamHandler(),  # 输出到控制台
        ],
    )


# 3. Constants
# 定义输入文件的Schema，强制数据类型可以提升读取速度并减少内存占用
SCHEMA = {
    "P1": pl.Utf8,  # 飞机ID
    "DTRQ": pl.Utf8,  # 日期时间字符串
    "JD": pl.Float64,  # 经度 (DMS格式)
    "WD": pl.Float64,  # 纬度 (DMS格式)
    "H": pl.Float64,  # 高度
}

# 4. Helper Functions


def dms_to_decimal_expr(dms_col: str) -> pl.Expr:
    """
    返回一个Polars表达式，用于将DMS格式(DD.MMSSff)的经纬度转换为十进制度。
    这是一个纯表达式实现，避免了低效的 .apply()。
    """
    degrees = pl.col(dms_col).floor()
    minutes = ((pl.col(dms_col) - degrees) * 100).floor()
    seconds = (((pl.col(dms_col) - degrees) * 100) - minutes) * 100
    return degrees + minutes / 60 + seconds / 3600


def _process_trajectory_worker_traffic(
    unique_id: str, temp_file_path: str, min_len_for_clean: int
) -> Optional[tuple[pl.DataFrame, pl.DataFrame]]:
    """
    使用 traffic 库处理单个航段的工作函数。
    遵循“预处理 -> 特征工程 -> 标准化 -> 全局平滑”的四阶段流程。
    """
    try:
        # 从临时文件中读取航段数据
        trajectory_df_polars = (
            pl.scan_parquet(temp_file_path)
            .filter(pl.col("Unique_ID") == unique_id)
            .collect()
        )

        if (
            trajectory_df_polars.is_empty()
            or len(trajectory_df_polars) < min_len_for_clean
        ):
            return None

        # 转换为 Pandas DataFrame 以便使用 traffic 库
        pandas_df = trajectory_df_polars.to_pandas()

        # 将填充值 0 替换为 NaN，以便 traffic 正确处理
        for col in ["Lon", "Lat", "H"]:
            pandas_df[col] = pandas_df[col].replace(0, np.nan)

        # 重命名列以符合 traffic 的标准
        flight_df = pandas_df.rename(
            columns={
                "Time": "timestamp",
                "Lon": "longitude",
                "Lat": "latitude",
                "H": "altitude",
                "Unique_ID": "flight_id",
            }
        )

        # 创建 Flight 对象
        flight = Flight(flight_df)

        # --- 第一阶段: 强力预处理 ---
        pre_filter = FilterAboveSigmaMedian() | FilterMedian()
        flight_pre_filtered = flight.filter(pre_filter, strategy=None)
        if flight_pre_filtered is None:
            return None

        # --- 第二阶段: 可靠的特征工程 ---
        flight_with_dynamics = flight_pre_filtered.cumulative_distance().rename(
            columns={"compute_gs": "groundspeed", "compute_track": "track"}
        )
        time_diff_seconds = (
            flight_with_dynamics.data.timestamp.diff().dt.total_seconds()
        )
        alt_diff_feet = flight_with_dynamics.data.altitude.diff()
        # 避免除以0，并将结果转换为 ft/min
        vertical_rate = (
            (alt_diff_feet / time_diff_seconds.replace(0, np.nan)) * 60
        ).bfill()
        flight_with_dynamics = flight_with_dynamics.assign(
            vertical_rate=vertical_rate
        )

        # --- 第三阶段: 标准化 ---
        resampled_flight = flight_with_dynamics.resample("1s")
        if resampled_flight is None:
            return None

        # --- 第四阶段: 全局最优平滑 ---
        # 卡尔曼滤波器在笛卡尔坐标系下工作，需要先进行坐标转换
        flight_with_xy = resampled_flight.compute_xy()
        final_smoother = KalmanSmoother6D()
        final_flight = flight_with_xy.filter(final_smoother)
        if final_flight is None:
            return None

        # 为对比准备预处理后的数据
        preprocessed_df_pandas = flight_pre_filtered.data.rename(
            columns={
                "timestamp": "Time",
                "longitude": "Lon",
                "latitude": "Lat",
                "altitude": "H",
                "flight_id": "Unique_ID",
            }
        )

        # 准备最终平滑后的数据
        final_df_pandas = final_flight.data.rename(
            columns={
                "timestamp": "Time",
                "longitude": "Lon",
                "latitude": "Lat",
                "altitude": "H",
                "flight_id": "Unique_ID",
            }
        )

        # 以元组形式返回两个DataFrame
        return (
            pl.from_pandas(preprocessed_df_pandas),
            pl.from_pandas(final_df_pandas),
        )

    except Exception as e:
        logging.error(
            f"使用 traffic 处理航段 {unique_id} 时发生错误: {e}", exc_info=True
        )
        return None


# 5. Main Processing Function
def process_flight_data(
    input_dir: str,
    output_dir: str,
    output_format: str,
    h_min: float,
    h_max: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    encoding_priority: str,
    max_workers: int,
    segment_split_minutes: int,
    log_level: str,
    min_len_for_clean: int,
    unique_id_strategy: str = "numeric",
    save_debug_segmented_file: bool = False,
):
    """
    主处理流程函数，包含了计划书中定义的四个阶段。
    """
    # 配置日志记录
    setup_logging(output_dir, log_level)

    logging.info("===== 开始飞行数据预处理流程 (Polars/Traffic版) =====")

    # --- 阶段一: 并行读取与预检 ---
    logging.info(f"--- 阶段一: 正在从 {input_dir} 扫描CSV文件... ---")
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not all_files:
        logging.error(f"在目录 {input_dir} 中未找到任何CSV文件。程序终止。")
        return

    # 根据用户指定的优先级确定编码尝试顺序
    if encoding_priority == "gbk":
        encodings_to_try = ["gbk", "utf8"]
    else:
        encodings_to_try = ["utf8", "gbk"]
    logging.info(
        f"编码检测顺序: {encodings_to_try[0]} -> {encodings_to_try[1]}"
    )

    lazy_frames = []
    failed_files = []
    for file in tqdm(all_files, desc="扫描文件"):
        df = None
        for encoding in encodings_to_try:
            try:
                # 使用 read_csv 进行稳健的编码尝试，成功后转为 lazy frame
                df = pl.read_csv(
                    file,
                    schema_overrides=SCHEMA,
                    ignore_errors=True,
                    separator=",",
                    encoding=encoding,
                )
                logging.info(
                    f"成功: 使用 '{encoding}' 编码读取文件: {os.path.basename(file)}"
                )
                break  # 成功读取，跳出内部循环
            except Exception:
                df = None  # 确保失败后 df 为 None
                continue  # 尝试下一种编码

        if df is not None and not df.is_empty():
            lazy_frames.append(df.lazy())
        else:
            logging.error(
                f"失败: 无法使用 {encodings_to_try} 解码文件 {os.path.basename(file)}，已跳过。"
            )
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
    transformed_lf = (
        lazy_df.with_columns(
            [
                # 日期解析
                pl.col("DTRQ")
                .str.to_datetime(format="%d-%b-%y %I.%M.%S%.f %p", strict=False)
                .alias("Time"),
                # 经纬度转换
                dms_to_decimal_expr("JD").alias("Lon"),
                dms_to_decimal_expr("WD").alias("Lat"),
                pl.col("P1").alias("ID"),
            ]
        )
        .select(["ID", "Time", "Lon", "Lat", "H"])
        .drop_nulls()
        .filter(
            # 注意：这里的0值过滤是第一道防线，更精细的处理在worker函数中完成
            (pl.col("H") != 0)
            & (pl.col("Lon") != 0)
            & (pl.col("Lat") != 0)
            &
            # 保留原有的地理和高度范围过滤
            (pl.col("H") >= h_min)
            & (pl.col("H") <= h_max)
            & (pl.col("Lon") >= lon_min)
            & (pl.col("Lon") <= lon_max)
            & (pl.col("Lat") >= lat_min)
            & (pl.col("Lat") <= lat_max)
        )
    )

    # --- 轨迹切分与ID生成 ---
    # 步骤1: 标记时间间隔超限的点，作为新航段的起点
    segmented_lf = transformed_lf.sort("ID", "Time").with_columns(
        (
            pl.col("Time").diff().over("ID")
            > timedelta(minutes=segment_split_minutes)
        )
        .fill_null(False)
        .alias("new_segment_marker")
    )

    # 步骤2: 使用累积和为每个航段生成一个数字ID
    segmented_lf = segmented_lf.with_columns(
        pl.col("new_segment_marker").cum_sum().over("ID").alias("segment_id")  # type: ignore
    )

    # 步骤3: 根据用户选择的策略生成最终的Unique_ID
    if unique_id_strategy == "numeric":
        logging.info("使用 'numeric' 策略生成轨迹ID (例如: PlaneA_0)")
        segmented_lf = segmented_lf.with_columns(
            (
                pl.col("ID").cast(pl.Utf8)
                + pl.lit("_")
                + pl.col("segment_id").cast(pl.Utf8)
            ).alias("Unique_ID")
        )
    elif unique_id_strategy == "timestamp":
        logging.info(
            "使用 'timestamp' 策略生成轨迹ID (例如: PlaneA_20230101T100000M123)"
        )
        # 获取每个航段的第一个时间点
        trajectory_start_time = pl.col("Time").first().over("ID", "segment_id")
        # 将起始时间格式化为 YYYYMMDDTHHMMSS.fff 的格式，然后用'M'替换'.'
        time_str_with_dot = trajectory_start_time.dt.strftime(
            "%Y%m%dT%H%M%S%.3f"
        )
        time_str_with_M = time_str_with_dot.str.replace(r".", "M", literal=True)

        segmented_lf = segmented_lf.with_columns(
            (pl.col("ID").cast(pl.Utf8) + pl.lit("_") + time_str_with_M).alias(
                "Unique_ID"
            )
        )

    # --- 增加健壮性检查: 确认有数据通过了过滤 ---
    logging.info("正在统计过滤和切分后的有效数据点总数...")
    # 注意：这次 collect 会触发完整的计算计划，但只为了获取行数，开销可控
    total_processed_rows = (
        segmented_lf.select(pl.len()).collect(streaming=True).item()
    )
    logging.info(
        f"经过过滤和轨迹切分后，剩余 {total_processed_rows:,} 个有效数据点。"
    )
    if total_raw_rows > 0:
        logging.info(f"数据保留率: {total_processed_rows / total_raw_rows:.2%}")

    if total_processed_rows == 0:
        logging.warning(
            "所有数据均在初始过滤阶段（如地理范围、有效日期等）被移除。没有可处理的航段。程序终止。"
        )
        return

    # --- (可选) 保存用于调试的中间文件 ---
    if save_debug_segmented_file:
        logging.info("正在以流式方式生成用于调试的航段分割文件 (内存安全)...")

        # 准备最终的 LazyFrame，包含所有需要的列选择和格式化逻辑
        debug_lf_to_save = segmented_lf.select(
            pl.col("Unique_ID").alias("ID"),
            pl.col("Time")
            .dt.strftime("%Y-%m-%d %H:%M:%S%.3f")
            .alias("Time"),  # 使用人类可读的时间格式
            "Lon",
            "Lat",
            "H",
            pl.col("ID").alias("Original_Plane_ID"),  # 保留原始飞机ID以供参考
        )

        # 定义输出路径并以流式方式写入，避免高内存占用
        debug_output_path = os.path.join(
            output_dir, "segmented_for_visualization.csv"
        )
        try:
            debug_lf_to_save.sink_csv(debug_output_path)
            logging.info(f"成功！已保存调试文件到: {debug_output_path}")
        except Exception as e:
            logging.error(f"保存调试文件失败: {e}", exc_info=True)

    # --- 阶段三: 真正的并行化平滑处理 ---
    logging.info("--- 阶段三: 准备进行并行化平滑处理... ---")

    temp_file = os.path.join(output_dir, "_temp_segmented_data.parquet")
    all_preprocessed_dfs = []
    all_final_dfs = []
    executor = ProcessPoolExecutor(max_workers=max_workers)

    try:
        logging.info("正在将分段数据写入临时文件以便并行读取...")
        # 在写入临时文件前，只选择后续处理需要的列
        final_segmented_lf = segmented_lf.select(
            "Unique_ID", "Time", "Lon", "Lat", "H"
        )
        final_segmented_lf.sink_parquet(temp_file, compression="zstd")

        unique_ids = (
            pl.scan_parquet(temp_file)
            .select("Unique_ID")
            .unique()
            .collect()["Unique_ID"]
            .to_list()
        )

        if not unique_ids:
            logging.warning("没有唯一的航段ID可供处理。")
            executor.shutdown()
            return

        logging.info(
            f"准备将 {len(unique_ids)} 个独立航段分发到 {max_workers} 个进程中进行处理。"
        )

        worker_func = partial(
            _process_trajectory_worker_traffic,
            temp_file_path=temp_file,
            min_len_for_clean=min_len_for_clean,
        )

        skipped_short_count = 0
        with tqdm(total=len(unique_ids), desc="并行处理航段") as pbar:
            for result in executor.map(worker_func, unique_ids):
                if result is None:
                    skipped_short_count += 1
                else:
                    preprocessed_df, final_df = result
                    if not preprocessed_df.is_empty():
                        all_preprocessed_dfs.append(preprocessed_df)
                    if not final_df.is_empty():
                        all_final_dfs.append(final_df)

                pbar.update(1)

        executor.shutdown(wait=True)  # Clean shutdown on success

        if skipped_short_count > 0:
            logging.info(
                f"共跳过 {skipped_short_count} 条因数据点过少(<{min_len_for_clean})的航段。"
            )

    except KeyboardInterrupt:
        logging.warning("\n--- 用户中断了处理流程。正在强制关闭工作进程... ---")
        # On interrupt, shutdown without waiting and cancel futures (Python 3.9+).
        executor.shutdown(wait=False, cancel_futures=True)
        return  # Exit the function
    finally:
        # This block ensures the temp file is always removed.
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logging.info(f"已清理临时文件: {temp_file}")

    if not all_final_dfs:
        logging.warning("所有航段处理后均为空，没有可保存的数据。")
        return

    # --- 阶段四: 保存结果 ---
    logging.info("--- 阶段四: 正在保存最终结果... ---")

    # 保存用于对比的预处理后文件
    if all_preprocessed_dfs:
        preprocessed_df = pl.concat(all_preprocessed_dfs)
        preprocessed_df = preprocessed_df.rename({"Unique_ID": "ID"}).select(
            ["ID", "Time", "Lon", "Lat", "H"]
        )
        # 修复：对中间文件也进行时间格式化
        preprocessed_df = preprocessed_df.with_columns(
            pl.col("Time").dt.strftime("%Y%m%d %H:%M:%S%.3f").alias("Time")
        )
        preprocessed_output_path = os.path.join(
            output_dir, f"preprocessed_for_comparison.{output_format}"
        )
        try:
            if output_format == "csv":
                preprocessed_df.write_csv(preprocessed_output_path)
            elif output_format == "parquet":
                preprocessed_df.write_parquet(preprocessed_output_path)
            logging.info(
                f"成功！已保存预处理对比文件到: {preprocessed_output_path}"
            )
        except Exception as e:
            logging.error(f"保存预处理对比文件时失败: {e}")

    # 保存最终平滑后的文件
    final_concat_df = pl.concat(all_final_dfs)
    if final_concat_df.is_empty():
        logging.warning("最终平滑处理后没有可保存的数据。")
    else:
        final_df = final_concat_df.rename({"Unique_ID": "ID"}).select(
            ["ID", "Time", "Lon", "Lat", "H"]
        )
        final_df = final_df.with_columns(
            pl.col("Time").dt.strftime("%Y%m%d %H:%M:%S%.3f").alias("Time")
        )
        output_filename = f"final_processed_trajectories.{output_format}"
        output_path = os.path.join(output_dir, output_filename)
        try:
            if output_format == "csv":
                final_df.write_csv(output_path)
            elif output_format == "parquet":
                final_df.write_parquet(output_path)

            logging.info(
                f"成功！已保存 {final_df['ID'].n_unique()} 条最终处理后的轨迹到 {output_path}"
            )
        except Exception as e:
            logging.error(f"保存最终文件到 {output_path} 时失败: {e}")

    if failed_files:
        logging.warning("以下文件在读取过程中失败，已被跳过:")
        for f in failed_files:
            logging.warning(f" - {f}")

    logging.info("===== 数据预处理流程全部完成 =====")


# 6. Main Execution Block
if __name__ == "__main__":
    # 根据Polars官方文档，为防止多进程死锁，推荐在Unix系统上使用 'spawn' 方法
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="高性能飞行数据预处理器 (Polars/Traffic版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="包含原始CSV文件的输入目录路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="用于存放处理后数据的输出目录路径",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="csv",
        choices=["csv", "parquet"],
        help="输出文件的格式",
    )
    parser.add_argument(
        "--encoding_priority",
        type=str,
        default="gbk",
        choices=["gbk", "utf8"],
        help="优先尝试的文件编码",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=os.cpu_count(),
        help="用于并行处理的最大工作进程数",
    )
    parser.add_argument(
        "--segment_split_minutes",
        type=int,
        default=5,
        help="用于切分轨迹的时间间隔（分钟）",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="设置日志记录级别",
    )
    parser.add_argument(
        "--min_len_for_clean",
        type=int,
        default=20,  # 降低最小长度要求，因为traffic可以处理更短的航段
        help="航段进行清洗（异常检测等）所需的最小数据点数",
    )
    parser.add_argument(
        "--unique_id_strategy",
        type=str,
        default="numeric",
        choices=["numeric", "timestamp"],
        help='生成轨迹唯一ID的策略: "numeric" (例如: PlaneA_0) 或 "timestamp" (例如: PlaneA_20230101T100000M123)',
    )
    parser.add_argument(
        "--save_debug_segmented_file",
        action="store_true",
        help="如果设置，将在航段切分后保存一个用于调试的CSV文件",
    )

    # 地理和高度过滤参数
    parser.add_argument("--h_min", type=float, default=0, help="最低高度 (米)")
    parser.add_argument(
        "--h_max", type=float, default=20000, help="最高高度 (米)"
    )
    parser.add_argument("--lon_min", type=float, default=-180, help="最小经度")
    parser.add_argument("--lon_max", type=float, default=180, help="最大经度")
    parser.add_argument("--lat_min", type=float, default=-90, help="最小纬度")
    parser.add_argument("--lat_max", type=float, default=90, help="最大纬度")

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
        save_debug_segmented_file=args.save_debug_segmented_file,
    )
