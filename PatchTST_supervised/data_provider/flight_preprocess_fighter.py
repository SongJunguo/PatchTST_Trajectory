# -*- coding: utf-8 -*-
# ==============================================================================
# ** 高性能飞行数据预处理器 (Polars/Traffic版) **
#
# ** 版本: 2.4 (Final with NaN trimming) **
#
# ** 依赖项: **
#   pip install polars numpy tqdm psutil traffic pandas
#
# ** 运行说明: **
#   python flight_data_preprocessor_traffic.py --input_dir [输入目录] --output_dir [输出目录]
#
# ** 核心改进点 (与原版对比): **
#   - 使用 traffic 库替换了自定义的平滑和清理函数。
#   - 采用多阶段滤波策略（去离群点 -> 预平滑 -> 特征工程 -> 全局平滑）。
#   - 增加保存预处理中间文件的功能，便于对比分析。
#   - 实现了健壮的错误捕获和日志记录机制。
#   - 增加删除边界NaN的功能，避免不合理的填充。
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

import polars as pl
from tqdm import tqdm

import numpy as np
from traffic.algorithms.filters.kalman import KalmanSmoother6D
from traffic.core import Flight

from .detect_height_anomalies import detect_height_anomalies


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
    "PLANETYPE": pl.Utf8,  # 飞机类型
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
    unique_id: str,
    temp_file_path: str,
    min_len_for_clean: int,
    min_valid_block_len: int,
    resample_freq: str,
    max_displacement_degrees: float,
    anomaly_max_duration: int,
    anomaly_max_rate: float,
    max_latlon_speed: float,
    max_alt_speed: float,
    log_level: str,
    output_dir: str,
    h_min: float,
    h_max: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> dict:
    """
    使用 traffic 库处理单个航段的工作函数。
    遵循“预处理 -> 特征工程 -> 标准化 -> 全局平滑”的四阶段流程。
    返回一个状态字典。
    """
    try:
        # ==================== 子进程日志初始化 ====================
        # 在每个工作进程开始时，必须重新配置日志记录器。
        # 这是因为子进程不会从父进程继承日志配置。
        # 通过在每个worker中独立设置，可以确保日志消息被正确捕获。
        setup_logging(output_dir, log_level)
        # =======================================================

        # 从临时文件中读取航段数据
        trajectory_df_polars = (
            pl.scan_parquet(temp_file_path)
            .filter(pl.col("Unique_ID") == unique_id)
            .collect()
        )

        if trajectory_df_polars.is_empty():
            return {
                "status": "skipped",
                "reason": "empty before processing",
            }

        # 转换为 Pandas DataFrame 以便使用 traffic 库
        pandas_df = trajectory_df_polars.to_pandas()

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

        # --- 数据清洗步骤 1 & 2: 处理重复和静态数据点 ---
        # 理由: 保证每个时间戳唯一，并移除冗余的静态数据点，以确保后续计算的有效性。

        # 步骤 1: 中位数聚合处理重复时间戳
        if flight_df.timestamp.duplicated().any():
            logging.info(
                f"轨迹ID '{unique_id}' 检测到重复时间戳，将执行中位数聚合..."
            )
            flight_df = (
                flight_df.groupby("timestamp")
                .agg(
                    {
                        "longitude": "median",
                        "latitude": "median",
                        "altitude": "median",
                        "flight_id": "first",
                    }
                )
                .reset_index()
            )

        # 步骤 2: 静态数据点去重
        static_cols = ["longitude", "latitude", "altitude"]
        has_changed = flight_df[static_cols].diff().abs().sum(axis=1) > 0

        # 使用.loc来避免FutureWarning
        shifted = has_changed.shift(1)
        shifted.loc[shifted.isna()] = True
        is_first_in_static_segment = shifted.astype(bool)

        rows_to_keep = has_changed | is_first_in_static_segment

        original_rows = len(flight_df)
        flight_df = flight_df[rows_to_keep]
        new_rows = len(flight_df)

        if original_rows > new_rows:
            logging.info(
                f"轨迹ID '{unique_id}' 清理了 {original_rows - new_rows} 个连续的静态数据点。"
            )

        # --- 清洗结束: 强制统一列顺序 ---
        # 理由: 确保无论是否经过groupby，列顺序都保持一致，避免后续concat失败。
        flight_df = flight_df[
            ["timestamp", "longitude", "latitude", "altitude", "flight_id"]
        ]

        # 创建 Flight 对象
        flight = Flight(flight_df)

        # --- 第一阶段: 强力预处理与边界清理 ---
        # pre_filter = FilterAboveSigmaMedian() | FilterMedian()
        # pre_filter = FilterMedian()
        # flight_pre_filtered = flight.filter(pre_filter, strategy=None)
        # if flight_pre_filtered is None:
        #     return {"status": "skipped", "reason": "empty after pre-filtering"}
        flight_pre_filtered = flight
        # 清理边界NaN: 根据至少3个连续有效点的原则，删除轨迹开头和结尾的连续NaN值
        pre_filtered_df = flight_pre_filtered.data

        # 步骤2.1: 处理轨迹首尾的连续缺失值
        # 理由: 确保每条轨迹都有一个“干净”的开头和结尾，至少包含3个连续的有效点，
        # 这是后续算法（如卡尔曼滤波器）正常初始化和运行的硬性要求。
        key_cols = ["altitude", "longitude", "latitude"]
        is_valid = pre_filtered_df[key_cols].notna().all(axis=1)

        # 至少需要 min_valid_block_len 个有效点才能形成一个块
        if is_valid.sum() < min_valid_block_len:
            return {
                "status": "skipped",
                "reason": f"fewer than {min_valid_block_len} valid data points in total",
            }

        # 使用滚动窗口找到所有连续 min_valid_block_len 个有效点的块的结束位置 (使用整数位置)
        rolling_sum = is_valid.rolling(
            window=min_valid_block_len, min_periods=min_valid_block_len
        ).sum()
        valid_block_end_ilocs = np.where(
            rolling_sum.to_numpy() == min_valid_block_len
        )[0]

        # 如果找不到任何这样的块，则跳过
        if len(valid_block_end_ilocs) == 0:
            return {
                "status": "skipped",
                "reason": f"no block of {min_valid_block_len} consecutive valid points found",
            }

        # 确定裁剪的开始和结束整数位置
        # 开始位置是第一个有效块的第一个点
        first_valid_block_end_iloc = valid_block_end_ilocs[0]
        start_iloc = first_valid_block_end_iloc - (min_valid_block_len - 1)

        # 结束位置是最后一个有效块的最后一个点
        end_iloc = valid_block_end_ilocs[-1]

        # 使用 .iloc 进行切片，确保我们选择的是一个连续的数据块
        trimmed_df = pre_filtered_df.iloc[start_iloc : end_iloc + 1]

        if trimmed_df.empty:
            return {
                "status": "skipped",
                "reason": "empty after trimming leading/trailing NaNs",
            }

        # --- (新增) 基于最大位移的过境航班过滤 ---
        # 理由: 排除长距离的过境航班，只保留区域内的起降或本地活动。
        first_point = trimmed_df.iloc[0]
        last_point = trimmed_df.iloc[-1]
        lon_displacement = abs(last_point.longitude - first_point.longitude)
        lat_displacement = abs(last_point.latitude - first_point.latitude)

        if (
            lon_displacement > max_displacement_degrees
            or lat_displacement > max_displacement_degrees
        ):
            return {
                "status": "skipped",
                "reason": f"Exceeded max displacement threshold ({max_displacement_degrees} degrees)",
            }
        # --- 过滤结束 ---

        # 在清理边界后，再次检查航段长度是否满足后续处理的要求
        if len(trimmed_df) < min_len_for_clean:
            return {
                "status": "skipped",
                "reason": f"too short after trimming (len={len(trimmed_df)} < {min_len_for_clean})",
            }

        flight_trimmed = Flight(trimmed_df)

        # --- 第二阶段: 可靠的特征工程 ---
        # 关键修正: traffic库默认高度单位为英尺，在此处将米转换为英尺
        flight_trimmed = flight_trimmed.assign(
            altitude=lambda df: df.altitude
            / 0.3048  # 1 foot = 0.3048 meters exactly
        )
        flight_with_dynamics = flight_trimmed.cumulative_distance().rename(
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
        # resample 会自动插值填充内部的 NaN
        resampled_flight = flight_with_dynamics.resample(resample_freq)
        if resampled_flight is None:
            return {"status": "skipped", "reason": "empty after resampling"}

        # ==================== 新增：高度异常检测 ====================
        resampled_df = resampled_flight.data

        detection_result = detect_height_anomalies(
            heights=resampled_df["altitude"].to_numpy(),
            times=resampled_df["timestamp"].to_numpy(),
            max_rate=anomaly_max_rate,
            max_duration=anomaly_max_duration,
        )

        # **关键逻辑：只有在成功找到并标记了异常段时才执行修改**
        if detection_result["success"]:
            logging.info(
                f"轨迹ID '{unique_id}' 检测到并成功界定了 {len(detection_result['anomaly_info'])} 个高度异常段。将进行清理和重新插值。"
            )

            cleaned_df = resampled_df[detection_result["mask"]]
            flight_after_anomaly_removal = Flight(cleaned_df)
            resampled_flight = flight_after_anomaly_removal.resample(
                resample_freq
            )

            if resampled_flight is None:
                return {
                    "status": "skipped",
                    "reason": "empty after anomaly removal",
                }
        else:
            logging.debug(
                f"轨迹ID '{unique_id}' 未检测到可处理的高度异常段，将继续执行后续流程。"
            )
            # **不执行任何操作，resampled_flight 保持原样**
        # ==================== 新增逻辑结束 ====================

        # ==================== 新增：基于速度的异常段检测与切分 ====================
        # 根据用户需求，在高度异常清理后，进行速度合规性检查。
        # 核心逻辑：识别速度超限的连续数据段，丢弃它们，并将剩余的正常数据段切分为新的独立轨迹。
        df_for_speed_check = resampled_flight.data.copy()
        original_flight_id = df_for_speed_check["flight_id"].iloc[0]

        # 步骤 1: 计算速度 (注意单位：高度为英尺)
        time_diff = df_for_speed_check["timestamp"].diff().dt.total_seconds()
        # 避免除以零
        time_diff_safe = time_diff.replace(0, np.nan)

        lon_speed = (
            df_for_speed_check["longitude"].diff().abs() / time_diff_safe
        )
        lat_speed = (
            df_for_speed_check["latitude"].diff().abs() / time_diff_safe
        )
        # 将用户输入的 m/s 阈值转换为 ft/s
        alt_speed_threshold_fts = max_alt_speed * 3.28084
        alt_speed = (
            df_for_speed_check["altitude"].diff().abs() / time_diff_safe
        )

        # 步骤 2: 标记每个点是否超速
        df_for_speed_check["is_abnormal"] = (
            (lon_speed > max_latlon_speed)
            | (lat_speed > max_latlon_speed)
            | (alt_speed > alt_speed_threshold_fts)
        ).fillna(False)  # 第一个点为NaN，填充为False

        # 步骤 3: 如果检测到异常，则执行切分和丢弃
        if df_for_speed_check["is_abnormal"].any():
            logging.info(
                f"轨迹ID '{original_flight_id}' 检测到速度超限段，将执行丢弃和切分..."
            )

            # 步骤 3.1: 识别连续的正常/异常段落
            state_change = df_for_speed_check["is_abnormal"].ne(
                df_for_speed_check["is_abnormal"].shift()
            )
            df_for_speed_check["block_id"] = state_change.cumsum()

            # 步骤 3.2: 丢弃所有异常段
            df_cleaned = df_for_speed_check[
                ~df_for_speed_check["is_abnormal"]
            ].copy()

            if df_cleaned.is_empty():
                return {
                    "status": "skipped",
                    "reason": "empty after velocity anomaly removal",
                }

            # 步骤 3.3: 为幸存的正常段生成新的唯一ID
            # 使用 groupby(..., group_keys=False) 来避免未来的警告
            df_cleaned["flight_id"] = df_cleaned.groupby(
                "block_id", group_keys=False
            ).ngroup()
            df_cleaned["flight_id"] = (
                f"{original_flight_id}_v"
                + df_cleaned["flight_id"].astype(str)
            )

            # 步骤 3.4: 丢弃长度过短的新航段
            # 注意：这里需要重新计算每个新ID的长度
            id_counts = df_cleaned["flight_id"].value_counts()
            valid_ids = id_counts[id_counts >= min_len_for_clean].index
            
            if len(valid_ids) == 0:
                return {
                    "status": "skipped",
                    "reason": f"all segments shorter than {min_len_for_clean} after velocity split",
                }

            df_final_split = df_cleaned[
                df_cleaned["flight_id"].isin(valid_ids)
            ].copy()
            
            logging.info(
                f"轨迹ID '{original_flight_id}' 被切分为 {len(valid_ids)} 个新的有效航段。"
            )

            # 用处理后的数据更新 resampled_flight
            resampled_flight = Flight(df_final_split)
        # ==================== 速度限制逻辑结束 ====================

        # --- (新增) 健壮性检查: 验证重采样和插值后是否存在NaN ---
        # 理由: 确保 traffic 的 resample/interpolate 行为符合预期，没有留下任何NaN。
        # 这有助于在早期阶段发现数据问题，而不是等到最终的卡尔曼滤波器步骤。
        resampled_df = resampled_flight.data
        # 检查除flight_id之外的所有列
        nan_check_cols = resampled_df.columns.drop("flight_id")
        nan_in_cols = [
            col for col in nan_check_cols if resampled_df[col].isnull().any()
        ]

        if nan_in_cols:
            logging.warning(
                f"严重警告: 轨迹ID '{unique_id}' 在重采样(resample)后，以下数据列仍检测到NaN值: {nan_in_cols}。"
                " 这可能表明插值未能覆盖所有间隙。后续流程将尝试修复。"
            )

        # --- 第四阶段: 全局最优平滑 ---
        # 卡尔曼滤波器在笛卡尔坐标系下工作，需要先进行坐标转换
        flight_with_xy = resampled_flight.compute_xy()
        final_smoother = KalmanSmoother6D()

        # --- 健壮性检查与强制清理: 在滤波前确保数据有效 ---
        # 理由: 卡尔曼滤波器对NaN或inf值非常敏感，会导致矩阵运算失败。
        # 即使经过resample，某些边缘情况或上游计算仍可能引入无效值。
        df_to_clean = flight_with_xy.data
        key_cols_for_smoothing = [
            "x",
            "y",
            "altitude",
            "groundspeed",
            "vertical_rate",
        ]

        # 检查是否存在NaN或inf
        problematic_cols = []
        for col in key_cols_for_smoothing:
            # 直接在Pandas DataFrame上检查，而不是在NumPy数组上
            if (
                df_to_clean[col].isnull().any()
                or np.isinf(df_to_clean[col]).any()
            ):
                problematic_cols.append(col)

        if problematic_cols:
            logging.critical(
                f"严重警告: 轨迹ID '{unique_id}' 在送入卡尔曼滤波器前检测到NaN或inf值！"
                f" 这很可能导致 'matmul' 错误。问题列: {problematic_cols}"
            )

        # 强制进行稳健的插值和填充，以消除任何剩余的无效值
        # 1. 线性插值
        # 2. 向前填充
        # 3. 向后填充 (确保开头没有NaN)
        df_to_clean[key_cols_for_smoothing] = (
            df_to_clean[key_cols_for_smoothing]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

        flight_with_xy = Flight(df_to_clean)
        # --- 清理结束 ---

        # --- (新增) 卡尔曼滤波器健壮性补丁 ---
        # 理由: 如果高度长时间不变，其标准差为0，会导致卡尔曼滤波器内部的R矩阵
        #      在对应维度上为0，进而可能导致S矩阵奇异，求逆失败产生NaN。
        # 补丁: 在送入滤波器前，检查高度标准差。如果接近于0，则加入微小噪声。
        df_for_smoothing = flight_with_xy.data
        # 使用一个足够小的阈值来判断标准差是否接近于零
        if df_for_smoothing["altitude"].std() < 1e-6:
            logging.warning(
                f"轨迹ID '{unique_id}' 的高度数据几乎没有变化。为防止卡尔曼滤波器数值不稳定，"
                "将向高度列添加微小的随机噪声。"
            )
            # 创建一个与DataFrame长度相同的、非常小的噪声
            noise = np.random.normal(0, 1e-6, len(df_for_smoothing))
            # 使用 .loc 避免 SettingWithCopyWarning
            df_for_smoothing.loc[:, "altitude"] = (
                df_for_smoothing["altitude"] + noise
            )
            flight_with_xy = Flight(df_for_smoothing)
        # --- 补丁结束 ---

        final_flight = flight_with_xy.filter(final_smoother)
        if final_flight is None:
            return {
                "status": "skipped",
                "reason": "empty after final smoothing",
            }

        # 新增：根据用户要求，在最终平滑后再次进行长度检查
        if len(final_flight.data) < min_len_for_clean:
            return {
                "status": "skipped",
                "reason": f"too short after final smoothing (len={len(final_flight.data)} < {min_len_for_clean})",
            }

        # --- 第五阶段: 单位转换与输出准备 ---
        # 在保存前，将高度从英尺转换回米，以保持与输入数据的一致性
        final_flight_meters = final_flight.assign(
            altitude=lambda df: df.altitude * 0.3048
        )
        # 对比文件也需要转换回来
        flight_trimmed_meters = flight_trimmed.assign(
            altitude=lambda df: df.altitude * 0.3048
        )

        # 为对比准备预处理后的数据
        # 注意：我们使用裁剪后的 flight_trimmed 的数据作为对比基准
        preprocessed_df_pandas = flight_trimmed_meters.data.rename(
            columns={
                "timestamp": "Time",
                "longitude": "Lon",
                "latitude": "Lat",
                "altitude": "H",
                "flight_id": "Unique_ID",
            }
        )

        # 准备最终平滑后的数据
        final_df_pandas = final_flight_meters.data.rename(
            columns={
                "timestamp": "Time",
                "longitude": "Lon",
                "latitude": "Lat",
                "altitude": "H",
                "flight_id": "Unique_ID",
            }
        )

        # --- 第六阶段 (新增): Worker内部的零容忍预检查 ---
        # 在将数据返回给主进程前，进行一次快速预检。
        # 这是第一道防线，确保从worker流出的数据是干净的。
        check_cols = ["Lon", "Lat", "H"]
        if final_df_pandas[check_cols].isnull().values.any():
            logging.critical(
                f"严重警告: 轨迹ID '{unique_id}' 在工作进程内部的最终检查中发现残留NaN值！"
                " 该轨迹将被丢弃。这表明平滑或插值步骤可能存在问题。"
            )
            return {
                "status": "error",
                "reason": "Post-smoothing NaN check failed inside worker",
            }

        # --- 第七阶段 (新增): Worker内部的最终范围验证 ---
        # 在将数据返回给主进程前，进行最后一次地理和高度范围检查。
        # 这是根据用户要求，在每个worker内部完成的最终验证。
        if not (
            final_df_pandas["Lon"].between(lon_min, lon_max).all()
            and final_df_pandas["Lat"].between(lat_min, lat_max).all()
            and final_df_pandas["H"].between(h_min, h_max).all()
        ):
            logging.warning(
                f"轨迹ID '{unique_id}' 在最终平滑后，有数据点超出了预设的地理或高度范围，将被丢弃。"
            )
            return {
                "status": "skipped",
                "reason": "Post-smoothing out-of-bounds check failed inside worker",
            }

        return {
            "status": "success",
            "data": (
                pl.from_pandas(preprocessed_df_pandas),
                pl.from_pandas(final_df_pandas),
            ),
        }

    except np.linalg.LinAlgError:
        # 捕获奇异矩阵错误，并作为特定失败原因返回
        return {"status": "error", "reason": "Singular matrix"}
    except Exception as e:
        # 捕获所有其他异常
        return {"status": "error", "reason": str(e)}


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
    segment_split_seconds: int,
    log_level: str,
    min_len_for_clean: int,
    min_valid_block_len: int,
    unique_id_strategy: str = "numeric",
    save_debug_segmented_file: bool = False,
    resample_freq: str = "1s",
    max_displacement_degrees: float = 2.0,
    anomaly_max_duration: int = 20,
    anomaly_max_rate: float = 100.0,
    max_latlon_speed: float = 0.01,
    max_alt_speed: float = 100.0,
):
    """
    主处理流程函数，包含了计划书中定义的四个阶段。
    """
    # 配置日志记录
    setup_logging(output_dir, log_level)

    logging.info("===== 开始飞行数据预处理流程 (Polars/Traffic版) =====")

    # --- 记录所有可调参数 ---
    logging.info("--- 运行参数配置 ---")
    args_dict = locals()
    for key, value in args_dict.copy().items():
        logging.info(f"  - {key}: {value}")
    logging.info("--------------------")

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

    # 新增：根据PLANETYPE筛选包含'歼'的行并随后删除该列
    logging.info("--- 新增筛选: 正在根据 PLANETYPE 列筛选含'歼'的机型... ---")
    try:
        # 为了日志记录，我们需要在这里触发一次计算来获取行数
        # 这会扫描数据，但对于监控数据处理流程是必要的
        # 使用 streaming 引擎来降低内存压力
        original_count = lazy_df.select(pl.len()).collect(engine="streaming").item()

        # 应用过滤器，只保留 PLANETYPE 列中包含 "歼" 的行
        lazy_df = lazy_df.filter(pl.col("PLANETYPE").str.contains("歼", literal=True))

        # 获取筛选后的行数
        filtered_count = lazy_df.select(pl.len()).collect(engine="streaming").item()
        
        dropped_count = original_count - filtered_count
        if original_count > 0:
            loss_rate = (dropped_count / original_count) * 100
            logging.info(f"  - PLANETYPE筛选前数据点: {original_count:,}")
            logging.info(f"  - 筛选后剩余: {filtered_count:,}")
            logging.info(f"  - 因PLANETYPE不含'歼'字丢弃: {dropped_count:,} ({loss_rate:.2f}%)")
        else:
            logging.info("  - PLANETYPE筛选前无数据，跳过统计。")
        
        # 筛选完成后，删除 PLANETYPE 列
        lazy_df = lazy_df.drop("PLANETYPE")
        logging.info("已成功删除 'PLANETYPE' 列。")

    except Exception as e:
        logging.error(f"在PLANETYPE筛选过程中发生错误: {e}", exc_info=True)
        logging.warning("由于PLANETYPE筛选时发生错误，将跳过此筛选步骤。")
    # --- 筛选结束 ---

    if log_level.upper() == "DEBUG":
        # --- DEBUG: 检查高度列的类型转换问题 ---
        logging.info(
            "--- [DEBUG]  检查 'H' 列是否存在无法转换为Float64的值 ---"
        )
        # 使用 strict=False 来将无法转换的值变为 null，然后统计 null 的数量
        # 我们在一个新的 LazyFrame 上操作，不影响主流程
        height_check_lf = lazy_df.with_columns(
            pl.col("H").cast(pl.Float64, strict=False).alias("H_casted")
        )

        # 计算转换失败的行数
        failed_cast_count = (
            height_check_lf.filter(
                pl.col("H_casted").is_null() & pl.col("H").is_not_null()
            )
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )

        if failed_cast_count > 0:
            logging.warning(
                f"发现 {failed_cast_count} 个 'H' 列的值无法被转换为Float64。"
            )
            # 显示一些转换失败的例子
            failed_examples = (
                height_check_lf.filter(
                    pl.col("H_casted").is_null() & pl.col("H").is_not_null()
                )
                .head(5)
                .collect(engine="streaming")
            )
            logging.warning(f"无法转换的 'H' 列值示例:\n{failed_examples}")
        else:
            logging.info("'H' 列的所有值都可以被成功转换为Float64。")
        # --- DEBUG END ---

    # --- 阶段二: 核心转换 (Lazy & Parallel) ---
    logging.info("正在统计原始数据点总数...")
    total_raw_rows = lazy_df.select(pl.len()).collect().item()
    logging.info(f"所有CSV文件共包含 {total_raw_rows:,} 个原始数据点。")

    logging.info("--- 阶段二: 正在执行惰性数据转换与轨迹切分... ---")

    # 核心转换表达式链
    transformed_lf = lazy_df.with_columns(
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
    ).select(["ID", "Time", "Lon", "Lat", "H"])

    if log_level.upper() == "DEBUG":
        logging.info("--- [DEBUG]  核心转换后，清理null值前的数据状态 ---")
        # .collect() 会触发计算，所以我们只在一个小的子集上操作
        transformed_sample_df = transformed_lf.head(5).collect(
            engine="streaming"
        )
        logging.info(f"转换后Schema: {transformed_sample_df.schema}")
        logging.info(f"转换后前5行数据:\n{transformed_sample_df}")

        # 统计null值，这会触发一次全表扫描，但对于调试是必要的
        null_counts = transformed_lf.select(
            [
                pl.col(c).is_null().sum().alias(f"{c}_null_count")
                for c in transformed_lf.schema.keys()
            ]
        ).collect(engine="streaming")
        logging.info(f"各列的Null值统计:\n{null_counts}")
        # --- DEBUG END ---

    # 步骤 1: 清理意外的 null 值
    df_after_drop_nulls = transformed_lf.drop_nulls()

    if log_level.upper() == "DEBUG":
        # --- DEBUG: 检查清理null值后的数据 ---
        logging.info("--- [DEBUG]  调用 drop_nulls() 后的数据状态 ---")
        # 同样，只对头部进行collect以避免大的性能开销
        df_after_drop_nulls_sample = df_after_drop_nulls.head(5).collect(
            engine="streaming"
        )
        logging.info(f"drop_nulls() 后前5行数据:\n{df_after_drop_nulls_sample}")
        # --- DEBUG END ---

    # 步骤 2: 增强日志统计
    rows_after_drop_nulls = (
        df_after_drop_nulls.select(pl.len()).collect(engine="streaming").item()
    )
    dropped_count = total_raw_rows - rows_after_drop_nulls
    loss_rate = (
        (dropped_count / total_raw_rows) * 100 if total_raw_rows > 0 else 0
    )

    logging.info("初始 `drop_nulls` 操作完成:")
    logging.info(f"  - 原始数据点: {total_raw_rows:,}")
    logging.info(f"  - 清理后剩余: {rows_after_drop_nulls:,}")
    logging.info(f"  - 因null值丢弃: {dropped_count:,} ({loss_rate:.2f}%)")

    # 步骤 3: 将 0 替换为 null，原始数据使用0代替nan进行缺失数值填充
    df_with_nulls_from_zeros = df_after_drop_nulls.with_columns(
        [
            pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c)
            for c in ["Lon", "Lat", "H"]
        ]
    )

    # 步骤 4: 进行地理和高度范围过滤
    transformed_lf = df_with_nulls_from_zeros.filter(
        (pl.col("H") >= h_min)
        & (pl.col("H") <= h_max)
        & (pl.col("Lon") >= lon_min)
        & (pl.col("Lon") <= lon_max)
        & (pl.col("Lat") >= lat_min)
        & (pl.col("Lat") <= lat_max)
    )

    # --- 轨迹切分与ID生成 ---
    # 步骤1: 标记时间间隔超限的点，作为新航段的起点
    segmented_lf = transformed_lf.sort("ID", "Time").with_columns(
        (
            pl.col("Time").diff().over("ID")
            > timedelta(seconds=segment_split_seconds)
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
                pl.col("ID").cast(pl.Utf8) + pl.col("segment_id").cast(pl.Utf8)
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
            "%Y%m%d%H%M%S%.3f"
        )
        time_str_with_M = time_str_with_dot.str.replace(r".", "", literal=True)

        segmented_lf = segmented_lf.with_columns(
            (pl.col("ID").cast(pl.Utf8) + time_str_with_M).alias("Unique_ID")
        )

    # --- 增加健壮性检查: 确认有数据通过了过滤 ---
    logging.info("正在统计过滤和切分后的有效数据点总数...")
    # 注意：这次 collect 会触发完整的计算计划，但只为了获取行数，开销可控
    total_processed_rows = (
        segmented_lf.select(pl.len()).collect(engine="streaming").item()
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
    failed_trajectories_log = []
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
            min_valid_block_len=min_valid_block_len,
            resample_freq=resample_freq,
            max_displacement_degrees=max_displacement_degrees,
            anomaly_max_duration=anomaly_max_duration,
            anomaly_max_rate=anomaly_max_rate,
            max_latlon_speed=max_latlon_speed,
            max_alt_speed=max_alt_speed,
            log_level=log_level,
            output_dir=output_dir,
            h_min=h_min,
            h_max=h_max,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
        )

        skipped_short_count = 0
        with tqdm(total=len(unique_ids), desc="并行处理航段") as pbar:
            for unique_id, result in zip(
                unique_ids, executor.map(worker_func, unique_ids)
            ):
                if result["status"] == "success":
                    preprocessed_df, final_df = result["data"]
                    if not preprocessed_df.is_empty():
                        all_preprocessed_dfs.append(preprocessed_df)
                    if not final_df.is_empty():
                        all_final_dfs.append(final_df)
                else:
                    # 记录失败或跳过的航段信息
                    original_id = unique_id.split("_")[0]
                    log_entry = {
                        "unique_id": unique_id,
                        "original_id": original_id,
                        "status": result["status"],
                        "reason": result["reason"],
                    }
                    failed_trajectories_log.append(log_entry)
                    if result["status"] == "skipped":
                        skipped_short_count += 1

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

    if not all_final_dfs and not all_preprocessed_dfs:
        logging.warning("所有航段处理后均为空，没有可保存的数据。")
        # 即使没有成功的数据，也要检查是否有错误日志需要保存
        if failed_trajectories_log:
            error_log_df = pl.DataFrame(failed_trajectories_log)
            error_log_path = os.path.join(output_dir, "error_log.csv")
            try:
                error_log_df.write_csv(error_log_path)
                logging.info(
                    f"检测到处理失败的航段，详情已记录到: {error_log_path}"
                )
            except Exception as e:
                logging.error(f"保存错误日志文件时失败: {e}")
        return

    # --- 阶段四: 保存结果 ---
    logging.info("--- 阶段四: 正在保存最终结果... ---")

    # 保存用于对比的预处理后文件
    if all_preprocessed_dfs:
        preprocessed_df = pl.concat(all_preprocessed_dfs)
        preprocessed_df = preprocessed_df.with_columns(
            Time=pl.col("Time").dt.strftime("%Y%m%d %H:%M:%S%.3f")
        ).select(
            pl.col("Unique_ID").alias("ID"),
            pl.col("H").alias("H"),
            pl.col("Lon").alias("Lon"),
            pl.col("Lat").alias("Lat"),
            pl.col("Time").alias("Time"),
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
    if all_final_dfs:
        final_concat_df = pl.concat(all_final_dfs)
        if final_concat_df.is_empty():
            logging.warning("最终平滑处理后没有可保存的数据。")
        else:
            # --- 第五阶段 (新增): 全局零容忍最终验证 ---
            # 这是写入文件前的最后一道关卡，确保100%的数据纯净性。
            logging.info("--- 阶段五: 正在执行全局零容忍最终验证... ---")

            # 1. 识别含有null值的轨迹
            key_cols_final_check = ["Lon", "Lat", "H"]

            # 创建一个布尔列，如果任何关键列为null，则为True
            is_contaminated_expr = pl.any_horizontal(
                [pl.col(c).is_null() for c in key_cols_final_check]
            )

            contaminated_ids_df = (
                final_concat_df.filter(is_contaminated_expr)
                .select("Unique_ID")
                .unique()
            )

            # 2. 如果存在受污染的轨迹，则记录日志并分离数据
            if not contaminated_ids_df.is_empty():
                contaminated_ids = contaminated_ids_df["Unique_ID"].to_list()
                for traj_id in contaminated_ids:
                    logging.critical(
                        f"轨迹ID '{traj_id}' 在完成所有处理步骤后，仍检测到残留的缺失值，已被强制丢弃。"
                        " 这强烈表明上游处理流程中存在未被覆盖的边缘案例，需要人工介入检查。"
                    )

                # 从主数据集中移除这些轨迹
                final_pure_df = final_concat_df.filter(
                    ~pl.col("Unique_ID").is_in(contaminated_ids)
                )
                logging.warning(
                    f"已强制丢弃 {len(contaminated_ids)} 条在最终验证中失败的轨迹。"
                )
            else:
                final_pure_df = final_concat_df
                logging.info("所有轨迹均通过最终完整性质检，数据100%纯净。")

            # 3. 只处理和保存纯净的数据
            if final_pure_df.is_empty():
                logging.warning("最终验证后，没有剩余的纯净轨迹可供保存。")
            else:
                final_df = final_pure_df.with_columns(
                    Time=pl.col("Time").dt.strftime("%Y%m%d %H:%M:%S%.3f")
                ).select(
                    pl.col("Unique_ID").alias("ID"),
                    pl.col("H").alias("H"),
                    pl.col("Lon").alias("Lon"),
                    pl.col("Lat").alias("Lat"),
                    pl.col("Time").alias("Time"),
                )
                output_filename = (
                    f"final_processed_trajectories.{output_format}"
                )
                output_path = os.path.join(output_dir, output_filename)
                try:
                    if output_format == "csv":
                        final_df.write_csv(output_path)
                    elif output_format == "parquet":
                        final_df.write_parquet(output_path)

                    logging.info(
                        f"成功！已保存 {final_df['ID'].n_unique()} 条通过最终验证的轨迹到 {output_path}"
                    )
                except Exception as e:
                    logging.error(f"保存最终文件到 {output_path} 时失败: {e}")

    # 保存错误日志
    if failed_trajectories_log:
        error_log_df = pl.DataFrame(failed_trajectories_log)
        error_log_path = os.path.join(output_dir, "error_log.csv")
        try:
            error_log_df.write_csv(error_log_path)
            logging.info(
                f"检测到处理失败的航段，详情已记录到: {error_log_path}"
            )
        except Exception as e:
            logging.error(f"保存错误日志文件时失败: {e}")

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
        "--segment_split_seconds",
        type=int,
        default=20,
        help="用于切分轨迹的时间间隔（秒）",
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

    parser.add_argument(
        "--min_valid_block_len",
        type=int,
        default=3,
        help="清理轨迹首尾NaN时，所需的最小连续有效数据点数",
    )

    # 地理和高度过滤参数
    parser.add_argument("--h_min", type=float, default=0, help="最低高度 (米)")
    parser.add_argument(
        "--h_max", type=float, default=30000, help="最高高度 (米)"
    )
    parser.add_argument("--lon_min", type=float, default=-180, help="最小经度")
    parser.add_argument("--lon_max", type=float, default=180, help="最大经度")
    parser.add_argument("--lat_min", type=float, default=-90, help="最小纬度")
    parser.add_argument("--lat_max", type=float, default=90, help="最大纬度")

    parser.add_argument(
        "--resample_freq",
        type=str,
        default="1s",
        choices=["1s", "2s", "5s"],
        help="重采样的时间频率 (例如: '1s', '2s', '5s')",
    )

    parser.add_argument(
        "--max_displacement_degrees",
        type=float,
        default=2.0,
        help="过滤过境航班的最大位移阈值（度）。如果航迹的经度或纬度变化超过此值，则被丢弃。",
    )

    # 高度异常检测参数
    parser.add_argument(
        "--anomaly_max_duration",
        type=int,
        default=20,
        help="高度异常检测的最大持续步数。",
    )
    parser.add_argument(
        "--anomaly_max_rate",
        type=float,
        default=100.0,
        help="高度异常检测的最大变化率（米/秒）。",
    )

    # 新增：速度限制参数
    parser.add_argument(
        "--max_latlon_speed",
        type=float,
        default=0.01,
        help="用于切分轨迹的最大水平速度（度/秒）。",
    )
    parser.add_argument(
        "--max_alt_speed",
        type=float,
        default=100.0,
        help="用于切分轨迹的最大垂直速度（米/秒）。",
    )

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
        segment_split_seconds=args.segment_split_seconds,
        log_level=args.log_level,
        min_len_for_clean=args.min_len_for_clean,
        min_valid_block_len=args.min_valid_block_len,
        unique_id_strategy=args.unique_id_strategy,
        save_debug_segmented_file=args.save_debug_segmented_file,
        resample_freq=args.resample_freq,
        max_displacement_degrees=args.max_displacement_degrees,
        anomaly_max_duration=args.anomaly_max_duration,
        anomaly_max_rate=args.anomaly_max_rate,
        max_latlon_speed=args.max_latlon_speed,
        max_alt_speed=args.max_alt_speed,
    )
