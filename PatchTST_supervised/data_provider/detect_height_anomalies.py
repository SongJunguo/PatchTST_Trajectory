import numpy as np
import pandas as pd


def detect_height_anomalies(
    heights: np.ndarray,
    times: np.ndarray,
    max_rate: float = 100.0,
    max_duration: int = 20,
) -> dict:
    """
    检测高度数据中的异常段（例如，由于数据错误导致的瞬间高度突变）。

    核心逻辑:
    1. 计算每两个连续点之间的高度变化率（米/秒）。
    2. 识别出变化率超过 `max_rate` 的点，这些是潜在的异常“起点”。
    3. 从每个起点开始，在 `max_duration` 步的窗口内，寻找一个反向的变化，
       使得从起点到该点的“净”高度变化率恢复到正常范围内。
    4. 只有当一个完整的“起点-终点”异常段被成功找到时，才将其标记为异常。
    5. 如果搜索超过 `max_duration` 步仍未找到终点，则放弃该起点，不视其为可处理的异常。

    Args:
        heights: 高度数据序列 (numpy array)。
        times: 时间戳序列 (numpy array of datetime64)。
        max_rate: 允许的最大高度变化率（米/秒）。
        max_duration: 寻找异常终点的最大搜索步数。

    Returns:
        一个字典，包含:
        - 'success': (bool) 是否成功找到了至少一个完整的异常段。
        - 'mask': (np.ndarray) 布尔掩码，True表示正常点，False表示异常点。
        - 'anomaly_info': (list) 包含已识别异常段详细信息的列表。
    """
    # 如果数据点太少，无法计算变化率，则直接返回
    if len(heights) < 2:
        return {
            "success": False,
            "mask": np.ones(len(heights), dtype=bool),
            "anomaly_info": [],
        }

    # 初始化掩码（所有点默认为正常）和成功标志
    mask = np.ones(len(heights), dtype=bool)
    anomaly_segments = []
    found_any_complete_anomaly = False

    # --- 1. 计算可变时间间隔下的高度变化率 ---
    height_diff = np.diff(heights)
    # 将 numpy.datetime64 转换为秒的浮点数差值
    time_diff_seconds = (
        np.diff(times).astype("timedelta64[ns]").astype(float) / 1e9
    )

    # 防止除以零错误：如果时间间隔为0，则变化率视为无穷大（如果高度有变化）或0（如果高度无变化）
    # 实际上，一个极小的时间间隔也会导致巨大的变化率，从而被捕捉到。
    rates = np.divide(
        height_diff,
        time_diff_seconds,
        out=np.full_like(height_diff, np.inf),
        where=time_diff_seconds != 0,
    )

    # --- 2. 找出所有变化率超过阈值的点作为潜在的异常起点 ---
    potential_anomaly_starts = np.where(np.abs(rates) > max_rate)[0]

    # 如果没有检测到任何高变化率的点，直接返回
    if len(potential_anomaly_starts) == 0:
        return {"success": False, "mask": mask, "anomaly_info": []}

    # --- 3. 遍历潜在起点，寻找完整的异常段 ---
    i = 0
    while i < len(potential_anomaly_starts):
        start_idx = potential_anomaly_starts[i]

        # 检查此起点是否已被之前的异常段覆盖
        if not mask[start_idx]:
            i += 1
            continue

        # 获取异常段的起始信息
        start_height = heights[start_idx]
        start_time = pd.Timestamp(
            times[start_idx].item()
        )  # .item() 将 numpy 类型转换为原生 Python 类型
        start_direction = np.sign(rates[start_idx])

        # --- 4. 在 max_duration 步内寻找异常结束点 ---
        found_end_for_this_start = False
        end_idx = -1  # 初始化 end_idx 以满足静态分析器
        # 搜索范围从异常发生后的第一个点开始，直到序列末尾或超出最大搜索步数
        search_window_end = min(start_idx + 1 + max_duration, len(rates))

        for j in range(start_idx + 1, search_window_end):
            # 检查当前点的变化方向是否与起始方向相反
            current_direction = np.sign(rates[j])
            if current_direction == -start_direction:
                # 找到了一个反向跳变，现在检查从起点到此点的整体斜率是否恢复正常
                end_idx = j + 1  # j是速率索引，对应的高度点索引是 j+1
                current_height = heights[end_idx]
                current_time = pd.Timestamp(times[end_idx].item())

                total_time_diff = (current_time - start_time).total_seconds()

                # 避免除以零
                if total_time_diff > 0:
                    overall_slope = (
                        current_height - start_height
                    ) / total_time_diff

                    # 如果整体斜率在允许范围内，我们认为找到了一个完整的异常段
                    if abs(overall_slope) <= max_rate:
                        # 标记从起点（不含）到终点（不含）之间的所有点为异常
                        # start_idx 是速率索引，对应的高度点是 start_idx 和 start_idx + 1
                        # 我们要移除的是 start_idx+1 到 end_idx-1
                        mask[start_idx + 1 : end_idx] = False

                        anomaly_segments.append(
                            {
                                "start_idx": start_idx,
                                "end_idx": end_idx,
                                "duration_seconds": total_time_diff,
                                "height_change": current_height - start_height,
                                "overall_rate": abs(overall_slope),
                            }
                        )

                        found_any_complete_anomaly = True
                        found_end_for_this_start = True

                        # 跳出内部的j循环，因为已经为当前start_idx找到了终点
                        break

        # 更新主循环索引 i
        # 如果找到了终点，就从终点之后继续搜索下一个异常
        if found_end_for_this_start:
            # np.searchsorted 会找到第一个大于等于 end_idx 的位置
            i = np.searchsorted(potential_anomaly_starts, end_idx)
        else:
            # 如果在窗口内没有找到终点，就直接检查下一个潜在起点
            i += 1

    # --- 5. 返回最终结果 ---
    return {
        "success": found_any_complete_anomaly,
        "mask": mask,
        "anomaly_info": anomaly_segments,
    }
