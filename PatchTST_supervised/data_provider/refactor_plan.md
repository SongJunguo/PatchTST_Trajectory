# 飞行轨迹数据处理重构计划

## 1. 背景与目标

当前 `my_flight_data_preprocessor_polars.py` 脚本中的 `_smooth_and_clean_udf` 函数在处理含有大量噪声（特别是经纬高为0的填充值和高频抖动）的私有飞行轨迹数据时，效果不佳且逻辑复杂。

本次重构的目标是利用专业航空数据处理库 `traffic`，以一个更稳健、更先进、更易于维护的流程，替换现有的核心处理逻辑，实现高质量的轨迹数据插值、重采样和滤波。

## 2. 核心思想

我们将采用一个**四阶段精细处理流程**，该流程结合了**剔除离群点、预平滑、特征工程**和**全局最优状态估计**。这种分层处理的策略，确保了每一步都为下一步提供尽可能干净、可靠的输入，从而在处理严重噪声数据时保证最终结果的质量和稳定性。

## 3. 最终方案流程图

```mermaid
graph TD
    A[原始航段数据] --> B{第一阶段: 强力预处理};
    subgraph 第一阶段
        B1[FilterAboveSigmaMedian: 剔除极端离群点]
        B2[FilterMedian: 对位置数据进行温和的平滑]
        B1 --> B2
    end
    B --> C{第二阶段: 可靠的特征工程};
    subgraph 第二阶段
        C1[cumulative_distance: 在【平滑后】的位置数据上计算速度/航迹]
    end
    C --> D{第三阶段: 标准化};
    subgraph 第三阶段
        D1[resample("1s"): 统一时间间隔并插值]
    end
    D --> E{第四阶段: 全局最优平滑};
    subgraph 第四阶段
        E1[KalmanSmoother6D: 融合所有信息进行最终精修]
    end
    E --> F[输出最终成品轨迹]
```

## 4. 详细步骤解析

### 第一阶段: 强力预处理

此阶段的目标是**在计算任何衍生数据（如速度）之前，最大程度地清洗原始的位置数据**。我们使用 `traffic` 库的管道操作符 `|` 组合两种滤波器：

1.  **`FilterAboveSigmaMedian`**: 首先上场，扮演“质检员”的角色。它使用自适应阈值，精确地识别并剔除那些完全错误的、影响巨大的**离群点**（例如，错误的GPS坐标），同时能容忍普遍存在的抖动噪声，避免误删。
2.  **`FilterMedian`**: 接着对剔除了离群点的数据进行一次**中位数平滑**。这一步至关重要，它能温和地平滑掉大部分高频抖动，为下一步计算导数提供一个更加稳定和可靠的位置序列。

**实现**: `flight.filter(FilterAboveSigmaMedian() | FilterMedian())`

### 第二阶段: 可靠的特征工程

在经过强力预处理的、更干净的位置数据上，我们进行特征工程：

1.  **`flight.cumulative_distance()`**: 调用此方法，根据平滑后的经纬度和时间戳，计算出飞机的**地速 (`groundspeed`)** 和 **航迹 (`track`)**。
2.  **手动计算垂直速率**: 通过高度差除以时间差 (`altitude.diff() / time.diff()`)，计算出**垂直速率 (`vertical_rate`)**。

由于输入的位置数据已经过平滑，此时计算出的动力学特征（速度、航迹等）的噪声已大大降低。

### 第三阶段: 标准化

1.  **`flight.resample("1s")`**: 此步骤有两个关键作用：
    *   **统一时间网格**: 将所有数据点对齐到标准的1秒时间间隔上。
    *   **插值填补**: 对第一阶段可能产生的 `NaN` 空位，以及重采样产生的新时间点，进行线性插值，确保数据序列的连续性，为最终的滤波器提供规整的输入。

### 第四阶段: 全局最优平滑

这是我们流水线的最后，也是最关键的一步。我们将一份高质量的、包含位置和动力学全套信息的规整数据，交给最专业的工具：

1.  **`KalmanSmoother6D`**: 这是一个双向的卡尔曼平滑器。它不再需要对抗极端的噪声，而是可以专注于在全局上，综合考量所有信息（位置、速度、航迹、垂直速率），并利用其内部的物理运动模型，找到一条**全局最优、物理上完全自洽**的平滑轨迹。

## 5. 核心代码逻辑参考

以下是新的 `_process_trajectory_worker_traffic` 函数的核心实现逻辑，它将替换现有的UDF。

```python
import traffic
import polars as pl
import pandas as pd
import numpy as np
from traffic.algorithms.filters import FilterAboveSigmaMedian, FilterMedian
from traffic.algorithms.filters.kalman import KalmanSmoother6D

def _process_trajectory_worker_traffic(polars_df: pl.DataFrame) -> pl.DataFrame:
    # 1. 转换为 Pandas, 并将无效0值替换为 np.nan
    pandas_df = polars_df.to_pandas()
    pandas_df[['longitude', 'latitude', 'altitude']] = pandas_df[['longitude', 'latitude', 'altitude']].replace(0, np.nan)

    # 2. 创建 Flight 对象 (列名需对齐)
    flight = traffic.Flight(pandas_df.rename(columns={
        "Time": "timestamp", "Lon": "longitude", "Lat": "latitude", "H": "altitude", "Unique_ID": "flight_id"
    }))

    # 3. 【第一阶段】强力预处理
    pre_filter = FilterAboveSigmaMedian() | FilterMedian()
    flight_pre_filtered = flight.filter(pre_filter, strategy=None) # strategy=None 保留NaN

    # 4. 【第二阶段】可靠的特征工程
    flight_with_dynamics = flight_pre_filtered.cumulative_distance().rename(columns={
        "compute_gs": "groundspeed", "compute_track": "track"
    })
    time_diff_seconds = flight_with_dynamics.data.timestamp.diff().dt.total_seconds()
    alt_diff_feet = flight_with_dynamics.data.altitude.diff()
    vertical_rate = (alt_diff_feet / time_diff_seconds) * 60
    flight_with_dynamics = flight_with_dynamics.assign(vertical_rate=vertical_rate)

    # 5. 【第三阶段】标准化
    resampled_flight = flight_with_dynamics.resample("1s") # resample会自动插值

    # 6. 【第四阶段】全局最优平滑
    final_smoother = KalmanSmoother6D()
    final_flight = resampled_flight.filter(final_smoother)

    # 7. 返回处理后的干净数据
    return pl.from_pandas(final_flight.data)