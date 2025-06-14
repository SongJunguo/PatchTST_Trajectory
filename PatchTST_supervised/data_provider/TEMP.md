PatchTST_supervised/data_provider/TEMP.md
有一个详细的计划书，你县阅读这个计划书和相关代码
PatchTST_supervised/data_provider/run_preprocessing.sh
PatchTST_supervised/data_provider/flight_data_preprocessor_multi.py

使用conda 虚拟环境 conda activate dl_data_env

PatchTST_supervised/data_provider/run_preprocessing_polars.sh
PatchTST_supervised/data_provider/flight_data_preprocessor_polars.py

详细的梳理这两个用于飞行轨迹数据集处理的脚本的内容，和详细的处理步骤和流程


还有一个问题，如果完全独立的处理每个文件，每个飞行轨迹，
由于原始是csv是按照日期储存，也就是每一天一个csv，格式大概是年月日，有小概率，一些飞机半夜飞行，导致一些轨迹分布在两个csv里面。


# **最终执行手册：高性能飞行数据预处理器**

**版本：4.0 (交接最终版)**

---

**致代码执行者：**

本文档是您执行此任务所需的 **唯一信息来源**。它包含了我们与用户经过多轮详细讨论后确定的最终需求、设计方案、技术细节和所有关键决策。请严格按照本手册的指引，在新文件中从头开始编写代码。

---

## **1. 项目概述与核心目标**

**目标：** 创建一个全新的、高性能的 Python 脚本，用于预处理大规模（~5.5GB, >6000万行, 100+ CSV 文件）飞行数据。该脚本需要解决原始脚本 `flight_data_preprocessor_multi.py` 在 I/O、内存、CPU 方面的严重性能瓶颈，并提供工业级的健壮性和用户体验。

**核心要求：**
*   **性能：** 最大化利用多核 CPU (96核/192线程)，128GB内存显著缩短处理时间。
*   **内存效率：** 避免将全部数据一次性加载到内存，能处理远超物理内存大小的数据集。
*   **健壮性：** 单个坏文件、坏数据行或处理失败的航段不应导致整个程序崩溃。
*   **用户体验：** 提供清晰的日志、进度报告和有用的错误信息。

## **2. 执行策略 (非破坏性)**

*   **新文件名：** `PatchTST_supervised/data_provider/flight_data_preprocessor_polars.py`
*   **原始文件：** `PatchTST_supervised/data_provider/flight_data_preprocessor_multi.py` **绝对不能修改**。
*   **原则：** 本次任务为 **非破坏性** 操作。

## **3. 原始脚本瓶颈分析 (用于代码注释)**

*   **I/O 瓶颈:** 单线程循环读取上百个 CSV 文件，无法利用并行 I/O。
*   **内存瓶颈:** `pd.concat` 将所有数据合并成一个巨大的 DataFrame，导致内存占用激增，极易溢出。数据类型未优化（如 `float64`）。
*   **CPU 瓶颈:**
    *   `pd.to_datetime` 在处理特定格式时，性能极差。
    *   `df.apply` 用于经纬度转换，是逐行操作，无法向量化，效率极低。
    *   轨迹切分和过滤使用 `groupby` + Python 循环，受 GIL 限制，无法真正并行。
*   **并行化瓶颈:** `ProcessPoolExecutor` 虽然实现了并行，但任务划分基于 `groupby` 的结果，数据在进程间的序列化/反序列化开销巨大。

## **4. 最终工作流与技术选型**

*   **核心库:** **Polars**
*   **依赖项 (在脚本顶部注释中说明):**
    ```python
    # pip install polars numpy scipy scikit-learn tqdm psutil
    ```

*   **工作流图:**
    ```mermaid
    graph TD
        subgraph "阶段一: 并行读取与预检"
            A[1. Glob 获取文件列表] --> B{2. 独立扫描每个文件<br>try-except, gbk优先};
            B --> C[3. 定义并强制Schema];
            C --> D[4. pl.concat 合并LazyFrames];
        end
        subgraph "阶段二: 核心转换 (Lazy & Parallel)"
            D --> E[5. 惰性表达式链<br>日期/经纬度/类型/范围];
            E --> F[6. 窗口函数切分轨迹];
        end
        subgraph "阶段三: 复杂平滑 (Streaming & Parallel)"
            F --> G[7. 触发流式计算<br>collect(streaming=True)];
            G --> H[8. 按航段ID分组];
            H --> I{9. 并行应用健壮的UDF<br>处理失败则返回空DF};
        end
        subgraph "阶段四: 保存结果"
            I --> J[10. 自动聚合结果];
            J --> K[11. 保存为CSV (默认)<br>或Parquet];
        end
    ```

## **5. 关键实现指令 (必须严格遵守)**

1.  **文件读取与容错 (`阶段一`)**
    *   用 `glob.glob` 获取文件列表。
    *   **循环** 遍历文件名，对每个文件独立执行 `pl.scan_csv` 并包裹在 `try-except` 块中。
    *   **编码：** 优先尝试 `encoding='gbk'`，若失败，在 `except` 中尝试 `encoding='utf-8-sig'`。使用 `ignore_errors=True`。
    *   **Schema:** 定义一个包含所有必需列的 `schema` 字典，并在 `scan_csv` 中使用它。
    *   记录所有无法读取的文件，并在流程结束时报告。

2.  **核心转换 (`阶段二`)**
    *   **日期解析:** `pl.col("DTRQ").str.to_datetime(format="%d-%b-%y %I.%M.%S%.f %p", strict=False)`。`strict=False` 会将错误转为 `NaT`。
    *   **经纬度转换 (DMS to Decimal):** **必须** 使用数学表达式，**严禁** 使用 `apply`。
        ```python
        # 假设输入格式为 DD.MMSSff (例如 116.3045)
        degrees = pl.col(dms_col).floor()
        minutes = ((pl.col(dms_col) - degrees) * 100).floor()
        seconds = (((pl.col(dms_col) - degrees) * 100) - minutes) * 100
        decimal_degrees = degrees + minutes / 60 + seconds / 3600
        ```
    *   **轨迹切分:** 使用窗口函数。
        ```python
        time_diff = pl.col('Time').diff().over('ID')
        new_segment_marker = time_diff > timedelta(minutes=15)
        segment_id = new_segment_marker.cast(pl.Int32).cumsum().over('ID')
        unique_id = pl.col("ID").cast(pl.Utf8) + pl.lit("_") + segment_id.cast(pl.Utf8)
        ```

3.  **复杂 UDF 处理 (`阶段三`)**
    *   **触发计算:** **必须** 使用 `lazy_df.collect(streaming=True)` 来控制内存。
    *   **UDF 函数 `_smooth_and_clean_udf`:**
        *   函数签名: `def _smooth_and_clean_udf(group_df: pl.DataFrame) -> pl.DataFrame:`
        *   **必须** 用 `try-except Exception as e:` 包裹所有核心逻辑。
        *   在 `try` 块中，用 `.to_numpy()` 将列转为 NumPy 数组，传递给 `savgol_filter`, `LocalOutlierFactor` 等。
        *   在 `except` 块中，**必须** 记录失败的 `Unique_ID` 和错误 `e`，然后 `return pl.DataFrame(schema=group_df.schema)` 返回一个 **空的、但具有正确 Schema 的 DataFrame**。
    *   **关于 `detect_height_anomalies` 的特别说明:**
        *   **决策:** 此函数将 **按原样从旧脚本中复制**。
        *   **理由:** 该函数接收 NumPy 数组作为输入，其内部逻辑（虽然复杂）是自包含的。在 Polars 的 UDF 中，我们可以通过 `.to_numpy()` 高效地将列转换为 NumPy 数组，然后调用此函数。其返回的布尔掩码 (mask) 也可以轻松地用于更新 Polars DataFrame。将其重写为纯 Polars 表达式将非常复杂且没有明显的性能优势。因此，保持其不变是风险最低、最高效的集成方式。

4.  **输出 (`阶段四`)**
    *   **默认输出 CSV。**
    *   使用 `df.write_csv(file_path)`。

## **6. 代码结构与命令行接口**

```python
# 1. Imports
import polars as pl
import numpy as np
# ... (scipy, sklearn, os, glob, logging, argparse, tqdm, psutil, datetime)

# 2. Logging Setup
# ... (配置 logging.basicConfig)

# 3. Constants
# ... (如 DMS_FORMAT_STRING)

# 4. Helper Functions
# def dms_to_decimal_expr(...) -> pl.Expr:
#     # 封装经纬度转换表达式
#
# def _smooth_and_clean_udf(group_df: pl.DataFrame) -> pl.DataFrame:
#     # 健壮的 UDF 实现
#
# def detect_height_anomalies(...) -> dict:
#     # 从旧脚本复制，无需修改

# 5. Main Processing Function
# def process_flight_data(input_dir: str, output_dir: str, ...):
#     # 包含所有处理阶段的主逻辑
#     # 阶段一: 读取
#     # 阶段二: 转换
#     # 阶段三: 平滑
#     # 阶段四: 保存

# 6. Main Execution Block
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(...)
#     # 必须包含以下所有参数:
#     parser.add_argument('--input_dir', type=str, required=True)
#     parser.add_argument('--output_dir', type=str, required=True)
#     parser.add_argument('--output_format', type=str, default='csv', choices=['csv', 'parquet'])
#     parser.add_argument('--h_min', type=float, default=0)
#     parser.add_argument('--h_max', type=float, default=20000)
#     parser.add_argument('--lon_min', type=float, default=110)
#     parser.add_argument('--lon_max', type=float, default=120)
#     parser.add_argument('--lat_min', type=float, default=33)
#     parser.add_argument('--lat_max', type=float, default=42)
#
#     args = parser.parse_args()
#
#     # 启动处理
#     process_flight_data(...)