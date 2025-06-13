# 技术文档：高性能飞行数据处理流水线

## 1. 执行摘要

本文档详细解析了一套为时序分析（如PatchTST模型）设计的高性能飞行数据预处理流水线。该流水线的核心目标是将大量、异构的原始飞行轨迹CSV数据，高效地转化为规整、干净、平滑且均匀采样的标准时间序列数据。

为实现这一目标，流水线采用了现代、高性能的技术栈，包括：

*   **Python**: 作为主要的编程和编排语言。
*   **Polars**: 核心的数据处理引擎，取代了传统的Pandas，以其卓越的性能、内存效率和原生的并行计算能力，成为整个流水线的基石。
*   **Scikit-learn**: 用于实现基于`LocalOutlierFactor`算法的多维离群点检测。
*   **`concurrent.futures.ProcessPoolExecutor`**: 用于实现真正的跨CPU核心并行处理，有效规避了Python的全局解释器锁（GIL）限制。

与传统的基于Pandas的实现相比，此Polars方案在以下方面取得了显著优势：

*   **高性能与可扩展性**: 通过**惰性计算（Lazy Evaluation）**和内置的多线程执行引擎，Polars能够优化整个计算图，大幅减少不必要的中间数据和内存分配。这使得它不仅处理速度更快，而且能够轻松处理远超物理内存大小的数据集。
*   **健壮性**: 流水线内置了对多种常见问题的容错机制，包括文件编码自动检测、CSV格式错误容忍、以及针对单个航段处理失败的精细化异常捕获，确保了大规模数据处理的稳定性和连续性。
*   **真并行架构**: 采用了一种创新的**“分段数据落盘，多进程分领任务”**策略。该策略将计算密集型的清洗和平滑任务（embarrassingly parallel）完美地分配到多个CPU核心上，实现了计算资源的充分利用。

## 2. 组件分析

流水线由两个核心脚本组成：一个用于配置和启动的Shell脚本，以及一个执行所有数据处理逻辑的Python脚本。

### 2.1. `run_preprocessing_polars.sh` - 执行脚本

此脚本是用户与数据处理流水线交互的前端，扮演着配置中心和一键启动器的角色。

*   **目的**: 提供一个简洁、易于理解的界面来配置所有关键参数，并以正确的参数组合启动后端的Python处理引擎。它将复杂的命令行调用封装起来，降低了使用门槛。

*   **可配置参数**:

| 参数 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `INPUT_DIR` | `./PatchTST_supervised/dataset/raw/` | 存放原始CSV文件的目录路径。 |
| `OUTPUT_DIR` | `./PatchTST_supervised/dataset/processed_data_polars/` | 存放处理后数据和日志文件的目录路径。 |
| `MAX_WORKERS` | `4` | 并行处理的工作进程数。建议设为CPU核心数以达最佳性能。 |
| `OUTPUT_FORMAT` | `csv` | 输出文件格式，可选 `'csv'` 或 `'parquet'`。Parquet是列式存储格式，读写更快、压缩率更高。 |
| `ENCODING_PRIORITY` | `utf8` | 文件编码检测的优先顺序。脚本会首先尝试此编码，失败后自动切换到另一种（`gbk`或`utf8`）。 |
| `SEGMENT_SPLIT_MINUTES` | `5` | 航迹切分的时间阈值（分钟）。若两个连续数据点的时间差超过此值，则视为一个新的航段。 |
| `LOG_LEVEL` | `INFO` | 日志记录的详细级别，可选 `'DEBUG'`, `'INFO'`, `'WARNING'`, `'ERROR'`。 |
| `MIN_LEN_FOR_CLEAN` | `792` | 一个航段在清洗和平滑前必须拥有的最小数据点数量。用于过滤掉无统计意义的过短航段。 |
| `H_MIN`/`H_MAX` | `0`/`20000` | 高度过滤范围（米）。 |
| `LON_MIN`/`LON_MAX` | `-180`/`180` | 经度过滤范围。 |
| `LAT_MIN`/`LAT_MAX` | `-90`/`90` | 纬度过滤范围。 |

*   **执行逻辑**:
    1.  **路径验证**: 脚本首先检查 `INPUT_DIR` 是否真实存在，如果不存在则报错并退出。
    2.  **命令构建与回显**: 脚本会将所有配置的参数拼接成一个完整的 `python` 命令，并将其打印到控制台。这为用户提供了清晰的执行预览，便于调试。
    3.  **脚本执行**: 最后，它执行构建好的命令，启动核心的Python数据处理脚本。

### 2.2. `flight_data_preprocessor_polars.py` - 核心处理引擎

这是整个流水线的大脑和动力所在，负责所有繁重的数据读取、转换、清洗和输出工作。

*   **代码结构**: 脚本采用了清晰的模块化设计，便于阅读和维护。
    1.  **Imports**: 导入所有必需的库。
    2.  **Logging Setup**: 初始化日志系统，将日志同时输出到文件和控制台。
    3.  **Constants**: 定义全局常量，如 `SCHEMA`，它预定义了CSV列的数据类型，这能显著提升读取速度并降低内存占用。
    4.  **Helper Functions**: 包含各种辅助函数，如坐标转换和异常检测。
    5.  **Main Processing Function (`process_flight_data`)**: 包含了流水线的主要业务逻辑，是整个流程的总指挥。
    6.  **Main Execution Block (`if __name__ == '__main__':`)**: 负责解析从Shell脚本传递过来的命令行参数，并启动主处理函数。

*   **关键函数深潜**:

    *   **`process_flight_data`**:
        此函数是流程的 orchestrator，严格按照设计的四个阶段执行操作：并行扫描与预检 -> 惰性转换与分段 -> 物化与并行平滑 -> 最终化与输出。

    *   **`dms_to_decimal_expr(dms_col: str) -> pl.Expr`**:
        这个函数是展示Polars表达能力的一个绝佳范例。它负责将度分秒格式（如 `DD.MMSSff`）的经纬度转换为十进制度。与Pandas中常见的 `.apply(lambda x: ...)` 写法相比，此函数构建并返回一个Polars表达式对象。这个表达式完全在Polars的Rust后端执行，实现了高效的向量化计算，避免了Python层面的行迭代，性能提升巨大。
        ```python
        # Polars Expression
        def dms_to_decimal_expr(dms_col: str) -> pl.Expr:
            degrees = pl.col(dms_col).floor()
            minutes = ((pl.col(dms_col) - degrees) * 100).floor()
            seconds = (((pl.col(dms_col) - degrees) * 100) - minutes) * 100
            return degrees + minutes / 60 + seconds / 3600
        ```

    *   **`_smooth_and_clean_udf(...)`**:
        这是流水线中计算最密集的部分，负责对单个航段（一个`Unique_ID`的所有数据）进行精细化处理。其逻辑遵循“先清洗、后插值、再平滑”的最佳实践。
        1.  **数据准备**: 对传入的航段数据按时间排序并去除重复的时间戳。
        2.  **长度校验**: 检查数据点数量是否达到 `min_len_for_clean` 阈值，若不足则直接放弃该航段。
        3.  **离群点检测 (`LocalOutlierFactor`)**: 在经度、纬度、高度三维空间中识别出行为异常的数据点。
        4.  **高度异常检测 (`detect_height_anomalies`)**: 使用一个遗留的、基于变率的规则来检测高度的突变。
        5.  **异常合并与标记**: 将上述两种检测方法识别出的异常点合并，并在DataFrame中将这些点对应的特征值（Lat, Lon, H）设置为 `null`。
        6.  **时间序列规整 (`upsample`)**: 将航段的时间轴强制重采样到1秒的固定间隔，这会产生许多新的、值为 `null` 的行。
        7.  **线性插值 (`interpolate`)**: 对数据中所有的 `null` 值（包括第5步标记的异常点和第6步重采样产生的空值）进行一次性的线性插值。
        8.  **数据平滑 (`savgol_filter`)**: 对插值后完整、干净的数据序列应用Savitzky-Golay滤波器，以消除噪声，使轨迹更加平滑。
        9.  **健壮的错误处理**: 整个函数被包裹在 `try...except` 块中。任何一个航段在处理过程中发生意外（如数据格式问题导致滤波失败），都会被捕获、记录日志，并返回一个空DataFrame，而不会中断整个批处理任务。

    *   **`_process_trajectory_worker` & `ProcessPoolExecutor`**:
        这部分是实现真并行的核心。
        *   `ProcessPoolExecutor` 创建一个包含多个独立Python进程的进程池。
        *   主进程将所有待处理的 `Unique_ID` 列表作为任务，通过 `executor.map` 方法分发给进程池。
        *   `_process_trajectory_worker` 是每个工作进程执行的函数。它接收一个 `Unique_ID`，然后从磁盘上的临时Parquet文件中**只读取和加载与该ID相关的数据**。这种设计避免了在进程间通过序列化传递大量数据，极大地降低了开销。加载数据后，它调用 `_smooth_and_clean_udf` 完成处理，并返回结果DataFrame。

## 3. 端到端数据处理工作流

下面是数据从原始文件到最终输出的完整旅程。

```mermaid
graph TD
    subgraph "用户侧"
        A[1. 用户执行 ./run_preprocessing_polars.sh] --> B{配置参数};
    end

    subgraph "Shell 脚本"
        B --> C[构造并执行 Python 命令];
    end

    subgraph "Python 主进程"
        C --> D[启动 flight_data_preprocessor_polars.py];
        D --> E[**Stage 1: 并行扫描与预检**<br>glob.glob 发现所有CSV<br>循环尝试 'gbk'/'utf8' 解码<br>pl.read_csv 读取有效文件];
        E --> F[**Stage 2: 惰性转换与分段**<br>pl.concat 合并 LazyFrames<br>链式调用 .with_columns (类型转换, DMS->Decimal)<br>.filter (地理围栏过滤)<br>.sort & .diff & .cum_sum (航段切分)];
        F --> G[**Stage 3: 物化与并行处理**<br>segmented_lf.sink_parquet(temp_file)<br>将分段数据写入临时Parquet文件];
        G --> H[获取所有 Unique_ID 列表];
        H --> I{创建 ProcessPoolExecutor};
        I -- 分发任务 --> J;
        K -- 收集结果 --> L[聚合所有返回的DataFrame];
        L --> M[**Stage 4: Finalization & Output**<br>pl.concat 合并所有处理好的航段<br>final_df.write_csv/parquet 保存最终结果];
        M --> N[**清理**<br>删除临时Parquet文件];
        N --> O[流程结束];
    end

    subgraph "并行工作进程 (Worker Processes)"
        J[**Worker Process 1..N**<br>接收 Unique_ID<br>pl.scan_parquet(temp_file).filter(ID)<br>只读取自己的数据<br>调用 _smooth_and_clean_udf 进行处理] --> K[返回处理后的DataFrame];
    end

    style A fill:#cde4ff
    style O fill:#cde4ff
```

*   **Step 1: 启动 (Initiation)**
    用户在终端执行 `run_preprocessing_polars.sh`。Shell脚本根据预设的参数，构建并执行一个 `python` 命令，启动 `flight_data_preprocessor_polars.py`。

*   **Step 2: Stage 1 - 并行扫描与预检 (Parallel Ingestion & Pre-check)**
    Python脚本启动后，首先使用 `glob` 找到输入目录下的所有CSV文件。它会遍历每个文件，并稳健地尝试用 `utf8` 或 `gbk` 编码进行读取。`pl.read_csv` 的 `ignore_errors=True` 参数确保了即使文件中存在少数几行格式损坏的数据，也不会导致整个文件读取失败。所有成功读取的文件都被转换成Polars的 `LazyFrame` 对象。

*   **Step 3: Stage 2 - 惰性转换与分段 (Lazy Transformation & Segmentation)**
    所有 `LazyFrame` 被合并成一个大的 `LazyFrame`。接着，一个链式的转换操作被定义（但尚未执行）：
    1.  **数据类型转换**: 将日期时间字符串解析为 `Datetime` 类型，使用 `dms_to_decimal_expr` 表达式转换经纬度。
    2.  **过滤**: 根据用户指定的地理和高度范围（`h_min`, `lon_max` 等）过滤数据。
    3.  **航段切分**: 这是最精巧的一步。数据首先按飞机ID和时间排序。然后通过 `diff()` 计算连续时间点的差值，并与 `segment_split_minutes` 阈值比较，生成一个布尔标记 `new_segment_marker`。最后，通过对这个标记进行累加求和（`cum_sum()`），为每个航段分配了一个唯一的 `segment_id`。飞机ID和`segment_id`组合起来，构成了全局唯一的 `Unique_ID`。

*   **Step 4: Stage 3 - 物化与并行平滑 (Materialization & Parallel Smoothing)**
    这是流水线的核心并行计算阶段。
    1.  **落盘 (Sink to Disk)**: `segmented_lf.sink_parquet(temp_file)` 是一个关键操作。它触发了之前定义的所有惰性计算，并将最终的分段后、但未清洗的数据一次性写入到一个临时的Parquet文件中。**这一步至关重要**，因为它创建了一个所有工作进程都可以访问的、共享的、只读的数据源，从而避免了在进程间传递大量数据的巨大开销。
    2.  **任务分配**: 主进程轻量级地扫描临时Parquet文件，提取出所有 `Unique_ID` 的列表。
    3.  **工人执行**: `ProcessPoolExecutor` 将 `Unique_ID` 列表分发给池中的工作进程。每个进程接收到ID后，执行 `_process_trajectory_worker` 函数。该函数利用Parquet文件的列式存储优势，高效地 `scan` 并 `filter` 出只属于自己任务ID的数据，然后调用 `_smooth_and_clean_udf` 进行密集的清洗和平滑计算。
    4.  **结果聚合**: 主进程收集所有工作进程返回的处理好的Polars DataFrame，并将它们存放在一个列表中。

*   **Step 5: Stage 4 - 最终化与输出 (Finalization & Output)**
    所有并行任务完成后，主进程使用 `pl.concat()` 将收集到的所有小DataFrame合并成一个最终的大DataFrame。然后，根据用户指定的 `OUTPUT_FORMAT`，将这个DataFrame保存为CSV或Parquet文件。

*   **Step 6: 清理 (Cleanup)**
    在 `finally` 块的保证下，临时的Parquet文件被安全删除，不留任何中间产物。

## 4. 架构优势和关键概念

*   **惰性计算 (Lazy Evaluation)**
    Polars的惰性计算是其高性能的基石。它允许开发者像定义一个配方一样链式地定义一系列操作，而不会立即执行它们。Polars的查询优化器会分析整个“配方”，重新排序操作，合并多个操作为一个，以最高效的方式执行。这最大限度地减少了内存使用和CPU周期，尤其是在处理需要多步过滤和转换的大型数据集时。

*   **真并行策略 (True Parallelism Strategy)**
    该流水线采用的“Sink-to-Shared-File”模式是一种非常实用且高效的单机并行处理策略。它巧妙地绕过了Python GIL的限制和多进程数据序列化的开销。与Dask等更重的分布式计算框架相比，此方法对于单机上“易于并行化”的分组处理任务（如本例中的按航段处理）来说，实现更简单、依赖更少、开销更低。

*   **健壮性与错误处理 (Robustness and Error Handling)**
    整个流水线在多个层面都体现了对现实世界“脏数据”的考量：
    *   **文件读取层**: 自动化的编码检测和对坏行的容忍。
    *   **数据过滤层**: 过滤掉无效或无意义的数据（如地理范围外、过短的航段）。
    *   **UDF处理层**: 对每个独立任务单元（航段）的计算都进行异常捕获，确保单个失败不会拖垮整个批处理流程。