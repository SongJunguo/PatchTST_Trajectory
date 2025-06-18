# 飞行数据预处理器优化方案

这是一个旨在解决当前性能警告并全面提升 `flight_data_preprocessor_traffic.py` 脚本性能的详细计划。

## 1. 修复 `PerformanceWarning` (首要任务)

*   **问题定位**: 在 `flight_data_preprocessor_traffic.py` 的第 549 行，代码 `for c in transformed_lf.columns` 触发了性能警告，因为它在 `LazyFrame` 上直接获取列名，这是一个潜在的昂贵操作。
*   **解决方案**: 采用 Polars 推荐的惰性方法来获取列名。将原来的代码：
    ```python
    # 位于 flight_data_preprocessor_traffic.py 第 546-551 行
    null_counts = transformed_lf.select(
        [
            pl.col(c).is_null().sum().alias(f"{c}_null_count")
            for c in transformed_lf.columns
        ]
    ).collect(engine="streaming")
    ```
    修改为：
    ```python
    # 建议的修改
    null_counts = transformed_lf.select(
        [
            pl.col(c).is_null().sum().alias(f"{c}_null_count")
            for c in transformed_lf.schema.keys()  # <-- 核心修改点
        ]
    ).collect(engine="streaming")
    ```
*   **理由**: `transformed_lf.schema` 会返回一个包含列名和类型的字典，而不会触发实际的计算。`.keys()` 操作可以安全、高效地获取所有列名，从而消除性能警告。

## 2. 全面性能瓶颈分析与优化建议

除了修复直接的警告，我们还可以从以下几个方面对脚本进行更深入的优化。

*   **A. 审视 `collect()` 调用**
    *   **观察**: 脚本中有多处 `.collect()` 调用，特别是在调试日志部分（例如第 541、551、561、646 行）。每次调用 `.collect()` 都会中断惰性计算链，强制 Polars 执行到目前为止的所有计算，这可能是主要的性能瓶颈。
    *   **建议**:
        1.  **合并统计**: 将多个分散的统计 `collect` 合并为一个。例如，可以一次性计算 `null` 值数量和行数统计，而不是多次触发计算。
        2.  **条件化调试**: 将调试相关的 `.collect()` 调用放在 `if log_level == "DEBUG":` 的条件块内。这样，在生产环境（如 `INFO` 级别）运行时，这些昂贵的调试计算将被完全跳过。

*   **B. 优化 Polars 与 Pandas 的转换**
    *   **观察**: 在并行工作函数 `_process_trajectory_worker_traffic` 中，数据首先从 Parquet 文件中读入 Polars `DataFrame`，然后立即用 `.to_pandas()` 转换为 Pandas `DataFrame` 以便 `traffic` 库使用。
    *   **建议**:
        1.  **延迟转换**: 尽可能在 Polars 中完成所有不依赖 `traffic` 库的预处理步骤（例如，额外的过滤、列重命名等），然后再执行 `.to_pandas()`。这可以减少在 Pandas 中处理的数据量和复杂性。
        2.  **探索 `traffic` 与 Polars 的兼容性**: 长期来看，可以研究 `traffic` 库是否有可能直接或间接地支持 Polars `DataFrame`（例如通过 `pyarrow` 格式交换），这将完全消除转换开销。

*   **C. 评估并行任务的数据分发机制**
    *   **观察**: 脚本将所有分段后的数据写入一个大的临时 Parquet 文件 (`_temp_segmented_data.parquet`)。然后，每个并行工作进程都从这个大文件中扫描并过滤出自己需要处理的 `Unique_ID`。当航段数量巨大时，每个进程重复扫描同一个大文件可能会导致 I/O 瓶颈。
    *   **建议**:
        1.  **按组分割文件**: 考虑一种替代策略，即在写入临时文件之前，就按 `Unique_ID` 对数据进行分组，并将每个航段（或一小组航段）保存为独立的 Parquet 文件。然后，将这些独立的文件路径分发给工作进程。这样，每个进程只需读取自己的小文件，避免了重复扫描大文件的问题。Polars 的 `sink_parquet` 配合 `partition_by` 参数可以轻松实现这一点。

## 3. 数据处理流程可视化

为了更直观地理解整个流程和我们建议的优化点，以下是数据处理管道的 Mermaid 流程图。

```mermaid
graph TD
    subgraph "阶段一 & 二: 数据加载与惰性转换"
        A[开始: 扫描所有CSV文件] --> B{创建 LazyFrames 列表};
        B --> C[合并为单个 LazyFrame: lazy_df];
        C --> D[执行核心转换 (with_columns)];
        D --> E[过滤与航段切分 (filter, sort, over)];
        E --> F{生成最终的 segmented_lf};
    end

    subgraph "阶段三: 并行处理 (当前实现)"
        G[写入单个临时Parquet文件\n(sink_parquet)]
        F --> G;
        G --> H{获取所有 Unique_IDs\n(scan_parquet -> unique -> collect)};
        H --> I[并行池: ProcessPoolExecutor];
        I --> J[每个Worker进程];
        J --> K{扫描整个临时文件\n(scan_parquet)};
        K --> L{过滤自己的 Unique_ID\n(filter)};
        L --> M{执行 collect};
        M --> N[转换为Pandas DataFrame\n(to_pandas)];
        N --> O[使用 traffic 库处理];
        O --> P[返回 Polars DataFrame];
    end

    subgraph "阶段四 & 五: 结果聚合与保存"
        Q[收集所有Worker返回的DataFrame];
        P --> Q;
        Q --> R[合并所有DataFrame\n(concat)];
        R --> S[最终验证与清理];
        S --> T[保存最终结果文件];
    end

    %% 标注性能瓶颈
    style H fill:#ffcccc,stroke:#333,stroke-width:2px
    style M fill:#ffcccc,stroke:#333,stroke-width:2px
    style N fill:#ffcccc,stroke:#333,stroke-width:2px
    style K fill:#ffe5cc,stroke:#333,stroke-width:2px