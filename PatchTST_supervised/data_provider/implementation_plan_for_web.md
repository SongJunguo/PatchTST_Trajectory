# 项目实施计划：为Web端实现轨迹预测与数据关联

**最后更新时间**: 2025-06-18

**目标**: 调整现有数据处理和模型推理流程，以生成并保存带有元数据的预测轨迹，从而实现历史数据与预测数据的有效关联，服务于Web前端可视化。

**核心原则**:

*   **非侵入式修改**: 不修改任何现有的、用于训练的脚本 (`flight_preprocess.py`, `data_loader.py`, `run_longExp.py`)。所有新功能都将在新创建的文件中实现。
*   **模块化**: 将数据清洗、推理数据加载、模型推理三个环节清晰地分离到不同的新文件中。
*   **可配置性**: 新的脚本应保持良好的可配置性，方便未来调整参数。

---

## 第一阶段：创建新的数据清洗流程 (`flight_data_preprocessor_for_web.py`)

此阶段的目标是实现完整的自定义数据清洗逻辑，将原始数据转换为格式统一、内容干净的“历史数据文件”。

### 步骤:

1.  **创建新文件**:
    *   创建 `PatchTST_supervised/data_provider/flight_data_preprocessor_for_web.py`。
    *   创建 `PatchTST_supervised/data_provider/run_preprocessing_for_web.sh` 来调用上述Python脚本。

2.  **实现 `flight_data_preprocessor_for_web.py`**:
    *   **读取数据**:
        *   使用 Polars 读取原始CSV目录中的所有文件。
        *   需要加载所有相关列，特别是 `PARTNO`, `P1`, `GP`, `H`, `JD`, `WD`, `TASK`, `PLANETYPE`, `DTRQ`。
        *   实现健壮的编码检测（如优先尝试GBK，失败后尝试UTF-8）。
    *   **时间转换与轨迹切分**:
        *   沿用 `flight_preprocess.py` 中的逻辑，使用 `DTRQ` 列生成标准时间戳 (`Time`)。
        *   根据 `P1` 和时间间隔 (`segment_split_seconds`) 对数据流进行排序和切分，识别出独立的航段。
    *   **核心清洗逻辑 (新功能)**:
        *   对每个切分好的航段 (DataFrame group) 并行执行以下操作：
            *   **ID 生成**: 获取该航段第一条有效数据的 `P1` 和 `Time`，组合成新的 `ID` (格式: `P1_YYYYMMDD_HHMMSS`)。将这个新 `ID` 填充到该航段的每一行。
            *   **众数填充**: 计算该航段内 `P1`, `TASK`, `PLANETYPE` 列的众数（出现次数最多的值），然后用此众数填充该航段内这些列的所有行。如果某列全为null，则保持为null。
            *   **前后填充**: 对 `GP` 列执行 `forward-fill` 然后 `backward-fill`，以填充内部的空值。
            *   **坐标转换**: 将 `JD`, `WD` 从度分秒格式 (`DD.MMSSff`) 转换为十进制度。
            *   **重采样**: 将每个航段的数据按指定频率（如 `1s` 或 `5s`）进行重采样和线性插值，以生成均匀的时间序列。
    *   **数据输出**:
        *   将所有处理好的航段合并成一个大的DataFrame。
        *   确保输出文件的编码为 **UTF-8**。
        *   最终列的顺序和名称应为: `ID`, `PARTNO`, `P1`, `GP`, `H`, `JD`, `WD`, `TASK`, `PLANETYPE`, `Time`。
        *   根据配置保存为 Parquet (推荐) 或 CSV 格式。

---

## 第二阶段：创建推理专用的数据加载器 (`data_loader_for_inference.py`)

此阶段的目标是创建一个新的 `Dataset` 类，它能在生成模型输入的同时，把关联所需的元数据也一并提供给推理流程。

### 步骤:

1.  **创建新文件**:
    *   创建 `PatchTST_supervised/data_provider/data_loader_for_inference.py`。

2.  **实现 `Dataset_Flight_Inference` 类**:
    *   此类将从 `torch.utils.data.Dataset` 继承。
    *   **`__init__` 方法**:
        *   接收数据文件路径、`seq_len`, `pred_len` 等参数。
        *   读取**第一阶段产出的历史数据文件** (`history_data.parquet`)。
        *   加载所有列，而不仅仅是数值特征 (`H`, `JD`, `WD`)。
        *   执行与 `Dataset_Flight` 类似的 `groupby('ID')` 和滑窗索引构建逻辑，以高效地定位每个样本。
    *   **`__getitem__` 方法 (核心修改)**:
        *   对于给定的 `index`，找到对应的滑窗数据。
        *   **返回一个元组或字典，包含**:
            1.  `seq_x`: 模型输入序列 (例如，归一化后的 `H`, `JD`, `WD`)。
            2.  `seq_y`: 标签序列 (在推理中可能不需要，但为了接口统一可以保留)。
            3.  `seq_x_mark`, `seq_y_mark`: 时间特征。
            4.  **`meta_info` (字典)**:
                *   `Pred_trajectory_id`: 滑窗所属的轨迹 `ID`。
                *   `prediction_anchor_time`: 输入滑窗 (`seq_x`) 的**最后一个时间点**的时间戳。
                *   `TASK`: 滑窗所属的 `TASK` 值。
                *   `PLANETYPE`: 滑窗所属的 `PLANETYPE` 值。

---

## 第三阶段：创建新的推理与结果保存主脚本 (`inference_for_web.py`)

此阶段的目标是整合新的数据加载器和现有模型，执行预测，并按照指定的格式保存最终结果。

### 步骤:

1.  **创建新文件**:
    *   创建 `PatchTST_supervised/inference_for_web.py`。
    *   创建 `PatchTST_supervised/scripts/PatchTST/run_inference_for_web.sh` 来调用该Python脚本。

2.  **实现 `inference_for_web.py`**:
    *   **参数解析**: 复制 `run_longExp.py` 的参数解析逻辑，以保持一致性。
    *   **创建 `Exp_Inference` 类**:
        *   此类可继承自 `Exp_Main` 或独立实现。
        *   在 `_get_data` 方法中，修改为导入并使用 `data_loader_for_inference.py` 中的 `Dataset_Flight_Inference`。
        *   **修改 `predict` 方法 (核心修改)**:
            *   初始化一个空的列表 `results = []` 用于收集所有预测片段。
            *   在遍历 `DataLoader` 的循环中：
                *   从 `DataLoader` 获取数据，解包出 `seq_x`, `seq_y`, `seq_x_mark`, `seq_y_mark` 和 **`meta_info`**。
                *   将 `seq_x` 等送入已加载的模型，得到预测输出 `outputs` (形状通常是 `[batch_size, pred_len, num_features]`)。
                *   对 `outputs` 进行反归一化。
                *   **循环处理 `batch_size` 中的每一条预测结果**:
                    *   对于 `batch` 中的第 `i` 条预测，提取其预测值 `H_predicted`, `JD_predicted`, `WD_predicted`。
                    *   从 `meta_info` 中提取对应的 `Pred_trajectory_id`, `prediction_anchor_time`, `TASK`, `PLANETYPE`。
                    *   将这些信息组合成一个字典或小的Pandas DataFrame，并添加到 `results` 列表中。
    *   **保存结果**:
        *   在 `predict` 方法的末尾，将 `results` 列表中的所有结果合并成一个大的 Pandas DataFrame。
        *   确保列名和格式符合要求: `Pred_trajectory_id`, `prediction_anchor_time`, `H_predicted`, `JD_predicted`, `WD_predicted`, `TASK`, `PLANETYPE`。
        *   将此 DataFrame 保存为 Parquet 文件 (`prediction_results.parquet`)。

---

## 数据流可视化 (Mermaid Diagram)

```mermaid
graph TD
    subgraph A[原始数据源]
        A1(原始CSV文件<br>GBK编码, 格式不一)
    end

    subgraph B[第一阶段: 数据清洗]
        B1["run_preprocessing_for_web.sh"] -- 调用 --> B2["flight_data_preprocessor_for_web.py"]
        B2 -- 1. 读取原始数据 --|> B3{清洗逻辑<br>- 生成新ID<br>- 众数/前后填充<br>- 重采样}
        B3 -- 2. 输出 --> B4(历史数据文件<br>history_data.parquet<br>UTF-8, 格式统一)
    end

    subgraph C[第二阶段: 推理]
        C1["run_inference_for_web.sh"] -- 调用 --> C2["inference_for_web.py"]
        C2 -- 使用 --> C3["data_loader_for_inference.py"]
        C3 -- 3. 读取 --> B4
        C3 -- 4. 生成滑窗 & 元数据 --> C4["PatchTST模型"]
        C4 -- 5. 进行预测 --> C5{结果整合}
        C5 -- 6. 组合预测值与元数据 --> C6(预测结果文件<br>prediction_results.parquet)
    end

    subgraph D[第三阶段: Web端使用]
        D1[Web前端] -- 加载 --> B4
        D1 -- 按需查找 --> C6
    end

    A --> B
    B --> C
    B4 --> D1
    C6 --> D1