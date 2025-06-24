#!/bin/bash

# ==============================================================================
# ** Web可视化数据预处理一键运行脚本 **
#
# ** 功能: **
#   - 调用 flight_data_preprocessor_for_web.py 脚本。
#   - 提供清晰、可配置的参数化接口。
#   - 自动创建输出目录。
# ==============================================================================

# --- 说明 ---
# 脚本应从项目根目录 (PatchTST/) 运行。
clear

# --- 激活 conda 虚拟环境 ---
echo "--- 激活 conda 虚拟环境 traffic ---"

# 初始化 conda（如果尚未初始化）
if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "警告: 未找到 conda 初始化脚本，尝试使用 conda init..."
    # 尝试运行 conda init
    if command -v conda &> /dev/null; then
        eval "$(conda shell.bash hook)"
    else
        echo "❌ 无法找到 conda 命令，请检查 conda 安装"
        exit 1
    fi
fi

# 激活 traffic 虚拟环境
conda activate traffic

# 检查激活是否成功
if [ "$CONDA_DEFAULT_ENV" = "traffic" ]; then
    echo "✓ 成功激活 conda 虚拟环境: $CONDA_DEFAULT_ENV"
else
    echo "❌ 激活 conda 虚拟环境失败，当前环境: $CONDA_DEFAULT_ENV"
    echo "请确保已创建名为 'traffic' 的 conda 虚拟环境"
    exit 1
fi

# --- 可配置参数 ---

# 1. 输入和输出目录 (路径相对于项目根目录)
#    确保您的原始数据放在 INPUT_DIR 中
INPUT_DIR="./PatchTST_supervised/dataset/raw/"
#    处理后的历史数据将保存在 OUTPUT_DIR
OUTPUT_DIR="./PatchTST_supervised/dataset/processed_for_web/"

# 2. 并行处理的工作进程数
#    建议设置为您的CPU核心数。
MAX_WORKERS=2

# 3. 输出格式
#    可选项: 'csv' 或 'parquet'。Parquet 格式更高效，推荐使用。
OUTPUT_FORMAT="parquet"

# 4. 编码检测优先级
#    可选项: 'gbk' 或 'utf8'。脚本会优先尝试此编码，失败后尝试另一种。
ENCODING_PRIORITY="gbk"

# 5. 轨迹切分时间间隔 (秒)
#    如果两个连续点的时间差超过此值，则切分为新航段。
SEGMENT_SPLIT_SECONDS=300

# 6. 日志级别
#    可选项: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL="INFO"

# 7. 清洗航段的最小长度
#    航段的数据点数量必须大于此值，才会被处理。
MIN_LEN_FOR_CLEAN=20

# 8. 地理和高度范围过滤参数
H_MIN=100
H_MAX=30000
LON_MIN=-180
LON_MAX=180
LAT_MIN=-90
LAT_MAX=90

# 9. 新增功能开关
#    ID生成策略。可选项: 'numeric' 或 'timestamp'
UNIQUE_ID_STRATEGY="timestamp"
#    保存调试文件。可选项: "true" 或 "false"
SAVE_DEBUG_FILE="true"

# 10. 重采样频率
#    可选项: '1s', '2s', '5s'
RESAMPLE_FREQ="5s"

# 11. 过境航班过滤阈值 (度)
#    如果航迹的经度或纬度总位移超过此值，则被视为过境航班并被丢弃。
MAX_DISPLACEMENT_DEGREES=2.0

# 12. 高度异常检测参数
#    设置异常检测时允许的最大持续步数。
ANOMALY_MAX_DURATION=20
#    设置异常检测时允许的最大变化率（米/秒）。
ANOMALY_MAX_RATE=100

# 13. 速度异常检测参数 (新增)
#    用于切分轨迹的最大水平速度（度/秒）。
MAX_LATLON_SPEED=0.01
#    用于切分轨迹的最大垂直速度（米/秒）。
MAX_ALT_SPEED=100.0

# 14. 其他核心参数 (新增)
#    清理轨迹首尾NaN时，所需的最小连续有效数据点数
MIN_VALID_BLOCK_LEN=3


# --- 脚本执行逻辑 (一般无需修改) ---

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "--- 错误: 输入目录 '$INPUT_DIR' 不存在。请检查路径。 ---"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 使用数组构建参数，以便更健壮地处理可选参数
CMD_ARGS=(
    --input_dir "$INPUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --max_workers "$MAX_WORKERS"
    --output_format "$OUTPUT_FORMAT"
    --encoding_priority "$ENCODING_PRIORITY"
    --h_min "$H_MIN" --h_max "$H_MAX"
    --lon_min "$LON_MIN" --lon_max "$LON_MAX"
    --lat_min "$LAT_MIN" --lat_max "$LAT_MAX"
    --segment_split_seconds "$SEGMENT_SPLIT_SECONDS"
    --log_level "$LOG_LEVEL"
    --min_len_for_clean "$MIN_LEN_FOR_CLEAN"
    --unique_id_strategy "$UNIQUE_ID_STRATEGY"
    --resample_freq "$RESAMPLE_FREQ"
    --max_displacement_degrees "$MAX_DISPLACEMENT_DEGREES"
    --anomaly_max_duration "$ANOMALY_MAX_DURATION"
    --anomaly_max_rate "$ANOMALY_MAX_RATE"
    --min_valid_block_len "$MIN_VALID_BLOCK_LEN"
    --max_latlon_speed "$MAX_LATLON_SPEED"
    --max_alt_speed "$MAX_ALT_SPEED"
)

# 如果 SAVE_DEBUG_FILE 设置为 "true"，则添加调试标志
if [ "$SAVE_DEBUG_FILE" = "true" ]; then
    CMD_ARGS+=(--save_debug_segmented_file)
fi

# 打印将要执行的命令，方便调试
echo ""
echo "--- 当前工作目录: $(pwd) ---"
echo "--- 将要执行的命令: ---"
# 使用 printf 来更安全地打印命令，避免特殊字符问题
printf "python -m PatchTST_supervised.data_provider.flight_data_preprocessor_for_web"
for arg in "${CMD_ARGS[@]}"; do
  printf " %q" "$arg"
done
echo ""
echo "------------------------------------"
echo ""

# 执行Python脚本
python -m PatchTST_supervised.data_provider.flight_data_preprocessor_for_web "${CMD_ARGS[@]}"

echo ""
echo "--- 脚本执行完毕。 ---"
echo "--- 清洗后的历史数据已保存到: $OUTPUT_DIR ---"