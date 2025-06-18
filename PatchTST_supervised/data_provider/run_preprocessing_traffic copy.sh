#!/bin/bash

# ==============================================================================
# 高性能飞行数据预处理一键运行脚本 (Traffic版) - v5 (重构)
# ==============================================================================

# --- 说明 ---
# 脚本应从项目根目录 (PatchTST/) 运行。
clear
# --- 可配置参数 ---

# 1. 输入和输出目录 (路径相对于项目根目录)
INPUT_DIR="./PatchTST_supervised/dataset/raw/"
OUTPUT_DIR="./PatchTST_supervised/dataset/processed_data_traffic/"

# 2. 并行处理的工作进程数
#    建议设置为您的CPU核心数。
MAX_WORKERS=16

# 3. 输出格式
#    可选项: 'csv' 或 'parquet'。Parquet 格式更高效，推荐使用。
OUTPUT_FORMAT="csv"

# 4. 编码检测优先级
#    可选项: 'gbk' 或 'utf8'。脚本会优先尝试此编码，失败后尝试另一种。
ENCODING_PRIORITY="gbk"

# 5. 轨迹切分时间间隔 (秒)
#    如果两个连续点的时间差超过此值，则切分为新航段。
SEGMENT_SPLIT_SECONDS=300

# 6. 日志级别
#    可选项: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL="DEBUG"

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


# --- 脚本执行逻辑 (一般无需修改) ---

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "--- 错误: 输入目录 '$INPUT_DIR' 不存在。请检查路径。 ---"
    exit 1
fi

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
    --unique_id_strategy "$UNIQUE_ID_STRATEGY"
    --resample_freq "$RESAMPLE_FREQ"
)

# 如果 SAVE_DEBUG_FILE 设置为 "true"，则添加调试标志
if [ "$SAVE_DEBUG_FILE" = "true" ]; then
    CMD_ARGS+=(--save_debug_segmented_file)
fi

# 打印将要执行的命令，方便调试
echo ""
echo "--- 当前工作目录: $(pwd) ---"
echo "--- 将要执行的命令 (Traffic版): ---"
# 使用 printf 来更安全地打印命令，避免特殊字符问题
printf "python PatchTST_supervised/data_provider/flight_data_preprocessor_traffic.py"
for arg in "${CMD_ARGS[@]}"; do
  printf " %q" "$arg"
done
echo ""
echo "------------------------------------"
echo ""

# 执行Python脚本
python PatchTST_supervised/data_provider/flight_data_preprocessor_traffic.py "${CMD_ARGS[@]}"

echo ""
echo "--- Traffic 脚本执行完毕。 ---"

