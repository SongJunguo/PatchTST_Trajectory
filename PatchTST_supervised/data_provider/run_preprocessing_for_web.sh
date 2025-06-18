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

# --- 可配置参数 ---

# 1. 输入和输出目录 (路径相对于项目根目录)
#    确保您的原始数据放在 INPUT_DIR 中
INPUT_DIR="./PatchTST_supervised/dataset/raw/"
#    处理后的历史数据将保存在 OUTPUT_DIR
OUTPUT_DIR="./PatchTST_supervised/dataset/processed_for_web/"

# 2. 并行处理的工作进程数
#    建议设置为您的CPU核心数。
MAX_WORKERS=16

# 3. 输出格式
#    可选项: 'csv' 或 'parquet'。Parquet 格式更高效，推荐使用。
OUTPUT_FORMAT="parquet"

# 4. 编码检测优先级
#    可选项: 'gbk' 或 'utf8'。脚本会优先尝试此编码，失败后尝试另一种。
ENCODING_PRIORITY="gbk"

# 5. 轨迹切分时间间隔 (秒)
#    如果两个连续点的时间差超过此值，则切分为新航段。
SEGMENT_SPLIT_SECONDS=300

# 6. 重采样频率
#    这是生成均匀时间序列的关键。'1s'代表1秒, '5s'代表5秒。
RESAMPLE_FREQ="1s"

# 7. 日志级别
#    可选项: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL="INFO"


# --- 脚本执行逻辑 (一般无需修改) ---

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "--- 错误: 输入目录 '$INPUT_DIR' 不存在。请检查路径。 ---"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 使用数组构建参数，以便更健壮地处理
CMD_ARGS=(
    --input_dir "$INPUT_DIR"
    --output_dir "$OUTPUT_DIR"
    --max_workers "$MAX_WORKERS"
    --output_format "$OUTPUT_FORMAT"
    --encoding_priority "$ENCODING_PRIORITY"
    --segment_split_seconds "$SEGMENT_SPLIT_SECONDS"
    --resample_freq "$RESAMPLE_FREQ"
    --log_level "$LOG_LEVEL"
)

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