#!/bin/bash

# ==============================================================================
# 飞行数据预处理一键运行脚本
# ==============================================================================

# --- 可配置参数 ---
# 0. 运行目录 为 PatchTST/PatchTST_supervised
# cd PatchTST_supervised/

# 1. 输入和输出目录
#    这些路径是相对于项目根目录 (PatchTST) 的。
#    请根据您的实际情况修改。
INPUT_DIR="./dataset/raw/"
OUTPUT_DIR="./dataset/processed_data/"

# 2. 并行处理的工作进程数
#    建议设置为您的CPU核心数或稍大一些。
MAX_WORKERS=16

# 3. 是否强制重新生成初始解析文件
#    - 设置为 "true" 来强制重新生成 _01_initial_parsed_data.csv 文件。
#    - 设置为 "false" 或留空，脚本会优先使用已存在的缓存文件。
FORCE_REGENERATE="true"

# 4. 地理和高度范围过滤参数
#    这些值将传递给Python脚本来过滤数据点。
H_MIN=100
H_MAX=30000
LON_MIN=-180
LON_MAX=180
LAT_MIN=-90
LAT_MAX=90

# H_MIN=0
# H_MAX=20000
# LON_MIN=110
# LON_MAX=120
# LAT_MIN=33
# LAT_MAX=42

# # 定义范围
# H_MIN, H_MAX = 0, 20000
# LON_MIN, LON_MAX = 110, 120
# LAT_MIN, LAT_MAX = 33, 42


# --- 脚本执行逻辑 (一般无需修改) ---

# 为了确保路径正确，脚本会始终切换到项目根目录 (PatchTST) 来执行Python程序
# `dirname "$0"` 获取脚本所在的目录 (PatchTST_supervised/data_provider)
# `cd ../..` 切换到其上上级目录 (PatchTST)
cd "$(dirname "$0")/.." || exit

# 构造 --force-regenerate 参数
REGENERATE_FLAG=""
if [ "$FORCE_REGENERATE" = "true" ]; then
    REGENERATE_FLAG="--force_regenerate"
    echo "--- 配置: 将强制重新生成初始数据。 ---"
else
    echo "--- 配置: 将优先使用缓存的初始数据（如果存在）。 ---"
fi

# 打印将要执行的命令，方便调试
echo ""
echo "--- 当前工作目录: $(pwd) ---"
echo "--- 将要执行的命令: ---"
echo "python data_provider/flight_data_preprocessor_multi.py \\"
echo "    --input_dir \"$INPUT_DIR\" \\"
echo "    --output_dir \"$OUTPUT_DIR\" \\"
echo "    --max_workers $MAX_WORKERS \\"
echo "    $REGENERATE_FLAG \\"
echo "    --h_min $H_MIN --h_max $H_MAX \\"
echo "    --lon_min $LON_MIN --lon_max $LON_MAX \\"
echo "    --lat_min $LAT_MIN --lat_max $LAT_MAX"
echo "--------------------------"
echo ""

# 执行Python脚本
python data_provider/flight_data_preprocessor_multi.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers "$MAX_WORKERS" \
    $REGENERATE_FLAG \
    --h_min "$H_MIN" --h_max "$H_MAX" \
    --lon_min "$LON_MIN" --lon_max "$LON_MAX" \
    --lat_min "$LAT_MIN" --lat_max "$LAT_MAX"

echo ""
echo "--- 脚本执行完毕。 ---"
