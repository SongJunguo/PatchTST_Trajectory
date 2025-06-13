#!/bin/bash

# ==============================================================================
# 高性能飞行数据预处理一键运行脚本 (Polars版) - v3 (编码可配置)
# ==============================================================================

# --- 说明 ---
# 脚本应从项目根目录 (PatchTST/) 运行。

# --- 可配置参数 ---

# 1. 输入和输出目录 (路径相对于项目根目录)
INPUT_DIR="./PatchTST_supervised/dataset/raw/"
OUTPUT_DIR="./PatchTST_supervised/dataset/processed_data_polars/"

# 2. 输出格式
#    可选项: 'csv' 或 'parquet'。Parquet 格式更高效，推荐使用。
OUTPUT_FORMAT="parquet"

# 3. 编码检测优先级
#    可选项: 'gbk' 或 'utf8'。脚本会优先尝试此编码，失败后尝试另一种。
ENCODING_PRIORITY="utf8"

# 4. 地理和高度范围过滤参数
H_MIN=0
H_MAX=20000
LON_MIN=-180
LON_MAX=180
LAT_MIN=-90
LAT_MAX=90

# --- 脚本执行逻辑 (一般无需修改) ---

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "--- 错误: 输入目录 '$INPUT_DIR' 不存在。请检查路径。 ---"
    exit 1
fi

# 打印将要执行的命令，方便调试
echo ""
echo "--- 当前工作目录: $(pwd) ---"
echo "--- 将要执行的命令 (Polars版): ---"
echo "python PatchTST_supervised/data_provider/flight_data_preprocessor_polars.py \\"
echo "    --input_dir \"$INPUT_DIR\" \\"
echo "    --output_dir \"$OUTPUT_DIR\" \\"
echo "    --output_format \"$OUTPUT_FORMAT\" \\"
echo "    --encoding_priority \"$ENCODING_PRIORITY\" \\"
echo "    --h_min $H_MIN --h_max $H_MAX \\"
echo "    --lon_min $LON_MIN --lon_max $LON_MAX \\"
echo "    --lat_min $LAT_MIN --lat_max $LAT_MAX"
echo "------------------------------------"
echo ""

# 执行Python脚本
python PatchTST_supervised/data_provider/flight_data_preprocessor_polars.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_format "$OUTPUT_FORMAT" \
    --encoding_priority "$ENCODING_PRIORITY" \
    --h_min "$H_MIN" --h_max "$H_MAX" \
    --lon_min "$LON_MIN" --lon_max "$LON_MAX" \
    --lat_min "$LAT_MIN" --lat_max "$LAT_MAX"

echo ""
echo "--- Polars 脚本执行完毕。 ---"