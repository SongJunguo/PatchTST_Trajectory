#!/bin/bash

# ==============================================================================
# 数据质量检查脚本
# ==============================================================================

# --- 可配置参数 ---
# 1. 输入文件和输出目录
#    这些路径是相对于 data_provider 目录的。
INPUT_FILE="../dataset/processed_data/_02_after_split_before_military_filter.csv"
OUTPUT_DIR="../dataset/processed_data/"

# 2. 并行处理的工作进程数
MAX_WORKERS=16

# 3. 地理和高度范围过滤参数 (与主脚本保持一致)
H_MIN=0
H_MAX=20000
LON_MIN=-180
LON_MAX=180
LAT_MIN=-90
LAT_MAX=90


# 4. 飞行剖面检查参数
MAX_SPEED_KMH=2470  # ~Mach 2
MAX_VS_MS=300       # 垂直速率 m/s

# --- 脚本执行逻辑 ---
cd "$(dirname "$0")" || exit

echo "--- 当前工作目录: $(pwd) ---"
echo "--- 将要执行的命令: ---"
echo "python check_split_data.py \\"
echo "    --input_file \"$INPUT_FILE\" \\"
echo "    --output_dir \"$OUTPUT_DIR\" \\"
echo "    --max_workers $MAX_WORKERS \\"
echo "    --h_min $H_MIN --h_max $H_MAX \\"
echo "    --lon_min $LON_MIN --lon_max $LON_MAX \\"
echo "    --lat_min $LAT_MIN --lat_max $LAT_MAX \\"
echo "    --max_speed_kmh $MAX_SPEED_KMH \\"
echo "    --max_vs_ms $MAX_VS_MS"
echo "--------------------------"
echo ""

# 执行Python脚本
python check_split_data.py \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers "$MAX_WORKERS" \
    --h_min "$H_MIN" --h_max "$H_MAX" \
    --lon_min "$LON_MIN" --lon_max "$LON_MAX" \
    --lat_min "$LAT_MIN" --lat_max "$LAT_MAX" \
    --max_speed_kmh "$MAX_SPEED_KMH" \
    --max_vs_ms "$MAX_VS_MS"

echo ""
echo "--- 检查脚本执行完毕。 ---"