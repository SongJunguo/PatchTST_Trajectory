#!/bin/bash

# ==============================================================================
# ** Dask 高性能内存预处理流程 启动脚本 **
#
# ** 功能: **
#   使用预设的参数调用 flight_data_preprocessor_dask.py 脚本，
#   以启动为大内存、多核心优化的数据预处理流程。
#
# ** 首次运行请务必安装依赖项: **
#   pip install "polars[numpy,pandas,pyarrow]" scipy scikit-learn "dask[complete]"
#
# ** 使用方法: **
#   1. 根据需要修改下面的 INPUT_DIR 和 OUTPUT_DIR 变量。
#   2. 直接在终端运行: ./run_preprocessing_dask.sh
# ==============================================================================
clear

# 当任何命令失败时，立即退出脚本
set -e

# --- 配置参数 ---

# 设置包含原始CSV文件的输入目录
# 请将此路径修改为您的实际数据存放目录
INPUT_DIR="./PatchTST_supervised/dataset/raw/"

# 设置存放处理后数据的输出目录
# 脚本会自动创建此目录
OUTPUT_DIR="./PatchTST_supervised/dataset/processed_data_dask/"

# Dask 使用的工作进程数 (推荐设置为您的CPU核心数)
# 针对您的96核设备，这里设置为96
N_WORKERS=4

# 输出格式 ('csv' 或 'parquet')
# 推荐使用 'parquet' 以获得更好的性能和更小的存储空间
OUTPUT_FORMAT="csv"

# --- 执行脚本 ---

echo "================================================="
echo "===      Dask 飞行数据预处理流程启动      ==="
echo "================================================="
echo "输入目录: ${INPUT_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "工作进程数: ${N_WORKERS}"
echo "输出格式: ${OUTPUT_FORMAT}"
echo "-------------------------------------------------"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 '${INPUT_DIR}' 不存在。"
    echo "请创建该目录并将CSV文件放入其中，或修改脚本中的 INPUT_DIR 变量。"
    exit 1
fi



# 调用Python脚本并传递所有参数
python PatchTST_supervised/data_provider/flight_data_preprocessor_dask.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_format "$OUTPUT_FORMAT" \
    --n_workers "$N_WORKERS" \
    --segment_split_minutes 5 \
    --log_level "INFO" \
    --min_len_for_clean 792 \
    --unique_id_strategy "timestamp" \
    --h_min 100 \
    --h_max 20000 \
    --lon_min 110 \
    --lon_max 120 \
    --lat_min 33 \
    --lat_max 42

echo "-------------------------------------------------"
echo "===           Dask 预处理流程执行完毕           ==="
echo "================================================="