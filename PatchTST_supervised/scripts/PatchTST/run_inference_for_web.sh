#!/bin/bash

# ==============================================================================
# ** Web可视化模型推理一键运行脚本 **
#
# ** 功能: **
#   - 激活 conda 虚拟环境 torch
#   - 调用 inference_for_web.py 脚本进行模型推理。
#   - 提供清晰、可配置的参数化接口。
#   - 指定要加载的预训练模型ID。
# ==============================================================================

# --- 说明 ---
# 脚本应从项目根目录 (PatchTST/) 运行。
clear

# --- 激活 conda 虚拟环境 ---
echo "--- 激活 conda 虚拟环境 torch ---"

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

# 激活 torch 虚拟环境
conda activate torch

# 检查激活是否成功
if [ "$CONDA_DEFAULT_ENV" = "torch" ]; then
    echo "✓ 成功激活 conda 虚拟环境: $CONDA_DEFAULT_ENV"
else
    echo "❌ 激活 conda 虚拟环境失败，当前环境: $CONDA_DEFAULT_ENV"
    echo "请确保已创建名为 'torch' 的 conda 虚拟环境"
    exit 1
fi

echo "--- Python 版本信息 ---"
python --version
echo "--- PyTorch 版本信息 ---"
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')" 2>/dev/null || echo "PyTorch 未安装或无法导入"
echo "------------------------------------"

# --- 可配置参数 ---

# 1. 预训练模型的ID
#    **重要**: 这个ID必须与你训练时使用的 `model_id` 完全一致。
#    脚本将从 './checkpoints/[MODEL_ID]_[...]' 路径加载模型。
MODEL_ID="20250612flight_dependent_test"

# 2. 数据路径
#    这些路径应指向第一阶段数据清洗脚本的输出。
ROOT_PATH="./PatchTST_supervised/dataset/processed_for_web/"
DATA_PATH="final_processed_trajectories.parquet"

# 3. 模型和任务参数
#    这些参数必须与训练时使用的参数保持一致。
MODEL_NAME="PatchTST"
SEQ_LEN=192
PRED_LEN=72
PATCH_LEN=16
STRIDE=8
PADDING_PATCH="end"
REVIN=1
AFFINE=0
SUBTRACT_LAST=0
DECOMPOSITION=0
KERNEL_SIZE=25
INDIVIDUAL=1
ENC_IN=3 # 输入特征数 (H, JD, WD)
C_OUT=3  # 输出特征数 (H, JD, WD)
FREQ='h' # 时间特征编码频率，必须与训练时一致
EMBED_TYPE=0 # 时间嵌入类型，必须与训练时一致

# 4. 归一化配置
#    **重要**: 必须指向训练时生成的归一化统计文件。
STATS_PATH="./PatchTST_supervised/dataset/normalization_stats/processed_trajectories_multi_normalization_stats.csv"

# 5. 硬件和批处理配置
# 🚀 基于性能测试的最优配置：70,865 samples/s 吞吐量
BATCH_SIZE=1024     # 大批处理大小，实测最佳性能配置
NUM_WORKERS=2      # 4个工作进程，实现最佳数据预加载效率
USE_MULTI_GPU=false
DEVICES="0"

# 6. 数据加载优化配置
PIN_MEMORY=true     # 启用内存锁定，加速GPU数据传输
PERSISTENT_WORKERS=true  # 保持工作进程存活，减少进程创建开销
Dataloader_STRIDE=10 # 🚀 新增：数据加载器滑窗步长
 
 # 7. 系统级优化 - 允许更多线程以提高并行度
 export OMP_NUM_THREADS=4        # 允许适量OpenMP线程
export MKL_NUM_THREADS=4        # 允许适量MKL线程
export NUMEXPR_NUM_THREADS=4    # 允许适量NumExpr线程
export PYTORCH_NUM_THREADS=4    # 设置PyTorch线程数

# --- 脚本执行逻辑 (一般无需修改) ---

# 根据是否使用多GPU设置设备参数
if [ "$USE_MULTI_GPU" = true ] ; then
  GPU_ARGS="--use_multi_gpu --devices $DEVICES"
else
  GPU_ARGS=""
fi

# 打印将要执行的命令，方便调试
echo ""
echo "--- 当前工作目录: $(pwd) ---"
echo "--- 开始为Web端进行模型推理 ---"
echo "--- 加载模型ID: $MODEL_ID ---"
echo "------------------------------------"
echo ""

# 执行Python推理脚本
python -u ./PatchTST_supervised/inference_for_web.py \
  --model_id "$MODEL_ID" \
  --model $MODEL_NAME \
  --data flight \
  --features M \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --label_len 0 \
  --patch_len $PATCH_LEN \
  --stride $STRIDE \
  --padding_patch $PADDING_PATCH \
  --revin $REVIN \
  --affine $AFFINE \
  --subtract_last $SUBTRACT_LAST \
  --decomposition $DECOMPOSITION \
  --kernel_size $KERNEL_SIZE \
  --individual $INDIVIDUAL \
  --enc_in $ENC_IN \
  --c_out $C_OUT \
  --freq "$FREQ" \
  --embed_type $EMBED_TYPE \
  --stats_path "$STATS_PATH" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --pin_memory $PIN_MEMORY \
  --persistent_workers $PERSISTENT_WORKERS \
  --dataloader_stride $Dataloader_STRIDE \
  $GPU_ARGS

echo ""
echo "--- 推理脚本执行完毕。 ---"
echo "--- 预测结果已保存到: ./results/${MODEL_ID}_[...] ---"