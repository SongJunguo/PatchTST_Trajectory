#!/bin/bash

# ==============================================================================
# ** Web可视化模型推理一键运行脚本 **
#
# ** 功能: **
#   - 调用 inference_for_web.py 脚本进行模型推理。
#   - 提供清晰、可配置的参数化接口。
#   - 指定要加载的预训练模型ID。
# ==============================================================================

# --- 说明 ---
# 脚本应从项目根目录 (PatchTST/) 运行。
clear

# --- 可配置参数 ---

# 1. 预训练模型的ID
#    **重要**: 这个ID必须与你训练时使用的 `model_id` 完全一致。
#    脚本将从 './checkpoints/[MODEL_ID]_[...]' 路径加载模型。
MODEL_ID="20250612flight_dependent_test"

# 2. 数据路径
#    这些路径应指向第一阶段数据清洗脚本的输出。
ROOT_PATH="./PatchTST_supervised/dataset/processed_for_web/"
DATA_PATH="history_data.parquet"

# 3. 模型和任务参数
#    这些参数必须与训练时使用的参数保持一致。
MODEL_NAME="PatchTST"
SEQ_LEN=192
PRED_LEN=72
PATCH_LEN=16
STRIDE=8
ENC_IN=3 # 输入特征数 (H, JD, WD)
C_OUT=3  # 输出特征数 (H, JD, WD)

# 4. 硬件和批处理配置
BATCH_SIZE=1024
NUM_WORKERS=16
USE_MULTI_GPU=true
DEVICES="0,1"

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
  --data flight_inference \
  --features M \
  --root_path "$ROOT_PATH" \
  --data_path "$DATA_PATH" \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --label_len 0 \
  --patch_len $PATCH_LEN \
  --stride $STRIDE \
  --enc_in $ENC_IN \
  --c_out $C_OUT \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  $GPU_ARGS

echo ""
echo "--- 推理脚本执行完毕。 ---"
echo "--- 预测结果已保存到: ./results/${MODEL_ID}_[...] ---"