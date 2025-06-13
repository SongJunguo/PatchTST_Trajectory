#!/bin/bash

# 优化后的飞行数据集训练脚本
# 主要优化：
# 1. 减小batch_size以减少数据加载压力
# 2. 增加num_workers以充分利用多核CPU
# 3. 数据加载器已添加pin_memory和persistent_workers优化

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# 设置变量
model_name=PatchTST
data_name=flight
seq_len=192
pred_len=72
timestamp=$(date +"%Y%m%d%H%M")

# 输出执行信息
echo "=== 飞行数据集训练开始 ==="
echo "模型: $model_name"
echo "数据集: $data_name"
echo "序列长度: $seq_len"
echo "预测长度: $pred_len"
echo "开始时间: $(date)"
echo "日志将保存到: logs/LongForecasting/${model_name}_${data_name}_${seq_len}_${pred_len}_${timestamp}.log"
echo ""

python -u run_longExp.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path processed_trajectories_multi.csv \
  --model_id 20250612flight_dependent_test \
  --model PatchTST \
  --data flight \
  --features M \
  --target H \
  --num_workers 32 \
  --seq_len 192 \
  --label_len 0 \
  --pred_len 72 \
  --e_layers 3 \
  --d_layers 1 \
  --n_heads 32 \
  --d_model 512 \
  --d_ff 1024 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0.0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Exp' \
  --individual 1 \
  --train_epochs 40 \
  --batch_size 1024 \
  --learning_rate 0.0001 \
  --itr 1 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --use_multi_gpu \
  --devices 0,1 >logs/LongForecasting/${model_name}_${data_name}_${seq_len}_${pred_len}_${timestamp}.log 2>&1

# 输出完成信息
echo ""
echo "=== 飞行数据集训练完成 ==="
echo "结束时间: $(date)"
echo "日志文件: logs/LongForecasting/${model_name}_${data_name}_${seq_len}_${pred_len}_${timestamp}.log"
echo "请查看日志文件了解训练详情"
