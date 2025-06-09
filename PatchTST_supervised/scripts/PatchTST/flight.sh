#!/bin/bash

# 优化后的飞行数据集训练脚本
# 主要优化：
# 1. 减小batch_size以减少数据加载压力
# 2. 增加num_workers以充分利用多核CPU
# 3. 数据加载器已添加pin_memory和persistent_workers优化

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path advanced_cleaned_2022-05-01.csv \
  --model_id 20250609flight_dependent \
  --model PatchTST \
  --data flight \
  --features M \
  --target H \
  --num_workers 64 \
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
  --devices 0,1
