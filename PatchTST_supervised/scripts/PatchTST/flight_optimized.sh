#!/bin/bash

# 飞行数据集GPU利用率优化脚本
# 解决GPU利用率从80%降到70%的问题

echo "=== 飞行数据集GPU利用率优化训练 ==="
echo "优化措施："
echo "1. 优化Dataset_Flight的__getitem__方法，从O(n)改为O(1)复杂度"
echo "2. 添加pin_memory=True加速GPU数据传输"
echo "3. 添加persistent_workers=True保持worker进程活跃"
echo "4. 调整batch_size和num_workers的平衡"
echo "5. 添加prefetch_factor提高数据预取效率"
echo ""

# 配置1：高吞吐量配置（推荐用于您的196核CPU）
echo "开始配置1：高吞吐量配置..."
python -u PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path ./PatchTST_supervised/dataset/ \
  --data_path advanced_cleaned_2022-05-01.csv \
  --model_id flight_192_72_high_throughput \
  --model PatchTST \
  --data flight \
  --features M \
  --target H \
  --num_workers 96 \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 72 \
  --e_layers 3 \
  --d_layers 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0.0 \
  --patch_len 16 \
  --stride 8 \
  --des 'HighThroughput' \
  --train_epochs 10 \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --itr 1 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --use_multi_gpu \
  --devices 0,1

echo ""
echo "配置1完成！"
echo ""

# 配置2：平衡配置
echo "开始配置2：平衡配置..."
python -u PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path ./PatchTST_supervised/dataset/ \
  --data_path advanced_cleaned_2022-05-01.csv \
  --model_id flight_192_72_balanced \
  --model PatchTST \
  --data flight \
  --features M \
  --target H \
  --num_workers 64 \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 72 \
  --e_layers 3 \
  --d_layers 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0.0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Balanced' \
  --train_epochs 10 \
  --batch_size 384 \
  --learning_rate 0.0001 \
  --itr 1 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --use_multi_gpu \
  --devices 0,1

echo ""
echo "配置2完成！"
echo ""

# 配置3：大批次配置（如果内存充足）
echo "开始配置3：大批次配置..."
python -u PatchTST_supervised/run_longExp.py \
  --is_training 1 \
  --root_path ./PatchTST_supervised/dataset/ \
  --data_path advanced_cleaned_2022-05-01.csv \
  --model_id flight_192_72_large_batch \
  --model PatchTST \
  --data flight \
  --features M \
  --target H \
  --num_workers 48 \
  --seq_len 192 \
  --label_len 48 \
  --pred_len 72 \
  --e_layers 3 \
  --d_layers 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0.0 \
  --patch_len 16 \
  --stride 8 \
  --des 'LargeBatch' \
  --train_epochs 10 \
  --batch_size 512 \
  --learning_rate 0.0001 \
  --itr 1 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --use_multi_gpu \
  --devices 0,1

echo ""
echo "所有配置测试完成！"
echo ""
echo "建议："
echo "1. 监控GPU利用率变化，选择最优配置"
echo "2. 如果内存不足，可以进一步减小batch_size"
echo "3. 可以使用nvidia-smi监控GPU使用情况"
echo "4. 可以使用htop监控CPU使用情况"
