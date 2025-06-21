# -*- coding: utf-8 -*-
# ==============================================================================
# ** 模型推理主脚本 (Web可视化专用) **
#
# ** 版本: 1.0 **
#
# ** 功能: **
#   - 加载预训练的PatchTST模型。
#   - 使用 data_loader_for_inference.py 中的专用数据加载器。
#   - 对历史数据进行滑窗预测。
#   - 将预测结果与元数据(ID, anchor_time, TASK, PLANETYPE)结合。
#   - 将最终结果保存为单个Parquet文件，供Web端使用。
#
# ** 运行说明: **
#   (通过 run_inference_for_web.sh 脚本调用)
# ==============================================================================

import os
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

# 导入我们自定义的推理数据加载器
from data_provider.data_loader_for_inference import Dataset_Flight_Inference
# 导入主要的实验类，我们将继承它
from exp.exp_main import Exp_Main

class Exp_Inference(Exp_Main):
    def __init__(self, args):
        super(Exp_Inference, self).__init__(args)

    def _get_data(self, flag):
        """
        重写 _get_data 方法，以使用我们为推理创建的专用Dataset。
        """
        if flag == 'pred':
            # 强制使用我们自定义的推理数据集
            data_set = Dataset_Flight_Inference(
                root_path=self.args.root_path,
                data_path=self.args.data_path,
                size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
                features=self.args.features,
                target=self.args.target,
                scale=True, # 推理时需要与训练时一致的归一化
                timeenc=self.args.embed_type,
                freq=self.args.freq,
                stride=self.args.dataloader_stride, # 🚀 传递滑窗步长
                stats_path=self.args.stats_path # 传递归一化统计文件路径
            )
            # 解析字符串参数为布尔值
            pin_memory = self.args.pin_memory.lower() == 'true'
            persistent_workers = self.args.persistent_workers.lower() == 'true' and self.args.num_workers > 0

            data_loader = torch.utils.data.DataLoader(
                data_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=False,
                pin_memory=pin_memory,  # 根据参数启用内存锁定
                persistent_workers=persistent_workers,  # 根据参数保持工作进程活跃
                prefetch_factor=4 if self.args.num_workers > 0 else 2  # 动态调整预取因子
            )
            return data_set, data_loader
        else:
            # 对于其他情况（如训练、验证），沿用父类的行为
            return super()._get_data(flag)

    def predict(self, setting, load=True):
        """
        重写 predict 方法，以实现我们的核心推理和保存逻辑。
        """
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            print(f"正在从 {best_model_path} 加载模型...")

            # 加载检查点
            checkpoint = torch.load(best_model_path, map_location=self.device)

            # 检查检查点格式
            if 'model_state_dict' in checkpoint:
                # 完整的训练检查点格式
                state_dict = checkpoint['model_state_dict']
                print(f"从训练检查点加载模型 (epoch: {checkpoint.get('epoch', 'unknown')})")
            else:
                # 仅模型状态字典格式
                state_dict = checkpoint
                print("从模型状态字典加载模型")

            # 处理 DataParallel 模型的键名问题
            if any(key.startswith('module.') for key in state_dict.keys()):
                # 移除 'module.' 前缀
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key[7:] if key.startswith('module.') else key  # 移除 'module.' 前缀
                    new_state_dict[new_key] = value
                state_dict = new_state_dict
                print("检测到 DataParallel 模型，已移除 'module.' 前缀")

            self.model.load_state_dict(state_dict)

        results_list = []
        self.model.eval()

        # 🚀 性能监控
        import time
        total_batches = len(pred_loader)
        print(f"开始推理，共 {total_batches} 个批次，批次大小: {self.args.batch_size}")

        overall_start = time.time()
        batch_times = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, meta_info) in enumerate(tqdm(pred_loader, desc="进行预测")):
                batch_start = time.time()
                # 使用非阻塞传输，提高GPU利用率
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)

                # PatchTST模型只需要输入序列
                outputs = self.model(batch_x)

                # 反归一化 - 处理3D数组
                outputs_np = outputs.detach().cpu().numpy()  # [batch_size, pred_len, num_features]
                batch_size, pred_len, num_features = outputs_np.shape

                # 重塑为2D进行反归一化
                outputs_2d = outputs_np.reshape(-1, num_features)  # [batch_size * pred_len, num_features]
                outputs_2d = pred_data.inverse_transform(outputs_2d)

                # 重塑回3D
                outputs = outputs_2d.reshape(batch_size, pred_len, num_features)
                
                # 🚀 优化：批量处理，避免逐个创建DataFrame
                batch_size = outputs.shape[0]

                # 批量重塑预测结果：[batch_size, pred_len, num_features] -> [batch_size * pred_len, num_features]
                outputs_flat = outputs.reshape(-1, outputs.shape[-1])  # [batch_size * pred_len, 3]

                # 批量创建基础数据
                batch_results = []
                for j in range(batch_size):
                    # 提取当前样本的预测结果
                    start_idx = j * pred_len
                    end_idx = start_idx + pred_len
                    sample_preds = outputs_flat[start_idx:end_idx]

                    # 创建当前样本的结果字典（避免DataFrame创建）
                    sample_result = {
                        'H_predicted': sample_preds[:, 0],
                        'JD_predicted': sample_preds[:, 1],
                        'WD_predicted': sample_preds[:, 2],
                        'Pred_trajectory_id': [meta_info['Pred_trajectory_id'][j]] * pred_len,
                        'prediction_anchor_time': [meta_info['prediction_anchor_time'][j]] * pred_len,
                        'TASK': [meta_info['TASK'][j]] * pred_len,
                        'PLANETYPE': [meta_info['PLANETYPE'][j]] * pred_len
                    }
                    batch_results.append(sample_result)

                # 批量创建DataFrame（一次性操作）
                if batch_results:
                    # 合并所有样本的数据
                    combined_data = {
                        'H_predicted': np.concatenate([r['H_predicted'] for r in batch_results]),
                        'JD_predicted': np.concatenate([r['JD_predicted'] for r in batch_results]),
                        'WD_predicted': np.concatenate([r['WD_predicted'] for r in batch_results]),
                        'Pred_trajectory_id': [item for r in batch_results for item in r['Pred_trajectory_id']],
                        'prediction_anchor_time': [item for r in batch_results for item in r['prediction_anchor_time']],
                        'TASK': [item for r in batch_results for item in r['TASK']],
                        'PLANETYPE': [item for r in batch_results for item in r['PLANETYPE']]
                    }

                    # 一次性创建DataFrame
                    batch_df = pd.DataFrame(combined_data)
                    results_list.append(batch_df)

                # 记录批次处理时间
                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)

                # 每10个批次输出一次性能统计
                if (i + 1) % 10 == 0:
                    avg_batch_time = sum(batch_times[-10:]) / min(10, len(batch_times))
                    print(f"批次 {i+1}/{total_batches}, 平均批次时间: {avg_batch_time:.3f}s")

        # 输出总体性能统计
        total_time = time.time() - overall_start
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        print(f"\n🚀 性能统计:")
        print(f"  总推理时间: {total_time:.2f}s")
        print(f"  平均批次时间: {avg_batch_time:.3f}s")
        print(f"  吞吐量: {total_batches/total_time:.2f} batches/s")
        print(f"  样本吞吐量: {total_batches*self.args.batch_size/total_time:.2f} samples/s")

        if not results_list:
            print("警告：没有生成任何预测结果。")
            return

        # 合并所有结果并保存
        final_results_df = pd.concat(results_list, ignore_index=True)
        
        # 调整列顺序
        final_results_df = final_results_df[[
            'Pred_trajectory_id', 
            'prediction_anchor_time', 
            'H_predicted', 
            'JD_predicted', 
            'WD_predicted', 
            'TASK', 
            'PLANETYPE'
        ]]

        # 结果保存
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        output_path = os.path.join(folder_path, 'prediction_results.parquet')
        final_results_df.to_parquet(output_path, index=False, compression='zstd')
        
        print(f"预测完成！结果已保存到: {output_path}")
        print(f"总共为 {final_results_df['Pred_trajectory_id'].nunique()} 条轨迹生成了 {len(final_results_df)} 个预测点。")

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PatchTST for Time Series Forecasting - Inference for Web')

    # 从 run_longExp.py 复制所有参数
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, required=True, help='model id')
    parser.add_argument('--model', type=str, default='PatchTST', help='model name')
    parser.add_argument('--data', type=str, default='flight', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./PatchTST_supervised/dataset/processed_for_web/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='final_processed_trajectories.parquet', help='data file')
    parser.add_argument('--features', type=str, default='M', help='M, S, MS')
    parser.add_argument('--target', type=str, default='H', help='target feature')
    parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./PatchTST_supervised/checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seq_len', type=int, default=192, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=72, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=3, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=32, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--fc_dropout', type=float, default=0.2, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')
    parser.add_argument('--embed_type', type=int, default=1, help='0: default 1: value embedding + temporal embedding + positional embedding')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size of train input data')
    parser.add_argument('--pin_memory', type=str, default='true', help='enable pin memory for faster GPU transfer')
    parser.add_argument('--persistent_workers', type=str, default='true', help='keep worker processes alive')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    # 🚀 新增：数据加载器滑窗步长参数
    parser.add_argument('--dataloader_stride', type=int, default=1, help='stride for dataloader sliding window')
    # 🚀 新增：归一化统计文件路径
    parser.add_argument('--stats_path', type=str, required=True, help='path to the normalization stats file')

    args = parser.parse_args()

    # 设置设备
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # 初始化并运行推理实验
    exp = Exp_Inference(args)
    # 使用与训练时相同的 setting 格式
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        1,  # factor (固定值)
        'timeF',  # embed (固定值)
        True,  # distil (固定值)
        args.des,
        0  # iteration (固定值)
    )
    
    print(f'>>>>>>>start predicting : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.predict(setting, load=True)

    torch.cuda.empty_cache()