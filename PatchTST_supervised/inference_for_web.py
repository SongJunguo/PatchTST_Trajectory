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
                freq=self.args.freq
            )
            data_loader = torch.utils.data.DataLoader(
                data_set,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=False
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
            # 加载模型到指定的设备
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        results_list = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, meta_info) in enumerate(tqdm(pred_loader, desc="进行预测")):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # PatchTST模型只需要输入序列
                outputs = self.model(batch_x)

                # 反归一化
                outputs = pred_data.inverse_transform(outputs.detach().cpu().numpy())
                
                # 处理batch中的每个样本
                batch_size = outputs.shape[0]
                for j in range(batch_size):
                    # 提取元数据
                    pred_id = meta_info['Pred_trajectory_id'][j]
                    anchor_time = meta_info['prediction_anchor_time'][j]
                    task = meta_info['TASK'][j]
                    planetype = meta_info['PLANETYPE'][j]
                    
                    # 提取预测值 (pred_len, num_features)
                    preds = outputs[j]
                    
                    # 创建一个DataFrame来存储这个滑窗的预测结果
                    # 假设特征顺序是 H, JD, WD
                    df = pd.DataFrame(preds, columns=['H_predicted', 'JD_predicted', 'WD_predicted'])
                    df['Pred_trajectory_id'] = pred_id
                    df['prediction_anchor_time'] = anchor_time
                    df['TASK'] = task
                    df['PLANETYPE'] = planetype
                    
                    results_list.append(df)

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
    parser.add_argument('--data', type=str, default='flight_inference', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./PatchTST_supervised/dataset/processed_for_web/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='history_data.parquet', help='data file')
    parser.add_argument('--features', type=str, default='M', help='M, S, MS')
    parser.add_argument('--target', type=str, default='H', help='target feature')
    parser.add_argument('--freq', type=str, default='s', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
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
    parser.add_argument('--revin', type=int, default=1, help='RevIN')
    parser.add_argument('--individual', type=int, default=1, help='individual head')
    parser.add_argument('--embed_type', type=int, default=1, help='0: default 1: value embedding + temporal embedding + positional embedding')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size of train input data')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--des', type=str, default='Inference', help='exp description')

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
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.des
    )
    
    print(f'>>>>>>>start predicting : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.predict(setting, load=True)

    torch.cuda.empty_cache()