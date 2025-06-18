# -*- coding: utf-8 -*-
# ==============================================================================
# ** 推理专用数据加载器 (Web可视化) **
#
# ** 版本: 1.0 **
#
# ** 功能: **
#   - 从预处理好的历史数据文件(Parquet/CSV)中加载数据。
#   - 为模型推理生成滑窗数据。
#   - 在每个样本中附加元数据(meta_info)，用于将预测结果与历史数据关联。
#
# ==============================================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Dataset_Flight_Inference(Dataset):
    def __init__(self, root_path, data_path='history_data.parquet', size=None,
                 features='M', target='H', scale=True, timeenc=1, freq='s'):
        """
        Args:
            root_path (str): 数据文件所在的根目录。
            data_path (str): 数据文件名。
            size (list): [seq_len, label_len, pred_len]。
            features (str): 'M' (多变量) 或 'S' (单变量)。
            target (str): 单变量预测时的目标列。
            scale (bool): 是否对数据进行标准化。
            timeenc (int): 时间编码方式 (0 for simple, 1 for detailed features)。
            freq (str): 时间特征编码的频率。
        """
        # 1. 初始化基本参数
        if size is None:
            self.seq_len = 96
            self.label_len = 0  # 推理时通常不需要
            self.pred_len = 72
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        
        # 2. 读取和预处理数据
        self.__read_data__()

    def __read_data__(self):
        file_path = os.path.join(self.root_path, self.data_path)
        
        # 根据文件扩展名选择读取方式
        if self.data_path.endswith('.parquet'):
            df_full = pd.read_parquet(file_path)
        else:
            df_full = pd.read_csv(file_path)

        # --- 特征选择 ---
        # 获取除ID和Time之外的所有数值列
        cols = list(df_full.columns)
        non_feature_cols = ['ID', 'Time', 'PARTNO', 'P1', 'GP', 'TASK', 'PLANETYPE']
        feature_cols = [c for c in cols if c not in non_feature_cols]

        if self.features == 'M':
            cols_data = feature_cols
        elif self.features == 'S':
            cols_data = [self.target]
        else: # MS
            cols_data = feature_cols

        df_data = df_full[cols_data]
        
        # --- 标准化 ---
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # --- 时间特征编码 ---
        df_stamp = df_full[['Time']]
        df_stamp.rename(columns={'Time': 'date'}, inplace=True)
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 1:
            data_stamp = time_features(df_stamp['date'], freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else: # 简易时间编码
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values

        # --- 构建索引映射 ---
        self.data_x = []
        self.data_stamp = []
        self.meta_data = [] # 存储元数据
        self.index_mapping = []

        grouped = df_full.groupby('ID')
        for traj_idx, (name, group) in enumerate(grouped):
            # 获取当前轨迹的数值数据和时间戳数据
            group_indices = group.index
            group_data = data[group_indices]
            stamp_data = data_stamp[group_indices]
            
            # 获取元数据
            meta_group = {
                'ID': name,
                'TASK': group['TASK'].iloc[0],
                'PLANETYPE': group['PLANETYPE'].iloc[0],
                'Time': group['Time'].values # 存储整个轨迹的时间戳
            }

            self.data_x.append(group_data)
            self.data_stamp.append(stamp_data)
            self.meta_data.append(meta_group)

            # 计算该轨迹可以产生的样本数量
            # 注意：推理时，我们只需要seq_len的长度
            traj_samples = len(group_data) - self.seq_len + 1
            if traj_samples > 0:
                for local_idx in range(traj_samples):
                    self.index_mapping.append((traj_idx, local_idx))

    def __getitem__(self, index):
        # 1. 使用预计算的索引映射，O(1)复杂度
        traj_idx, local_idx = self.index_mapping[index]

        # 2. 获取对应轨迹的数据
        traj_data_x = self.data_x[traj_idx]
        traj_data_stamp = self.data_stamp[traj_idx]
        traj_meta = self.meta_data[traj_idx]

        # 3. 定义滑窗
        s_begin = local_idx
        s_end = s_begin + self.seq_len
        
        # 4. 提取模型输入
        seq_x = traj_data_x[s_begin:s_end]
        seq_x_mark = traj_data_stamp[s_begin:s_end]

        # 5. 准备要传递的元数据
        # 预测的锚点是输入序列的最后一个点的时间
        anchor_time_index = s_end - 1
        prediction_anchor_time = traj_meta['Time'][anchor_time_index]

        meta_info = {
            'Pred_trajectory_id': traj_meta['ID'],
            'prediction_anchor_time': prediction_anchor_time,
            'TASK': traj_meta['TASK'],
            'PLANETYPE': traj_meta['PLANETYPE']
        }
        
        # 6. 返回模型输入和元数据
        # 为了与训练时的格式保持一致，返回一个空的seq_y和seq_y_mark
        # 形状: [pred_len, num_features]
        seq_y = np.zeros((self.pred_len, seq_x.shape[-1])) 
        # 形状: [pred_len, time_features]
        seq_y_mark = np.zeros((self.pred_len, seq_x_mark.shape[-1]))

        return seq_x, seq_y, seq_x_mark, seq_y_mark, meta_info

    def __len__(self):
        return len(self.index_mapping)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_scaler(self):
        return self.scaler