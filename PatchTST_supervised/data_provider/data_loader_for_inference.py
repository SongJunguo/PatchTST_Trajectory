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
                 features='M', target='H', scale=True, timeenc=1, freq='s', stride=1):
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
            stride (int): 数据加载器的滑窗步长。
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
        self.stride = stride # 🚀 保存步长

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
        
        # --- 标准化（优化：使用float32减少内存使用） ---
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_data.values.astype(np.float32))
            data = self.scaler.transform(df_data.values.astype(np.float32))
        else:
            data = df_data.values.astype(np.float32)

        # --- 时间特征编码 ---
        df_stamp = df_full[['Time']]
        df_stamp.rename(columns={'Time': 'date'}, inplace=True)
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 1:
            # 将 Series 转换为 DatetimeIndex
            datetime_index = pd.DatetimeIndex(df_stamp['date'])
            data_stamp = time_features(datetime_index, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else: # 简易时间编码
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            data_stamp = df_stamp.drop(['date'], axis=1).values

        # --- 🚀 优化：构建索引映射 ---
        self.data_x = []
        self.data_stamp = []
        self.meta_data = [] # 存储元数据
        self.index_mapping = []

        print(f"正在处理 {df_full['ID'].nunique()} 条轨迹的数据...")

        # 🚀 优化：使用更高效的groupby处理
        grouped = df_full.groupby('ID', sort=False)  # sort=False 可以提高性能

        for traj_idx, (name, group) in enumerate(grouped):
            # 🚀 优化：直接使用连续索引，避免复杂的索引操作
            if group.index.is_monotonic_increasing:
                # 如果索引是连续的，直接切片
                start_idx = group.index[0]
                end_idx = group.index[-1] + 1
                group_data = data[start_idx:end_idx]
                stamp_data = data_stamp[start_idx:end_idx]
            else:
                # 如果索引不连续，使用原来的方法
                group_indices = group.index
                group_data = data[group_indices]
                stamp_data = data_stamp[group_indices]

            # 获取元数据（优化：减少iloc调用）
            first_row = group.iloc[0]
            meta_group = {
                'ID': name,
                'TASK': first_row['TASK'],
                'PLANETYPE': first_row['PLANETYPE'],
                'Time': group['Time'].values # 存储整个轨迹的时间戳
            }

            self.data_x.append(group_data)
            self.data_stamp.append(stamp_data)
            self.meta_data.append(meta_group)

            # 计算该轨迹可以产生的样本数量
            traj_samples = len(group_data) - self.seq_len + 1
            if traj_samples > 0:
                # 🚀 优化：批量添加索引映射，并使用步长
                traj_indices = [(traj_idx, local_idx) for local_idx in range(0, traj_samples, self.stride)]
                self.index_mapping.extend(traj_indices)

        print(f"数据处理完成，共生成 {len(self.index_mapping)} 个样本 (滑窗步长: {self.stride})")

    def __getitem__(self, index):
        # 1. 使用预计算的索引映射，O(1)复杂度
        traj_idx, local_idx = self.index_mapping[index]

        # 2. 获取对应轨迹的数据（优化：直接索引，避免额外变量）
        traj_data_x = self.data_x[traj_idx]
        traj_data_stamp = self.data_stamp[traj_idx]
        traj_meta = self.meta_data[traj_idx]

        # 3. 定义滑窗（优化：直接计算，减少变量赋值）
        s_begin = local_idx
        s_end = s_begin + self.seq_len

        # 4. 🚀 优化：使用视图而非复制，减少内存开销
        seq_x = traj_data_x[s_begin:s_end]  # 使用视图，避免不必要的复制
        seq_x_mark = traj_data_stamp[s_begin:s_end]  # 使用视图

        # 5. 准备要传递的元数据（优化：减少字典查找）
        prediction_anchor_time = traj_meta['Time'][s_end - 1]

        meta_info = {
            'Pred_trajectory_id': traj_meta['ID'],
            'prediction_anchor_time': prediction_anchor_time,
            'TASK': traj_meta['TASK'],
            'PLANETYPE': traj_meta['PLANETYPE']
        }

        # 6. 🚀 优化：使用预分配的全局零数组，避免重复创建
        if not hasattr(self, '_seq_y_template'):
            self._seq_y_template = np.zeros((self.pred_len, seq_x.shape[-1]), dtype=np.float32)
            self._seq_y_mark_template = np.zeros((self.pred_len, seq_x_mark.shape[-1]), dtype=np.float32)

        # 返回模板的副本（比每次创建新数组快）
        seq_y = self._seq_y_template.copy()
        seq_y_mark = self._seq_y_mark_template.copy()

        return seq_x, seq_y, seq_x_mark, seq_y_mark, meta_info

    def __len__(self):
        return len(self.index_mapping)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_scaler(self):
        return self.scaler