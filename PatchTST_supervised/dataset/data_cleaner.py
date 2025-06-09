#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清理器
处理经纬高数据的缺失值：
1. ID开头的缺失值：直接删除
2. ID中间的缺失值：进行插补
3. ID末尾的缺失值：直接删除
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')


def clean_missing_data(input_file="./dataset/processed_2022-05-01.csv", output_file="./dataset/cleaned_2022-05-01.csv"):
    """
    清理缺失数据
    
    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    
    返回:
    bool: 处理是否成功
    """
    try:
        print(f"开始清理数据: {input_file}")
        
        # 读取数据
        print("正在读取数据...")
        df = pd.read_csv(input_file)
        
        print(f"原始数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        
        # 检查关键列的缺失情况
        key_columns = ['H', 'Lon', 'Lat']
        print(f"\n原始缺失值统计:")
        for col in key_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                print(f"{col}: {missing_count} ({missing_pct:.2f}%)")
        
        # 按ID分组处理
        print("\n开始按ID分组处理缺失值...")
        cleaned_data = []
        unique_ids = df['ID'].unique()
        
        total_ids = len(unique_ids)
        processed_ids = 0
        
        for flight_id in unique_ids:
            processed_ids += 1
            if processed_ids % 1000 == 0:
                print(f"已处理 {processed_ids}/{total_ids} 个ID...")
            
            # 获取当前ID的数据
            id_data = df[df['ID'] == flight_id].copy().reset_index(drop=True)
            
            # 处理当前ID的缺失值
            cleaned_id_data = process_id_missing_values(id_data, key_columns)
            
            if len(cleaned_id_data) > 0:
                cleaned_data.append(cleaned_id_data)
        
        # 合并所有清理后的数据
        if cleaned_data:
            final_data = pd.concat(cleaned_data, ignore_index=True)
        else:
            final_data = pd.DataFrame(columns=df.columns)
        
        print(f"\n清理后数据形状: {final_data.shape}")
        
        # 检查清理后的缺失情况
        print(f"\n清理后缺失值统计:")
        for col in key_columns:
            if col in final_data.columns:
                missing_count = final_data[col].isna().sum()
                missing_pct = (missing_count / len(final_data)) * 100 if len(final_data) > 0 else 0
                print(f"{col}: {missing_count} ({missing_pct:.2f}%)")
        
        # 保存清理后的数据
        print(f"\n正在保存到文件: {output_file}")
        final_data.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"数据清理完成! 文件已保存为: {output_file}")
        print(f"数据行数从 {len(df)} 减少到 {len(final_data)}")
        
        return True
        
    except Exception as e:
        print(f"数据清理过程中发生错误: {str(e)}")
        return False


def process_id_missing_values(id_data, key_columns):
    """
    处理单个ID的缺失值
    
    参数:
    id_data: 单个ID的数据DataFrame
    key_columns: 需要处理的关键列列表
    
    返回:
    DataFrame: 处理后的数据
    """
    if len(id_data) == 0:
        return pd.DataFrame()
    
    # 创建一个组合的缺失值标记（任一关键列缺失就标记为缺失）
    missing_mask = pd.Series([False] * len(id_data))
    for col in key_columns:
        if col in id_data.columns:
            missing_mask |= id_data[col].isna()
    
    # 如果所有数据都缺失，返回空DataFrame
    if missing_mask.all():
        return pd.DataFrame()
    
    # 找到第一个和最后一个非缺失值的位置
    valid_indices = np.where(~missing_mask)[0]
    if len(valid_indices) == 0:
        return pd.DataFrame()
    
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]
    
    # 删除开头和结尾的缺失值
    trimmed_data = id_data.iloc[first_valid_idx:last_valid_idx+1].copy()
    
    # 如果删除后没有数据，返回空DataFrame
    if len(trimmed_data) == 0:
        return pd.DataFrame()
    
    # 对中间的缺失值进行插补
    for col in key_columns:
        if col in trimmed_data.columns:
            trimmed_data[col] = interpolate_missing_values(trimmed_data[col])
    
    return trimmed_data


def interpolate_missing_values(series):
    """
    对序列中的缺失值进行插补
    
    参数:
    series: pandas Series
    
    返回:
    pandas Series: 插补后的序列
    """
    if series.isna().all():
        return series
    
    # 使用线性插值
    interpolated = series.interpolate(method='linear', limit_direction='both')
    
    # 如果还有缺失值（比如开头或结尾），使用前向填充和后向填充
    interpolated = interpolated.fillna(method='ffill').fillna(method='bfill')
    
    return interpolated


def analyze_data_quality(file_path):
    """
    分析数据质量
    """
    try:
        df = pd.read_csv(file_path)
        print(f"\n=== 数据质量分析: {file_path} ===")
        print(f"总行数: {len(df)}")
        print(f"总ID数: {df['ID'].nunique()}")
        
        # 每个ID的平均数据点数
        id_counts = df['ID'].value_counts()
        print(f"每个ID平均数据点数: {id_counts.mean():.2f}")
        print(f"每个ID数据点数范围: {id_counts.min()} - {id_counts.max()}")
        
        # 缺失值统计
        key_columns = ['H', 'Lon', 'Lat']
        for col in key_columns:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                missing_pct = (missing_count / len(df)) * 100
                print(f"{col}缺失值: {missing_count} ({missing_pct:.2f}%)")
        
        return True
    except Exception as e:
        print(f"数据质量分析失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("数据清理器")
    print("处理策略:")
    print("1. ID开头的缺失值：直接删除")
    print("2. ID中间的缺失值：线性插补")
    print("3. ID末尾的缺失值：直接删除")
    print("=" * 60)
    
    input_file = "./dataset/processed_2022-05-01.csv"
    output_file = "./dataset/cleaned_2022-05-01.csv"
    
    # 分析原始数据质量
    print("分析原始数据质量...")
    analyze_data_quality(input_file)
    
    # 执行数据清理
    success = clean_missing_data(input_file, output_file)
    
    if success:
        print("\n✅ 数据清理完成!")
        # 分析清理后数据质量
        analyze_data_quality(output_file)
    else:
        print("\n❌ 数据清理失败!")


if __name__ == "__main__":
    # 检查依赖
    try:
        import pandas as pd
        import numpy as np
        from scipy import interpolate
    except ImportError:
        print("错误: 需要安装pandas, numpy, scipy库")
        print("请运行: pip install pandas numpy scipy")
        exit(1)
    
    main()
