#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV数据处理器
读取2022-05-01.csv文件，提取指定列并重新格式化时间

输出列：ID, H, Lon, Lat, Time
Time格式：20220502 13:27:15.000
"""

import pandas as pd
import os
from datetime import datetime
import numpy as np


def process_csv_data(input_file="2022-05-01.csv", output_file="processed_2022-05-01.csv"):
    """
    处理CSV数据，提取指定列并格式化时间
    
    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    
    返回:
    bool: 处理是否成功
    """
    try:
        print(f"开始处理文件: {input_file}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误: 输入文件 '{input_file}' 不存在")
            return False
        
        # 读取CSV文件
        print("正在读取CSV文件...")
        df = pd.read_csv(input_file)
        
        print(f"原始数据形状: {df.shape}")
        print(f"原始列名: {list(df.columns)}")
        
        # 显示前几行数据
        print("\n原始数据前5行:")
        print(df.head())
        
        # 创建新的DataFrame，包含所需的列
        processed_data = pd.DataFrame()
        
        # ID列 - 使用flight_id或icao24作为ID
        if 'flight_id' in df.columns:
            processed_data['ID'] = df['flight_id']
        elif 'icao24' in df.columns:
            processed_data['ID'] = df['icao24']
        else:
            # 如果没有合适的ID列，创建序号作为ID
            processed_data['ID'] = range(1, len(df) + 1)
            print("警告: 未找到flight_id或icao24列，使用序号作为ID")
        
        # H列 - 使用altitude作为高度
        if 'altitude' in df.columns:
            processed_data['H'] = df['altitude']
        else:
            processed_data['H'] = np.nan
            print("警告: 未找到altitude列，H列设为空值")
        
        # Lon列 - 使用longitude
        if 'longitude' in df.columns:
            processed_data['Lon'] = df['longitude']
        else:
            processed_data['Lon'] = np.nan
            print("警告: 未找到longitude列，Lon列设为空值")
        
        # Lat列 - 使用latitude
        if 'latitude' in df.columns:
            processed_data['Lat'] = df['latitude']
        else:
            processed_data['Lat'] = np.nan
            print("警告: 未找到latitude列，Lat列设为空值")
        
        # Time列 - 格式化timestamp
        if 'timestamp' in df.columns:
            print("正在格式化时间列...")
            processed_data['Time'] = format_timestamp(df['timestamp'])
        else:
            # 如果没有timestamp列，生成示例时间
            print("警告: 未找到timestamp列，生成示例时间")
            base_time = datetime(2022, 5, 2, 13, 27, 15)
            processed_data['Time'] = [
                (base_time + pd.Timedelta(seconds=i)).strftime("%Y%m%d %H:%M:%S.000")
                for i in range(len(df))
            ]
        
        print(f"\n处理后数据形状: {processed_data.shape}")
        print("处理后数据前5行:")
        print(processed_data.head())
        
        # 显示数据统计信息
        print(f"\n数据统计:")
        print(f"总行数: {len(processed_data)}")
        print(f"非空ID数量: {processed_data['ID'].notna().sum()}")
        print(f"非空H数量: {processed_data['H'].notna().sum()}")
        print(f"非空Lon数量: {processed_data['Lon'].notna().sum()}")
        print(f"非空Lat数量: {processed_data['Lat'].notna().sum()}")
        print(f"非空Time数量: {processed_data['Time'].notna().sum()}")
        
        # 保存处理后的数据
        print(f"\n正在保存到文件: {output_file}")
        processed_data.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"处理完成! 文件已保存为: {output_file}")
        print(f"输出文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
        
        return True
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return False


def format_timestamp(timestamp_series):
    """
    将时间戳格式化为指定格式: 20220502 13:27:15.000
    
    参数:
    timestamp_series: pandas Series，包含时间戳数据
    
    返回:
    pandas Series: 格式化后的时间字符串
    """
    formatted_times = []
    
    for timestamp in timestamp_series:
        try:
            if pd.isna(timestamp) or timestamp == '':
                formatted_times.append('')
                continue
            
            # 解析时间戳
            if isinstance(timestamp, str):
                # 处理带时区的时间戳
                if '+' in timestamp:
                    timestamp = timestamp.split('+')[0]
                elif 'Z' in timestamp:
                    timestamp = timestamp.replace('Z', '')
                
                # 解析时间
                dt = pd.to_datetime(timestamp)
            else:
                dt = pd.to_datetime(timestamp)
            
            # 格式化为目标格式: 20220502 13:27:15.000
            formatted_time = dt.strftime("%Y%m%d %H:%M:%S.000")
            formatted_times.append(formatted_time)
            
        except Exception as e:
            print(f"时间格式化错误: {timestamp}, 错误: {e}")
            formatted_times.append('')
    
    return pd.Series(formatted_times)


def main():
    """主函数"""
    print("=" * 60)
    print("CSV数据处理器")
    print("提取列: ID, H, Lon, Lat, Time")
    print("时间格式: 20220502 13:27:15.000")
    print("=" * 60)
    
    # 输入和输出文件路径
    input_file = "./dataset/2022-05-02.csv"
    output_file = "./dataset/processed_2022-05-02.csv"
    
    # 执行处理
    success = process_csv_data(input_file, output_file)
    
    if success:
        print("\n✅ 数据处理完成!")
        print(f"输出文件: {output_file}")
        print("包含列: ID, H, Lon, Lat, Time")
    else:
        print("\n❌ 数据处理失败!")


if __name__ == "__main__":
    # 检查pandas是否已安装
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("错误: 需要安装pandas和numpy库")
        print("请运行: pip install pandas numpy")
        exit(1)
    
    main()
