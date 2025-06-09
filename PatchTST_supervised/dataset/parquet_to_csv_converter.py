#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parquet to CSV Converter
将parquet格式文件转换为CSV格式文件的工具

使用方法:
python parquet_to_csv_converter.py

默认会将 '2022-05-01.parquet' 转换为 '2022-05-01.csv'
"""

import pandas as pd
import os
import sys
from pathlib import Path


def convert_parquet_to_csv(input_file, output_file=None, encoding='utf-8'):
    """
    将parquet文件转换为CSV文件
    
    参数:
    input_file (str): 输入的parquet文件路径
    output_file (str, optional): 输出的CSV文件路径，如果不指定则自动生成
    encoding (str): CSV文件的编码格式，默认为utf-8
    
    返回:
    bool: 转换是否成功
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(f"错误: 输入文件 '{input_file}' 不存在")
            return False
        
        # 如果没有指定输出文件名，则自动生成
        if output_file is None:
            input_path = Path(input_file)
            output_file = input_path.with_suffix('.csv')
        
        print(f"开始转换: {input_file} -> {output_file}")
        
        # 读取parquet文件
        print("正在读取parquet文件...")
        df = pd.read_parquet(input_file)
        
        # 显示数据基本信息
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"数据类型:")
        print(df.dtypes)
        print(f"\n前5行数据预览:")
        print(df.head())
        
        # 保存为CSV文件
        print(f"\n正在保存为CSV文件...")
        df.to_csv(output_file, index=False, encoding=encoding)
        
        print(f"转换成功! 文件已保存为: {output_file}")
        print(f"输出文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        return False


def main():
    """主函数"""
    # 默认的输入文件
    default_input_file = "./dataset/2022-05-02.parquet"
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = default_input_file
    
    # 检查输出文件参数
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = None
    
    print("=" * 50)
    print("Parquet to CSV 转换器")
    print("=" * 50)
    
    # 执行转换
    success = convert_parquet_to_csv(input_file, output_file)
    
    if success:
        print("\n转换完成!")
    else:
        print("\n转换失败!")
        sys.exit(1)


if __name__ == "__main__":
    # 检查pandas是否已安装
    try:
        import pandas as pd
    except ImportError:
        print("错误: 需要安装pandas库")
        print("请运行: pip install pandas")
        sys.exit(1)
    
    # 检查pyarrow是否已安装（读取parquet文件需要）
    try:
        import pyarrow
    except ImportError:
        print("错误: 需要安装pyarrow库来读取parquet文件")
        print("请运行: pip install pyarrow")
        sys.exit(1)
    
    main()
