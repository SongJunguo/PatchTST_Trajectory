import os
import pandas as pd

def convert_csv_encoding():
    """
    将指定的CSV文件列表从UTF-8编码转换为GBK编码。
    """
    # 定义源目录和目标目录
    source_dir = 'PatchTST_supervised/dataset/raw/'
    output_dir = 'PatchTST_supervised/dataset/raw_GBK/'

    # 定义需要转换的文件列表
    files_to_convert = [
        'converted_data_final_20250611.csv',
        'converted_data_final_20250612.csv',
        'converted_data_final_20250613.csv',
        'converted_data_final_20250614.csv',
        'converted_data_final_20250615.csv'
    ]

    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录 '{output_dir}' 已准备就绪。")

    # 遍历文件列表并进行转换
    for filename in files_to_convert:
        source_path = os.path.join(source_dir, filename)
        
        # 生成新的文件名 (例如: converted_data_final_20250611_gbk.csv)
        base_name, extension = os.path.splitext(filename)
        output_filename = f"{base_name}_gbk{extension}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            print(f"\n正在处理文件: {source_path}")
            
            # 使用 pandas 读取 UTF-8 编码的 CSV 文件
            df = pd.read_csv(source_path, encoding='utf-8')
            
            # 将数据写入新的 GBK 编码的 CSV 文件，不包含索引
            df.to_csv(output_path, encoding='gbk', index=False)
            
            print(f"成功转换并保存至: {output_path}")

        except FileNotFoundError:
            print(f"错误: 文件未找到 - {source_path}")
        except Exception as e:
            print(f"处理文件 {source_path} 时发生错误: {e}")

    print("\n所有文件处理完毕。")

if __name__ == '__main__':
    convert_csv_encoding()