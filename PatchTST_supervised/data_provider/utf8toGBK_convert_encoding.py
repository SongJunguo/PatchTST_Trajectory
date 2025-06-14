import os
import pandas as pd
import argparse

def convert_csv_encoding(direction):
    """
    根据指定的方向，在UTF-8和GBK编码之间转换CSV文件。

    :param direction: 转换方向, 'utf8_to_gbk' 或 'gbk_to_utf8'
    """
    if direction == 'utf8_to_gbk':
        source_dir = 'PatchTST_supervised/dataset/raw/'
        output_dir = 'PatchTST_supervised/dataset/raw_GBK/'
        source_encoding = 'utf-8'
        target_encoding = 'gbk'
        suffix = '_gbk'
    elif direction == 'gbk_to_utf8':
        source_dir = 'PatchTST_supervised/dataset/raw/'
        output_dir = 'PatchTST_supervised/dataset/raw_UTF8/'
        source_encoding = 'gbk'
        target_encoding = 'utf-8'
        suffix = '_utf8'
    else:
        print("错误: 无效的转换方向。请选择 'utf8_to_gbk' 或 'gbk_to_utf8'。")
        return

    # 确保输出目录存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    print(f"源目录: '{source_dir}'")
    print(f"输出目录: '{output_dir}' 已准备就绪。")

    # 查找源目录下的所有CSV文件
    try:
        files_to_convert = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
        if not files_to_convert:
            print(f"在 '{source_dir}' 中未找到CSV文件。")
            return
    except FileNotFoundError:
        print(f"错误: 源目录未找到 - {source_dir}")
        return

    print(f"找到 {len(files_to_convert)} 个CSV文件进行转换。")

    # 遍历文件列表并进行转换
    for filename in files_to_convert:
        source_path = os.path.join(source_dir, filename)
        
        # 生成新的文件名
        base_name, extension = os.path.splitext(filename)
        # 移除可能存在的旧后缀
        if direction == 'gbk_to_utf8' and base_name.endswith('_gbk'):
            base_name = base_name[:-4]
        
        output_filename = f"{base_name}{suffix}{extension}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            print(f"\n正在处理文件: {source_path}")
            
            # 读取源CSV文件
            df = pd.read_csv(source_path, encoding=source_encoding)
            
            # 写入目标CSV文件
            df.to_csv(output_path, encoding=target_encoding, index=False)
            
            print(f"成功转换并保存至: {output_path}")

        except FileNotFoundError:
            print(f"错误: 文件未找到 - {source_path}")
        except Exception as e:
            print(f"处理文件 {source_path} 时发生错误: {e}")

    print("\n所有文件处理完毕。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="转换CSV文件的编码。")
    parser.add_argument(
        '--direction',
        type=str,
        default='gbk_to_utf8',
        choices=['utf8_to_gbk', 'gbk_to_utf8'],
        help="编码转换的方向: 'utf8_to_gbk' 或 'gbk_to_utf8' (默认: 'gbk_to_utf8')"
    )
    args = parser.parse_args()
    convert_csv_encoding(args.direction)