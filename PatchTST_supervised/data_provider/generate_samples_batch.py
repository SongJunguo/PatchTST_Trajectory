import os
import shutil # 用于高效的文件操作
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

# =========================================================
# 所有的辅助函数和工作函数都必须在这里定义
# =========================================================

def z_score_normalize(data, means, stds):
    """Z-score归一化"""
    stds[stds == 0] = 1.0
    return (data - means) / stds

def append_to_csv(filepath, data_array):
    """将Numpy数组追加到CSV文件，不含表头和索引。"""
    if data_array.size == 0:
        return
    data_to_save = data_array.reshape(-1, data_array.shape[-1])
    # 'a'模式为追加，如果文件不存在则创建
    pd.DataFrame(data_to_save).to_csv(filepath, mode='a', header=False, index=False)

def process_file(file_path):
    """(并行工作函数) 读取单个CSV文件，返回轨迹字典。"""
    try:
        df = pd.read_csv(file_path, dtype={0: str}, low_memory=False)
        df_data = df.iloc[:, :4]
        df_data.columns = ['id', 'lon', 'lat', 'h']
        trajectories = {}
        for traj_id, group in df_data.groupby('id'):
            trajectories[traj_id] = group[['lon', 'lat', 'h']].values.astype(np.float32)
        return trajectories
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return {}

def create_windows_and_save_to_temp(args):
    """
    (最终版并行工作函数) 
    为单个轨迹创建窗口，并直接将结果保存到唯一的临时文件中。
    返回临时文件的路径和统计信息，而不是巨大的数据本身。
    """
    (traj_id, traj_data, sequence_length, prediction_steps, time_step, 
     means, stds, temp_dir, set_name) = args
    
    # 创建窗口
    total_length = sequence_length + prediction_steps
    windows = []
    for i in range(0, len(traj_data) - total_length + 1, time_step):
        window = traj_data[i : i + total_length, :].copy()
        normalized_window = z_score_normalize(window, means, stds)
        windows.append(normalized_window)
    
    if not windows:
        return None

    windows_array = np.array(windows, dtype=np.float32)
    
    # 定义唯一的临时文件路径
    temp_x_path = os.path.join(temp_dir, f"{set_name}_x_{os.getpid()}_{traj_id}.csv")
    temp_y_path = os.path.join(temp_dir, f"{set_name}_y_{os.getpid()}_{traj_id}.csv")
    
    # 直接写入临时文件
    append_to_csv(temp_x_path, windows_array[:, :sequence_length, :])
    append_to_csv(temp_y_path, windows_array[:, sequence_length:, :])
    
    # 只返回轻量级的结果
    return {
        'set_name': set_name,
        'x_path': temp_x_path,
        'y_path': temp_y_path,
        'id': traj_id,
        'count': len(windows)
    }

# =========================================================
# 最终的主函数
# =========================================================
def process_all_files_final(file_path, save_path, sequence_length, prediction_steps, time_step, num_workers):
    """
    最终版：并行计算并由工作进程直接写入临时文件，主进程最后合并。
    """
    print(f"🚀 最终版并行处理 (最多{num_workers}核)...")
    
    # 准备一个临时目录来存放中间文件
    temp_dir = os.path.join(save_path, "temp_files")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 清理最终的输出文件
    output_files_paths = {
        'train_x': os.path.join(save_path, 'train_x.csv'), 'train_y': os.path.join(save_path, 'train_y.csv'),
        'val_x': os.path.join(save_path, 'val_x.csv'), 'val_y': os.path.join(save_path, 'val_y.csv'),
        'test_x': os.path.join(save_path, 'test_x.csv'), 'test_y': os.path.join(save_path, 'test_y.csv'),
        'train_ids': os.path.join(save_path, 'train_ids.csv'), 'val_ids': os.path.join(save_path, 'val_ids.csv'),
        'test_ids': os.path.join(save_path, 'test_ids.csv')
    }
    for f in output_files_paths.values():
        if os.path.exists(f):
            os.remove(f)

    # 步骤 1 & 2: 读取数据和计算统计量
    print("\n[步骤 1 & 2: 读取数据与计算统计量...]")
    all_files = []
    if os.path.isdir(file_path):
        all_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]
    elif os.path.isfile(file_path) and file_path.endswith('.csv'):
        all_files = [file_path]
    if not all_files: 
        print(f"错误：在路径 '{file_path}' 中未找到任何 .csv 文件。")
        return False

    all_trajectories_dict = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in all_files}
        for future in tqdm(as_completed(futures), total=len(all_files), desc="读取文件"):
            all_trajectories_dict.update(future.result())
    
    all_ids = list(all_trajectories_dict.keys())
    train_ids, temp_ids = train_test_split(all_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.33, random_state=42)
    
    train_ids_set, val_ids_set, test_ids_set = set(train_ids), set(val_ids), set(test_ids)
    
    if not train_ids:
        print("错误：没有轨迹被分配到训练集，无法计算统计数据。")
        return False

    train_data_points = np.concatenate([all_trajectories_dict[tid] for tid in train_ids])
    means = np.mean(train_data_points, axis=0, dtype=np.float32)
    stds = np.std(train_data_points, axis=0, dtype=np.float32)
    pd.DataFrame([means]).to_csv(os.path.join(save_path, 'mean.csv'), index=False, header=None)
    pd.DataFrame([stds]).to_csv(os.path.join(save_path, 'std.csv'), index=False, header=None)
    del train_data_points

    # 步骤 3: 创建并行任务，让每个任务自己保存
    print("\n[步骤 3: 并行创建窗口并写入临时文件...]")
    tasks = []
    for traj_id, traj_data in all_trajectories_dict.items():
        if len(traj_data) < sequence_length + prediction_steps:
            continue
        
        set_name = ''
        if traj_id in train_ids_set: set_name = 'train'
        elif traj_id in val_ids_set: set_name = 'val'
        else: set_name = 'test'
        
        tasks.append((traj_id, traj_data, sequence_length, prediction_steps, time_step, means, stds, temp_dir, set_name))
    
    # 执行并行处理
    temp_file_results = {'train': [], 'val': [], 'test': []}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(create_windows_and_save_to_temp, tasks), total=len(tasks), desc="并行处理轨迹"))

    # 收集返回的轻量级结果
    for res in results:
        if res:
            temp_file_results[res['set_name']].append(res)
            
    # 步骤 4: 合并所有临时文件
    print("\n[步骤 4: 合并临时文件...]")
    for set_name in ['train', 'val', 'test']:
        print(f"合并 {set_name} 数据集...")
        id_list = []
        
        with open(output_files_paths[f'{set_name}_x'], 'wb') as wfd_x, open(output_files_paths[f'{set_name}_y'], 'wb') as wfd_y:
            for result in tqdm(temp_file_results[set_name], desc=f"合并 {set_name} 文件"):
                if os.path.exists(result['x_path']):
                    with open(result['x_path'], 'rb') as rfd:
                        shutil.copyfileobj(rfd, wfd_x)
                if os.path.exists(result['y_path']):
                    with open(result['y_path'], 'rb') as rfd:
                        shutil.copyfileobj(rfd, wfd_y)
                id_list.extend([result['id']] * result['count'])
        
        pd.DataFrame({'id': id_list}).to_csv(output_files_paths[f'{set_name}_ids'], index=False)

    # 步骤 5: 清理临时文件
    print("\n[步骤 5: 清理临时文件...]")
    shutil.rmtree(temp_dir)
    
    print("\n--- ✅ 处理完成 ---")
    return True

if __name__ == '__main__':
    seq_length = 192
    pred_steps = 72
    time_step = 1 # 设置为1来测试最极端的情况

    NUM_WORKERS = 64 # 建议设为物理核心数

    save_paths = f"./dataset/input-{seq_length}_output-{pred_steps}_timestep-{time_step}_final"
    if not os.path.exists(save_paths):
        os.makedirs(save_paths)

    process_all_files_final(file_path='./dataset/advanced_cleaned_2022-05-01.csv',
                            save_path=save_paths,
                            sequence_length=seq_length,
                            prediction_steps=pred_steps,
                            time_step=time_step,
                            num_workers=NUM_WORKERS)