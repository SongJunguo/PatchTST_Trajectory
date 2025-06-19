#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能瓶颈诊断脚本
专门分析推理过程中的性能瓶颈
"""

import os
import sys
import time
import psutil
import threading
import subprocess
from datetime import datetime
import numpy as np

def profile_dataloader_performance():
    """分析数据加载器性能"""
    print("=== 数据加载器性能分析 ===")
    
    # 导入必要的模块
    sys.path.append('./PatchTST_supervised')
    from data_provider.data_loader_for_inference import Dataset_Flight_Inference
    import torch
    from torch.utils.data import DataLoader
    
    # 测试不同配置
    configs = [
        {"num_workers": 0, "batch_size": 512, "pin_memory": False},
        {"num_workers": 1, "batch_size": 512, "pin_memory": True},
        {"num_workers": 2, "batch_size": 512, "pin_memory": True},
        {"num_workers": 4, "batch_size": 512, "pin_memory": True},
        {"num_workers": 4, "batch_size": 1024, "pin_memory": True},
        {"num_workers": 4, "batch_size": 2048, "pin_memory": True},
    ]
    
    # 检查数据文件是否存在
    data_path = "./PatchTST_supervised/dataset/processed_for_web/history_data.parquet"
    if not os.path.exists(data_path):
        print(f"错误：找不到数据文件 {data_path}")
        return
    
    print(f"使用数据文件: {data_path}")
    
    for config in configs:
        print(f"\n--- 测试配置: {config} ---")
        
        try:
            # 创建数据集
            dataset_start = time.time()
            dataset = Dataset_Flight_Inference(
                root_path="./PatchTST_supervised/dataset/processed_for_web/",
                data_path="history_data.parquet",
                size=[192, 0, 72],
                features='M',
                target='H',
                scale=True,
                timeenc=1,
                freq='s'
            )
            dataset_time = time.time() - dataset_start
            print(f"数据集创建时间: {dataset_time:.2f}s")
            print(f"数据集大小: {len(dataset)} 个样本")
            
            # 创建数据加载器
            dataloader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                drop_last=False,
                pin_memory=config['pin_memory'],
                persistent_workers=config['num_workers'] > 0,
                prefetch_factor=4 if config['num_workers'] > 0 else 2
            )
            
            # 测试数据加载性能
            print(f"开始测试数据加载，共 {len(dataloader)} 个批次...")
            
            load_times = []
            start_time = time.time()
            
            for i, batch in enumerate(dataloader):
                batch_start = time.time()
                
                # 模拟数据传输到GPU
                if torch.cuda.is_available():
                    batch_x, batch_y, batch_x_mark, batch_y_mark, meta_info = batch
                    batch_x = batch_x.float().cuda(non_blocking=True)
                    batch_x_mark = batch_x_mark.float().cuda(non_blocking=True)
                    torch.cuda.synchronize()  # 确保传输完成
                
                batch_time = time.time() - batch_start
                load_times.append(batch_time)
                
                # 只测试前20个批次
                if i >= 19:
                    break
            
            total_time = time.time() - start_time
            avg_batch_time = np.mean(load_times)
            
            print(f"结果:")
            print(f"  总时间: {total_time:.2f}s")
            print(f"  平均批次时间: {avg_batch_time:.3f}s")
            print(f"  吞吐量: {len(load_times)/total_time:.2f} batches/s")
            print(f"  样本吞吐量: {len(load_times)*config['batch_size']/total_time:.2f} samples/s")
            
        except Exception as e:
            print(f"配置测试失败: {e}")
        
        time.sleep(1)  # 让系统休息一下

def profile_model_inference():
    """分析模型推理性能"""
    print("\n=== 模型推理性能分析 ===")
    
    try:
        import torch
        import torch.nn as nn
        
        # 创建一个简单的模型来测试
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(192*3, 72*3)
                
            def forward(self, x):
                batch_size = x.shape[0]
                x = x.view(batch_size, -1)
                x = self.linear(x)
                return x.view(batch_size, 72, 3)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleModel().to(device)
        model.eval()
        
        # 测试不同批次大小的推理性能
        batch_sizes = [256, 512, 1024, 2048]
        
        for batch_size in batch_sizes:
            print(f"\n--- 测试批次大小: {batch_size} ---")
            
            # 创建随机输入
            x = torch.randn(batch_size, 192, 3).to(device)
            
            # 预热
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # 测试推理时间
            inference_times = []
            
            with torch.no_grad():
                for _ in range(20):
                    start_time = time.time()
                    output = model(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
            
            avg_inference_time = np.mean(inference_times)
            print(f"平均推理时间: {avg_inference_time:.4f}s")
            print(f"吞吐量: {batch_size/avg_inference_time:.2f} samples/s")
            
    except Exception as e:
        print(f"模型推理测试失败: {e}")

def monitor_system_during_inference():
    """在推理过程中监控系统资源"""
    print("\n=== 系统资源监控 ===")
    
    def monitor_resources(duration=30):
        """监控系统资源"""
        cpu_usage = []
        memory_usage = []
        process_count = []
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            active_cores = sum(1 for usage in cpu_percent if usage > 5.0)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            
            # Python进程数
            python_processes = 0
            for proc in psutil.process_iter(['name']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            cpu_usage.append({
                'avg': np.mean(cpu_percent),
                'max': np.max(cpu_percent),
                'active_cores': active_cores
            })
            memory_usage.append(memory.percent)
            process_count.append(python_processes)
            
            time.sleep(1)
        
        return {
            'cpu': cpu_usage,
            'memory': memory_usage,
            'processes': process_count
        }
    
    # 启动监控
    print("开始30秒系统监控...")
    data = monitor_resources(30)
    
    # 分析结果
    avg_cpu = np.mean([d['avg'] for d in data['cpu']])
    max_cpu = np.max([d['max'] for d in data['cpu']])
    avg_active_cores = np.mean([d['active_cores'] for d in data['cpu']])
    avg_memory = np.mean(data['memory'])
    avg_processes = np.mean(data['processes'])
    
    print(f"\n监控结果:")
    print(f"  平均CPU使用率: {avg_cpu:.1f}%")
    print(f"  最大CPU使用率: {max_cpu:.1f}%")
    print(f"  平均活跃核心: {avg_active_cores:.1f}/{psutil.cpu_count()}")
    print(f"  平均内存使用率: {avg_memory:.1f}%")
    print(f"  平均Python进程数: {avg_processes:.1f}")
    
    # 诊断建议
    print(f"\n诊断建议:")
    if avg_cpu < 20:
        print("  ⚠️  CPU使用率过低，可能存在I/O瓶颈或进程等待")
    if avg_active_cores < psutil.cpu_count() * 0.3:
        print("  ⚠️  活跃CPU核心过少，多线程效果不佳")
    if avg_memory > 80:
        print("  ⚠️  内存使用率过高，可能影响性能")
    if avg_processes > 20:
        print("  ⚠️  Python进程过多，可能存在资源竞争")

def analyze_dataloader_bottleneck():
    """分析数据加载器的具体瓶颈"""
    print("\n=== 数据加载器瓶颈分析 ===")
    
    try:
        sys.path.append('./PatchTST_supervised')
        from data_provider.data_loader_for_inference import Dataset_Flight_Inference
        
        # 测试数据集初始化时间
        print("测试数据集初始化时间...")
        start_time = time.time()
        
        dataset = Dataset_Flight_Inference(
            root_path="./PatchTST_supervised/dataset/processed_for_web/",
            data_path="history_data.parquet",
            size=[192, 0, 72],
            features='M',
            target='H',
            scale=True,
            timeenc=1,
            freq='s'
        )
        
        init_time = time.time() - start_time
        print(f"数据集初始化时间: {init_time:.2f}s")
        print(f"数据集大小: {len(dataset)} 个样本")
        
        # 测试单个样本获取时间
        print("\n测试单个样本获取时间...")
        sample_times = []
        
        for i in range(100):
            start_time = time.time()
            sample = dataset[i]
            sample_time = time.time() - start_time
            sample_times.append(sample_time)
        
        avg_sample_time = np.mean(sample_times)
        print(f"平均单样本获取时间: {avg_sample_time:.4f}s")
        print(f"单样本吞吐量: {1/avg_sample_time:.2f} samples/s")
        
        # 分析瓶颈
        if init_time > 10:
            print("⚠️  数据集初始化时间过长，可能是文件读取或预处理瓶颈")
        if avg_sample_time > 0.01:
            print("⚠️  单样本获取时间过长，可能是__getitem__方法瓶颈")
        
    except Exception as e:
        print(f"数据加载器分析失败: {e}")

def main():
    """主函数"""
    print("🔍 开始性能瓶颈诊断...")
    print(f"系统信息:")
    print(f"  CPU核心数: {psutil.cpu_count()} (逻辑)")
    print(f"  物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"  总内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  CUDA可用: {'是' if __import__('torch').cuda.is_available() else '否'}")
    
    try:
        # 1. 分析数据加载器瓶颈
        analyze_dataloader_bottleneck()
        
        # 2. 分析数据加载器性能
        profile_dataloader_performance()
        
        # 3. 分析模型推理性能
        profile_model_inference()
        
        # 4. 监控系统资源
        monitor_system_during_inference()
        
        print("\n🎯 诊断完成！")
        print("根据以上分析结果，主要瓶颈可能在于:")
        print("1. 数据加载器配置不当（num_workers, batch_size）")
        print("2. 数据预处理过程耗时")
        print("3. I/O瓶颈（文件读取速度）")
        print("4. 内存带宽限制")
        print("5. CPU核心利用率不足")
        
    except Exception as e:
        print(f"诊断过程中出现错误: {e}")

if __name__ == "__main__":
    main()
