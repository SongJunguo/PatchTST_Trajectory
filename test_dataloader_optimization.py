#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器优化测试脚本
用于测试不同配置下的数据加载性能和CPU利用率
"""

import os
import sys
import time
import torch
import psutil
import threading
from torch.utils.data import DataLoader, Dataset
import numpy as np

class DummyDataset(Dataset):
    """模拟数据集，用于测试数据加载性能"""
    def __init__(self, size=10000, seq_len=192, features=3):
        self.size = size
        self.seq_len = seq_len
        self.features = features
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 模拟一些计算开销
        x = torch.randn(self.seq_len, self.features)
        y = torch.randn(72, self.features)  # pred_len=72
        x_mark = torch.randn(self.seq_len, 4)
        y_mark = torch.randn(72, 4)
        meta_info = {'id': f'test_{idx}', 'anchor_time': '2024-01-01 00:00:00'}
        
        # 添加一些CPU计算来模拟真实的数据预处理
        time.sleep(0.001)  # 模拟1ms的处理时间
        
        return x, y, x_mark, y_mark, meta_info

def monitor_system_resources(duration=60, interval=1):
    """监控系统资源使用情况"""
    cpu_usage = []
    memory_usage = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_usage.append(psutil.cpu_percent(interval=None))
        memory_usage.append(psutil.virtual_memory().percent)
        time.sleep(interval)
    
    return {
        'avg_cpu': np.mean(cpu_usage),
        'max_cpu': np.max(cpu_usage),
        'avg_memory': np.mean(memory_usage),
        'max_memory': np.max(memory_usage)
    }

def test_dataloader_config(num_workers, batch_size, pin_memory, persistent_workers):
    """测试特定配置下的数据加载性能"""
    print(f"\n=== 测试配置 ===")
    print(f"num_workers: {num_workers}")
    print(f"batch_size: {batch_size}")
    print(f"pin_memory: {pin_memory}")
    print(f"persistent_workers: {persistent_workers}")
    
    # 创建数据集和数据加载器
    dataset = DummyDataset(size=5000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else 2
    )
    
    # 启动资源监控
    monitor_duration = 30
    monitor_thread = threading.Thread(
        target=lambda: monitor_system_resources(monitor_duration),
        daemon=True
    )
    
    # 开始测试
    start_time = time.time()
    total_batches = 0
    
    print("开始数据加载测试...")
    monitor_thread.start()
    
    try:
        for batch_idx, (x, y, x_mark, y_mark, meta_info) in enumerate(dataloader):
            total_batches += 1
            
            # 模拟GPU计算
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
                x_mark = x_mark.cuda(non_blocking=True)
                y_mark = y_mark.cuda(non_blocking=True)
                
                # 模拟一些GPU计算
                _ = torch.matmul(x, x.transpose(-1, -2))
            
            # 限制测试时间
            if time.time() - start_time > monitor_duration:
                break
                
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 等待监控线程结束
    monitor_thread.join(timeout=5)
    
    # 计算性能指标
    throughput = total_batches / elapsed_time if elapsed_time > 0 else 0
    
    print(f"测试结果:")
    print(f"  总耗时: {elapsed_time:.2f}秒")
    print(f"  处理批次: {total_batches}")
    print(f"  吞吐量: {throughput:.2f} batches/sec")
    
    return {
        'config': {
            'num_workers': num_workers,
            'batch_size': batch_size,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers
        },
        'performance': {
            'elapsed_time': elapsed_time,
            'total_batches': total_batches,
            'throughput': throughput
        }
    }

def main():
    """主测试函数"""
    print("=== 数据加载器优化测试 ===")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CPU核心数: {psutil.cpu_count()}")
    print(f"物理CPU核心数: {psutil.cpu_count(logical=False)}")
    
    # 测试不同配置
    test_configs = [
        # (num_workers, batch_size, pin_memory, persistent_workers)
        (0, 512, False, False),      # 单线程基准
        (1, 512, True, False),       # 单工作进程
        (2, 512, True, False),       # 2个工作进程
        (4, 512, True, False),       # 4个工作进程（推荐）
        (4, 512, True, True),        # 4个工作进程 + persistent_workers
        (4, 1024, True, True),       # 4个工作进程 + 大批次
        (8, 512, True, True),        # 8个工作进程
        (12, 512, True, True),       # 12个工作进程（原配置）
    ]
    
    results = []
    
    for config in test_configs:
        try:
            result = test_dataloader_config(*config)
            results.append(result)
            time.sleep(2)  # 让系统稍作休息
        except Exception as e:
            print(f"配置 {config} 测试失败: {e}")
    
    # 输出汇总结果
    print("\n=== 测试汇总 ===")
    print(f"{'Workers':<8} {'BatchSize':<10} {'PinMem':<8} {'Persist':<8} {'Throughput':<12}")
    print("-" * 60)
    
    best_config = None
    best_throughput = 0
    
    for result in results:
        config = result['config']
        perf = result['performance']
        
        print(f"{config['num_workers']:<8} {config['batch_size']:<10} "
              f"{config['pin_memory']:<8} {config['persistent_workers']:<8} "
              f"{perf['throughput']:<12.2f}")
        
        if perf['throughput'] > best_throughput:
            best_throughput = perf['throughput']
            best_config = config
    
    if best_config:
        print(f"\n最佳配置:")
        print(f"  num_workers: {best_config['num_workers']}")
        print(f"  batch_size: {best_config['batch_size']}")
        print(f"  pin_memory: {best_config['pin_memory']}")
        print(f"  persistent_workers: {best_config['persistent_workers']}")
        print(f"  吞吐量: {best_throughput:.2f} batches/sec")

if __name__ == "__main__":
    main()
