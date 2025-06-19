#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU使用情况分析脚本
分析为什么多个工作进程只有一个在使用CPU
"""

import os
import sys
import time
import psutil
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def cpu_intensive_task(task_id, duration=5):
    """CPU密集型任务"""
    print(f"任务 {task_id} 开始执行 (PID: {os.getpid()})")
    start_time = time.time()
    
    # 执行CPU密集型计算
    result = 0
    while time.time() - start_time < duration:
        result += sum(range(1000))
    
    print(f"任务 {task_id} 完成 (PID: {os.getpid()})")
    return task_id, result

def io_intensive_task(task_id, duration=5):
    """I/O密集型任务（模拟数据加载）"""
    print(f"I/O任务 {task_id} 开始执行 (PID: {os.getpid()})")
    start_time = time.time()
    
    # 模拟I/O操作
    while time.time() - start_time < duration:
        # 模拟文件读取
        time.sleep(0.1)
        # 模拟一些CPU计算
        _ = sum(range(100))
    
    print(f"I/O任务 {task_id} 完成 (PID: {os.getpid()})")
    return task_id

def monitor_cpu_usage(duration=20, interval=0.5):
    """监控CPU使用情况"""
    print(f"开始监控CPU使用情况，持续 {duration} 秒...")
    
    cpu_data = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # 获取每个CPU核心的使用率
        cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
        timestamp = time.time() - start_time
        
        cpu_data.append({
            'time': timestamp,
            'cpu_usage': cpu_percent,
            'active_cores': sum(1 for usage in cpu_percent if usage > 5.0),
            'max_usage': max(cpu_percent),
            'avg_usage': sum(cpu_percent) / len(cpu_percent)
        })
        
        # 实时显示
        active_cores = sum(1 for usage in cpu_percent if usage > 5.0)
        print(f"\r时间: {timestamp:6.1f}s | "
              f"活跃核心: {active_cores:2d}/{len(cpu_percent)} | "
              f"最大使用率: {max(cpu_percent):5.1f}% | "
              f"平均使用率: {sum(cpu_percent)/len(cpu_percent):5.1f}%", 
              end="", flush=True)
    
    print("\n监控完成")
    return cpu_data

def test_multiprocessing_scenarios():
    """测试不同多进程场景"""
    
    print("=== CPU使用情况分析测试 ===")
    print(f"系统信息:")
    print(f"  CPU核心数: {psutil.cpu_count()} (逻辑)")
    print(f"  物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"  当前CPU使用率: {psutil.cpu_percent()}%")
    print()
    
    scenarios = [
        {
            'name': 'CPU密集型任务 (4进程)',
            'task_func': cpu_intensive_task,
            'num_workers': 4,
            'duration': 8
        },
        {
            'name': 'I/O密集型任务 (8进程)',
            'task_func': io_intensive_task,
            'num_workers': 8,
            'duration': 8
        },
        {
            'name': 'CPU密集型任务 (12进程)',
            'task_func': cpu_intensive_task,
            'num_workers': 12,
            'duration': 8
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- 测试场景: {scenario['name']} ---")
        
        # 启动CPU监控
        monitor_thread = threading.Thread(
            target=monitor_cpu_usage,
            args=(scenario['duration'] + 2, 0.5),
            daemon=True
        )
        monitor_thread.start()
        
        time.sleep(1)  # 让监控先启动
        
        # 执行多进程任务
        with ProcessPoolExecutor(max_workers=scenario['num_workers']) as executor:
            tasks = [
                executor.submit(scenario['task_func'], i, scenario['duration'])
                for i in range(scenario['num_workers'])
            ]
            
            # 等待所有任务完成
            results = [task.result() for task in tasks]
        
        # 等待监控完成
        monitor_thread.join(timeout=5)
        
        print(f"\n任务完成，共处理 {len(results)} 个任务")
        time.sleep(2)  # 让系统稍作休息

def analyze_dataloader_bottleneck():
    """分析数据加载器的瓶颈"""
    print("\n=== 数据加载器瓶颈分析 ===")
    
    # 模拟PyTorch DataLoader的行为
    def simulate_data_loading(worker_id, num_batches=50):
        """模拟数据加载过程"""
        print(f"数据加载器工作进程 {worker_id} 启动 (PID: {os.getpid()})")
        
        for batch_id in range(num_batches):
            # 模拟数据读取（I/O操作）
            time.sleep(0.02)  # 20ms的I/O延迟
            
            # 模拟数据预处理（CPU操作）
            data = np.random.randn(32, 192, 3)  # 模拟batch数据
            processed_data = np.mean(data, axis=1)  # 简单的预处理
            
            # 模拟数据传输延迟
            time.sleep(0.005)  # 5ms的传输延迟
            
            if batch_id % 10 == 0:
                print(f"工作进程 {worker_id} 处理了 {batch_id+1} 个批次")
        
        print(f"数据加载器工作进程 {worker_id} 完成")
        return worker_id
    
    # 测试不同数量的工作进程
    for num_workers in [1, 4, 8, 12]:
        print(f"\n--- 测试 {num_workers} 个数据加载工作进程 ---")
        
        # 启动监控
        monitor_thread = threading.Thread(
            target=monitor_cpu_usage,
            args=(15, 0.5),
            daemon=True
        )
        monitor_thread.start()
        
        time.sleep(1)
        
        # 启动数据加载工作进程
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            tasks = [
                executor.submit(simulate_data_loading, i, 30)
                for i in range(num_workers)
            ]
            
            results = [task.result() for task in tasks]
        
        monitor_thread.join(timeout=5)
        print(f"完成 {len(results)} 个工作进程")
        time.sleep(2)

def main():
    """主函数"""
    print("开始CPU使用情况分析...")
    
    try:
        # 测试多进程场景
        test_multiprocessing_scenarios()
        
        # 分析数据加载器瓶颈
        analyze_dataloader_bottleneck()
        
        print("\n=== 分析结论 ===")
        print("1. 如果CPU密集型任务能充分利用多核，说明系统支持真正的并行")
        print("2. 如果I/O密集型任务只有少数核心活跃，说明存在I/O瓶颈")
        print("3. 数据加载器测试可以揭示实际的数据处理瓶颈")
        print("\n建议:")
        print("- 如果I/O是瓶颈，减少num_workers并增加batch_size")
        print("- 如果CPU是瓶颈，优化数据预处理代码")
        print("- 如果内存是瓶颈，减少num_workers和batch_size")
        
    except KeyboardInterrupt:
        print("\n\n用户中断分析")
    except Exception as e:
        print(f"\n分析过程中出现错误: {e}")

if __name__ == "__main__":
    main()
