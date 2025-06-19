#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化后的数据加载器性能
"""

import os
import sys
import time
import psutil
import threading
import numpy as np

# 添加路径
sys.path.append('./PatchTST_supervised')

def test_dataloader_performance():
    """测试数据加载器性能"""
    print("=== 测试优化后的数据加载器性能 ===")
    
    try:
        from data_provider.data_loader_for_inference import Dataset_Flight_Inference
        import torch
        from torch.utils.data import DataLoader
        
        # 检查数据文件
        data_path = "./PatchTST_supervised/dataset/processed_for_web/final_processed_trajectories.parquet"
        if not os.path.exists(data_path):
            print(f"错误：找不到数据文件 {data_path}")
            return
        
        print(f"使用数据文件: {data_path}")
        
        # 测试配置（基于我们的优化）
        configs = [
            {"name": "原始配置", "num_workers": 24, "batch_size": 512, "pin_memory": False, "persistent_workers": False},
            {"name": "优化配置1", "num_workers": 4, "batch_size": 1024, "pin_memory": True, "persistent_workers": True},
            {"name": "优化配置2", "num_workers": 4, "batch_size": 2048, "pin_memory": True, "persistent_workers": True},
            {"name": "优化配置3", "num_workers": 6, "batch_size": 2048, "pin_memory": True, "persistent_workers": True},
        ]
        
        results = []
        
        for config in configs:
            print(f"\n--- 测试 {config['name']} ---")
            print(f"配置: num_workers={config['num_workers']}, batch_size={config['batch_size']}")
            
            try:
                # 创建数据集
                dataset_start = time.time()
                dataset = Dataset_Flight_Inference(
                    root_path="./PatchTST_supervised/dataset/processed_for_web/",
                    data_path="final_processed_trajectories.parquet",
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
                    persistent_workers=config['persistent_workers'] and config['num_workers'] > 0,
                    prefetch_factor=4 if config['num_workers'] > 0 else 2
                )
                
                print(f"数据加载器创建完成，共 {len(dataloader)} 个批次")
                
                # 启动CPU监控
                cpu_usage = []
                memory_usage = []
                monitoring = True
                
                def monitor_system():
                    while monitoring:
                        cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
                        active_cores = sum(1 for usage in cpu_percent if usage > 5.0)
                        memory = psutil.virtual_memory()
                        
                        cpu_usage.append({
                            'avg': np.mean(cpu_percent),
                            'max': np.max(cpu_percent),
                            'active_cores': active_cores
                        })
                        memory_usage.append(memory.percent)
                        time.sleep(0.5)
                
                monitor_thread = threading.Thread(target=monitor_system, daemon=True)
                monitor_thread.start()
                
                # 测试数据加载性能
                print("开始数据加载测试...")
                load_times = []
                start_time = time.time()
                
                for i, batch in enumerate(dataloader):
                    batch_start = time.time()
                    
                    # 模拟数据处理
                    batch_x, batch_y, batch_x_mark, batch_y_mark, meta_info = batch
                    
                    # 模拟GPU传输（如果有GPU）
                    if torch.cuda.is_available():
                        batch_x = batch_x.float().cuda(non_blocking=True)
                        batch_x_mark = batch_x_mark.float().cuda(non_blocking=True)
                        torch.cuda.synchronize()
                    
                    batch_time = time.time() - batch_start
                    load_times.append(batch_time)
                    
                    # 只测试前20个批次
                    if i >= 19:
                        break
                
                # 停止监控
                monitoring = False
                monitor_thread.join(timeout=2)
                
                total_time = time.time() - start_time
                avg_batch_time = np.mean(load_times)
                
                # 计算CPU统计
                if cpu_usage:
                    avg_cpu = np.mean([d['avg'] for d in cpu_usage])
                    max_cpu = np.max([d['max'] for d in cpu_usage])
                    avg_active_cores = np.mean([d['active_cores'] for d in cpu_usage])
                    avg_memory = np.mean(memory_usage)
                else:
                    avg_cpu = max_cpu = avg_active_cores = avg_memory = 0
                
                result = {
                    'config': config,
                    'dataset_time': dataset_time,
                    'total_time': total_time,
                    'avg_batch_time': avg_batch_time,
                    'throughput_batches': len(load_times) / total_time,
                    'throughput_samples': len(load_times) * config['batch_size'] / total_time,
                    'avg_cpu': avg_cpu,
                    'max_cpu': max_cpu,
                    'avg_active_cores': avg_active_cores,
                    'avg_memory': avg_memory
                }
                
                results.append(result)
                
                print(f"结果:")
                print(f"  数据集初始化: {dataset_time:.2f}s")
                print(f"  总加载时间: {total_time:.2f}s")
                print(f"  平均批次时间: {avg_batch_time:.3f}s")
                print(f"  批次吞吐量: {result['throughput_batches']:.2f} batches/s")
                print(f"  样本吞吐量: {result['throughput_samples']:.2f} samples/s")
                print(f"  平均CPU使用: {avg_cpu:.1f}%")
                print(f"  平均活跃核心: {avg_active_cores:.1f}/24")
                print(f"  平均内存使用: {avg_memory:.1f}%")
                
            except Exception as e:
                print(f"配置测试失败: {e}")
                import traceback
                traceback.print_exc()
            
            time.sleep(2)  # 让系统休息
        
        # 输出对比结果
        if results:
            print(f"\n=== 性能对比总结 ===")
            print(f"{'配置':<12} {'批次吞吐量':<12} {'样本吞吐量':<12} {'CPU使用':<10} {'活跃核心':<10}")
            print("-" * 70)
            
            best_config = None
            best_throughput = 0
            
            for result in results:
                config = result['config']
                throughput = result['throughput_samples']
                
                print(f"{config['name']:<12} {result['throughput_batches']:<12.2f} "
                      f"{throughput:<12.2f} {result['avg_cpu']:<10.1f} "
                      f"{result['avg_active_cores']:<10.1f}")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_config = config
            
            if best_config:
                print(f"\n🏆 最佳配置: {best_config['name']}")
                print(f"   num_workers: {best_config['num_workers']}")
                print(f"   batch_size: {best_config['batch_size']}")
                print(f"   样本吞吐量: {best_throughput:.2f} samples/s")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_single_sample_performance():
    """测试单样本获取性能"""
    print("\n=== 测试单样本获取性能 ===")
    
    try:
        from data_provider.data_loader_for_inference import Dataset_Flight_Inference
        
        # 创建数据集
        print("创建数据集...")
        dataset = Dataset_Flight_Inference(
            root_path="./PatchTST_supervised/dataset/processed_for_web/",
            data_path="final_processed_trajectories.parquet",
            size=[192, 0, 72],
            features='M',
            target='H',
            scale=True,
            timeenc=1,
            freq='s'
        )
        
        print(f"数据集大小: {len(dataset)} 个样本")
        
        # 测试单样本获取时间
        print("测试单样本获取时间...")
        sample_times = []
        
        for i in range(min(100, len(dataset))):
            start_time = time.time()
            sample = dataset[i]
            sample_time = time.time() - start_time
            sample_times.append(sample_time)
        
        avg_sample_time = np.mean(sample_times)
        max_sample_time = np.max(sample_times)
        min_sample_time = np.min(sample_times)
        
        print(f"单样本获取性能:")
        print(f"  平均时间: {avg_sample_time:.4f}s")
        print(f"  最大时间: {max_sample_time:.4f}s")
        print(f"  最小时间: {min_sample_time:.4f}s")
        print(f"  单样本吞吐量: {1/avg_sample_time:.2f} samples/s")
        
        # 分析瓶颈
        if avg_sample_time > 0.01:
            print("⚠️  单样本获取时间较长，可能存在__getitem__方法瓶颈")
        else:
            print("✅ 单样本获取性能良好")
            
    except Exception as e:
        print(f"单样本测试失败: {e}")

def main():
    """主函数"""
    print("🚀 开始测试优化后的数据加载器性能...")
    print(f"系统信息:")
    print(f"  CPU核心数: {psutil.cpu_count()} (逻辑)")
    print(f"  物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"  总内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    try:
        # 测试单样本性能
        test_single_sample_performance()
        
        # 测试数据加载器性能
        test_dataloader_performance()
        
        print("\n🎯 测试完成！")
        print("根据测试结果选择最佳配置以解决单线程CPU使用问题。")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

if __name__ == "__main__":
    main()
