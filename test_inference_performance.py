#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理性能测试脚本
测试不同配置下的推理性能和CPU利用率
"""

import os
import sys
import time
import psutil
import threading
import subprocess
from datetime import datetime
import numpy as np

def monitor_system_performance(duration=30, interval=1):
    """监控系统性能"""
    print(f"开始监控系统性能，持续 {duration} 秒...")
    
    cpu_data = []
    memory_data = []
    process_data = []
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        timestamp = time.time() - start_time
        
        # CPU使用情况
        cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
        active_cores = sum(1 for usage in cpu_percent if usage > 5.0)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        
        # Python进程信息
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'num_threads': proc.info['num_threads']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # 记录数据
        cpu_data.append({
            'time': timestamp,
            'overall_cpu': sum(cpu_percent) / len(cpu_percent),
            'max_cpu': max(cpu_percent),
            'active_cores': active_cores,
            'total_cores': len(cpu_percent)
        })
        
        memory_data.append({
            'time': timestamp,
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3)
        })
        
        process_data.append({
            'time': timestamp,
            'python_processes': len(python_processes),
            'total_cpu_by_python': sum(p['cpu_percent'] for p in python_processes),
            'total_memory_by_python': sum(p['memory_percent'] for p in python_processes)
        })
        
        # 实时显示
        print(f"\r时间: {timestamp:6.1f}s | "
              f"活跃核心: {active_cores:2d}/{len(cpu_percent)} | "
              f"平均CPU: {sum(cpu_percent)/len(cpu_percent):5.1f}% | "
              f"最大CPU: {max(cpu_percent):5.1f}% | "
              f"内存: {memory.percent:5.1f}% | "
              f"Python进程: {len(python_processes)}", 
              end="", flush=True)
        
        time.sleep(interval)
    
    print("\n监控完成")
    
    return {
        'cpu_data': cpu_data,
        'memory_data': memory_data,
        'process_data': process_data
    }

def analyze_performance_data(data):
    """分析性能数据"""
    cpu_data = data['cpu_data']
    memory_data = data['memory_data']
    process_data = data['process_data']
    
    if not cpu_data:
        print("没有性能数据可分析")
        return
    
    # CPU分析
    avg_cpu = np.mean([d['overall_cpu'] for d in cpu_data])
    max_cpu = np.max([d['max_cpu'] for d in cpu_data])
    avg_active_cores = np.mean([d['active_cores'] for d in cpu_data])
    max_active_cores = np.max([d['active_cores'] for d in cpu_data])
    total_cores = cpu_data[0]['total_cores']
    
    # 内存分析
    avg_memory = np.mean([d['percent'] for d in memory_data])
    max_memory = np.max([d['percent'] for d in memory_data])
    
    # 进程分析
    avg_python_processes = np.mean([d['python_processes'] for d in process_data])
    max_python_processes = np.max([d['python_processes'] for d in process_data])
    avg_python_cpu = np.mean([d['total_cpu_by_python'] for d in process_data])
    
    print("\n=== 性能分析结果 ===")
    print(f"CPU使用情况:")
    print(f"  平均CPU使用率: {avg_cpu:.1f}%")
    print(f"  最大CPU使用率: {max_cpu:.1f}%")
    print(f"  平均活跃核心: {avg_active_cores:.1f}/{total_cores}")
    print(f"  最大活跃核心: {max_active_cores}/{total_cores}")
    print(f"  核心利用率: {(avg_active_cores/total_cores)*100:.1f}%")
    
    print(f"\n内存使用情况:")
    print(f"  平均内存使用率: {avg_memory:.1f}%")
    print(f"  最大内存使用率: {max_memory:.1f}%")
    
    print(f"\n进程情况:")
    print(f"  平均Python进程数: {avg_python_processes:.1f}")
    print(f"  最大Python进程数: {max_python_processes}")
    print(f"  Python进程平均CPU使用: {avg_python_cpu:.1f}%")
    
    # 诊断建议
    print(f"\n=== 诊断建议 ===")
    if avg_active_cores < total_cores * 0.3:
        print("⚠️  CPU核心利用率较低，可能存在以下问题:")
        print("   - I/O瓶颈：数据加载速度限制了CPU使用")
        print("   - 进程间竞争：过多工作进程导致资源竞争")
        print("   - GIL限制：某些操作受到Python GIL限制")
        print("   建议：减少num_workers，增加batch_size")
    elif avg_active_cores > total_cores * 0.8:
        print("✅ CPU核心利用率良好，系统负载均衡")
    else:
        print("📊 CPU核心利用率中等，可以进一步优化")
    
    if avg_memory > 80:
        print("⚠️  内存使用率较高，可能影响性能")
        print("   建议：减少batch_size或num_workers")
    
    if max_python_processes > 20:
        print("⚠️  Python进程数过多，可能导致资源竞争")
        print("   建议：减少num_workers")

def test_inference_configurations():
    """测试不同的推理配置"""
    
    # 检查是否有必要的文件
    script_path = "./PatchTST_supervised/scripts/PatchTST/run_inference_for_web.sh"
    if not os.path.exists(script_path):
        print(f"错误：找不到推理脚本 {script_path}")
        return
    
    print("=== 推理性能测试 ===")
    print(f"系统信息:")
    print(f"  CPU核心数: {psutil.cpu_count()} (逻辑)")
    print(f"  物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"  总内存: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # 测试配置
    test_configs = [
        {"num_workers": 1, "batch_size": 512, "description": "单进程基准测试"},
        {"num_workers": 4, "batch_size": 512, "description": "4进程中等批次"},
        {"num_workers": 6, "batch_size": 1024, "description": "6进程大批次（当前配置）"},
        {"num_workers": 8, "batch_size": 512, "description": "8进程中等批次"},
        {"num_workers": 12, "batch_size": 512, "description": "12进程中等批次"},
    ]
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n--- 测试配置 {i+1}/{len(test_configs)}: {config['description']} ---")
        print(f"num_workers: {config['num_workers']}, batch_size: {config['batch_size']}")
        
        # 修改配置文件
        modify_config_file(script_path, config['num_workers'], config['batch_size'])
        
        # 启动性能监控
        monitor_duration = 45  # 给推理足够的时间
        performance_data = None
        
        def run_monitoring():
            nonlocal performance_data
            performance_data = monitor_system_performance(monitor_duration, 1)
        
        monitor_thread = threading.Thread(target=run_monitoring, daemon=True)
        monitor_thread.start()
        
        # 等待监控启动
        time.sleep(2)
        
        # 运行推理（限制时间）
        try:
            print("开始推理...")
            process = subprocess.Popen(
                [script_path], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                timeout=40  # 40秒超时
            )
            
            # 读取输出但不阻塞
            output_lines = []
            while process.poll() is None:
                line = process.stdout.readline()
                if line:
                    output_lines.append(line.strip())
                    if len(output_lines) <= 5:  # 只显示前几行
                        print(f"[推理] {line.strip()}")
            
            return_code = process.returncode
            
        except subprocess.TimeoutExpired:
            print("推理超时，终止进程")
            process.kill()
            return_code = -1
        except Exception as e:
            print(f"推理执行错误: {e}")
            return_code = -1
        
        # 等待监控完成
        monitor_thread.join(timeout=5)
        
        if performance_data:
            print(f"\n推理完成，返回码: {return_code}")
            analyze_performance_data(performance_data)
            results.append({
                'config': config,
                'performance_data': performance_data,
                'return_code': return_code
            })
        
        print("\n" + "="*60)
        time.sleep(3)  # 让系统休息一下
    
    # 输出最终建议
    print("\n=== 最终建议 ===")
    print("基于测试结果，建议选择CPU核心利用率最高且内存使用合理的配置")

def modify_config_file(script_path, num_workers, batch_size):
    """修改配置文件中的参数"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换NUM_WORKERS和BATCH_SIZE
        import re
        content = re.sub(r'NUM_WORKERS=\d+', f'NUM_WORKERS={num_workers}', content)
        content = re.sub(r'BATCH_SIZE=\d+', f'BATCH_SIZE={batch_size}', content)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"已更新配置: NUM_WORKERS={num_workers}, BATCH_SIZE={batch_size}")
        
    except Exception as e:
        print(f"修改配置文件失败: {e}")

def main():
    """主函数"""
    if len(sys.argv) > 1 and sys.argv[1] == "--monitor-only":
        # 仅监控模式
        print("仅监控模式，监控60秒...")
        data = monitor_system_performance(60, 1)
        analyze_performance_data(data)
    else:
        # 完整测试模式
        test_inference_configurations()

if __name__ == "__main__":
    main()
