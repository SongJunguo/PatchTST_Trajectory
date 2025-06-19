#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理性能监控脚本
实时监控CPU、GPU使用情况和数据加载性能
"""

import os
import sys
import time
import psutil
import threading
import subprocess
from datetime import datetime
import json

class PerformanceMonitor:
    def __init__(self, log_file="inference_performance.log"):
        self.log_file = log_file
        self.monitoring = False
        self.monitor_thread = None
        self.data = []
        
    def start_monitoring(self, interval=1):
        """开始性能监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"开始性能监控，间隔: {interval}秒")
        
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("性能监控已停止")
        
    def _monitor_loop(self, interval):
        """监控循环"""
        while self.monitoring:
            try:
                timestamp = datetime.now().isoformat()
                
                # CPU使用情况
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
                
                # 内存使用情况
                memory = psutil.virtual_memory()
                
                # 进程信息
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        if 'python' in proc.info['name'].lower():
                            processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # GPU信息（如果可用）
                gpu_info = self._get_gpu_info()
                
                # 记录数据
                data_point = {
                    'timestamp': timestamp,
                    'cpu': {
                        'overall': cpu_percent,
                        'per_core': cpu_per_core,
                        'active_cores': sum(1 for usage in cpu_per_core if usage > 5.0)
                    },
                    'memory': {
                        'percent': memory.percent,
                        'available_gb': memory.available / (1024**3),
                        'used_gb': memory.used / (1024**3)
                    },
                    'processes': processes,
                    'gpu': gpu_info
                }
                
                self.data.append(data_point)
                
                # 实时输出关键指标
                active_cores = data_point['cpu']['active_cores']
                total_cores = len(cpu_per_core)
                
                print(f"\r[{timestamp[:19]}] "
                      f"CPU: {cpu_percent:5.1f}% "
                      f"活跃核心: {active_cores}/{total_cores} "
                      f"内存: {memory.percent:5.1f}% "
                      f"Python进程: {len(processes)}", end="", flush=True)
                
                if gpu_info:
                    print(f" GPU: {gpu_info.get('utilization', 'N/A')}%", end="", flush=True)
                
            except Exception as e:
                print(f"\n监控错误: {e}")
                
            time.sleep(interval)
    
    def _get_gpu_info(self):
        """获取GPU信息"""
        try:
            # 尝试使用nvidia-smi获取GPU信息
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    gpu_util, mem_used, mem_total = lines[0].split(', ')
                    return {
                        'utilization': int(gpu_util),
                        'memory_used_mb': int(mem_used),
                        'memory_total_mb': int(mem_total),
                        'memory_percent': (int(mem_used) / int(mem_total)) * 100
                    }
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        
        return None
    
    def save_report(self):
        """保存性能报告"""
        if not self.data:
            print("没有监控数据可保存")
            return
            
        # 计算统计信息
        cpu_usage = [d['cpu']['overall'] for d in self.data]
        active_cores = [d['cpu']['active_cores'] for d in self.data]
        memory_usage = [d['memory']['percent'] for d in self.data]
        
        report = {
            'summary': {
                'duration_seconds': len(self.data),
                'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
                'max_cpu_usage': max(cpu_usage),
                'avg_active_cores': sum(active_cores) / len(active_cores),
                'max_active_cores': max(active_cores),
                'avg_memory_usage': sum(memory_usage) / len(memory_usage),
                'max_memory_usage': max(memory_usage)
            },
            'detailed_data': self.data
        }
        
        # 保存到文件
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n性能报告已保存到: {self.log_file}")
        
        # 打印摘要
        print("\n=== 性能监控摘要 ===")
        print(f"监控时长: {report['summary']['duration_seconds']} 秒")
        print(f"平均CPU使用率: {report['summary']['avg_cpu_usage']:.1f}%")
        print(f"最大CPU使用率: {report['summary']['max_cpu_usage']:.1f}%")
        print(f"平均活跃CPU核心: {report['summary']['avg_active_cores']:.1f}")
        print(f"最大活跃CPU核心: {report['summary']['max_active_cores']}")
        print(f"平均内存使用率: {report['summary']['avg_memory_usage']:.1f}%")
        print(f"最大内存使用率: {report['summary']['max_memory_usage']:.1f}%")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python monitor_inference_performance.py <推理脚本命令>")
        print("例如: python monitor_inference_performance.py ./PatchTST_supervised/scripts/PatchTST/run_inference_for_web.sh")
        sys.exit(1)
    
    # 获取要监控的命令
    command = sys.argv[1:]
    
    print(f"准备监控命令: {' '.join(command)}")
    
    # 创建性能监控器
    monitor = PerformanceMonitor()
    
    try:
        # 开始监控
        monitor.start_monitoring(interval=1)
        
        # 执行推理命令
        print(f"\n开始执行推理命令...")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True)
        
        # 实时输出推理脚本的输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"\n[推理] {output.strip()}")
        
        # 等待进程结束
        return_code = process.poll()
        
    except KeyboardInterrupt:
        print("\n\n用户中断监控")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"\n执行错误: {e}")
    finally:
        # 停止监控并保存报告
        monitor.stop_monitoring()
        monitor.save_report()
        
        if 'return_code' in locals():
            print(f"\n推理脚本退出码: {return_code}")

if __name__ == "__main__":
    main()
