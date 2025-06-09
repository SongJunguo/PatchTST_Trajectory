import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random
from matplotlib.backends.backend_pdf import PdfPages

def visual_multi(true, preds, num_features, feature_names=None, index=None, disable_scientific_notation=False, input_data=None, plot_type='trajectory'):
    """
    Results visualization for all features. Returns the figure object.
    """
    num_cols = 1  # 两列布局，更紧凑
    num_rows = int(np.ceil(num_features / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows)) #预测60s，所以说图片扩大了3倍

    # 处理单行或单列的情况
    if num_rows == 1:
        axes = axes.reshape(1, -1) #如果只有一行，确保 axes 是二维的
    if num_cols == 1:
        axes = axes.reshape(-1, 1) #如果只有一列，确保 axes 是二维的

    # 定义维度名称 - 修改为高经纬坐标系（按照原始数据顺序）
    dimension_names = ['Altitude', 'Longitude', 'Latitude']

    for i, ax in enumerate(axes.flatten()):
        if i < num_features:
            if plot_type == 'trajectory':
                # 绘制轨迹图
                if input_data is not None:
                    plot_data_true = np.concatenate((input_data[:, i], true[:, i]), axis=0)
                    ax.plot(plot_data_true, label='GroundTruth', linewidth=2)
                    prediction_start_index = len(input_data)
                    ax.plot(range(prediction_start_index, prediction_start_index + len(preds)), preds[:, i], label='Prediction', linewidth=2)
                else:
                    ax.plot(true[:, i], label='GroundTruth', linewidth=2)
                    ax.plot(preds[:, i], label='Prediction', linewidth=2)
            else:  # plot_type == 'error'
                # 绘制绝对误差图
                error = np.abs(preds[:, i] - true[:, i])
                ax.plot(error, label='Absolute Error', linewidth=2, color='red')
                ax.axhline(y=np.mean(error), color='green', linestyle='--', label='Mean Error')

            # 使用维度名称作为标题
            if i < len(dimension_names):
                title = dimension_names[i]
                # 为不同坐标添加单位（按照高经纬顺序）
                if dimension_names[i] == 'Altitude':
                    title += ' (m)'
                elif dimension_names[i] == 'Longitude':
                    title += ' (°)'
                elif dimension_names[i] == 'Latitude':
                    title += ' (°)'
            else:
                title = f'Feature {i + 1}'

            if index is not None:
                title += f' (Test Index: {index})'
            ax.set_title(title, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True)

            # 对经纬高坐标使用合适的显示格式
            if disable_scientific_notation:
                ax.ticklabel_format(style='plain', axis='both', useOffset=False)
                # 为经纬度设置更精确的显示格式
                if i < len(dimension_names) and dimension_names[i] in ['Longitude', 'Latitude']:
                    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
                    # 设置y轴标签格式，显示更多小数位
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.6f}'))
        else:  # 关闭多余的子图
            ax.axis('off')

    plt.tight_layout()
    return fig

def visualize_test_results(results_folder, image_folder, exclude_input=True, sample_strategy='random', num_samples_to_plot=10, sample_indices=None, feature_names=None):
    """
    Visualizes test results from saved numpy files with optional sampling, saving to two PDFs:
    1. Trajectory plots (inverse transformed)
    2. Absolute error plots (inverse transformed)
    """
    pred_file = os.path.join(results_folder, 'pred.npy')
    true_file = os.path.join(results_folder, 'true.npy')
    inverse_pred_file = os.path.join(results_folder, 'inverse_pred.npy')
    inverse_true_file = os.path.join(results_folder, 'inverse_true.npy')
    input_file = os.path.join(results_folder, 'inverse_input.npy') # 加载输入数据

    if not os.path.exists(pred_file) or not os.path.exists(true_file) or not os.path.exists(inverse_pred_file) or not os.path.exists(inverse_true_file):
        print(f"Error: Missing prediction or ground truth files in {results_folder}")
        return

    # 加载归一化数据（用于MAE计算，避免量纲不匹配问题）
    preds = np.load(pred_file)
    trues = np.load(true_file)

    # 加载反归一化数据（用于可视化）
    inverse_preds = np.load(inverse_pred_file)
    inverse_trues = np.load(inverse_true_file)
    input_data = None
    if exclude_input and os.path.exists(input_file):
        input_data = np.load(input_file)
    elif exclude_input:
        print(f"Warning: Input data file not found at {input_file}, input will not be included in ground truth plots.")

    num_samples = inverse_trues.shape[0]
    num_features = inverse_trues.shape[2]

    # 确定要可视化的样本索引
    if sample_strategy == 'random':
        if num_samples_to_plot >= num_samples:
            plot_indices = list(range(num_samples))
            print("Warning: num_samples_to_plot is greater than or equal to the total number of samples. Plotting all samples.")
        else:
            plot_indices = sorted(random.sample(range(num_samples), num_samples_to_plot))
    elif sample_strategy == 'indices' and sample_indices:
        plot_indices = sorted([int(i) for i in sample_indices.split(',') if 0 <= int(i) < num_samples])  # 将字符串转换为整数列表并确保索引有效
        if len(plot_indices) == 0:
            print("Warning: No valid indices provided. Plotting no samples.")
            return
    elif sample_strategy == 'fixed_flight':  # 添加固定飞行轨迹的选项
        plot_indices = [50, 150, 250, 350, 450, 13134, 13135, 5342, 2210]  # 固定打印这5个索引的图片
        print(f"将绘制固定飞行轨迹样本: {plot_indices}")
    elif sample_strategy == 'interval':
        if num_samples_to_plot >= num_samples:
            plot_indices = list(range(num_samples))
            print("Warning: num_samples_to_plot is greater than or equal to the total number of samples. Plotting all samples.")
        elif num_samples_to_plot <= 0:
            print("Warning: num_samples_to_plot must be a positive integer for interval sampling. Plotting no samples.")
            return
        else:
            indices = np.linspace(0, num_samples - 1, num_samples_to_plot, dtype=int)
            plot_indices = sorted(list(set(indices))) # 去重并排序
    else:
        plot_indices = list(range(num_samples))

    print(f"Plotting the following indices: {plot_indices}")

    mae_per_sample = np.mean(np.abs(preds - trues), axis=(1, 2))
    mean_mae = np.mean(mae_per_sample)
    
    # 找出MAE最大、最小和最接近平均值的5个轨迹
    sorted_indices = np.argsort(mae_per_sample)
    best_indices = sorted_indices[:5]  # MAE最小的5个
    worst_indices = sorted_indices[-20:]  # MAE最大的5个
    avg_indices = sorted_indices[np.abs(mae_per_sample - mean_mae).argsort()[:5]]  # MAE最接近平均值的5个

    # 创建两个PDF文件
    trajectory_pdf = PdfPages(os.path.join(image_folder, 'trajectory_plots.pdf'))
    error_pdf = PdfPages(os.path.join(image_folder, 'error_plots.pdf'))

    # 绘制按采样策略选择的轨迹
    for i in plot_indices:
        trajectory_fig = visual_multi(inverse_trues[i], inverse_preds[i], num_features, 
                                    feature_names=feature_names, index=i, 
                                    disable_scientific_notation=True, 
                                    input_data=input_data[i] if input_data is not None else None,
                                    plot_type='trajectory')
        trajectory_pdf.savefig(trajectory_fig)
        plt.close(trajectory_fig)

        error_fig = visual_multi(inverse_trues[i], inverse_preds[i], num_features,
                                feature_names=feature_names, index=i,
                                disable_scientific_notation=True,
                                plot_type='error')
        error_pdf.savefig(error_fig)
        plt.close(error_fig)

    # 绘制最佳轨迹
    for i, idx in enumerate(best_indices):
        trajectory_fig = visual_multi(inverse_trues[idx], inverse_preds[idx], num_features, 
                                    feature_names=feature_names, index=f'Best_{i+1}_{idx}', 
                                    disable_scientific_notation=True, 
                                    input_data=input_data[idx] if input_data is not None else None,
                                    plot_type='trajectory')
        trajectory_pdf.savefig(trajectory_fig)
        plt.close(trajectory_fig)

        error_fig = visual_multi(inverse_trues[idx], inverse_preds[idx], num_features,
                                 feature_names=feature_names, index=f'Best_{i+1}_{idx}',
                                 disable_scientific_notation=True,
                                 plot_type='error')
        error_pdf.savefig(error_fig)
        plt.close(error_fig)
    # 绘制平均轨迹
    for i, idx in enumerate(avg_indices):
        trajectory_fig = visual_multi(inverse_trues[idx], inverse_preds[idx], num_features, 
                                    feature_names=feature_names, index=f'Mean_{i+1}_{idx}', 
                                    disable_scientific_notation=True, 
                                    input_data=input_data[idx] if input_data is not None else None,
                                    plot_type='trajectory')
        trajectory_pdf.savefig(trajectory_fig)
        plt.close(trajectory_fig)

        error_fig = visual_multi(inverse_trues[idx], inverse_preds[idx], num_features,
                                 feature_names=feature_names, index=f'Mean_{i + 1}_{idx}',
                                 disable_scientific_notation=True,
                                 plot_type='error')
        error_pdf.savefig(error_fig)
        plt.close(error_fig)
    # 绘制最差轨迹
    for i, idx in enumerate(worst_indices):
        trajectory_fig = visual_multi(inverse_trues[idx], inverse_preds[idx], num_features, 
                                    feature_names=feature_names, index=f'Worst_{i+1}_{idx}', 
                                    disable_scientific_notation=True, 
                                    input_data=input_data[idx] if input_data is not None else None,
                                    plot_type='trajectory')
        trajectory_pdf.savefig(trajectory_fig)
        plt.close(trajectory_fig)

        error_fig = visual_multi(inverse_trues[idx], inverse_preds[idx], num_features,
                                 feature_names=feature_names, index=f'Worst_{i + 1}_{idx}',
                                 disable_scientific_notation=True,
                                 plot_type='error')
        error_pdf.savefig(error_fig)
        plt.close(error_fig)
    # 计算并绘制误差分布图
    errors_3d = np.abs(inverse_preds - inverse_trues).reshape(-1, num_features)
    plot_error_distribution_3d(errors_3d, os.path.join(image_folder, 'error_distribution.png'))

    trajectory_pdf.close()
    error_pdf.close()

def plot_error_distribution_3d(errors_3d, save_path=None):
    """
    绘制三维误差分布图
    :param errors_3d: 三维误差数据 [n_samples, 3]
    :param save_path: 保存路径
    """
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 维度名称 - 修改为高经纬坐标系（按照原始数据顺序）
    dimensions = ['Altitude', 'Longitude', 'Latitude']
    
    # 绘制每个维度的误差分布
    for i, ax in enumerate(axes):
        # 计算误差统计信息
        mean_error = np.mean(errors_3d[:, i])
        median_error = np.median(errors_3d[:, i])
        std_error = np.std(errors_3d[:, i])
        max_error = np.max(errors_3d[:, i])
        
        # 计算分位数
        percentile_95 = np.percentile(errors_3d[:, i], 95)
        percentile_90 = np.percentile(errors_3d[:, i], 90)
        
        # 绘制直方图
        counts, bins, _ = ax.hist(errors_3d[:, i], bins=50, alpha=0.7, color='blue')
        
        # 在右上角添加统计信息
        stats_text = (
            f'Total Points: {len(errors_3d[:, i])}\n'
            f'95% of the samples error < {percentile_95:.4f}\n'
            f'90% of the samples error < {percentile_90:.4f}\n'
            f'Mean: {mean_error:.4f}\n'
            f'Median: {median_error:.4f}\n'
            f'Max: {max_error:.4f}'
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 在柱状图上添加数值标签（竖着排列）
        for j in range(len(counts)):
            if counts[j] > 0:  # 只显示非零值
                # 计算柱子的中心位置
                bin_center = (bins[j] + bins[j+1]) / 2
                # 添加数值标签，竖着排列
                ax.text(bin_center, counts[j], f'{int(counts[j])}', 
                        ha='center', va='bottom', rotation=90)
        
        # 添加标题和坐标轴标签，包含单位信息
        if dimensions[i] == 'Altitude':
            ax.set_title(f'{dimensions[i]} Error Distribution (m)')
            ax.set_xlabel('Error (m)')
        elif dimensions[i] == 'Longitude' or dimensions[i] == 'Latitude':
            ax.set_title(f'{dimensions[i]} Error Distribution (°)')
            ax.set_xlabel('Error (°)')
        else:
            ax.set_title(f'{dimensions[i]} Error Distribution')
            ax.set_xlabel('Error')
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize test results.')
    parser.add_argument('--results_folder', type=str, required=True, help='Path to the test results folder.')
    parser.add_argument('--exclude_input', action='store_false', default=True, help='Do not include input data for the ground truth plots (default: include).')
    parser.add_argument('--sample_strategy', type=str, default='interval', 
                        choices=['random', 'indices', 'interval', 'fixed_flight'], 
                        help='Sampling strategy: random, indices, interval, or fixed_flight.')
    parser.add_argument('--num_samples_to_plot', type=int, default=10, help='Number of samples to plot (for random or interval sampling).')
    parser.add_argument('--sample_indices', type=str, help='Comma-separated list of indices to plot (for indices sampling).')
    parser.add_argument('--feature_names', type=str, help='Comma-separated list of feature names.')
    args = parser.parse_args()

    feature_names_list = None
    if args.feature_names:
        feature_names_list = [name.strip() for name in args.feature_names.split(',')]

    visualize_test_results(args.results_folder, args.exclude_input, args.sample_strategy, 
                         args.num_samples_to_plot, args.sample_indices, feature_names_list)


# 使用例子：
# python visualize.py --sample_strategy interval --num_samples_to_plot 3 --exclude_input --feature_names longitude,latitude,Altitude,TAS --results_folder /workspace/PatchTST/PatchTST_supervised/test_results/2025_01_21_15_25_PatchTST2_climb_01_21_1934pm_PatchTST_processed_aircraft_ftM_sl100_ll48_pl20_dm256_nh32_el6_dl1_df512_fc1_ebtimeF_dtTrue_Exp_scaleTrue_0


# 1. 使用自定义特征名称进行等间隔抽样
# python visualize.py --sample_strategy interval --num_samples_to_plot 3 --exclude_input --results_folder /path/to/results --feature_names 经度,纬度,海拔,真实空速

# 2. 使用部分自定义特征名称（多余的特征将使用默认名称）
# python visualize.py --sample_strategy interval --num_samples_to_plot 3 --exclude_input --results_folder /path/to/results --feature_names Temp,Pressure

# 3. 不指定特征名称，使用默认名称
# python visualize.py --sample_strategy interval --num_samples_to_plot 3 --exclude_input --results_folder /path/to/results


# 使用例子与之前相同，例如：
# python visualize.py --sample_strategy interval --num_samples_to_plot 3 --exclude_input --results_folder /workspace/PatchTST/PatchTST_supervised/test_results/your_results_folder

# 1. 等间隔抽样 3 条轨迹
# python visualize.py --exclude_input --results_folder /path/to/results --sample_strategy interval --num_samples_to_plot 3

# 2. 随机抽样 15 条轨迹
# python visualize.py --exclude_input --results_folder /path/to/results --sample_strategy random --num_samples_to_plot 15

# 3. 指定绘制索引为 1, 10, 50, 200 的轨迹
# python visualize.py --exclude_input --results_folder /path/to/results --sample_strategy indices --sample_indices 1,10,50,200

# 4. 默认随机抽样 10 条轨迹
# python visualize.py --exclude_input --results_folder /path/to/results
#
#
#
