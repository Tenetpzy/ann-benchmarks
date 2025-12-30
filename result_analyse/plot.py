from data import *
from simulator import simulate_performance
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


def get_simplified_bottleneck(result: dict) -> str:
    """根据bottleneck概率获取简化的瓶颈名称"""
    bottleneck = result.get('bottleneck', {})
    disk_prob = bottleneck.get('Disk_Bound_Prob', 0)
    cpu_prob = bottleneck.get('CPU_Bound_Prob', 0)
    buffer_prob = bottleneck.get('Buffer_Bound_Prob', 0)
    ssd_computing_prob = bottleneck.get('SSD_Computing_Prob', 0)

    # 获取所有瓶颈及其概率
    candidates = [
        ('SSD_IO', disk_prob),
        ('CPU', cpu_prob),
        ('Buffer', buffer_prob),
        ('SSD_Computing', ssd_computing_prob)
    ]

    # 返回概率最大的瓶颈
    return max(candidates, key=lambda x: x[1])[0]

def plot_results_table(title: str, results: list, save_path: str):
    """
    生成标准学术三线表（修复版）
    原理：强制表格填满画布，根据行数计算精确的分割线坐标。
    """
    # --------------------------
    # 1. 准备数据
    # --------------------------
    columns = ['Recall', 'QPS', 'Bottleneck']
    table_data = []
    
    for r in results:
        recall = r.get('input_summary_recall_target', 0)
        qps = r.get('qps_mean', 0)
        # 兼容 bottleneck 获取逻辑
        bottleneck = r.get('bottleneck', 'N/A')
        if 'get_simplified_bottleneck' in globals():
            try:
                bottleneck = get_simplified_bottleneck(r)
            except:
                pass
        
        table_data.append([recall, f"{qps:,}", bottleneck])

    # --------------------------
    # 2. 动态计算画布高度
    # --------------------------
    # 设定每行的高度（英寸），确保不管数据多少，行间距都合适
    row_height_inch = 0.4  
    n_data_rows = len(table_data)
    total_rows = n_data_rows + 1 # 加上表头
    
    # 计算画布总高度
    fig_height = (total_rows * row_height_inch) 
    fig, ax = plt.subplots(figsize=(6, fig_height)) # 宽度固定为6英寸
    
    # 移除坐标轴
    ax.axis('off')
    
    # --------------------------
    # 3. 创建表格（强制填满）
    # --------------------------
    # bbox=[0, 0, 1, 1] 强制表格占满整个 Axes，坐标系归一化 (0~1)
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        bbox=Bbox.from_bounds(0, 0, 1, 1)
    )

    # --------------------------
    # 4. 样式美化
    # --------------------------
    # 设置字体（学术常用衬线体）
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    
    cells = table.get_celld()
    for (row, col), cell in cells.items():
        cell.set_text_props(fontfamily='serif') # 设置衬线字体
        cell.set_linewidth(0) # 去掉所有默认边框（关键！）
        
        # 表头加粗
        if row == 0:
            cell.set_text_props(weight='bold')

    # --------------------------
    # 5. 画“三线” (坐标绝对精确)
    # --------------------------
    # 因为表格填满了画布 [0,1]，我们可以直接通过比例计算线条位置
    
    # 计算表头底部位置：总高度的 (1 - 1/总行数) 处
    header_bottom_y = 1.0 - (1.0 / total_rows)
    
    # 线条参数
    line_color = 'black'
    thick_width = 1.5  # 顶线/底线粗细
    thin_width = 0.75  # 栏目线粗细
    
    # 绘制顶线 (y=1)
    ax.plot([0, 1], [1, 1], color=line_color, lw=thick_width, transform=ax.transAxes)
    
    # 绘制栏目线 (表头下方)
    ax.plot([0, 1], [header_bottom_y, header_bottom_y], color=line_color, lw=thin_width, transform=ax.transAxes)
    
    # 绘制底线 (y=0)
    ax.plot([0, 1], [0, 0], color=line_color, lw=thick_width, transform=ax.transAxes)

    # --------------------------
    # 6. 添加表题
    # --------------------------
    # 放在顶线 (y=1) 上方适当距离
    ax.text(0.5, 1.02, title, 
            transform=ax.transAxes, 
            ha='center', va='bottom', 
            fontsize=12, weight='bold', fontfamily='serif')

    # 保存
    # pad_inches=0.2 留出白边，防止标题或线条太贴边被切掉
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close(fig)


def plot_result_of_baseline_and_reorder():
    # 获取两个算法的9组模拟结果
    baseline_results = []
    reorder_results = []

    for recall in recall_target:
        idx = recall_target.index(recall)
        baseline_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=hnsw_baseline_cpu_cores,
            beam_width=hnsw_baseline_beam_width,
            cache_hit_rate=hnsw_baseline_cache_hit_rate[idx],
            C=hnsw_baseline_C,
            B=hnsw_baseline_B,
            T_cpu_base_us=hnsw_baseline_cpu_base_us,
            T_cpu_per_node_us=hnsw_baseline_cpu_per_node_us
        ))
        reorder_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=csdann_reorder_cpu_cores,
            beam_width=csdann_reorder_beam_width,
            cache_hit_rate=csdann_reorder_cache_hit_rate[idx],
            C=csdann_reorder_C,
            B=csdann_reorder_B,
            T_cpu_base_us=csdann_reorder_cpu_base_us,
            T_cpu_per_node_us=csdann_reorder_cpu_per_node_us
        ))

    # 提取数据
    cache_hit_rate_baseline = hnsw_baseline_cache_hit_rate
    cache_hit_rate_reorder = csdann_reorder_cache_hit_rate
    latency_baseline = [r['latency_ms_mean'] for r in baseline_results]
    latency_reorder = [r['latency_ms_mean'] for r in reorder_results]
    qps_baseline = [r['qps_mean'] for r in baseline_results]
    qps_reorder = [r['qps_mean'] for r in reorder_results]

    # 设置学术论文风格
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (6, 4.5),
        'font.family': 'serif',
    })

    x = np.arange(len(recall_target))
    width = 0.35
    x_left = x - width / 2  # 使用变量避免 IDE 类型警告
    x_right = x + width / 2

    # 图1: cache_hit_rate柱状图
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.bar(x_left, [c * 100 for c in cache_hit_rate_baseline], width,
            label='hnsw_baseline', color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax1.bar(x_right, [c * 100 for c in cache_hit_rate_reorder], width,
            label='csdann_reorder', color='#E94F37', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Cache Hit Rate (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(r) for r in recall_target], rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylim(0, 60)
    plt.tight_layout()
    fig1.savefig('baseline_vs_reorder/cache_hit_rate.png', dpi=300)
    plt.close(fig1)

    # 计算 error_rate 作为横坐标（更均匀的分布）
    error_rates = [1 - r for r in recall_target]

    # 图2: latency折线图（使用log scale）
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.plot(error_rates, latency_baseline, 'o-', label='hnsw_baseline',
             color='#2E86AB', linewidth=2, markersize=6)
    ax2.plot(error_rates, latency_reorder, 's-', label='csdann_reorder',
             color='#E94F37', linewidth=2, markersize=6)
    ax2.set_xscale('log')
    ax2.set_xlabel('Error Rate (1 - Recall)')
    ax2.set_ylabel('Latency (ms)')
    ax2.invert_xaxis()  # 反转使高recall在右
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    fig2.savefig('baseline_vs_reorder/latency.png', dpi=300)
    plt.close(fig2)

    # 图3: qps折线图（使用log scale）
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    ax3.plot(error_rates, qps_baseline, 'o-', label='hnsw_baseline',
             color='#2E86AB', linewidth=2, markersize=6)
    ax3.plot(error_rates, qps_reorder, 's-', label='csdann_reorder',
             color='#E94F37', linewidth=2, markersize=6)
    ax3.set_xscale('log')
    ax3.set_xlabel('Error Rate (1 - Recall)')
    ax3.set_ylabel('QPS')
    ax3.invert_xaxis()  # 反转使高recall在右
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    fig3.savefig('baseline_vs_reorder/qps.png', dpi=300)
    plt.close(fig3)

    print("图表已保存到 baseline_vs_reorder/ 目录:")
    print("  - cache_hit_rate.png")
    print("  - latency.png")
    print("  - qps.png")

    # 生成结果表格
    plot_results_table(
        title='HNSW Baseline Results',
        results=baseline_results,
        save_path='baseline_vs_reorder/baseline_table.png'
    )
    plot_results_table(
        title='CSDANN Reorder Results',
        results=reorder_results,
        save_path='baseline_vs_reorder/reorder_table.png'
    )
    print("  - baseline_table.png")
    print("  - reorder_table.png")

def plot_result_of_hnsw_baseline_and_csdann_opt_storage():
    # 获取两个算法的9组模拟结果
    baseline_results = []
    opt_storage_results = []

    for recall in recall_target:
        idx = recall_target.index(recall)
        baseline_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=hnsw_baseline_cpu_cores,
            beam_width=hnsw_baseline_beam_width,
            cache_hit_rate=hnsw_baseline_cache_hit_rate[idx],
            C=hnsw_baseline_C,
            B=hnsw_baseline_B,
            T_cpu_base_us=hnsw_baseline_cpu_base_us,
            T_cpu_per_node_us=hnsw_baseline_cpu_per_node_us
        ))
        opt_storage_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=csdann_opt_storage_cpu_cores,
            beam_width=csdann_opt_storage_beam_width,
            cache_hit_rate=csdann_opt_storage_cache_hit_rate[idx],
            C=csdann_opt_storage_C,
            B=csdann_opt_storage_B,
            T_cpu_base_us=csdann_opt_storage_cpu_base_us,
            T_cpu_per_node_us=csdann_opt_storage_cpu_per_node_us
        ))

    # 提取数据
    io_baseline = [r['input_summary_avg_physical_io'] for r in baseline_results]
    io_opt_storage = [r['input_summary_avg_physical_io'] for r in opt_storage_results]
    latency_baseline = [r['latency_ms_mean'] for r in baseline_results]
    latency_opt_storage = [r['latency_ms_mean'] for r in opt_storage_results]
    qps_baseline = [r['qps_mean'] for r in baseline_results]
    qps_opt_storage = [r['qps_mean'] for r in opt_storage_results]

    # 设置学术论文风格
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (6, 4.5),
        'font.family': 'serif',
    })

    x = np.arange(len(recall_target) - 1)
    width = 0.35
    x_left = x - width / 2
    x_right = x + width / 2

    # 图1: I/O Operation Num柱状图（去掉最后一个数据点）
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.bar(x_left, io_baseline[:-1], width,
            label='hnsw_baseline', color='#2E86AB', edgecolor='black', linewidth=0.5)
    ax1.bar(x_right, io_opt_storage[:-1], width,
            label='csdann_opt_storage', color='#E94F37', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('I/O Operation Num')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(r) for r in recall_target[:-1]], rotation=45, ha='right')
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig1.savefig('baseline_vs_opt_storage/io_operations.png', dpi=300)
    plt.close(fig1)

    # 计算 error_rate 作为横坐标（更均匀的分布）
    error_rates = [1 - r for r in recall_target]

    # 图2: latency折线图（使用log scale）
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.plot(error_rates, latency_baseline, 'o-', label='hnsw_baseline',
             color='#2E86AB', linewidth=2, markersize=6)
    ax2.plot(error_rates, latency_opt_storage, 's-', label='csdann_opt_storage',
             color='#E94F37', linewidth=2, markersize=6)
    ax2.set_xscale('log')
    ax2.set_xlabel('Error Rate (1 - Recall)')
    ax2.set_ylabel('Latency (ms)')
    ax2.invert_xaxis()  # 反转使高recall在右
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    fig2.savefig('baseline_vs_opt_storage/latency.png', dpi=300)
    plt.close(fig2)

    # 图3: qps折线图（使用log scale）
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    ax3.plot(error_rates, qps_baseline, 'o-', label='hnsw_baseline',
             color='#2E86AB', linewidth=2, markersize=6)
    ax3.plot(error_rates, qps_opt_storage, 's-', label='csdann_opt_storage',
             color='#E94F37', linewidth=2, markersize=6)
    ax3.set_xscale('log')
    ax3.set_xlabel('Error Rate (1 - Recall)')
    ax3.set_ylabel('QPS')
    ax3.invert_xaxis()  # 反转使高recall在右
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    fig3.savefig('baseline_vs_opt_storage/qps.png', dpi=300)
    plt.close(fig3)

    print("图表已保存到 baseline_vs_opt_storage/ 目录:")
    print("  - io_operations.png")
    print("  - latency.png")
    print("  - qps.png")

    # 生成结果表格
    plot_results_table(
        title='HNSW Baseline Results',
        results=baseline_results,
        save_path='baseline_vs_opt_storage/baseline_table.png'
    )
    plot_results_table(
        title='CSDANN Opt Storage Results',
        results=opt_storage_results,
        save_path='baseline_vs_opt_storage/opt_storage_table.png'
    )
    print("  - baseline_table.png")
    print("  - opt_storage_table.png")

def plot_result_of_baseline_and_csdann_opt_storage_and_csdann_offload():
    # 获取三个算法的9组模拟结果
    baseline_results = []
    opt_storage_results = []
    offload_results = []

    for recall in recall_target:
        idx = recall_target.index(recall)
        baseline_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=hnsw_baseline_cpu_cores,
            beam_width=hnsw_baseline_beam_width,
            cache_hit_rate=hnsw_baseline_cache_hit_rate[idx],
            C=hnsw_baseline_C,
            B=hnsw_baseline_B,
            T_cpu_base_us=hnsw_baseline_cpu_base_us,
            T_cpu_per_node_us=hnsw_baseline_cpu_per_node_us
        ))
        opt_storage_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=csdann_opt_storage_cpu_cores,
            beam_width=csdann_opt_storage_beam_width,
            cache_hit_rate=csdann_opt_storage_cache_hit_rate[idx],
            C=csdann_opt_storage_C,
            B=csdann_opt_storage_B,
            T_cpu_base_us=csdann_opt_storage_cpu_base_us,
            T_cpu_per_node_us=csdann_opt_storage_cpu_per_node_us
        ))
        offload_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=csdann_offload_cpu_cores,
            beam_width=csdann_offload_beam_width,
            cache_hit_rate=csdann_offload_cache_hit_rate[idx],
            C=csdann_offload_storage_C,
            B=csdann_offload_storage_B,
            T_cpu_base_us=csdann_offload_storage_cpu_base_us,
            T_cpu_per_node_us=csdann_offload_storage_cpu_per_node_us,
            ssd_cpu_cores=csdann_offload_ssd_cpu_cores,
            csd_schedular_on=csdann_offload_csd_schedular_on
        ))

    # 提取数据
    latency_baseline = [r['latency_ms_mean'] for r in baseline_results]
    latency_opt_storage = [r['latency_ms_mean'] for r in opt_storage_results]
    latency_offload = [r['latency_ms_mean'] for r in offload_results]
    qps_baseline = [r['qps_mean'] for r in baseline_results]
    qps_opt_storage = [r['qps_mean'] for r in opt_storage_results]
    qps_offload = [r['qps_mean'] for r in offload_results]

    # 设置学术论文风格
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (6, 4.5),
        'font.family': 'serif',
    })

    # 计算 error_rate 作为横坐标（更均匀的分布）
    error_rates = [1 - r for r in recall_target]

    # 图1: latency折线图（使用log scale）
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.plot(error_rates, latency_baseline, 'o-', label='hnsw_baseline',
             color='#2E86AB', linewidth=2, markersize=6)
    ax1.plot(error_rates, latency_opt_storage, 's-', label='csdann_opt_storage',
             color='#E94F37', linewidth=2, markersize=6)
    ax1.plot(error_rates, latency_offload, '^-', label='csdann_offload',
             color='#44AF69', linewidth=2, markersize=6)
    ax1.set_xscale('log')
    ax1.set_xlabel('Error Rate (1 - Recall)')
    ax1.set_ylabel('Latency (ms)')
    ax1.invert_xaxis()  # 反转使高recall在右
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    fig1.savefig('baseline_vs_opt_store_vs_offload/latency.png', dpi=300)
    plt.close(fig1)

    # 图2: qps折线图（使用log scale）
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.plot(error_rates, qps_baseline, 'o-', label='hnsw_baseline',
             color='#2E86AB', linewidth=2, markersize=6)
    ax2.plot(error_rates, qps_opt_storage, 's-', label='csdann_opt_storage',
             color='#E94F37', linewidth=2, markersize=6)
    ax2.plot(error_rates, qps_offload, '^-', label='csdann_offload',
             color='#44AF69', linewidth=2, markersize=6)
    ax2.set_xscale('log')
    ax2.set_xlabel('Error Rate (1 - Recall)')
    ax2.set_ylabel('QPS')
    ax2.invert_xaxis()  # 反转使高recall在右
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    fig2.savefig('baseline_vs_opt_store_vs_offload/qps.png', dpi=300)
    plt.close(fig2)

    print("图表已保存到 baseline_vs_opt_store_vs_offload/ 目录:")
    print("  - latency.png")
    print("  - qps.png")

    # 生成结果表格
    plot_results_table(
        title='HNSW Baseline Results',
        results=baseline_results,
        save_path='baseline_vs_opt_store_vs_offload/baseline_table.png'
    )
    plot_results_table(
        title='CSDANN Opt Storage Results',
        results=opt_storage_results,
        save_path='baseline_vs_opt_store_vs_offload/opt_storage_table.png'
    )
    plot_results_table(
        title='CSDANN Offload Results',
        results=offload_results,
        save_path='baseline_vs_opt_store_vs_offload/offload_table.png'
    )
    print("  - baseline_table.png")
    print("  - opt_storage_table.png")
    print("  - offload_table.png")

def plot_result_of_baseline_and_csdann_opt_storage_and_csdann_sched_offload():
    # 获取三个算法的9组模拟结果
    baseline_results = []
    opt_storage_results = []
    sched_offload_results = []

    for recall in recall_target:
        idx = recall_target.index(recall)
        baseline_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=hnsw_baseline_cpu_cores,
            beam_width=hnsw_baseline_beam_width,
            cache_hit_rate=hnsw_baseline_cache_hit_rate[idx],
            C=hnsw_baseline_C,
            B=hnsw_baseline_B,
            T_cpu_base_us=hnsw_baseline_cpu_base_us,
            T_cpu_per_node_us=hnsw_baseline_cpu_per_node_us
        ))
        opt_storage_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=csdann_opt_storage_cpu_cores,
            beam_width=csdann_opt_storage_beam_width,
            cache_hit_rate=csdann_opt_storage_cache_hit_rate[idx],
            C=csdann_opt_storage_C,
            B=csdann_opt_storage_B,
            T_cpu_base_us=csdann_opt_storage_cpu_base_us,
            T_cpu_per_node_us=csdann_opt_storage_cpu_per_node_us
        ))
        sched_offload_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=csdann_sched_offload_cpu_cores,
            beam_width=csdann_sched_offload_beam_width,
            cache_hit_rate=csdann_sched_offload_cache_hit_rate[idx],
            C=csdann_sched_offload_C,
            B=csdann_sched_offload_B,
            T_cpu_base_us=csdann_sched_offload_cpu_base_us,
            T_cpu_per_node_us=csdann_sched_offload_cpu_per_node_us,
            ssd_cpu_cores=csdann_sched_offload_ssd_cpu_cores,
            csd_schedular_on=csdann_sched_offload_csd_schedular_on
        ))

    # 提取数据
    latency_baseline = [r['latency_ms_mean'] for r in baseline_results]
    latency_opt_storage = [r['latency_ms_mean'] for r in opt_storage_results]
    latency_sched_offload = [r['latency_ms_mean'] for r in sched_offload_results]
    qps_baseline = [r['qps_mean'] for r in baseline_results]
    qps_opt_storage = [r['qps_mean'] for r in opt_storage_results]
    qps_sched_offload = [r['qps_mean'] for r in sched_offload_results]

    # 设置学术论文风格
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (6, 4.5),
        'font.family': 'serif',
    })

    # 计算 error_rate 作为横坐标（更均匀的分布）
    error_rates = [1 - r for r in recall_target]

    # 图1: latency折线图（使用log scale）
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.plot(error_rates, latency_baseline, 'o-', label='hnsw_baseline',
             color='#2E86AB', linewidth=2, markersize=6)
    ax1.plot(error_rates, latency_opt_storage, 's-', label='csdann_opt_storage',
             color='#E94F37', linewidth=2, markersize=6)
    ax1.plot(error_rates, latency_sched_offload, '^-', label='csdann_sched_offload',
             color='#44AF69', linewidth=2, markersize=6)
    ax1.set_xscale('log')
    ax1.set_xlabel('Error Rate (1 - Recall)')
    ax1.set_ylabel('Latency (ms)')
    ax1.invert_xaxis()  # 反转使高recall在右
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    fig1.savefig('baseline_vs_opt_store_vs_sched_offload/latency.png', dpi=300)
    plt.close(fig1)

    # 图2: qps折线图（使用log scale）
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    ax2.plot(error_rates, qps_baseline, 'o-', label='hnsw_baseline',
             color='#2E86AB', linewidth=2, markersize=6)
    ax2.plot(error_rates, qps_opt_storage, 's-', label='csdann_opt_storage',
             color='#E94F37', linewidth=2, markersize=6)
    ax2.plot(error_rates, qps_sched_offload, '^-', label='csdann_sched_offload',
             color='#44AF69', linewidth=2, markersize=6)
    ax2.set_xscale('log')
    ax2.set_xlabel('Error Rate (1 - Recall)')
    ax2.set_ylabel('QPS')
    ax2.invert_xaxis()  # 反转使高recall在右
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    fig2.savefig('baseline_vs_opt_store_vs_sched_offload/qps.png', dpi=300)
    plt.close(fig2)

    print("图表已保存到 baseline_vs_opt_store_vs_sched_offload/ 目录:")
    print("  - latency.png")
    print("  - qps.png")

    # 生成结果表格
    plot_results_table(
        title='HNSW Baseline Results',
        results=baseline_results,
        save_path='baseline_vs_opt_store_vs_sched_offload/baseline_table.png'
    )
    plot_results_table(
        title='CSDANN Opt Storage Results',
        results=opt_storage_results,
        save_path='baseline_vs_opt_store_vs_sched_offload/opt_storage_table.png'
    )
    plot_results_table(
        title='CSDANN Sched Offload Results',
        results=sched_offload_results,
        save_path='baseline_vs_opt_store_vs_sched_offload/sched_offload_table.png'
    )
    print("  - baseline_table.png")
    print("  - opt_storage_table.png")
    print("  - sched_offload_table.png")

# plot_result_of_baseline_and_reorder()
# plot_result_of_hnsw_baseline_and_csdann_opt_storage()
# plot_result_of_baseline_and_csdann_opt_storage_and_csdann_offload()
plot_result_of_baseline_and_csdann_opt_storage_and_csdann_sched_offload()