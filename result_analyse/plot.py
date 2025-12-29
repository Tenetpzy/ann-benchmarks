from data import *
from simulator import simulate_performance
import matplotlib.pyplot as plt
import numpy as np


def plot_result_of_baseline_and_reorder():
    # 获取两个算法的9组模拟结果
    baseline_results = []
    reorder_results = []

    for recall in recall_target:
        idx = recall_target.index(recall)
        baseline_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=cpu_cores,
            beam_width=beam_width,
            cache_hit_rate=hnsw_baseline_cache_hit_rate[idx],
            C=hnsw_baseline_C,
            T_cpu_base_us=hnsw_baseline_cpu_base_us
        ))
        reorder_results.append(simulate_performance(
            recall_target=recall,
            available_cache_pages=available_cache_pages,
            cpu_cores=cpu_cores,
            beam_width=beam_width,
            cache_hit_rate=hnsw_reorder_cache_hit_rate[idx],
            C=hnsw_reorder_C,
            T_cpu_base_us=hnsw_reorder_cpu_base_us
        ))

    # 提取数据
    cache_hit_rate_baseline = hnsw_baseline_cache_hit_rate
    cache_hit_rate_reorder = hnsw_reorder_cache_hit_rate
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

    # 图1: cache_hit_rate柱状图
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    bars1 = ax1.bar(x - width/2, [c * 100 for c in cache_hit_rate_baseline], width,
                    label='hnsw_baseline', color='#2E86AB', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, [c * 100 for c in cache_hit_rate_reorder], width,
                    label='hnsw_reorder', color='#E94F37', edgecolor='black', linewidth=0.5)
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
    ax2.plot(error_rates, latency_reorder, 's-', label='hnsw_reorder',
             color='#E94F37', linewidth=2, markersize=6)
    ax2.set_xscale('log')
    ax2.set_xlabel('Error Rate (1 - Recall)')
    ax2.set_ylabel('Latency (ms)')
    ax2.invert_xaxis()  # 反转使高recall在右
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    fig2.savefig('baseline_vs_reorder/latency.png', dpi=300)
    plt.close(fig2)

    # 图3: qps折线图（使用log scale）
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    ax3.plot(error_rates, qps_baseline, 'o-', label='hnsw_baseline',
             color='#2E86AB', linewidth=2, markersize=6)
    ax3.plot(error_rates, qps_reorder, 's-', label='hnsw_reorder',
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

plot_result_of_baseline_and_reorder()