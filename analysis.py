from __future__ import absolute_import

import os
import h5py
import numpy as np
from typing import List, Tuple, Literal, Optional, Dict, Any

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

from ann_benchmarks.datasets import get_dataset


# ============================================================
# 结果加载函数
# ============================================================

def load_result_set(result_dir: str) -> Tuple[str, List[dict]]:
    """
    加载指定结果集目录，获取所有指标数据。

    Args:
        result_dir: 结果集目录路径，例如 "results/sift-128-euclidean/10/hnswlib-batch"

    Returns:
        Tuple[str, List[dict]]: (结果集标识, 各组数据的指标字典列表)
            - 结果集标识为目录名
            - 每个字典包含所有计算得到的指标
    """
    result_id = os.path.basename(result_dir)

    # 从路径中解析 dataset name 和 count
    # 路径格式: results/{dataset}/{count}/{algorithm}
    parts = result_dir.split(os.sep)
    dataset_name = parts[1]  # e.g., "sift-128-euclidean"
    count = int(parts[2])    # e.g., 10

    # 获取数据集的真实距离（ground truth）
    dataset, _ = get_dataset(dataset_name)
    true_nn_distances = np.array(dataset["distances"])

    results = []

    for filename in os.listdir(result_dir):
        if not filename.endswith(".hdf5"):
            continue

        filepath = os.path.join(result_dir, filename)
        with h5py.File(filepath, "r") as f:
            properties = dict(f.attrs)
            run_distances = np.array(f["distances"])

            # 计算 recall（严格按照 metrics.py 中的 knn 指标计算方式）
            recall = compute_knn_metric(true_nn_distances, run_distances, count)

            # 构建结果字典
            result_dict = {
                "algo": properties.get("algo", ""),
                "name": properties.get("name", ""),
                "count": count,
                # 基础指标
                "recall": recall["mean"],
                "recall_std": recall["std"],
                # 时间指标
                "best_search_time": properties.get("best_search_time", 0),
                "build_time": properties.get("build_time", 0),
                # 索引大小
                "index_size": properties.get("index_size", 0),
            }

            # 添加额外指标（从属性中获取，存储在 f.attrs 中）
            for key, value in properties.items():
                if key not in result_dict:
                    result_dict[key] = value

            results.append(result_dict)

    return result_id, results


def compute_knn_metric(dataset_distances: np.ndarray, run_distances: np.ndarray, count: int) -> dict:
    """
    计算 recall 指标，严格按照 ann_benchmarks/plotting/metrics.py 中的 knn 函数逻辑。

    Args:
        dataset_distances: 真实的最近邻距离
        run_distances: 算法返回的距离
        count: 近邻数量

    Returns:
        dict: 包含 mean, std 的字典
    """
    # 计算 recall（与 get_recall_values 相同逻辑）
    recalls = np.zeros(len(run_distances))
    epsilon = 1e-3

    for i in range(len(run_distances)):
        # 阈值：第 count-1 个真实距离 + epsilon
        threshold = dataset_distances[i][count - 1] + epsilon
        actual = 0
        for d in run_distances[i][:count]:
            if d <= threshold:
                actual += 1
        recalls[i] = actual

    mean_recall = np.mean(recalls) / float(count)
    std_recall = np.std(recalls) / float(count)

    return {"mean": float(mean_recall), "std": float(std_recall)}


# ============================================================
# 学术论文规范绘图函数
# ============================================================

def draw_plot(
    result_sets: List[Tuple[str, List[Dict[str, Any]]]],
    x_metric: str,
    y_metric: str,
    plot_style: Literal["line", "bar"] = "line",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    output_path: str = "plot.png",
    title: Optional[str] = None,
    log_scale_x: bool = False,
    log_scale_y: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    color_scheme: str = "academic",
    markers: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    error_metric: Optional[str] = None,
) -> None:
    """
    绘制符合学术论文规范的对比图。

    Args:
        result_sets: 结果集列表，每个元素为 (标识符, 指标字典列表)
        x_metric: x轴对应的指标名（如 "recall", "best_search_time", "index_size"）
        y_metric: y轴对应的指标名（如 "best_search_time", "recall", "index_size"）
        plot_style: 图形样式，"line" 为折线图，"bar" 为柱状图
        x_label: x轴标签，默认使用 x_metric
        y_label: y轴标签，默认使用 y_metric
        output_path: 输出图片路径
        title: 图表标题，默认不显示标题
        log_scale_x: 是否对x轴使用对数刻度
        log_scale_y: 是否对y轴使用对数刻度
        xlim: x轴范围限制 (min, max)
        ylim: y轴范围限制 (min, max)
        color_scheme: 配色方案，"academic" 为学术配色，"viridis" 为渐变色
        markers: 折线图标记样式列表
        linestyles: 折线图线型列表
        error_metric: 误差指标列名（如 "recall_std"），用于显示误差棒
    """
    # 学术论文配色方案（专业、易区分）
    academic_colors = [
        "#1f77b4",  # 蓝色
        "#ff7f0e",  # 橙色
        "#2ca02c",  # 绿色
        "#d62728",  # 红色
        "#9467bd",  # 紫色
        "#8c564b",  # 棕色
        "#e377c2",  # 粉色
        "#7f7f7f",  # 灰色
    ]

    viridis_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(result_sets)))

    colors = viridis_colors if color_scheme == "viridis" else academic_colors

    # 默认标记和线型
    default_markers = ["o", "s", "^", "D", "v", "p", "*", "h"]
    default_linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    if markers is None:
        markers = default_markers[:len(result_sets)]
    if linestyles is None:
        linestyles = default_linestyles[:len(result_sets)]

    # 设置学术论文标准字体和大小
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, (result_id, results) in enumerate(result_sets):
        # 提取数据并按 x_metric 排序
        valid_results = [r for r in results if x_metric in r and y_metric in r]
        valid_results.sort(key=lambda r: r[x_metric])

        if not valid_results:
            continue

        x_data = [r[x_metric] for r in valid_results]
        y_data = [r[y_metric] for r in valid_results]

        # 获取误差数据（用于误差棒）
        error_data = None
        if error_metric and plot_style == "line":
            error_data = [r.get(error_metric, 0) for r in valid_results]

        # 绘制
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]

        if plot_style == "line":
            ax.plot(
                x_data, y_data,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=6,
                linewidth=1.5,
                label=result_id,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
            # 添加误差棒（如果指定了误差指标）
            if error_data is not None and any(e > 0 for e in error_data):
                ax.errorbar(
                    x_data, y_data,
                    yerr=error_data,
                    color=color,
                    fmt="none",
                    capsize=3,
                    capthick=1,
                    elinewidth=1,
                )
        elif plot_style == "bar":
            bar_width = 0.8 / len(result_sets)
            offset = (idx - len(result_sets) / 2 + 0.5) * bar_width
            x_positions = [x + offset for x in range(len(x_data))]
            ax.bar(
                x_positions, y_data,
                width=bar_width * 0.8,
                color=color,
                label=result_id,
                edgecolor="white",
                linewidth=0.5,
            )
            # 更新x轴刻度标签
            ax.set_xticks(range(len(x_data)))
            ax.set_xticklabels(x_data, rotation=45, ha="right")

        else:
            raise ValueError(f"Unsupported plot_style: {plot_style}")

    # 设置轴标签
    ax.set_xlabel(x_label if x_label else x_metric)
    ax.set_ylabel(y_label if y_label else y_metric)

    # 添加标题
    if title:
        ax.set_title(title)

    # 设置对数刻度
    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")

    # 设置轴范围
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # 添加图例（自动选择最佳位置）
    ax.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        edgecolor="gray",
    )

    # 优化网格线（仅在需要时显示）
    if plot_style == "line":
        ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    # 添加边框（学术论文标准）
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # 保存图片
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, format=os.path.splitext(output_path)[1][1:] or "png")
    plt.close()


if __name__ == "__main__":
    # 示例用法
    from analysis import load_result_set, draw_plot

    # 加载多个结果集
    result_sets = [
        load_result_set("results/sift-128-euclidean/10/csdann"),
        load_result_set("results/sift-128-euclidean/10/baseline"),
    ]

    # 示例 1：折线图（Recall vs QPS）
    draw_plot(
        result_sets=result_sets,
        x_metric="recall",
        y_metric="io_operations",
        plot_style="line",
        x_label="Recall",
        y_label="IO Operations",
        output_path="output/recall_vs_io_operations.png",
        log_scale_y=True,
        title="Recall vs IO Operations Comparison",
    )


