import matplotlib as mpl

mpl.use("Agg")  # noqa

# 三号字约为 16 磅
FONT_SIZE = 16
mpl.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
    'legend.fontsize': FONT_SIZE,
})

import argparse

import matplotlib.pyplot as plt
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics
from ann_benchmarks.plotting.utils import (compute_metrics, create_linestyles,
                                           create_pointset, get_plot_label)
from ann_benchmarks.results import get_unique_algorithms, load_all_results


def parse_algorithm_specs(specs):
    """Parse algorithm specs like 'algo' or 'algo:name' into (algo_names, legend_names).

    Args:
        specs: List of algorithm specification strings

    Returns:
        tuple: (set of algorithm names, dict mapping algo to custom legend name)
    """
    algo_names = set()
    legend_names = {}
    for spec in (specs or []):
        if ':' in spec:
            algo, name = spec.split(':', 1)
            algo_names.add(algo)
            legend_names[algo] = name
        else:
            algo_names.add(spec)
    return algo_names, legend_names


def create_plot(all_data, raw, x_scale, y_scale, xn, yn, fn_out, linestyles, legend_names=None):
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        return -np.log(np.array(ys)).mean()

    # Find range for logit x-scale
    min_x, max_x = 1, 0
    for algo in sorted(all_data.keys(), key=mean_y):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        min_x = min([min_x] + [x for x in xs if x > 0])
        max_x = max([max_x] + [x for x in xs if x < 1])
        color, faded, linestyle, marker = linestyles[algo]
        label = legend_names.get(algo, algo) if legend_names else algo
        (handle,) = plt.plot(
            xs, ys, "-", label=label, color=color, ms=7, mew=3, lw=3, marker=marker
        )
        handles.append(handle)
        if raw:
            (handle2,) = plt.plot(
                axs, ays, "-", label=algo, color=faded, ms=5, mew=2, lw=2, marker=marker
            )
        labels.append(label)

    ax = plt.gca()
    ax.set_ylabel(ym["description"])
    ax.set_xlabel(xm["description"])
    # Custom scales of the type --x-scale a3
    if x_scale[0] == "a":
        alpha = float(x_scale[1:])

        def fun(x):
            return 1 - (1 - x) ** (1 / alpha)

        def inv_fun(x):
            return 1 - (1 - x) ** alpha

        ax.set_xscale("function", functions=(fun, inv_fun))
        if alpha <= 3:
            ticks = [inv_fun(x) for x in np.arange(0, 1.2, 0.2)]
            plt.xticks(ticks)
        if alpha > 3:
            from matplotlib import ticker

            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            # plt.xticks(ticker.LogitLocator().tick_values(min_x, max_x))
            plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
    # Other x-scales
    else:
        ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(get_plot_label(xm, ym))
    plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc="best")
    plt.grid(visible=True, which="major", color="0.65", linestyle="-")
    plt.setp(ax.get_xminorticklabels(), visible=True)

    # Logit scale has to be a subset of (0,1)
    if "lim" in xm and x_scale != "logit":
        x0, x1 = xm["lim"]
        plt.xlim(max(x0, 0), min(x1, 1))
    elif x_scale == "logit":
        plt.xlim(min_x, max_x)
    if "lim" in ym:
        plt.ylim(ym["lim"])

    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines["bottom"]._adjust_location()

    plt.savefig(fn_out, bbox_inches="tight", dpi=144)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", metavar="DATASET", default="glove-100-angular")
    parser.add_argument("--count", default=10)
    parser.add_argument(
        "--definitions", metavar="FILE", help="load algorithm definitions from FILE", default="algos.yaml"
    )
    parser.add_argument("--limit", default=-1)
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "-x", "--x-axis", help="Which metric to use on the X-axis", choices=metrics.keys(), default="k-nn"
    )
    parser.add_argument(
        "-y", "--y-axis", help="Which metric to use on the Y-axis", choices=metrics.keys(), default="qps"
    )
    parser.add_argument(
        "-X", "--x-scale", help="Scale to use when drawing the X-axis. Typically linear, logit or a2", default="linear",
    )
    parser.add_argument(
        "-Y",
        "--y-scale",
        help="Scale to use when drawing the Y-axis",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
    )
    parser.add_argument(
        "--raw", help="Show raw results (not just Pareto frontier) in faded colours", action="store_true"
    )
    parser.add_argument("--recompute", help="Clears the cache and recomputes the metrics", action="store_true")
    parser.add_argument(
        "--algorithm", nargs="+",
        help="Filter to specific algorithms with optional legend names. "
             "Format: algo[:name] (e.g., csd-normal-2t or csd-normal-2t:\"CSD Normal\")"
    )
    args = parser.parse_args()

    if not args.output:
        args.output = "results/%s.png" % args.dataset
        print("writing output to %s" % args.output)

    dataset, _ = get_dataset(args.dataset)
    count = int(args.count)

    # Parse algorithm filter and custom legend names
    algo_filter, legend_names = parse_algorithm_specs(args.algorithm)

    # Load all results, both batch and non-batch, with optional filter
    results = load_all_results(args.dataset, count, batch_mode=None, algo_filter=algo_filter)

    # Get unique algorithms for linestyles (use filtered if specified, otherwise all)
    if algo_filter:
        unique_algorithms = algo_filter
    else:
        unique_algorithms = get_unique_algorithms()

    linestyles = create_linestyles(sorted(unique_algorithms))
    runs = compute_metrics(np.array(dataset["distances"]), results, args.x_axis, args.y_axis, args.recompute)
    if not runs:
        raise Exception("Nothing to plot")

    create_plot(
        runs, args.raw, args.x_scale, args.y_scale, args.x_axis, args.y_axis, args.output, linestyles, legend_names
    )
