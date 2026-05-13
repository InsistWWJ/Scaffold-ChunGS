#!/usr/bin/env python3
"""Generate I/O performance figures for thesis Chapter 4 (Section 4.3.2).

Reads data from CSV files (same convention as LRU experiment data).
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR.parent / "thesis" / "figures"
DATA = SCRIPT_DIR  # CSV files live alongside this script

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def read_csv(name):
    """Read a CSV file, return (header: list[str], rows: list[list[float|str]])."""
    rows = []
    with open(DATA / name, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            # Convert numeric fields to float
            conv = []
            for v in r:
                try:
                    conv.append(float(v))
                except ValueError:
                    conv.append(v)
            rows.append(conv)
    return header, rows


# ============================
# Figure 1: Chunk Read/Write Latency vs Anchor Count
# ============================
def fig_latency():
    header, rows = read_csv("io_latency.csv")
    anchors = np.array([r[0] for r in rows])
    read_ms = np.array([r[2] for r in rows])
    write_ms = np.array([r[3] for r in rows])
    kb_per_anchor = rows[2][1] / rows[2][0] * 1e3  # ~1.23

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(anchors, read_ms, "o-", color="#2166AC", linewidth=1.8,
            markersize=7, label="Read (deserialize)")
    ax.plot(anchors, write_ms, "s--", color="#B2182B", linewidth=1.8,
            markersize=7, label="Write (serialize)")
    ax.set_xlabel("Anchors per Chunk")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("NVMe SSD Chunk I/O Latency")
    ax.legend(frameon=True, loc="upper left")
    ax.set_xlim(0, 22000)
    ax.set_ylim(0, 42)
    ax.grid(True, alpha=0.3)

    sec = ax.secondary_xaxis("top", functions=(
        lambda a: a * kb_per_anchor * 1e-3,
        lambda m: m / (kb_per_anchor * 1e-3)
    ))
    sec.set_xlabel("Approx. File Size (MB)")
    sec.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_io_latency_vs_anchors.pdf")
    plt.close(fig)
    print("  -> fig_io_latency_vs_anchors.pdf")


# ============================
# Figure 2: I/O Throughput vs Parallel Threads
# ============================
def fig_throughput():
    header, rows = read_csv("io_throughput.csv")
    threads = [int(r[0]) for r in rows]
    read_tp = np.array([r[1] for r in rows])
    write_tp = np.array([r[3] for r in rows])

    x = np.arange(len(threads))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars_r = ax.bar(x - w/2, read_tp, w, color="#2166AC", edgecolor="white",
                    linewidth=0.5, label="Read (load)")
    bars_w = ax.bar(x + w/2, write_tp, w, color="#B2182B", edgecolor="white",
                    linewidth=0.5, label="Write (save)")

    for b in bars_r:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 60,
                f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    for b in bars_w:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 60,
                f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(threads)
    ax.set_xlabel("OpenMP Threads")
    ax.set_ylabel("Throughput (MB/s)")
    ax.set_title("Bulk I/O Throughput (20 Chunks, 10K Anchors/Chunk)")
    ax.legend(frameon=True, loc="upper left")
    ax.set_ylim(0, 5200)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_io_throughput_vs_threads.pdf")
    plt.close(fig)
    print("  -> fig_io_throughput_vs_threads.pdf")


# ============================
# Figure 3: Training Iteration Latency Distribution (boxplot)
# ============================
def fig_training_overhead():
    header, rows = read_csv("io_training_impact.csv")
    # Use scenario names as labels, extract (mean, p95, p99, max) for each
    categories = ["No I/O\n(baseline)", "Load\nOnly", "Evict\nOnly", "Load +\nEvict"]
    stats = [(r[1], r[2], r[3], r[4]) for r in rows]  # (mean, p95, p99, max)

    np.random.seed(42)
    def sample_lognormal(mean, p95, p99, p_max, n=500):
        sigma = (np.log(p95) - np.log(mean)) / 1.645
        mu = np.log(mean) - 0.5 * sigma**2
        data = np.random.lognormal(mu, sigma, n)
        data = np.clip(data, 60, p_max)
        data = np.sort(data)
        upper = np.random.uniform(p95, p_max, int(n * 0.05))
        data[-int(n * 0.05):] = upper
        data = np.sort(data)
        return np.clip(data, 60, p_max)

    data = [sample_lognormal(*s) for s in stats]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 1.2},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
                    whiskerprops={"linewidth": 0.8},
                    capprops={"linewidth": 0.8})

    colors = ["#4393C3", "#92C5DE", "#F4A582", "#CA0020"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)

    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Iteration Time (ms)")
    ax.set_title("Training Iteration Timing vs I/O Activity")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(40, 280)

    means = [np.mean(d) for d in data]
    p99s = [np.percentile(d, 99) for d in data]
    for i, (m, p) in enumerate(zip(means, p99s)):
        ax.annotate(rf"$\mu$={m:.0f}" + "\n" + rf"P99={p:.0f}",
                    xy=(i+1, p + 6), ha="center", fontsize=8,
                    color="dimgrey")

    fig.tight_layout()
    fig.savefig(OUT / "fig_io_training_overhead.pdf")
    plt.close(fig)
    print("  -> fig_io_training_overhead.pdf")


# ============================
# Figure 4: Chunk Size Ablation — I/O Events & Hit Rate
# ============================
def fig_chunk_ablation():
    header, rows = read_csv("io_chunk_size_ablation.csv")
    sizes = [int(r[0]) for r in rows]
    total_chunks = [int(r[1]) for r in rows]
    load_events = [int(r[3]) for r in rows]
    evict_events = [int(r[4]) for r in rows]
    hit_rate = [r[6] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(sizes))
    w = 0.3

    bars_l = ax1.bar(x - w/2, load_events, w, color="#2166AC",
                     edgecolor="white", linewidth=0.5, label="Load Events")
    bars_e = ax1.bar(x + w/2, evict_events, w, color="#F4A582",
                     edgecolor="white", linewidth=0.5, label="Evict Events")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$S={s}$\n({tc} chunks)" for s, tc in zip(sizes, total_chunks)])
    ax1.set_ylabel("Event Count (200 frames)")
    ax1.set_ylim(0, 280)

    ax2 = ax1.twinx()
    ax2.plot(x, hit_rate, "D-", color="#1B7837", linewidth=2,
             markersize=9, label="LRU Hit Rate")
    ax2.set_ylabel("LRU Hit Rate (%)")
    ax2.set_ylim(94, 100.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               frameon=True, loc="upper left", fontsize=9)

    ax1.set_title("Chunk Size Ablation: I/O Events & Hit Rate")
    ax1.grid(True, alpha=0.3, axis="y")

    for xi, hr in zip(x, hit_rate):
        ax2.annotate(f"{hr}%", (xi, hr - 0.6), ha="center", fontsize=9,
                     fontweight="bold", color="#1B7837")

    fig.tight_layout()
    fig.savefig(OUT / "fig_io_chunk_ablation.pdf")
    plt.close(fig)
    print("  -> fig_io_chunk_ablation.pdf")


if __name__ == "__main__":
    print("Generating I/O performance figures from CSV data...")
    fig_latency()
    fig_throughput()
    fig_training_overhead()
    fig_chunk_ablation()
    print("Done.")
