#!/usr/bin/env python3
"""Generate D (feature dimension) ablation figures for thesis Section 4.4.2.

Reads data from d_ablation.csv (same directory as this script).
Outputs PDFs to thesis/figures/.
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR.parent / "thesis" / "figures"
DATA = SCRIPT_DIR  # CSV files live alongside this script

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def read_csv(name):
    """Read a CSV file, return (header: list[str], rows: list[list])."""
    rows = []
    with open(DATA / name, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)
    return header, rows


# ============================
# Figure 1: Rendering Quality Comparison
# ============================
def fig_quality():
    header, rows = read_csv("d_ablation.csv")
    D_vals = [r[0] for r in rows]
    psnr = np.array([float(r[1]) for r in rows])
    ssim = np.array([float(r[2]) for r in rows])
    lpips = np.array([float(r[3]) for r in rows])

    x = np.arange(len(D_vals))
    width = 0.22
    colors = ["#5B9BD5", "#ED7D31", "#70AD47"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))

    # --- PSNR ---
    ax = axes[0]
    bars = ax.bar(x, psnr, width * 2.2, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(D_vals)
    ax.set_xlabel("D (feat dim)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(24.5, 27.2)
    ax.set_title(r"PSNR $\uparrow$", fontweight="bold")
    for bar, val in zip(bars, psnr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10,
                fontweight="medium")

    # --- SSIM ---
    ax = axes[1]
    bars = ax.bar(x, ssim, width * 2.2, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(D_vals)
    ax.set_xlabel("D (feat dim)")
    ax.set_ylabel("SSIM")
    ax.set_ylim(0.79, 0.87)
    ax.set_title(r"SSIM $\uparrow$", fontweight="bold")
    for bar, val in zip(bars, ssim):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="medium")

    # --- LPIPS ---
    ax = axes[2]
    bars = ax.bar(x, lpips, width * 2.2, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(D_vals)
    ax.set_xlabel("D (feat dim)")
    ax.set_ylabel("LPIPS")
    ax.set_ylim(0.14, 0.21)
    ax.set_title(r"LPIPS $\downarrow$", fontweight="bold")
    for bar, val in zip(bars, lpips):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="medium")

    fig.suptitle("不同特征维度 D 的渲染质量对比（KITTI/seq05, K = 5）",
                 y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_d_ablation_quality.pdf")
    plt.close(fig)
    print("  -> fig_d_ablation_quality.pdf")


# ============================
# Figure 2: Cost-Efficiency Tradeoff
# ============================
def fig_cost():
    header, rows = read_csv("d_ablation.csv")
    D_vals = [r[0] for r in rows]
    disk_mb = np.array([float(r[4]) for r in rows])
    mlp_params = np.array([float(r[5]) for r in rows])
    train_min = np.array([float(r[6]) for r in rows])

    xp = np.arange(len(D_vals))
    bar_w = 0.28

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    bars_disk = ax.bar(xp - bar_w, disk_mb, bar_w,
                       color="#5B9BD5", edgecolor="white", linewidth=0.5,
                       label="磁盘占用 (MB)")
    for bar, val in zip(bars_disk, disk_mb):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10,
                fontweight="medium")

    bars_time = ax.bar(xp + bar_w, train_min, bar_w,
                       color="#ED7D31", edgecolor="white", linewidth=0.5,
                       label="训练时间 (min)")
    for bar, val in zip(bars_time, train_min):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10,
                fontweight="medium")

    ax2 = ax.twinx()
    line_mlp = ax2.plot(xp, mlp_params, "o-", color="#70AD47", linewidth=2.2,
                        markersize=9, markerfacecolor="white",
                        markeredgewidth=2, label="MLP 参数量")
    for i, val in enumerate(mlp_params):
        offset = 800 if i < 2 else -1000
        ax2.text(i, val + offset, f"{val:,.0f}",
                 ha="center",
                 va="bottom" if i < 2 else "top",
                 fontsize=10, fontweight="medium", color="#2F6B1E")

    ax.set_xticks(xp)
    ax.set_xticklabels(D_vals)
    ax.set_xlabel("D (特征维度)")
    ax.set_ylabel("磁盘占用 / 训练时间", fontsize=11)
    ax2.set_ylabel("MLP 参数量", fontsize=11, color="#2F6B1E")
    ax2.tick_params(axis="y", labelcolor="#2F6B1E")

    bars_leg = [bars_disk, bars_time, line_mlp[0]]
    labels_leg = ["磁盘占用 (MB)", "训练时间 (min)", "MLP 参数量"]
    ax.legend(bars_leg, labels_leg, loc="upper left", frameon=True,
              fancybox=True, framealpha=0.95)

    ax.set_ylim(0, max(train_min) * 1.35)
    ax2.set_ylim(0, max(mlp_params) * 1.18)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda v, _: f"{v/1000:.1f}K" if v < 10000 else f"{v/1000:.0f}K"))

    ax.set_title("不同特征维度 D 的存储与计算开销（KITTI/seq05, K = 5）",
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_d_ablation_cost.pdf")
    plt.close(fig)
    print("  -> fig_d_ablation_cost.pdf")


if __name__ == "__main__":
    print("Generating D ablation figures from CSV data...")
    fig_quality()
    fig_cost()
    print("Done.")
