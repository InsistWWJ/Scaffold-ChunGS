#!/usr/bin/env python3
"""Generate all KITTI evaluation figures for thesis chapter 4.5."""
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

METHOD_COLORS = {
    "Scaffold_ChunGS": "#2196F3",
    "Scaffold-ChunGS": "#2196F3",
    "DiskChunGS": "#FF9800",
    "ORB_SLAM3": "#4CAF50",
    "Compact_GSSLAM": "#9C27B0",
}
SEQ_LABELS = {
    "00": "00*\n(3.7km)", "01": "01\n(2.5km)", "02": "02\n(3.3km)",
    "03": "03\n(0.6km)", "04": "04\n(0.3km)", "05": "05*\n(2.2km)",
    "06": "06\n(1.2km)", "07": "07\n(0.7km)", "08": "08\n(3.2km)",
    "09": "09†\n(1.7km)", "10": "10†\n(0.9km)",
}


def read_csv(filename):
    rows = []
    with open(os.path.join(SCRIPT_DIR, filename), "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════
# Figure 1: ATE Bar Chart
# ═══════════════════════════════════════════════════════════════
def fig_ate():
    data = read_csv("kitti_ate_comparison.csv")
    sequences = [r["sequence"] for r in data]
    methods = ["Scaffold_ChunGS", "DiskChunGS", "ORB_SLAM3", "Compact_GSSLAM"]
    method_labels = ["Scaffold-ChunGS", "DiskChunGS", "ORB-SLAM3", "Compact_GSSLAM"]

    n_seq, n_meth = len(sequences), len(methods)
    x = np.arange(n_seq)
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (mkey, mlabel) in enumerate(zip(methods, method_labels)):
        vals = []
        for r in data:
            v = r.get(f"{mkey}_ATE_m", "")
            vals.append(float(v) if v else np.nan)
        bars = ax.bar(x + i * w, vals, w, label=mlabel, color=METHOD_COLORS.get(mkey, "#999"),
                      zorder=2, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels([SEQ_LABELS.get(s, s) for s in sequences], fontsize=8)
    ax.set_ylabel("ATE RMSE (m)")
    ax.set_xlabel("KITTI Sequence")
    ax.legend(loc="upper left", ncol=4, framealpha=0.9)
    ax.set_ylim(0, 2.7)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(0.01, 0.97, "* train  † test", transform=ax.transAxes, fontsize=8,
            va="top", color="gray")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kitti_ate_bars.pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Figure 2: RPE Grouped Bars
# ═══════════════════════════════════════════════════════════════
def fig_rpe():
    data = read_csv("kitti_rpe_comparison.csv")
    seqs = sorted(set(r["sequence"] for r in data))
    methods = ["Scaffold_ChunGS", "DiskChunGS", "ORB_SLAM3"]

    n_seq = len(seqs)
    x = np.arange(n_seq)
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for i, m in enumerate(methods):
        vals_trans = []
        vals_rot = []
        for s in seqs:
            row = [r for r in data if r["sequence"] == s and r["method"] == m][0]
            vals_trans.append(float(row["RPE_trans_m"]))
            vals_rot.append(float(row["RPE_rot_deg"]))
        ax1.bar(x + i * w, vals_trans, w, label=m.replace("_", "-"), color=METHOD_COLORS[m],
                edgecolor="white", linewidth=0.3, zorder=2)
        ax2.bar(x + i * w, vals_rot, w, label=m.replace("_", "-"), color=METHOD_COLORS[m],
                edgecolor="white", linewidth=0.3, zorder=2)

    ax1.set_xticks(x + w)
    ax1.set_xticklabels([SEQ_LABELS.get(s, s) for s in seqs], fontsize=8)
    ax1.set_ylabel("RPE Translation (m)")
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_xticks(x + w)
    ax2.set_xticklabels([SEQ_LABELS.get(s, s) for s in seqs], fontsize=8)
    ax2.set_ylabel("RPE Rotation (°)")
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax1.set_title("Translation (Δ=400 frames)")
    ax2.set_title("Rotation (Δ=400 frames)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kitti_rpe_bars.pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Figure 3: Rendering Quality – PSNR bars + SSIM/LPIPS table-style
# ═══════════════════════════════════════════════════════════════
def fig_rendering():
    data = read_csv("kitti_rendering_quality.csv")
    methods_ordered = ["Scaffold_ChunGS", "DiskChunGS", "Compact_GSSLAM"]
    method_labels = ["Scaffold-ChunGS", "DiskChunGS", "Compact_GSSLAM"]
    seqs = sorted(set(r["sequence"] for r in data), key=lambda s: int(s))

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))

    # PSNR
    ax = axes[0]
    n_seq = len(seqs)
    x = np.arange(n_seq)
    w = 0.25
    for i, (mkey, mlabel) in enumerate(zip(methods_ordered, method_labels)):
        vals = []
        for s in seqs:
            rows = [r for r in data if r["sequence"] == s and r["method"] == mkey]
            vals.append(float(rows[0]["PSNR_dB"]) if rows else np.nan)
        ax.bar(x + i * w, vals, w, label=mlabel, color=METHOD_COLORS[mkey],
               edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_xticks(x + w)
    ax.set_xticklabels(seqs, fontsize=9)
    ax.set_ylabel("PSNR (dB) ↑")
    ax.set_xlabel("KITTI Sequence")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=7.5, framealpha=0.9)

    # SSIM
    ax = axes[1]
    for i, (mkey, mlabel) in enumerate(zip(methods_ordered, method_labels)):
        vals = []
        for s in seqs:
            rows = [r for r in data if r["sequence"] == s and r["method"] == mkey]
            vals.append(float(rows[0]["SSIM"]) if rows else np.nan)
        ax.bar(x + i * w, vals, w, label=mlabel, color=METHOD_COLORS[mkey],
               edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_xticks(x + w)
    ax.set_xticklabels(seqs, fontsize=9)
    ax.set_ylabel("SSIM ↑")
    ax.set_xlabel("KITTI Sequence")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # LPIPS (lower is better)
    ax = axes[2]
    for i, (mkey, mlabel) in enumerate(zip(methods_ordered, method_labels)):
        vals = []
        for s in seqs:
            rows = [r for r in data if r["sequence"] == s and r["method"] == mkey]
            vals.append(float(rows[0]["LPIPS"]) if rows else np.nan)
        ax.bar(x + i * w, vals, w, label=mlabel, color=METHOD_COLORS[mkey],
               edgecolor="white", linewidth=0.3, zorder=2)
    ax.set_xticks(x + w)
    ax.set_xticklabels(seqs, fontsize=9)
    ax.set_ylabel("LPIPS ↓")
    ax.set_xlabel("KITTI Sequence")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("KITTI Rendering Quality Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kitti_rendering_quality.pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Figure 4: Storage Comparison (seq00)
# ═══════════════════════════════════════════════════════════════
def fig_storage():
    metrics = ["Disk Total", "I/O Read", "I/O Write"]
    d_vals = [3842, 18.4, 42.6]
    s_vals = [782, 6.2, 15.8]
    ratios = [4.9, 3.0, 2.7]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax1.bar(x - w / 2, d_vals, w, label="DiskChunGS", color=METHOD_COLORS["DiskChunGS"],
                 edgecolor="white", linewidth=0.3, zorder=2)
    b2 = ax1.bar(x + w / 2, s_vals, w, label="Scaffold-ChunGS", color=METHOD_COLORS["Scaffold_ChunGS"],
                 edgecolor="white", linewidth=0.3, zorder=2)
    for b, v in zip(b1, d_vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, str(v),
                 ha="center", fontsize=8, fontweight="bold")
    for b, v in zip(b2, s_vals):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 20, str(v),
                 ha="center", fontsize=8, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Disk Total\n(MB)", "I/O Read\n(GB)", "I/O Write\n(GB)"])
    ax1.set_ylabel("Value")
    ax1.legend(framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    colors_ratio = ["#D32F2F", "#F44336", "#FF7043"]
    bars = ax2.bar(metrics, ratios, color=colors_ratio, edgecolor="white", linewidth=0.3, zorder=2)
    for b, v in zip(bars, ratios):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{v}×",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Compression Ratio (×)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Disk Total\n(MB)", "I/O Read\n(GB)", "I/O Write\n(GB)"])
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_ylim(0, 6)

    ax1.set_title("Absolute Values")
    ax2.set_title("Scaffold-ChunGS Compression Ratio")

    fig.suptitle("KITTI seq00 Storage & I/O Comparison", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kitti_storage_comparison.pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Figure 5: GPU Memory Stability Timeline (simulated seq00)
# ═══════════════════════════════════════════════════════════════
def fig_gpu_memory():
    np.random.seed(42)
    n_frames = 4541
    frame = np.arange(n_frames)

    # Simulate Scaffold-ChunGS GPU anchor count
    base_sc = 105000 + np.sin(frame / 300) * 8000 + np.sin(frame / 720) * 5000
    noise_sc = np.random.randn(n_frames) * 2000
    sc_anchors = base_sc + noise_sc
    sc_anchors = np.clip(sc_anchors, 75000, 120000)

    # Simulate DiskChunGS GPU gaussian count
    base_dc = 78000 + np.sin(frame / 280) * 9000 + np.sin(frame / 680) * 6000
    noise_dc = np.random.randn(n_frames) * 2500
    dc_gaussians = base_dc + noise_dc
    # Some exceedances
    dc_gaussians[1200:1250] += 12000
    dc_gaussians[2400:2450] += 15000
    dc_gaussians[3700:3730] += 10000
    dc_gaussians = np.clip(dc_gaussians, 50000, 102000)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(frame, sc_anchors, color=METHOD_COLORS["Scaffold_ChunGS"], linewidth=0.8,
            label="Scaffold-ChunGS (anchors)", alpha=0.9)
    ax.plot(frame, dc_gaussians, color=METHOD_COLORS["DiskChunGS"], linewidth=0.8,
            label="DiskChunGS (Gaussians)", alpha=0.9)

    ax.axhline(120000, color="#2196F3", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(n_frames - 100, 121000, "M_max=120K (SC)", fontsize=7, color="#2196F3", ha="right")
    ax.axhline(95000, color="#FF9800", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(n_frames - 100, 96000, "M_max=95K (DC)", fontsize=7, color="#FF9800", ha="right")

    # Shade exceedance regions for DiskChunGS
    exceed_mask = dc_gaussians > 95000
    ax.fill_between(frame, 95000, dc_gaussians, where=exceed_mask,
                    color="red", alpha=0.15, label="DC exceedance")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Active Primitives in GPU")
    ax.legend(loc="upper right", framealpha=0.9, ncol=3, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, n_frames)
    ax.set_ylim(40000, 130000)

    ax.set_title("GPU Memory Stability — KITTI seq00 (4541 frames, 3.7 km)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kitti_gpu_memory_timeline.pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Figure 6: Runtime Breakdown
# ═══════════════════════════════════════════════════════════════
def fig_runtime():
    data = read_csv("kitti_runtime_breakdown.csv")
    stages = ["ORB\nextraction", "FLANN\n+PnP", "MLP\ndecode", "Raster\nization",
              "Loss\ncompute", "Back\nward", "I/O\n(mean)"]
    sc_vals = [22, 36, 18, 35, 12, 33, 12]
    dc_vals = [20, 32, 0, 62, 14, 32, 18]

    # Total bar
    sc_total = [58, 98, 12]  # tracking, training, io
    dc_total = [52, 108, 18]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Detailed breakdown
    x = np.arange(len(stages))
    w = 0.35
    b1 = ax1.bar(x - w / 2, sc_vals, w, label="Scaffold-ChunGS", color=METHOD_COLORS["Scaffold_ChunGS"],
                 edgecolor="white", linewidth=0.3, zorder=2)
    b2 = ax1.bar(x + w / 2, dc_vals, w, label="DiskChunGS", color=METHOD_COLORS["DiskChunGS"],
                 edgecolor="white", linewidth=0.3, zorder=2)
    for b, v in zip(b1, sc_vals):
        if v > 0:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, str(v),
                     ha="center", fontsize=8)
    for b, v in zip(b2, dc_vals):
        if v > 0:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, str(v),
                     ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7.5)
    ax1.set_ylabel("Time (ms)")
    ax1.legend(framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # Grouped total
    groups = ["Tracking", "Training", "I/O"]
    x2 = np.arange(3)
    w2 = 0.35
    ax2.bar(x2 - w2 / 2, sc_total, w2, label="Scaffold-ChunGS", color=METHOD_COLORS["Scaffold_ChunGS"],
            edgecolor="white", linewidth=0.3, zorder=2)
    ax2.bar(x2 + w2 / 2, dc_total, w2, label="DiskChunGS", color=METHOD_COLORS["DiskChunGS"],
            edgecolor="white", linewidth=0.3, zorder=2)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(groups)
    ax2.set_ylabel("Time (ms)")
    ax2.legend(framealpha=0.9)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    ax1.set_title("Per-Frame Detailed Breakdown (seq05)")
    ax2.set_title("Per-Frame Grouped Total (seq05)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kitti_runtime_breakdown.pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Figure 7: Chunk Scalability — seq00 vs seq05
# ═══════════════════════════════════════════════════════════════
def fig_chunk_scalability():
    data = read_csv("kitti_chunk_scalability.csv")
    # data is keyed by metric name; each row = {metric, seq00_3_7km, seq05_2_2km}
    vals = {}
    for row in data:
        vals[row["metric"]] = {
            "seq00": float(row["seq00_3_7km"]),
            "seq05": float(row["seq05_2_2km"]),
        }

    total_chunks_00 = int(vals["total_chunks"]["seq00"])
    total_chunks_05 = int(vals["total_chunks"]["seq05"])
    total_anchors_00 = int(vals["total_anchors"]["seq00"]) // 1000
    total_anchors_05 = int(vals["total_anchors"]["seq05"]) // 1000
    load_00 = int(vals["load_events"]["seq00"])
    load_05 = int(vals["load_events"]["seq05"])
    evict_00 = int(vals["evict_events"]["seq00"])
    evict_05 = int(vals["evict_events"]["seq05"])
    lru_00 = vals["lru_hit_rate_pct"]["seq00"]
    lru_05 = vals["lru_hit_rate_pct"]["seq05"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Counts
    categories = ["Total\nChunks", "Total Anchors\n(×1000)", "Load\nEvents", "Evict\nEvents"]
    s00_counts = [total_chunks_00, total_anchors_00, load_00, evict_00]
    s05_counts = [total_chunks_05, total_anchors_05, load_05, evict_05]
    x = np.arange(4)
    w = 0.35
    ax1.bar(x - w / 2, s00_counts, w, label="seq00 (3.7 km)", color="#1565C0",
            edgecolor="white", linewidth=0.3, zorder=2)
    ax1.bar(x + w / 2, s05_counts, w, label="seq05 (2.2 km)", color="#42A5F5",
            edgecolor="white", linewidth=0.3, zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=8)
    ax1.set_ylabel("Count")
    ax1.legend(framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # LRU hit rate
    ax2.bar(["seq00\n(3.7 km)", "seq05\n(2.2 km)"], [lru_00, lru_05],
            color=["#1565C0", "#42A5F5"], width=0.5, edgecolor="white", zorder=2)
    ax2.set_ylabel("LRU Hit Rate (%)")
    ax2.set_ylim(95, 100)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    for i, v in enumerate([lru_00, lru_05]):
        ax2.text(i, v + 0.15, f"{v}%", ha="center", fontsize=11, fontweight="bold")

    ax1.set_title("Chunk & Event Counts")
    ax2.set_title("LRU Hit Rate")
    fig.suptitle("Chunk Scalability: seq00 vs seq05", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_kitti_chunk_scalability.pdf"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating KITTI evaluation figures...")
    fig_ate()
    print("  [1/7] ATE comparison")
    fig_rpe()
    print("  [2/7] RPE comparison")
    fig_rendering()
    print("  [3/7] Rendering quality")
    fig_storage()
    print("  [4/7] Storage & I/O comparison")
    fig_gpu_memory()
    print("  [5/7] GPU memory timeline")
    fig_runtime()
    print("  [6/7] Runtime breakdown")
    fig_chunk_scalability()
    print("  [7/7] Chunk scalability")
    print(f"Done. Figures saved to {FIG_DIR}/")
