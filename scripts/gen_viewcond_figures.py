#!/usr/bin/env python3
"""Generate viewpoint-condition ablation figures for thesis Section 4.4.1."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR.parent / "thesis" / "figures"
DATA = SCRIPT_DIR

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def read_csv(name):
    rows = []
    with open(DATA / name, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


# ============================
# Figure 1: Ablation — PSNR/SSIM/LPIPS bar chart
# ============================
def fig_ablation():
    rows = read_csv("viewcond_ablation.csv")
    labels_full = [r["config"] for r in rows]
    labels_short = ["Feat only", "Feat+ViewDir", "Full"]
    psnr = np.array([float(r["psnr"]) for r in rows])
    ssim = np.array([float(r["ssim"]) for r in rows])
    lpips = np.array([float(r["lpips"]) for r in rows])

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(11, 4.8),
                                   gridspec_kw={"width_ratios": [2.5, 1]})

    x = np.arange(len(labels_full))
    w = 0.25

    # --- Left: PSNR + SSIM grouped bars ---
    bars_p = ax1.bar(x - w, psnr, w, color="#2166AC", edgecolor="white",
                     linewidth=0.5, label="PSNR (dB)")
    ax1.set_ylabel("PSNR (dB) $\\uparrow$", color="#2166AC")
    ax1.tick_params(axis="y", labelcolor="#2166AC")
    ax1.set_ylim(22, 28)

    ax2 = ax1.twinx()
    bars_s = ax2.bar(x, ssim, w, color="#4393C3", edgecolor="white",
                     linewidth=0.5, label="SSIM")
    ax2.set_ylabel("SSIM $\\uparrow$", color="#4393C3")
    ax2.tick_params(axis="y", labelcolor="#4393C3")
    ax2.set_ylim(0.72, 0.90)

    for bar, val in zip(bars_p, psnr):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars_s, ssim):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_short, fontsize=10)
    ax1.set_title("Rendering Quality by Input Configuration")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    # --- Right: LPIPS bars with short labels ---
    colors_lp = ["#D73027", "#F46D43", "#4575B4"]
    bars_l = ax3.bar(x, lpips, w * 1.5, color=colors_lp, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars_l, lpips):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_short, fontsize=10)
    ax3.set_ylabel("LPIPS $\\downarrow$")
    ax3.set_title("Perceptual Quality (LPIPS)")
    ax3.set_ylim(0.14, 0.26)
    ax3.grid(True, alpha=0.3, axis="y")

    # Legend for LPIPS subplot
    legend_labels = labels_full
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=c, edgecolor="white",
                                     linewidth=0.5) for c in colors_lp]
    ax3.legend(legend_patches, legend_labels, loc="upper right", frameon=True,
               fontsize=8, title="Input Config")

    fig.tight_layout()
    fig.savefig(OUT / "fig_viewcond_ablation.pdf")
    plt.close(fig)
    print("  -> fig_viewcond_ablation.pdf")


# ============================
# Figure 2: Viewpoint Robustness — near vs far angle
# ============================
def fig_robustness():
    rows = read_csv("viewcond_robustness.csv")
    labels = [r["config"] for r in rows]
    near = np.array([float(r["psnr_near_5deg"]) for r in rows])
    far = np.array([float(r["psnr_far_10deg"]) for r in rows])
    gap = np.array([float(r["gap_db"]) for r in rows])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    x = np.arange(len(labels))
    w = 0.30

    bars_n = ax.bar(x - w/2, near, w, color="#2166AC", edgecolor="white",
                    linewidth=0.5, label=r"Near viewpoints ($\theta < 5^\circ$)")
    bars_f = ax.bar(x + w/2, far, w, color="#F4A582", edgecolor="white",
                    linewidth=0.5, label=r"Far viewpoints ($\theta > 10^\circ$)")

    # Annotate bars
    for bar, val in zip(bars_n, near):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#2166AC")
    for bar, val in zip(bars_f, far):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#B2182B")

    # Draw gap annotations
    for i in range(len(labels)):
        mid = x[i]
        top = max(near[i], far[i]) + 0.8
        ax.annotate("",
            xy=(mid - w/2, top), xytext=(mid + w/2, top),
            arrowprops=dict(arrowstyle="<->", color="grey", lw=1.3))
        ax.text(mid, top + 0.1, f"$\\Delta$={gap[i]:.1f}dB",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="dimgrey")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("PSNR (dB) $↑$")
    ax.set_title("Viewpoint Robustness: PSNR Across Viewing Angles")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.set_ylim(23, 27.8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / "fig_viewcond_robustness.pdf")
    plt.close(fig)
    print("  -> fig_viewcond_robustness.pdf")


if __name__ == "__main__":
    print("Generating viewpoint-condition ablation figures...")
    fig_ablation()
    fig_robustness()
    print("Done.")
