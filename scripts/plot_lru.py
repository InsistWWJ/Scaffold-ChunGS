#!/usr/bin/env python3
"""
Plot LRU eviction experiment figures for thesis Section 4.3.1 (KITTI seq00).
Reads lru_frame_trace.csv and lru_chunk_inventory.csv, outputs PDF figures.
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Chinese font setup for thesis
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["SimSun", "Times New Roman"],
    "font.size": 10,
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR.parent / "thesis" / "figures"
OUT_DIR.mkdir(exist_ok=True)

MAX_ANCHORS = 120_000   # KITTI seq00, Jetson Orin Nano
BUFFER = 6_000           # 5% hysteresis


# ---------------------------------------------------------------------------
# CSV reader helpers
# ---------------------------------------------------------------------------
def read_csv(name):
    with open(SCRIPT_DIR / name, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_pipe(val, dtype=float):
    """Parse pipe-separated field into list."""
    if not val or not val.strip():
        return []
    result = []
    for v in str(val).split("|"):
        v = v.strip()
        if v:
            result.append(dtype(v))
    return result


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
trace_rows = read_csv("lru_frame_trace.csv")
inventory_rows = read_csv("lru_chunk_inventory.csv")

frames = np.array([int(r["frame"]) for r in trace_rows])
gpu_before = np.array([int(r["gpu_anchors_before"]) for r in trace_rows])
gpu_after = np.array([int(r["gpu_anchors_after"]) for r in trace_rows])
num_visible = np.array([int(r["num_visible"]) for r in trace_rows])
num_loaded = np.array([int(r["num_loaded"]) for r in trace_rows])
num_evicted = np.array([int(r["num_evicted"]) for r in trace_rows])

# Parse pipe-separated fields for eviction details
evicted_atimes_all = []
evicted_acounts_all = []
scatter_frames_list = []
scatter_atimes_list = []
scatter_sizes_list = []

for row in trace_rows:
    evicted = row.get("evicted_chunks", "")
    atimes_str = row.get("evicted_access_times", "")
    acounts_str = row.get("evicted_anchor_counts", "")
    f = int(row["frame"])

    if evicted and evicted.strip():
        atimes = parse_pipe(atimes_str, float)
        acounts = parse_pipe(acounts_str, int)
        for at, ac in zip(atimes, acounts):
            scatter_frames_list.append(f)
            scatter_atimes_list.append(at)
            scatter_sizes_list.append(max(3, np.sqrt(ac) * 0.3))
            evicted_atimes_all.append(at)
            evicted_acounts_all.append(ac)


# ===========================================================================
# Figure 1: GPU Anchor Count Stability
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 3.2))

ax.fill_between(frames, 0, MAX_ANCHORS, alpha=0.08, color="tab:red",
                label="超出上限区域 (Over-limit)")
ax.axhline(y=MAX_ANCHORS, color="tab:red", linewidth=1.0, linestyle="--",
           label=f"$M_{{\\max}}$ = {MAX_ANCHORS:,}")
ax.axhline(y=MAX_ANCHORS + BUFFER, color="tab:orange", linewidth=0.7, linestyle=":",
           label=f"+5% 滞后缓冲区 (Hysteresis)")

ax.plot(frames, gpu_before, linewidth=0.5, alpha=0.35, color="tab:blue",
        label="逐帧前 (before load/evict)")
ax.plot(frames, gpu_after, linewidth=1.2, color="tab:blue",
        label="逐帧后 (after load/evict)")

ax.set_xlabel("帧序号 (Frame)")
ax.set_ylabel("GPU常驻锚点数 (GPU-resident anchors)")
ax.set_title("GPU锚点数量稳定性 — KITTI seq00 (GPU Anchor Count Stability)")
ax.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.8)
ax.set_ylim(0, gpu_after.max() * 1.12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_gpu_anchor_stability.pdf")
plt.close(fig)

# ===========================================================================
# Figure 2: Load / Evict Activity Per Frame
# ===========================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)

width = 0.6
ax1.bar(frames, num_loaded, width, color="tab:green", alpha=0.7,
        label="加载块数 (Loaded)")
ax1.bar(frames, -num_evicted, width, color="tab:red", alpha=0.7,
        label="淘汰块数 (Evicted)")
ax1.axhline(y=0, color="black", linewidth=0.5)
ax1.set_ylabel("块数 (Chunks)")
ax1.set_title("逐帧加载与淘汰活动 — KITTI seq00 (Load & Evict Activity)")
ax1.legend(fontsize=8, loc="upper right")
ax1.grid(axis="y", alpha=0.3)

ax2.plot(frames, num_visible, linewidth=1.0, color="tab:blue", marker=".",
         markersize=2)
ax2.set_xlabel("帧序号 (Frame)")
ax2.set_ylabel("可见块数 (Visible)")
ax2.set_title("逐帧可见块数 — KITTI seq00 (Visible Chunks per Frame)")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_load_evict_activity.pdf")
plt.close(fig)

# ===========================================================================
# Figure 3: LRU Discipline — Evicted Chunk Access Times
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 3.2))

if scatter_frames_list:
    sc = ax.scatter(scatter_frames_list, scatter_atimes_list,
                    c=scatter_atimes_list, cmap="YlOrRd",
                    s=scatter_sizes_list, alpha=0.7, edgecolors="gray",
                    linewidths=0.2)
    cbar = fig.colorbar(sc, ax=ax, label="访问时间 (Access time)")

    ax.annotate("陈旧块被淘汰\n(Stale chunks evicted)",
                xy=(scatter_frames_list[0] if scatter_frames_list else 18, 2),
                xytext=(40, 8),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=8, color="gray")

ax.set_xlabel("帧序号 (Frame)")
ax.set_ylabel("被淘汰时的访问时间 (Access time at eviction)")
ax.set_title("LRU淘汰纪律 — KITTI seq00 (LRU Discipline)")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_lru_discipline.pdf")
plt.close(fig)

# ===========================================================================
# Figure 4: Chunk Access Timeline (heatmap)
# ===========================================================================
# Build chunk visibility matrix
all_chunk_ids = sorted([int(r["chunk_id"]) for r in inventory_rows])

# Count chunk visibility
chunk_frame_counts = {}
for row in trace_rows:
    vis_str = row.get("visible_chunks", "")
    if vis_str and vis_str.strip():
        for c in vis_str.split("|"):
            cid = int(c.strip())
            chunk_frame_counts[cid] = chunk_frame_counts.get(cid, 0) + 1

# Pick top 30 most active chunks
top_chunks = sorted(chunk_frame_counts, key=chunk_frame_counts.get, reverse=True)[:30]
top_chunks.sort()

# Build matrix: 0=evicted, 1=loaded but not visible, 2=visible
chunk_matrix = np.zeros((len(top_chunks), len(frames)), dtype=int)
cid_to_row = {cid: i for i, cid in enumerate(top_chunks)}

# Initial frame
first_vis = parse_pipe(trace_rows[0].get("visible_chunks", ""), int)
for cid in first_vis:
    if cid in cid_to_row:
        chunk_matrix[cid_to_row[cid], 0] = 2

for i in range(1, len(frames)):
    # Carry over previous state
    chunk_matrix[:, i] = chunk_matrix[:, i - 1]
    # Downgrade visible->loaded
    for r in range(len(top_chunks)):
        if chunk_matrix[r, i] == 2:
            chunk_matrix[r, i] = 1

    row = trace_rows[i]
    # Visible -> state 2
    vis_cids = parse_pipe(row.get("visible_chunks", ""), int)
    for cid in vis_cids:
        if cid in cid_to_row:
            chunk_matrix[cid_to_row[cid], i] = 2

    # Evicted -> state 0
    ev_cids = parse_pipe(row.get("evicted_chunks", ""), int)
    for cid in ev_cids:
        if cid in cid_to_row:
            chunk_matrix[cid_to_row[cid], i] = 0

fig, ax = plt.subplots(figsize=(7, 4.5))
cmap = matplotlib.colors.ListedColormap(["#e0e0e0", "#a0d2ff", "#ff6b35"])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
im = ax.imshow(chunk_matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.85)
cbar.ax.set_yticklabels(["已淘汰/不在GPU (Evicted)", "已加载不在视野 (Loaded, invisible)",
                          "可见 (Visible)"])

ax.set_xlabel("帧序号 (Frame)")
ax.set_ylabel("块ID (Chunk ID)")
ax.set_title("高频访问块的生命周期热力图 — KITTI seq00 (Chunk Access Timeline)")
ax.set_yticks(range(len(top_chunks)))
ax.set_yticklabels(top_chunks, fontsize=6)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig_chunk_timeline.pdf")
plt.close(fig)

# ===========================================================================
# Figure 5: Anchor Count Distribution across Chunks
# ===========================================================================
fig, ax = plt.subplots(figsize=(6, 2.8))
counts = np.array([int(r["num_anchors"]) for r in inventory_rows])
ax.hist(counts, bins=20, color="tab:blue", alpha=0.7, edgecolor="white", linewidth=0.5)
ax.axvline(x=counts.mean(), color="tab:red", linestyle="--", linewidth=1.0,
           label=f"均值 = {counts.mean():.0f}")

ax.set_xlabel("每块锚点数 (Anchors per chunk)")
ax.set_ylabel("块数 (Chunk count)")
ax.set_title("96个块的锚点数量分布 — KITTI seq00 (Anchor Distribution Across Chunks)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_anchor_distribution.pdf")
plt.close(fig)

# ===========================================================================
# Figure 6: Summary dashboard (4-panel for thesis)
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))

# (a) GPU anchor count
ax = axes[0, 0]
ax.fill_between(frames, 0, MAX_ANCHORS, alpha=0.06, color="tab:red")
ax.axhline(y=MAX_ANCHORS, color="tab:red", linewidth=0.8, linestyle="--")
ax.plot(frames, gpu_after, linewidth=1.0, color="tab:blue")
ax.set_xlabel("帧序号 (Frame)")
ax.set_ylabel("GPU锚点数")
ax.set_title("(a) GPU锚点数量稳定性 (seq00)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))
ax.grid(alpha=0.3)

# (b) Load/evict per frame
ax = axes[0, 1]
ax.bar(frames, num_loaded, 0.6, color="tab:green", alpha=0.6, label="加载")
ax.bar(frames, -num_evicted, 0.6, color="tab:red", alpha=0.6, label="淘汰")
ax.axhline(y=0, color="black", linewidth=0.4)
ax.set_xlabel("帧序号 (Frame)")
ax.set_ylabel("块数")
ax.set_title("(b) 逐帧加载与淘汰 (seq00)")
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)

# (c) LRU discipline scatter
ax = axes[1, 0]
if scatter_frames_list:
    sc = ax.scatter(scatter_frames_list, scatter_atimes_list,
                    c=scatter_atimes_list, cmap="YlOrRd",
                    s=scatter_sizes_list, alpha=0.6, edgecolors="gray",
                    linewidths=0.15)
ax.set_xlabel("帧序号 (Frame)")
ax.set_ylabel("淘汰时访问时间")
ax.set_title("(c) 被淘汰块访问时间分布 (seq00)")
ax.grid(alpha=0.3)

# (d) Chunk lifetime histogram
ax = axes[1, 1]
chunk_lifetimes = {}
chunk_load_frame = {}
for row in trace_rows:
    f = int(row["frame"])
    lds = parse_pipe(row.get("loaded_chunks", ""), int)
    evs = parse_pipe(row.get("evicted_chunks", ""), int)
    for cid in lds:
        if cid not in chunk_load_frame:
            chunk_load_frame[cid] = f
    for cid in evs:
        if cid in chunk_load_frame:
            lt = f - chunk_load_frame[cid]
            chunk_lifetimes.setdefault(cid, []).append(lt)
            del chunk_load_frame[cid]

all_lifetimes = [lt for v in chunk_lifetimes.values() for lt in v]
ax.hist(all_lifetimes, bins=25, color="tab:purple", alpha=0.7,
        edgecolor="white", linewidth=0.5)
ax.axvline(x=np.mean(all_lifetimes), color="tab:red", linestyle="--",
           linewidth=1.0, label=f"均值 = {np.mean(all_lifetimes):.1f} 帧")
ax.set_xlabel("驻留时间 (帧)")
ax.set_ylabel("淘汰次数")
ax.set_title("(d) 块GPU驻留时长分布 (seq00)")
ax.legend(fontsize=7)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("LRU分块淘汰机制验证 — KITTI seq00 (LRU Chunk Eviction Verification)",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_lru_dashboard.pdf")
plt.close(fig)

print(f"Figures saved to {OUT_DIR.resolve()}:")
for f in sorted(OUT_DIR.glob("fig_*.pdf")):
    print(f"  {f.name}")
