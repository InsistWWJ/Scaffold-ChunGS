"""Draw a publication-style schematic for the 3DGS rendering pipeline.

Figure contract:
Core conclusion: 3DGS rendering can be understood as a compact four-stage
pipeline whose critical operations are projection, tile rasterization, and
alpha compositing.
Archetype: schematic-led composite.
Exports: SVG/PDF/TIFF/PNG with editable vector text where supported.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "thesis" / "figures" / "fig_3dgs_pipeline_nature"

COLORS = {
    "ink": "#2f3437",
    "muted": "#667178",
    "line": "#596166",
    "panel_edge": "#c7ced3",
    "panel_fill": "#fbfcfd",
    "panel_head": "#edf3f6",
    "stage_fill": "#f2f6f8",
    "stage_edge": "#b8c4cb",
    "blue": "#9dbbd0",
    "blue_dark": "#486f86",
    "blue_fill": "#edf5fa",
    "warm": "#d7c9b6",
    "warm_dark": "#8f7e67",
    "green": "#a9c9ad",
    "green_dark": "#4f7b58",
    "green_fill": "#eef6f0",
    "orange": "#edc98d",
    "orange_dark": "#9b6a2f",
    "orange_fill": "#fbf1e1",
    "red": "#c85c5c",
    "red_light": "#f5dada",
    "gray": "#d9dee2",
    "grid": "#d4d9dd",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.linewidth": 0.6,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_box(ax, xy, w, h, label, sublabel=None, fc=None, ec=None, lw=0.45, text_x_frac=0.5):
    fc = fc or COLORS["stage_fill"]
    ec = ec or COLORS["stage_edge"]
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=3,
    )
    ax.add_patch(box)
    tx = x + w * text_x_frac
    ax.text(tx, y + h * 0.67, label, ha="center", va="center", weight="bold", color=COLORS["ink"])
    if sublabel:
        ax.text(tx, y + h * 0.35, sublabel, ha="center", va="center", color=COLORS["muted"])
    return box


def arrow(ax, start, end, lw=0.6, color=None, dashed=False, zorder=2, connectionstyle=None):
    color = color or COLORS["line"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=lw,
        color=color,
        linestyle=(0, (3, 2)) if dashed else "solid",
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(patch)


def draw_main_gaussian(ax, cx, cy, scale=1.0):
    ax.add_patch(Ellipse((cx, cy), 0.55 * scale, 0.28 * scale, angle=18, fc=COLORS["blue"], ec=COLORS["blue_dark"], lw=0.6, alpha=0.72, zorder=5))
    ax.plot([cx - 0.33 * scale, cx + 0.33 * scale], [cy - 0.13 * scale, cy + 0.16 * scale], color=COLORS["blue_dark"], lw=0.55, zorder=6)
    ax.plot([cx - 0.27 * scale, cx + 0.27 * scale], [cy + 0.16 * scale, cy - 0.14 * scale], color=COLORS["blue_dark"], lw=0.55, zorder=6)
    ax.add_patch(Circle((cx, cy), 0.025 * scale, fc=COLORS["ink"], ec="none", zorder=7))


def draw_main_projection(ax, x, y):
    ax.add_patch(Ellipse((x, y + 0.12), 0.35, 0.19, angle=12, fc=COLORS["blue"], ec=COLORS["blue_dark"], lw=0.5, alpha=0.7, zorder=5))
    ax.plot([x + 0.18, x + 0.65], [y + 0.13, y - 0.10], color=COLORS["line"], lw=0.5, zorder=5)
    ax.plot([x + 0.18, x + 0.65], [y + 0.13, y + 0.35], color=COLORS["line"], lw=0.5, zorder=5)
    ax.add_patch(Rectangle((x + 0.65, y - 0.20), 0.34, 0.65, fc="white", ec=COLORS["line"], lw=0.5, zorder=5))
    ax.add_patch(Ellipse((x + 0.82, y + 0.10), 0.16, 0.08, fc=COLORS["warm"], ec=COLORS["warm_dark"], lw=0.45, zorder=6))


def draw_main_grid(ax, x, y):
    ax.add_patch(Rectangle((x - 0.45, y - 0.33), 0.90, 0.72, fc="white", ec=COLORS["line"], lw=0.5, zorder=5))
    for dx in [-0.225, 0.0, 0.225]:
        ax.plot([x + dx, x + dx], [y - 0.33, y + 0.39], color=COLORS["grid"], lw=0.35, zorder=5)
    for dy in [-0.15, 0.03, 0.21]:
        ax.plot([x - 0.45, x + 0.45], [y + dy, y + dy], color=COLORS["grid"], lw=0.35, zorder=5)
    ax.add_patch(Ellipse((x, y + 0.04), 0.52, 0.22, fc=COLORS["green"], ec=COLORS["green_dark"], lw=0.45, alpha=0.75, zorder=6))


def draw_main_image(ax, x, y):
    ax.add_patch(Rectangle((x - 0.48, y - 0.34), 0.96, 0.70, fc="white", ec=COLORS["line"], lw=0.5, zorder=5))
    ax.add_patch(Polygon([(x - 0.38, y - 0.22), (x - 0.12, y + 0.05), (x + 0.04, y - 0.08), (x + 0.28, y + 0.16), (x + 0.40, y - 0.22)], fc=COLORS["warm"], ec=COLORS["warm_dark"], lw=0.45, alpha=0.7, zorder=6))
    ax.add_patch(Circle((x + 0.30, y + 0.20), 0.055, fc=COLORS["blue"], ec=COLORS["blue_dark"], lw=0.4, zorder=6))


def draw_mini_projection(ax, x, y):
    ax.add_patch(Ellipse((x, y + 0.08), 0.28, 0.15, angle=12, fc=COLORS["blue"], ec=COLORS["blue_dark"], lw=0.45, alpha=0.72, zorder=5))
    ax.plot([x + 0.14, x + 0.46], [y + 0.08, y - 0.08], color=COLORS["line"], lw=0.42, zorder=5)
    ax.plot([x + 0.14, x + 0.46], [y + 0.08, y + 0.24], color=COLORS["line"], lw=0.42, zorder=5)
    ax.add_patch(Rectangle((x + 0.46, y - 0.17), 0.24, 0.50, fc="white", ec=COLORS["line"], lw=0.45, zorder=5))
    ax.add_patch(Ellipse((x + 0.58, y + 0.04), 0.11, 0.055, fc=COLORS["warm"], ec=COLORS["warm_dark"], lw=0.38, zorder=6))


def draw_mini_image(ax, x, y):
    ax.add_patch(Rectangle((x - 0.36, y - 0.25), 0.72, 0.52, fc="white", ec=COLORS["line"], lw=0.45, zorder=5))
    ax.add_patch(Polygon([(x - 0.28, y - 0.16), (x - 0.10, y + 0.04), (x + 0.03, y - 0.07), (x + 0.21, y + 0.11), (x + 0.30, y - 0.16)], fc=COLORS["warm"], ec=COLORS["warm_dark"], lw=0.38, alpha=0.72, zorder=6))
    ax.add_patch(Circle((x + 0.23, y + 0.16), 0.045, fc=COLORS["blue"], ec=COLORS["blue_dark"], lw=0.35, zorder=6))


def add_stage(ax, xy, w, h, title, subtitle, fill, edge, title_y=0.88):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            fc=fill,
            ec=edge,
            lw=0.38,
            zorder=2,
        )
    )
    ax.text(x + w / 2, y + h * title_y, title, ha="center", va="center", weight="bold", fontsize=6.9, color=COLORS["ink"], zorder=8)
    ax.text(x + w / 2, y + h * 0.10, subtitle, ha="center", va="center", fontsize=5.8, color=COLORS["muted"], zorder=8)


def draw_parameter_cylinder(ax, x, y, w=0.92, h=0.62):
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc="#d9e8f1", ec=COLORS["blue_dark"], lw=0.55, zorder=4))
    ax.add_patch(Ellipse((x, y + h / 2), w, 0.20, fc="#c6dcea", ec=COLORS["blue_dark"], lw=0.55, zorder=5))
    ax.add_patch(Ellipse((x, y - h / 2), w, 0.20, fc="#d9e8f1", ec=COLORS["blue_dark"], lw=0.55, zorder=3))
    labels = [r"$\mu$", r"$\Sigma$", r"$\alpha$", r"$\mathrm{SH}$"]
    for i, lab in enumerate(labels):
        ax.text(x - 0.36 + 0.24 * i, y + 0.02, lab, ha="center", va="center", fontsize=6.6, color=COLORS["ink"], zorder=7)


def draw_camera_with_pose(ax, x, y):
    ax.add_patch(Polygon([(x - 0.42, y - 0.22), (x + 0.05, y), (x - 0.42, y + 0.22)], closed=True, fc="#f4dfc6", ec=COLORS["orange_dark"], lw=0.55, zorder=5))
    ax.add_patch(Rectangle((x + 0.04, y - 0.29), 0.44, 0.58, fc="white", ec=COLORS["orange_dark"], lw=0.55, zorder=5))
    ax.add_patch(Circle((x + 0.26, y), 0.07, fc=COLORS["orange"], ec=COLORS["orange_dark"], lw=0.45, zorder=6))
    ax.text(x + 0.58, y + 0.20, r"$T_{cw}$", ha="left", va="center", fontsize=7.3, color=COLORS["ink"], zorder=7)
    ax.text(x + 0.58, y - 0.07, r"$[R\ |\ t]$", ha="left", va="center", fontsize=5.9, color=COLORS["muted"], zorder=7)


def draw_projection_process(ax, x, y):
    ax.add_patch(Ellipse((x - 0.34, y + 0.08), 0.50, 0.26, angle=18, fc=COLORS["blue"], ec=COLORS["blue_dark"], lw=0.55, alpha=0.70, zorder=5))
    ax.plot([x - 0.10, x + 0.36], [y + 0.08, y - 0.16], color=COLORS["line"], lw=0.48, zorder=5)
    ax.plot([x - 0.10, x + 0.36], [y + 0.08, y + 0.30], color=COLORS["line"], lw=0.48, zorder=5)
    ax.add_patch(Rectangle((x + 0.36, y - 0.28), 0.42, 0.72, fc="white", ec=COLORS["line"], lw=0.52, zorder=5))
    ax.add_patch(Ellipse((x + 0.57, y + 0.06), 0.26, 0.12, fc=COLORS["orange"], ec=COLORS["orange_dark"], lw=0.45, alpha=0.75, zorder=6))
    ax.text(x + 0.10, y - 0.36, r"$J$", ha="center", va="center", fontsize=8.2, weight="bold", color=COLORS["orange_dark"], zorder=8)


def draw_raster_sort(ax, x, y):
    ax.add_patch(Rectangle((x - 0.55, y - 0.36), 1.10, 0.72, fc="white", ec=COLORS["green_dark"], lw=0.50, zorder=5))
    for dx in [-0.275, 0, 0.275]:
        ax.plot([x + dx, x + dx], [y - 0.36, y + 0.36], color=COLORS["grid"], lw=0.30, zorder=5)
    for dy in [-0.18, 0, 0.18]:
        ax.plot([x - 0.55, x + 0.55], [y + dy, y + dy], color=COLORS["grid"], lw=0.30, zorder=5)
    ax.add_patch(Ellipse((x - 0.16, y + 0.06), 0.42, 0.18, angle=8, fc=COLORS["green"], ec=COLORS["green_dark"], lw=0.45, alpha=0.75, zorder=6))
    ax.add_patch(Ellipse((x + 0.22, y - 0.12), 0.34, 0.15, angle=-12, fc=COLORS["blue"], ec=COLORS["blue_dark"], lw=0.42, alpha=0.52, zorder=6))
    arrow(ax, (x - 0.45, y + 0.47), (x + 0.45, y + 0.47), lw=0.45, color=COLORS["green_dark"], zorder=8)
    ax.text(x, y + 0.63, r"$z$ 排序", ha="center", va="center", fontsize=6.5, color=COLORS["green_dark"], zorder=8)


def draw_alpha_ellipses(ax, x, y):
    ellipses = [
        (x - 0.18, y - 0.02, COLORS["blue"], COLORS["blue_dark"], 0.55),
        (x + 0.02, y + 0.06, COLORS["green"], COLORS["green_dark"], 0.56),
        (x + 0.22, y - 0.02, COLORS["orange"], COLORS["orange_dark"], 0.58),
    ]
    for cx, cy, fc, ec, alpha in ellipses:
        ax.add_patch(Ellipse((cx, cy), 0.62, 0.30, angle=15, fc=fc, ec=ec, lw=0.45, alpha=alpha, zorder=6))


def draw_film_frame(ax, x, y):
    ax.add_patch(Rectangle((x - 0.54, y - 0.38), 1.08, 0.76, fc="white", ec=COLORS["line"], lw=0.55, zorder=5))
    for i in range(5):
        yy = y - 0.30 + i * 0.15
        ax.add_patch(Rectangle((x - 0.50, yy), 0.08, 0.055, fc=COLORS["gray"], ec="none", zorder=6))
        ax.add_patch(Rectangle((x + 0.42, yy), 0.08, 0.055, fc=COLORS["gray"], ec="none", zorder=6))
    ax.add_patch(Polygon([(x - 0.30, y - 0.18), (x - 0.08, y + 0.06), (x + 0.07, y - 0.06), (x + 0.28, y + 0.18), (x + 0.36, y - 0.18)], fc=COLORS["warm"], ec=COLORS["warm_dark"], lw=0.42, alpha=0.75, zorder=6))
    ax.add_patch(Circle((x + 0.30, y + 0.22), 0.055, fc=COLORS["blue"], ec=COLORS["blue_dark"], lw=0.38, zorder=7))


def draw_training_node(ax, x, y):
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.94, y - 0.28),
            1.88,
            0.56,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=COLORS["red_light"],
            ec=COLORS["red"],
            lw=0.50,
            zorder=5,
        )
    )
    ax.text(x, y + 0.08, "训练", ha="center", va="center", weight="bold", fontsize=7.0, color=COLORS["ink"], zorder=7)
    ax.text(x, y - 0.13, r"$\partial\mathcal{L}/\partial\mu,\ \partial\mathcal{L}/\partial\Sigma,\ \partial\mathcal{L}/\partial\alpha$", ha="center", va="center", fontsize=5.3, color=COLORS["red"], zorder=7)


def draw_projection_detail(ax, x0, y0, w, h):
    ax.add_patch(Rectangle((x0, y0 + h - 0.42), w, 0.42, fc=COLORS["panel_head"], ec="none", zorder=2))
    ax.text(x0 + 0.17, y0 + h - 0.21, "a  Projection", ha="left", va="center", weight="bold", color=COLORS["ink"])
    draw_main_gaussian(ax, x0 + 0.70, y0 + 0.98, scale=1.12)
    ax.plot([x0 + 1.04, x0 + 1.66], [y0 + 0.98, y0 + 0.72], color=COLORS["line"], lw=0.48)
    ax.plot([x0 + 1.04, x0 + 1.66], [y0 + 0.98, y0 + 1.25], color=COLORS["line"], lw=0.48)
    ax.add_patch(Rectangle((x0 + 1.66, y0 + 0.46), 0.80, 1.10, fc="white", ec=COLORS["line"], lw=0.5))
    ax.add_patch(Ellipse((x0 + 2.06, y0 + 1.02), 0.42, 0.20, fc=COLORS["warm"], ec=COLORS["warm_dark"], lw=0.45, alpha=0.75))
    arrow(ax, (x0 + 1.17, y0 + 0.98), (x0 + 1.64, y0 + 0.98), lw=0.55)
    ax.text(x0 + 0.33, y0 + 0.28, r"$\Sigma_{2D}=J R\Sigma R^\top J^\top$", ha="left", va="bottom", fontsize=6.8, color=COLORS["ink"])


def draw_tile_detail(ax, x0, y0, w, h):
    ax.add_patch(Rectangle((x0, y0 + h - 0.42), w, 0.42, fc=COLORS["panel_head"], ec="none", zorder=2))
    ax.text(x0 + 0.17, y0 + h - 0.21, "b  Tile rasterization", ha="left", va="center", weight="bold", color=COLORS["ink"])
    gx, gy = x0 + 0.45, y0 + 0.55
    gw, gh = w - 0.90, h - 0.96
    ax.add_patch(Rectangle((gx, gy), gw, gh, fc="white", ec=COLORS["line"], lw=0.48))
    for i in range(1, 7):
        xx = gx + gw * i / 7
        ax.plot([xx, xx], [gy, gy + gh], color=COLORS["grid"], lw=0.26)
    for j in range(1, 5):
        yy = gy + gh * j / 5
        ax.plot([gx, gx + gw], [yy, yy], color=COLORS["grid"], lw=0.26)
    ax.add_patch(Ellipse((gx + gw * 0.36, gy + gh * 0.62), 1.05, 0.42, angle=8, fc=COLORS["green"], ec=COLORS["green_dark"], lw=0.45, alpha=0.75))
    ax.add_patch(Ellipse((gx + gw * 0.68, gy + gh * 0.38), 0.80, 0.32, angle=-12, fc=COLORS["blue"], ec=COLORS["blue_dark"], lw=0.45, alpha=0.55))
    ax.text(gx + gw * 0.52, gy - 0.14, "screen-space bins", ha="center", va="top", fontsize=6.8, color=COLORS["muted"])


def draw_blending_detail(ax, x0, y0, w, h):
    ax.add_patch(Rectangle((x0, y0 + h - 0.42), w, 0.42, fc=COLORS["panel_head"], ec="none", zorder=2))
    ax.text(x0 + 0.17, y0 + h - 0.21, "c  Alpha blending", ha="left", va="center", weight="bold", color=COLORS["ink"])
    layers = [(COLORS["blue"], 0.0), (COLORS["green"], 0.28), (COLORS["warm"], 0.56)]
    for color, off in layers:
        ax.add_patch(Rectangle((x0 + 0.48 + off, y0 + 0.70 + off * 0.50), 0.92, 0.64, fc=color, ec=COLORS["line"], lw=0.42, alpha=0.72))
    ax.text(x0 + 1.18, y0 + 1.86, r"$\sum_i T_i\alpha_i c_i$", ha="center", va="center", fontsize=7.0, color=COLORS["ink"])
    arrow(ax, (x0 + 1.84, y0 + 1.15), (x0 + 2.42, y0 + 1.15), lw=0.55)
    draw_main_image(ax, x0 + 2.95, y0 + 1.15)
    ax.text(x0 + 2.95, y0 + 0.30, "composited image", ha="center", va="top", fontsize=6.8, color=COLORS["muted"])


def make_figure() -> plt.Figure:
    setup_style()
    fig = plt.figure(figsize=(8.2, 4.95))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 8.25)
    ax.axis("off")

    def float_card(x, y, w, h, fill, edge, title):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.025,rounding_size=0.07",
                fc=fill,
                ec=edge,
                lw=0.34,
                zorder=2,
            )
        )
        ax.text(x + 0.18, y + h - 0.26, title, ha="left", va="center", fontsize=5.8, weight="bold", color=edge, zorder=8)

    # ------------------------------------------------------------------
    # CVPR-style trunk: four dominant modules with lightweight detail insets
    # ------------------------------------------------------------------
    trunk_y = 3.35
    main_h = 2.05
    main = {
        "gauss": (0.62, trunk_y, 2.45, main_h),
        "proj": (4.30, trunk_y + 0.18, 2.78, main_h + 0.18),
        "rast": (8.22, trunk_y, 2.72, main_h),
        "image": (12.06, trunk_y + 0.08, 2.50, main_h - 0.08),
    }

    add_stage(ax, (main["gauss"][0], main["gauss"][1]), main["gauss"][2], main["gauss"][3], "Gaussians", "高斯参数表示", COLORS["blue_fill"], COLORS["blue_dark"], title_y=0.86)
    add_stage(ax, (main["proj"][0], main["proj"][1]), main["proj"][2], main["proj"][3], "Projection", "三维到二维投影", COLORS["orange_fill"], COLORS["orange_dark"], title_y=0.87)
    add_stage(ax, (main["rast"][0], main["rast"][1]), main["rast"][2], main["rast"][3], "Rasterization", "tile 光栅化", COLORS["green_fill"], COLORS["green_dark"], title_y=0.86)
    add_stage(ax, (main["image"][0], main["image"][1]), main["image"][2], main["image"][3], "Image", "渲染图像", "#f3f5f6", COLORS["line"], title_y=0.86)

    draw_parameter_cylinder(ax, 1.84, 4.30, w=1.22, h=0.64)
    draw_projection_process(ax, 5.68, 4.35)
    draw_raster_sort(ax, 9.58, 4.18)
    draw_film_frame(ax, 13.31, 4.22)

    arrow(ax, (3.12, 4.40), (4.24, 4.48), lw=0.70)
    arrow(ax, (7.12, 4.47), (8.16, 4.40), lw=0.70)
    arrow(ax, (10.98, 4.38), (12.00, 4.38), lw=0.70)

    # Projection detail: camera pose and local covariance projection.
    float_card(4.12, 5.98, 3.20, 1.00, COLORS["orange_fill"], COLORS["orange_dark"], "局部 detail: 位姿与投影")
    draw_camera_with_pose(ax, 4.78, 6.42)
    ax.text(6.24, 6.45, r"$\Sigma_{2D}=J R\Sigma R^\top J^\top$", ha="center", va="center", fontsize=6.1, color=COLORS["ink"], zorder=8)
    arrow(ax, (5.92, 5.98), (5.92, 5.72), lw=0.46, color=COLORS["orange_dark"])

    # Gaussian detail: parameters are retained as a compact side branch.
    float_card(0.62, 1.70, 3.02, 1.18, COLORS["blue_fill"], COLORS["blue_dark"], "局部 detail: 可优化参数")
    draw_main_gaussian(ax, 1.18, 2.20, scale=1.10)
    ax.text(2.30, 2.36, r"$\mu$: 位置", ha="center", va="center", fontsize=5.9, color=COLORS["ink"], zorder=8)
    ax.text(2.30, 2.09, r"$\Sigma$: 协方差", ha="center", va="center", fontsize=5.9, color=COLORS["ink"], zorder=8)
    ax.text(2.30, 1.82, r"$\alpha,\ \mathrm{SH}$: 外观", ha="center", va="center", fontsize=5.9, color=COLORS["ink"], zorder=8)
    arrow(ax, (1.90, 2.88), (1.88, 3.30), lw=0.46, color=COLORS["blue_dark"])

    # Rasterization/detail branch: sorted tile bins.
    float_card(8.36, 1.64, 3.05, 1.26, COLORS["green_fill"], COLORS["green_dark"], "局部 detail: tile 与深度顺序")
    ax.add_patch(Rectangle((8.70, 1.92), 1.06, 0.64, fc="white", ec=COLORS["green_dark"], lw=0.45, zorder=5))
    for dx in [8.96, 9.23, 9.49]:
        ax.plot([dx, dx], [1.92, 2.56], color=COLORS["grid"], lw=0.28, zorder=5)
    for dy in [2.13, 2.34]:
        ax.plot([8.70, 9.76], [dy, dy], color=COLORS["grid"], lw=0.28, zorder=5)
    ax.text(10.55, 2.32, r"$z_1<z_2<z_3$", ha="center", va="center", fontsize=6.2, color=COLORS["green_dark"], zorder=8)
    arrow(ax, (9.82, 2.24), (10.20, 2.24), lw=0.46, color=COLORS["green_dark"])
    arrow(ax, (9.70, 2.90), (9.62, 3.30), lw=0.46, color=COLORS["green_dark"])

    # Image/detail branch: alpha blending and differentiable training.
    float_card(11.72, 1.32, 3.22, 1.60, COLORS["green_fill"], COLORS["green_dark"], "局部 detail: 透明度混合")
    draw_alpha_ellipses(ax, 12.56, 2.15)
    ax.text(13.86, 2.18, r"$\mathbf{C}=\sum_i c_i\alpha_iT_i$", ha="center", va="center", fontsize=6.2, color=COLORS["ink"], zorder=8)
    arrow(ax, (13.10, 2.92), (13.18, 3.28), lw=0.46, color=COLORS["green_dark"])

    draw_training_node(ax, 6.02, 1.03)
    arrow(ax, (13.30, 3.32), (7.06, 1.22), lw=0.62, color=COLORS["red"], dashed=True, zorder=1, connectionstyle="arc3,rad=-0.14")
    arrow(ax, (5.00, 0.96), (2.10, 3.30), lw=0.62, color=COLORS["red"], dashed=True, zorder=1, connectionstyle="arc3,rad=-0.18")
    ax.text(9.86, 1.86, r"$\mathcal{L}(I,\hat I)$", ha="center", va="center", fontsize=6.4, color=COLORS["red"], zorder=8)
    ax.text(4.05, 1.08, r"$\partial\mathcal{L}/\partial\mu,\ \partial\mathcal{L}/\partial\Sigma,\ \partial\mathcal{L}/\partial\alpha$", ha="center", va="center", fontsize=5.4, color=COLORS["red"], zorder=8)

    ax.text(0.62, 7.78, "3DGS 可微渲染主干", ha="left", va="center", weight="bold", fontsize=8.5, color=COLORS["ink"])
    ax.text(0.62, 7.46, "主模块强调渲染路径，浮动 detail 解释关键参数、投影、排序与混合。", ha="left", va="center", fontsize=6.2, color=COLORS["muted"])
    return fig


def save_all(fig: plt.Figure) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}.tiff", dpi=600, bbox_inches="tight")


def qa_exports() -> None:
    for suffix in [".svg", ".pdf", ".png", ".tiff"]:
        path = OUT.with_suffix(suffix)
        if not path.exists() or path.stat().st_size < 10_000:
            raise RuntimeError(f"Export failed or too small: {path}")

    from PIL import Image

    png = OUT.with_suffix(".png")
    with Image.open(png) as img:
        arr = np.asarray(img.convert("L"))
    if arr.std() < 3:
        raise RuntimeError("PNG QA failed: figure appears blank or nearly uniform")


def main() -> None:
    fig = make_figure()
    save_all(fig)
    plt.close(fig)
    qa_exports()
    print(f"Wrote {OUT}.svg/.pdf/.png/.tiff")


if __name__ == "__main__":
    main()
