"""Generate thumbnails for examples that do not produce a static figure.

Run from the repository root, for example::

    python doc/generate_gallery_thumbnails.py font_indexing multiprocess

The generated images are intentionally made from Matplotlib primitives so that
they can be recreated or adjusted without relying on an external image editor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT_DIR = Path(__file__).parent / "_static"


def _save(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=100, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _box(ax, xy, width, height, *, facecolor, edgecolor, linewidth=2, radius=0.08):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def font_indexing() -> None:
    """Show characters being resolved to their glyph shapes."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    _box(ax, (0.75, 1.1), 3.25, 4.9, facecolor="#e8f1fb", edgecolor="#4c78a8",
         linewidth=3)
    _box(ax, (6.0, 1.1), 3.25, 4.9, facecolor="#e8f5e9", edgecolor="#3a9d5d",
         linewidth=3)
    ax.text(1.6, 3.4, "A", ha="center", va="center", fontsize=72, weight="bold",
            color="#4c78a8")
    ax.text(3.1, 3.4, "V", ha="center", va="center", fontsize=72, weight="bold",
            color="#4c78a8")
    ax.text(6.85, 3.4, "A", ha="center", va="center", fontsize=72, weight="bold",
            color="#3a9d5d")
    ax.text(8.35, 3.4, "V", ha="center", va="center", fontsize=72, weight="bold",
            color="#3a9d5d")
    ax.add_patch(FancyArrowPatch((4.2, 3.4), (5.8, 3.4), arrowstyle="-|>",
                                 mutation_scale=24, linewidth=3, color="#607d8b"))
    _save(fig, "font_indexing.png")


def multiprocess() -> None:
    """Show data moving between two independent processes."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    _box(ax, (0.45, 1.0), 3.25, 4.9, facecolor="#eaf6ee",
         edgecolor="#35a661", linewidth=3)
    _box(ax, (6.3, 1.0), 3.25, 4.9, facecolor="#edf4fb",
         edgecolor="#4295d1", linewidth=3)
    rng = np.random.default_rng(3)
    xs = np.linspace(1.0, 3.1, 6)
    ys = 2.0 + 1.6 * rng.random(6)
    ax.scatter(xs, ys, s=120, color="#ef6a64", zorder=3)

    ax.add_patch(Rectangle((6.8, 1.55), 2.25, 2.75, facecolor="white",
                           edgecolor="#79b7e8", linewidth=2))
    plot_x = np.linspace(7.1, 8.75, 6)
    plot_y = 1.9 + 1.7 * np.array([0.2, 0.38, 0.28, 0.58, 0.52, 0.85])
    ax.plot(plot_x, plot_y, color="#4c9fd7", linewidth=3)
    ax.scatter(plot_x, plot_y, color="#ef6a64", s=80, zorder=3)

    ax.add_patch(Rectangle((3.9, 3.0), 2.2, 1.2, facecolor="#fff8e6",
                           edgecolor="#e49428", linewidth=2.5))
    ax.add_patch(FancyArrowPatch(
        (3.65, 3.55), (6.25, 3.55), arrowstyle="-|>", mutation_scale=18,
        linewidth=2.2, color="#e49428"))
    ax.scatter(np.linspace(4.15, 5.8, 5), np.full(5, 3.55), s=55,
               color="#e49428", zorder=3)
    _save(fig, "multiprocess.png")


def frame_grabbing() -> None:
    """Show captured frames and their corresponding frame-strip colors."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    x = np.linspace(0.8, 9.2, 300)
    y = 4.25 + 1.25 * np.sin(x * 1.2)
    ax.plot(x, y, color="#8d9be8", linewidth=4, alpha=0.35)
    ax.plot(x, y, color="#7181d8", linewidth=2)
    sample_x = np.array([1.5, 3.75, 6.0, 8.25])
    sample_y = 4.25 + 1.25 * np.sin(sample_x * 1.2)
    colors = ["#ed7c9e", "#f2a26f", "#86d69b", "#bb8ce8"]
    for sx, sy, color in zip(sample_x, sample_y, colors):
        ax.scatter([sx], [sy], s=360, color=color, edgecolor="white",
                   linewidth=2.5, zorder=3)
        ax.add_patch(FancyArrowPatch((sx, sy - 0.42), (sx, 1.95), arrowstyle="-|>",
                                     mutation_scale=17, linewidth=2.0, color=color,
                                     connectionstyle="arc3,rad=0.08"))

    ax.add_patch(Rectangle((0.75, 0.55), 8.5, 1.0, facecolor="#f5f6fb",
                           edgecolor="#9da7c6", linewidth=2))
    for x0, color in zip([1.05, 3.1, 5.15, 7.2], colors):
        ax.add_patch(Rectangle((x0, 0.78), 1.5, 0.55, facecolor=color,
                               edgecolor="white", linewidth=1.5))
    _save(fig, "frame_grabbing.png")


def lasso_selector() -> None:
    """Show a LassoSelector outline and the points it selects."""
    rng = np.random.default_rng(19680801)
    points = rng.random((55, 2))
    polygon = np.array([[0.08, 0.54], [0.18, 0.88], [0.43, 0.96], [0.62, 0.79],
                        [0.67, 0.54], [0.48, 0.40], [0.22, 0.42], [0.08, 0.54]])
    from matplotlib.path import Path

    selected = Path(polygon).contains_points(points)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(points[~selected, 0], points[~selected, 1], s=46,
               color="#8bbbd8", alpha=0.55)
    ax.scatter(points[selected, 0], points[selected, 1], s=58,
               color="#ef6b66", label="selected points")
    ax.plot(polygon[:, 0], polygon[:, 1], color="#d83f47", linewidth=2.5, label="lasso")
    ax.fill(polygon[:, 0], polygon[:, 1], color="#ef6b66", alpha=0.12)
    ax.set_title("LassoSelector", fontsize=18, weight="bold")
    ax.text(0.5, -0.14, "draw a loop to select points", transform=ax.transAxes,
            ha="center", color="#607d8b")
    ax.legend(loc="upper right", frameon=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, "lasso_selector_demo.png")


GENERATORS = {
    "font_indexing": font_indexing,
    "multiprocess": multiprocess,
    "frame_grabbing": frame_grabbing,
    "lasso_selector": lasso_selector,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=sorted(GENERATORS),
                        help="thumbnail names to generate; defaults to all")
    args = parser.parse_args()
    names = args.names or list(GENERATORS)
    for name in names:
        GENERATORS[name]()


if __name__ == "__main__":
    main()
