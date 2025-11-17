from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as patheffects


def _render_png_array(make_fig, threshold: float) -> np.ndarray:
    """
    Renders the figure returned by make_fig() with a given path.simplify_threshold, 
    and returns a NumPy array of the image (RGBA).
    """
    mpl.rcParams["path.simplify"] = True
    mpl.rcParams["path.simplify_threshold"] = threshold

    fig = make_fig()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = mpimg.imread(buf)  # (H, W, 4)
    return img


def render_thr0_vs_thr1(make_fig, output_path: Path, title_prefix: str = "") -> None:
    """
    3-panel PNG: thr=0.0, thr=1.0, absolute difference.
    """
    img0 = _render_png_array(make_fig, 0.0)
    img1 = _render_png_array(make_fig, 1.0)

    # assure same size
    h = min(img0.shape[0], img1.shape[0])
    w = min(img0.shape[1], img1.shape[1])
    img0 = img0[:h, :w, :]
    img1 = img1[:h, :w, :]

    diff = np.abs(img1[..., :3] - img0[..., :3]).max(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    if not title_prefix:
        title_prefix = ""

    axes[0].imshow(img0)
    axes[0].set_title(f"{title_prefix} thr=0.0", fontsize=16)
    axes[0].axis("off")

    axes[1].imshow(img1)
    axes[1].set_title(f"{title_prefix} thr=1.0", fontsize=16)
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="magma")
    axes[2].set_title("abs diff", fontsize=16)
    axes[2].axis("off")

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[ok] saved {output_path}")


# TESTS

def case_hatch_shadow():
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

    x = [1, 2, 3]
    y = [1, 5, 3]
    ax.plot(x, y, lw=10, color="tab:blue",
            path_effects=[patheffects.withStroke(linewidth=20, foreground="k")])

    ax.text(2, 5, "testing", fontsize=20, color="red",
            ha="center", va="bottom",
            path_effects=[patheffects.withStroke(linewidth=8, foreground="k")])

    circ = plt.Circle((2, 3), 0.5, color="red", ec="none")
    ax.add_patch(circ)

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 6)

    return fig


def case_colour_contour():
    x = np.linspace(-3, 5, 300)
    y = np.linspace(-3, 5, 300)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(2 * X) + np.cos(3 * Y)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    cs = ax.contour(X, Y, Z, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax)
    return fig


def case_scatter_offset():
    # no RNG
    x = np.linspace(0, 10, 2000)
    y = np.sin(x) + 0.1 * np.cos(20 * x)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.scatter(x, y, s=5, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 2)
    return fig



def main():
    outdir = Path(__file__).parent

    tests = [
        ("hatch_shadow", case_hatch_shadow),
        ("colour_contour", case_colour_contour),
        ("scatter_offset", case_scatter_offset),
    ]

    for name, fn in tests:
        outfile = outdir / f"compare_{name}_thr0_vs_thr1.png"
        print(f">>> gerando {outfile.name}")
        render_thr0_vs_thr1(fn, outfile, title_prefix=name)


if __name__ == "__main__":
    main()
