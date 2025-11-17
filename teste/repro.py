# repro.py
from __future__ import annotations

import sys, os, time, csv
from pathlib import Path
import numpy as np
import matplotlib as mpl

BASE = Path(__file__).resolve().parent

mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0

# Monkey-patch SVG
from matplotlib.backends.backend_svg import RendererSVG as _RSVG

_svg_orig = _RSVG.draw_path_collection


def _svg_patch(
    self,
    gc,
    master_transform,
    paths,
    all_transforms,
    offsets,
    offset_trans,
    facecolors,
    edgecolors,
    linewidths,
    linestyles,
    antialiaseds,
    urls,
    offset_position,
    *,
    hatchcolors=None,
):
    if (
        mpl.rcParams.get("path.simplify", False)
        and mpl.rcParams.get("path.simplify_threshold", 0) > 0
        and paths
    ):
        try:
            paths = [p.cleaned(simplify=True) for p in paths]
        except Exception:
            pass

    return _svg_orig(
        self,
        gc,
        master_transform,
        paths,
        all_transforms,
        offsets,
        offset_trans,
        facecolors,
        edgecolors,
        linewidths,
        linestyles,
        antialiaseds,
        urls,
        offset_position,
        hatchcolors=hatchcolors,
    )


_RSVG.draw_path_collection = _svg_patch

# Monkey-patch PDF
from matplotlib.backends.backend_pdf import RendererPdf as _RPDF

_pdf_orig = _RPDF.draw_path_collection


def _pdf_patch(
    self,
    gc,
    master_transform,
    paths,
    all_transforms,
    offsets,
    offset_trans,
    facecolors,
    edgecolors,
    linewidths,
    linestyles,
    antialiaseds,
    urls,
    offset_position,
    *,
    hatchcolors=None,
):
    if (
        mpl.rcParams.get("path.simplify", False)
        and mpl.rcParams.get("path.simplify_threshold", 0) > 0
        and paths
    ):
        try:
            paths = [p.cleaned(simplify=True) for p in paths]
        except Exception:
            pass

    return _pdf_orig(
        self,
        gc,
        master_transform,
        paths,
        all_transforms,
        offsets,
        offset_trans,
        facecolors,
        edgecolors,
        linewidths,
        linestyles,
        antialiaseds,
        urls,
        offset_position,
        hatchcolors=hatchcolors,
    )


_RPDF.draw_path_collection = _pdf_patch

print(">>> monkey-patch SVG/PDF ON")

import matplotlib.pyplot as plt

T_MAX = 20_000
SEED = 424242
mpl.rcParams["path.simplify"] = True
mpl.rcParams["path.simplify_threshold"] = 1.0


def count_tag(path: Path, tag: str) -> int:
    tag = tag.lower()
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if tag in line.lower())
    except Exception:
        return -1


def save_and_stats(fig, fname_svg: Path, *, save_png: bool = True, png_dpi: int = 100):
    """
    Saves SVG to fname_svg, optionally saves PNG with the same name,
    and returns metrics based on the SVG.
    """
    # check if dir exists
    fname_svg.parent.mkdir(parents=True, exist_ok=True)

    # SVG
    fig.savefig(fname_svg)

    # PNG
    if save_png:
        png_path = fname_svg.with_suffix(".png")
        fig.savefig(png_path, dpi=png_dpi)

    plt.close(fig)

    size = fname_svg.stat().st_size
    images = count_tag(fname_svg, "<image")
    paths = count_tag(fname_svg, "<path")
    return size, images, paths

def scenario_lines():
    """Two Line2D (min/max)."""
    rng = np.random.default_rng(SEED)
    t = np.arange(1, T_MAX + 1, dtype=float)
    y = rng.normal(size=(10, T_MAX))
    y1, y2 = y.min(0), y.max(0)

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(t, y1, color="tab:blue", alpha=0.75)
    ax.plot(t, y2, color="tab:blue", alpha=0.75)
    ax.set_xlim(1, T_MAX)

    svg = BASE / "plot_minmax.svg"
    size, images, paths = save_and_stats(fig, svg)


    return dict(
        file=str(svg),
        size=size,
        svg_images=images,
        svg_paths=paths,
        verts_before=None,
        verts_after=None,
        note="lines",
    )


def scenario_fill():
    """fill_between."""
    rng = np.random.default_rng(SEED)
    t = np.arange(1, T_MAX + 1, dtype=float)
    y = rng.normal(size=(10, T_MAX))
    y1, y2 = y.min(0), y.max(0)

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    coll = ax.fill_between(t, y1, y2, color="tab:blue", alpha=0.75)
    ax.set_xlim(1, T_MAX)

    # Diagnosis: number of vertices before/after cleaned()
    verts_before = verts_after = None
    try:
        p = coll.get_paths()[0]
        verts_before = len(p.vertices)
        verts_after = len(p.cleaned(simplify=True).vertices)
    except Exception:
        pass

    svg = BASE / "plot_fill.svg"
    pdf = BASE / "cleaplot_fill.pdf"

    size_svg, images_svg, paths_svg = save_and_stats(fig, svg)
    # png: plot_fill.png (via save_and_stats)

    # new fig for PDF (no PNG here, just pdf)
    fig2 = plt.figure(dpi=200)
    ax2 = fig2.add_subplot(111)
    ax2.fill_between(t, y1, y2, color="tab:blue", alpha=0.75)
    ax2.set_xlim(1, T_MAX)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(pdf)
    plt.close(fig2)

    size_pdf = pdf.stat().st_size

    return (
        dict(
            file=str(svg),
            size=size_svg,
            svg_images=images_svg,
            svg_paths=paths_svg,
            verts_before=verts_before,
            verts_after=verts_after,
            note="fill (SVG)",
        ),
        dict(
            file=str(pdf),
            size=size_pdf,
            svg_images=None,
            svg_paths=None,
            verts_before=verts_before,
            verts_after=verts_after,
            note="fill (PDF)",
        ),
    )

def main():
    pyver = (
        f"{sys.version_info.major}."
        f"{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )
    mplver = mpl.__version__
    backend = mpl.get_backend()
    ts = time.strftime("%Y%m%d-%H%M%S")

    rows = []

    rows.append(
        dict(
            python=pyver,
            mpl=mplver,
            backend=backend,
            tmax=T_MAX,
            thresh=mpl.rcParams["path.simplify_threshold"],
            timestamp=ts,
            **scenario_lines(),
        )
    )

    svg_row, pdf_row = scenario_fill()
    rows.append(
        dict(
            python=pyver,
            mpl=mplver,
            backend=backend,
            tmax=T_MAX,
            thresh=mpl.rcParams["path.simplify_threshold"],
            timestamp=ts,
            **svg_row,
        )
    )
    rows.append(
        dict(
            python=pyver,
            mpl=mplver,
            backend=backend,
            tmax=T_MAX,
            thresh=mpl.rcParams["path.simplify_threshold"],
            timestamp=ts,
            **pdf_row,
        )
    )

    # CSV on standards output
    header = [
        "python",
        "mpl",
        "backend",
        "tmax",
        "thresh",
        "timestamp",
        "file",
        "size",
        "svg_images",
        "svg_paths",
        "verts_before",
        "verts_after",
        "note",
    ]
    print(",".join(header))
    for r in rows:
        print(",".join(str(r.get(k, "")) for k in header))


if __name__ == "__main__":
    main()
