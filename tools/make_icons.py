#!/usr/bin/env python
"""
Generates PDF and PNG variants of the Matplotlib app icon and toolbar icons.
"""

# ------------------------------------------------------------------------------------
#
# Toolbar Icon Notes:
#
# Historically, the toolbar icons were generated directly from the FontAwesome 4.7
# font (released under the SIL OFL 1.1 license):
#
# Code Point  | Icon Name
# ------------+---------------------
# 0xf015      | home
# 0xf060      | back
# 0xf061      | forward
# 0xf002      | zoom_to_rect
# 0xf047      | move
# 0xf0c7      | filesave
# 0xf1de      | subplots
# 0xf201      | qt4_editor_options
# 0xf128      | help
#
# This resulted in fuzzy edges (due to scaling) and vertical misalignment.
#
# The glyphs were redrawn by hand and pixel-aligned to a 48x48 canvas.
#
# ------------------------------------------------------------------------------------
#
# App Icon Notes:
#
# The app icons were generated using a modified logos2.py.
#
# For the 256px icon, the following parameters were used:
# height_px=256, lw_bars=1.5, lw_grid=2, lw_border=4, rgrid=[1, 3, 5, 7],
# dpi=72, ax_positions=(0.0891, 0.0891, 0.8218, 0.8218)
#
# For the 32px icon:
# height_px=32, lw_bars=1, lw_grid=0, lw_border=0, rgrid=[1, 3, 5, 7],
# dpi=72, ax_positions=(0.0891, 0.0891, 0.8218, 0.8218)
#
# Both matplotlib.svg and matplotlib_small.svg were manually cleaned up and
# optimized using svgo
#
# ------------------------------------------------------------------------------------
#
# Conversion and Compression Notes:
#
# This script requires `inkscape` and `optipng` binaries. These were chosen due to
# being dependencies of the documentation build process.
#
# Optionally, this script uses `pngcrush` if present to further optimize PNG files.
#
# Output PNG files have the following pixel dimensions:
#
# matplotlib.png           256
# matplotlib_small.png     32
# {toolbar_name}.png       24
# {toolbar_name}_large.png 48
#
# ------------------------------------------------------------------------------------

import argparse
import subprocess
import shutil
from pathlib import Path

from lxml import etree
import pikepdf
from PIL import Image


INKSCAPE_BIN = "inkscape"
OPTIPNG_BIN = "optipng"

DEFAULT_IMAGES_PATH = "../lib/matplotlib/mpl-data/images"

TOOLBAR_ICON_NAMES = [
    "home",
    "back",
    "forward",
    "zoom_to_rect",
    "move",
    "filesave",
    "subplots",
    "qt4_editor_options",
    "help"
]


def run_command(*args: str) -> None:
    """Run a subprocess and raise RuntimeError with stderr on failure."""
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(err.stderr.strip()) from err


def run_inkscape(actions: list[str]) -> None:
    actions_str = "; ".join(actions) + ";"
    run_command(INKSCAPE_BIN, "--actions", actions_str)


def optimize_png(png_path: Path) -> None:
    """Removes all metadata from a PNG file and then runs `optipng`"""
    img = Image.open(png_path)
    img.save(png_path)  # Should save without metadata
    run_command(OPTIPNG_BIN, str(png_path))


def optimize_pdf(pdf_path: Path) -> None:
    """Removes all metadata from a PDF file."""
    with pikepdf.Pdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        # Each PDF gets a fresh CreationDate when we run make_icons.py
        # This dirties the git working tree directory, so strip all metadata
        # that could change.
        for key in ("/CreationDate", "/ModDate", "/Creator", "/Producer"):
            if key in pdf.docinfo:
                del pdf.docinfo[key]

        if "/Metadata" in pdf.Root:
            del pdf.Root["/Metadata"]

        pdf.save(pdf_path, deterministic_id=True)


def has_black_fill_group(svg_path: Path) -> bool:
    """Checks that the SVG file has a root-level group with 'fill:black'"""
    tree = etree.parse(svg_path)
    root = tree.getroot()
    for child in root:
        if etree.QName(child).localname != "g":
            continue
        style = child.get("style", "")
        declarations = [d.strip() for d in style.split(";") if d.strip()]
        if "fill:black" in declarations or style.strip() == "fill:black":
            return True
    return False


def process_toolbar_icons(
    source_dir: Path,
    dest_dir: Path,
    png_paths: list[Path],
    pdf_paths: list[Path],
) -> None:
    actions = []

    def add_actions(name: str) -> None:
        svg_path = source_dir / f"{name}.svg"
        pdf_path = dest_dir / f"{name}.pdf"
        png_large_path = dest_dir / f"{name}_large.png"
        png_small_path = dest_dir / f"{name}.png"

        if not has_black_fill_group(svg_path):
            raise ValueError(
                f"SVG file missing <g> with style='fill:black': {svg_path}")

        actions.extend([
            f"file-open:{svg_path}",
            f"export-filename:{pdf_path}; export-do",
            f"export-filename:{png_large_path}; "
                f"export-png-color-mode:GrayAlpha_8; "
                f"export-width:48; export-height:48; export-do",
            f"export-filename:{png_small_path}; "
                f"export-png-color-mode:GrayAlpha_8; "
                f"export-width:24; export-height:24; export-do",
        ])

        png_paths.extend([png_small_path, png_large_path])
        pdf_paths.append(pdf_path)

    for toolbar_icon_name in TOOLBAR_ICON_NAMES:
        add_actions(toolbar_icon_name)

    run_inkscape(actions)


def process_matplotlib_icons(
    source_dir: Path,
    dest_dir: Path,
    png_paths: list[Path],
    pdf_paths: list[Path]
) -> None:
    actions = []

    def add_actions(name: str) -> None:
        svg_path = source_dir / f"{name}.svg"
        pdf_path = dest_dir / f"{name}.pdf"
        png_path = dest_dir / f"{name}.png"

        actions.extend([
            f"file-open:{svg_path}",
            f"export-filename:{pdf_path}; export-do;",
            f"export-filename:{png_path}; export-do;"
        ])

        png_paths.append(png_path)
        pdf_paths.append(pdf_path)

    add_actions("matplotlib")
    add_actions("matplotlib_small")

    run_inkscape(actions)


def make_icons() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and convert SVG icons to PNG/PDF.")

    parser.add_argument(
        "-s", "--source-dir",
        type=Path,
        default=Path(__file__).parent / DEFAULT_IMAGES_PATH,
        help="Directory where to read the SVG files.")
    parser.add_argument(
        "-d", "--dest-dir",
        type=Path,
        default=Path(__file__).parent / DEFAULT_IMAGES_PATH,
        help="Directory where to write the PNG/PDF files.")
    args = parser.parse_args()

    source_dir = args.source_dir
    dest_dir = args.dest_dir

    if not source_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {source_dir}")
    if not dest_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {dest_dir}")

    if shutil.which(INKSCAPE_BIN) is None:
        raise FileNotFoundError(f"Could not locate the `{INKSCAPE_BIN}` binary")
    if shutil.which(OPTIPNG_BIN) is None:
        raise FileNotFoundError(f"Could not locate the `{OPTIPNG_BIN}` binary")

    png_paths = []
    pdf_paths = []
    process_toolbar_icons(source_dir, dest_dir, png_paths, pdf_paths)
    process_matplotlib_icons(source_dir, dest_dir, png_paths, pdf_paths)

    for png_path in png_paths:
        optimize_png(png_path)

    for pdf_path in pdf_paths:
        optimize_pdf(pdf_path)


if __name__ == "__main__":
    make_icons()
