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
import xml.etree.ElementTree as ElementTree
from xml.etree.ElementTree import Element

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


def svg_size(element: Element) -> tuple[int, int]:
    width = element.get("width")
    height = element.get("height")
    viewbox = element.get("viewBox")

    if width and height:
        return int(width.removesuffix("px")), int(height.removesuffix("px"))

    if viewbox:
        _, _, w, h = map(int, viewbox.split())
        return w, h

    raise ValueError("Could not determine SVG size from element")


class ImageConverter:
    def __init__(self) -> None:
        self._pdf_paths = []
        self._png_paths = []
        self._actions = []
        self._original_size = None

    def _optimize_png(self, png_path: Path) -> None:
        """Removes all metadata from a PNG file and then runs `optipng`"""
        with Image.open(png_path) as img:
            img.save(png_path)  # Should save without metadata
        run_command(OPTIPNG_BIN, str(png_path))

    def _optimize_pdf(self, pdf_path: Path) -> None:
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

    def open_svg(self, svg_path: Path) -> None:
        """Adds an action to open a SVG file."""
        self._original_size = svg_size(ElementTree.parse(svg_path).getroot())
        self._actions.append(f"file-open:{svg_path}")

    def export_pdf(self, pdf_path: Path) -> None:
        """Adds an action to export a PDF file."""
        self._pdf_paths.append(pdf_path)
        self._actions.extend([f"export-filename:{pdf_path}", "export-do"])

    def export_png(
        self,
        png_path: Path,
        width: int | None = None,
        height: int | None = None,
        mode: str | None = None
    ) -> None:
        """Adds an action to export a PNG file."""
        mode = "RGBA_8" if mode is None else mode
        original_width, original_height = self._original_size

        if width is None and height is None:
            width, height = original_width, original_height
        elif width is None:
            width = round(height * (original_width / original_height))
        elif height is None:
            height = round(width * (original_height / original_width))

        self._png_paths.append(png_path)
        self._actions.extend([
            f"export-filename:{png_path}",
            f"export-png-color-mode:{mode}",
            f"export-width:{width}",
            f"export-height:{height}",
            "export-do"
        ])

    def run(self) -> None:
        """Runs all actions and then optimizes PNG/PDF files."""
        actions_str = ";".join(self._actions) + ";"
        run_command(INKSCAPE_BIN, "--actions", actions_str)

        for png_path in self._png_paths:
            self._optimize_png(png_path)

        for pdf_path in self._pdf_paths:
            self._optimize_pdf(pdf_path)


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


def process_toolbar_icon(
    name: str,
    source_dir: Path,
    dest_dir: Path,
    converter: ImageConverter
) -> None:
    svg_path = source_dir / f"{name}.svg"
    pdf_path = dest_dir / f"{name}.pdf"
    png_large_path = dest_dir / f"{name}_large.png"
    png_small_path = dest_dir / f"{name}.png"

    if not has_black_fill_group(svg_path):
        raise ValueError(
            f"SVG file missing <g> with style='fill:black': {svg_path}")

    converter.open_svg(svg_path)
    converter.export_pdf(pdf_path)
    converter.export_png(png_small_path, width=24, mode="GrayAlpha_8")
    converter.export_png(png_large_path, width=48, mode="GrayAlpha_8")


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

    converter = ImageConverter()

    for name in TOOLBAR_ICON_NAMES:
        process_toolbar_icon(name, source_dir, dest_dir, converter)

    for name in ("matplotlib", "matplotlib_small"):
        converter.open_svg(source_dir / f"{name}.svg")
        converter.export_pdf(dest_dir / f"{name}.pdf")
        converter.export_png(dest_dir / f"{name}.png")

    converter.run()


if __name__ == "__main__":
    make_icons()
