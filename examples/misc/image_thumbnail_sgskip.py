"""
===============
Image Thumbnail
===============

You can use Matplotlib to generate thumbnails from existing images.
Matplotlib relies on Pillow_ for reading images, and thus supports all formats
supported by Pillow.

.. _Pillow: https://python-pillow.org/
"""

from argparse import ArgumentParser
from pathlib import Path
import sys
import matplotlib.image as image


parser = ArgumentParser(
    description="Build thumbnails of all images in a directory.")
parser.add_argument("imagedir", type=Path)
args = parser.parse_args()
if not args.imagedir.isdir():
    sys.exit(f"Could not find input directory {args.imagedir}")

outdir = Path("thumbs")
outdir.mkdir(parents=True, exist_ok=True)

for path in args.imagedir.glob("*.png"):
    outpath = outdir / path.name
    fig = image.thumbnail(path, outpath, scale=0.15)
    print(f"saved thumbnail of {path} to {outpath}")
