"""
Script to generate a squircle-compliant macOS icon (.icns) for matplotlib.

Generates light and dark mode variants of the matplotlib logo
placed on a squircle-shaped background, as required by macOS Big Sur+.

Usage:
    python tools/make_macos_icon.py
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import importlib.util
import io

# ── Load make_logo from logos2.py ──────────────────────────────────────────
_logos2_path = os.path.join(os.path.dirname(__file__),
                             '../galleries/examples/misc/logos2.py')
spec = importlib.util.spec_from_file_location('logos2', _logos2_path)
logos2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logos2)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_logo_png(size=1024):
    """Render the matplotlib logo on a transparent background."""
    fig, ax = logos2.make_logo(
        height_px=size, lw_bars=4, lw_grid=3, lw_border=2,
        rgrid=[1, 3, 5, 7]
    )
    ax.set_position([0.0, 0.0, 1.0, 1.0])  # force full canvas

    # Remove the hardcoded white rectangle background
    for patch in ax.patches:
        if isinstance(patch, Rectangle) and patch.get_facecolor() == (1.0, 1.0, 1.0, 1.0):
            patch.set_visible(False)

    ax.spines['polar'].set_visible(False)  # removes blue circle border
    ax.grid(False)                          # removes grid lines

    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True,
                dpi=100) #""", bbox_inches='tight', pad_inches=0"""

    plt.close(fig)
    buf.seek(0)
    #return Image.open(buf).convert('RGBA')
    img = Image.open(buf).convert('RGBA')

# Crop to actual content (non-transparent pixels)
    bbox = img.getbbox()
    img_cropped = img.crop(bbox)

# Paste centered onto a fresh square canvas
    canvas = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
    offset_x = (1024 - img_cropped.width) // 2 - 20
    offset_y = (1024 - img_cropped.height) // 2 - 44
    canvas.paste(img_cropped, (offset_x, offset_y), img_cropped)

    return canvas


def make_squircle_mask(size=1024, n=5):
    """
    Generate a squircle mask.
    n=5 approximates Apple's superellipse corner shape.
    """
    mask = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(mask)

    cx = cy = size / 2
    r = size / 2 * 0.90  # slight inset so edges aren't clipped

    xs, ys = np.meshgrid(np.arange(size), np.arange(size))
    data = (((np.abs(xs - cx) / r) ** n + (np.abs(ys - cy) / r) ** n) <= 1).astype(np.uint8) * 255
    mask = Image.fromarray(data, mode='L')

    return mask


def make_icon(logo, bg_color, size=1024):
    """Paste logo onto a squircle background."""
    icon = Image.new('RGBA', (size, size), (0, 0, 0, 0))

    # Draw squircle background
    background = Image.new('RGBA', (size, size), bg_color)
    mask = make_squircle_mask(size)
    icon.paste(background, mask=mask)

    # Resize logo to fit with padding (~10%)
    pad = int(size * 0.10)
    logo_size = size - 2 * pad
    logo_resized = logo.resize((logo_size, logo_size), Image.LANCZOS)

    # Paste logo centered
    icon.paste(logo_resized, (pad, pad), logo_resized)

    return icon


def save_icns(icon, path):
    """Save icon as .icns with all required sizes."""
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    icons = {size: icon.resize((size, size), Image.LANCZOS) for size in sizes}
    icons[1024].save(
        path,
        format='icns',
        append_images=list(icons.values())
    )
    print(f'Saved: {path}')


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__),
                           '../lib/matplotlib/mpl-data/images')

    print('Rendering logo...')
    logo = get_logo_png(size=1024)

    # Light mode: white background
    print('Generating light mode icon...')
    light = make_icon(logo, bg_color=(255, 255, 255, 255))
    save_icns(light, os.path.join(out_dir, 'matplotlib.icns'))

    # Dark mode: dark background
    print('Generating dark mode icon...')
    dark = make_icon(logo, bg_color=(30, 30, 30, 255))
    save_icns(dark, os.path.join(out_dir, 'matplotlib_dark.icns'))

    print('Done.')