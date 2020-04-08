#!/usr/bin/env python
"""
Generates the Matplotlib icon, and the toolbar icon images from the FontAwesome
font.

Generates SVG, PDF in one size (since they are vectors), and PNG in 24x24 and
48x48.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from io import BytesIO
from pathlib import Path
import tarfile
import urllib.request

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


plt.rcdefaults()
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['pdf.fonttype'] = 3
plt.rcParams['pdf.compression'] = 9


def get_fontawesome():
    cached_path = Path(mpl.get_cachedir(), "FontAwesome.otf")
    if not cached_path.exists():
        with urllib.request.urlopen(
                "https://github.com/FortAwesome/Font-Awesome"
                "/archive/v4.7.0.tar.gz") as req, \
             tarfile.open(fileobj=BytesIO(req.read()), mode="r:gz") as tf:
            cached_path.write_bytes(tf.extractfile(tf.getmember(
                "Font-Awesome-4.7.0/fonts/FontAwesome.otf")).read())
    return cached_path


def save_icon(fig, dest_dir, name):
    fig.savefig(dest_dir / (name + '.svg'))
    fig.savefig(dest_dir / (name + '.pdf'))
    for dpi, suffix in [(24, ''), (48, '_large')]:
        fig.savefig(dest_dir / (name + suffix + '.png'), dpi=dpi)


def make_icon(font_path, ccode):
    fig = plt.figure(figsize=(1, 1))
    fig.patch.set_alpha(0.0)
    text = fig.text(0.5, 0.48, chr(ccode), ha='center', va='center',
                    font=font_path, fontsize=68)
    return fig


def make_matplotlib_icon():
    fig = plt.figure(figsize=(1, 1))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95], projection='polar')
    ax.set_axisbelow(True)

    N = 7
    arc = 2 * np.pi
    theta = np.arange(0, arc, arc / N)
    radii = 10 * np.array([0.2, 0.6, 0.8, 0.7, 0.4, 0.5, 0.8])
    width = np.pi / 4 * np.array([0.4, 0.4, 0.6, 0.8, 0.2, 0.5, 0.3])
    bars = ax.bar(theta, radii, width=width, bottom=0.0, linewidth=1,
                  edgecolor='k')

    for r, bar in zip(radii, bars):
        bar.set_facecolor(mpl.cm.jet(r / 10))

    ax.tick_params(labelleft=False, labelright=False,
                   labelbottom=False, labeltop=False)
    ax.grid(lw=0.0)

    ax.set_yticks(np.arange(1, 9, 2))
    ax.set_rmax(9)

    return fig


icon_defs = [
    ('home', 0xf015),
    ('back', 0xf060),
    ('forward', 0xf061),
    ('zoom_to_rect', 0xf002),
    ('move', 0xf047),
    ('filesave', 0xf0c7),
    ('subplots', 0xf1de),
    ('qt4_editor_options', 0xf201),
    ('help', 0xf128),
]


def make_icons():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-d", "--dest-dir",
        type=Path,
        default=Path(__file__).parent / "../lib/matplotlib/mpl-data/images",
        help="Directory where to store the images.")
    args = parser.parse_args()
    font_path = get_fontawesome()
    for name, ccode in icon_defs:
        fig = make_icon(font_path, ccode)
        save_icon(fig, args.dest_dir, name)
    fig = make_matplotlib_icon()
    save_icon(fig, args.dest_dir, 'matplotlib')


if __name__ == "__main__":
    make_icons()
