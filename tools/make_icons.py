#!/usr/bin/env python
"""
Generates the toolbar icon images from the FontAwesome font.

First download and extract FontAwesome from http://fontawesome.io/.
Place the FontAwesome.otf file in the tools directory (same directory
as this script).

Generates SVG, PDF in one size (size they are vectors) and PNG, PPM and GIF in
24x24 and 48x48.
"""

import matplotlib
matplotlib.use('agg')  # noqa

import os

from PIL import Image

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import cm
import matplotlib
import matplotlib.patheffects as PathEffects
matplotlib.rcdefaults()

matplotlib.rcParams['svg.fonttype'] = 'path'
matplotlib.rcParams['pdf.fonttype'] = 3
matplotlib.rcParams['pdf.compression'] = 9


IMAGES_ROOT = os.path.join(
    os.path.dirname(__file__), '..', 'lib', 'matplotlib', 'mpl-data', 'images')
FONT_PATH = os.path.join(
    os.path.dirname(__file__), 'FontAwesome.otf')


def save_icon(fig, name):
    fig.savefig(os.path.join(IMAGES_ROOT, name + '.svg'))
    fig.savefig(os.path.join(IMAGES_ROOT, name + '.pdf'))

    for dpi, suffix in [(24, ''), (48, '_large')]:
        fig.savefig(os.path.join(IMAGES_ROOT, name + suffix + '.png'), dpi=dpi)

        img = Image.open(os.path.join(IMAGES_ROOT, name + suffix + '.png'))
        img.save(os.path.join(IMAGES_ROOT, name + suffix + '.ppm'))


def make_icon(fontfile, ccode):
    prop = FontProperties(fname=fontfile, size=68)

    fig = plt.figure(figsize=(1, 1))
    fig.patch.set_alpha(0.0)
    text = fig.text(0.5, 0.48, chr(ccode), ha='center', va='center',
                    fontproperties=prop)
    text.set_path_effects([PathEffects.Normal()])

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
        bar.set_facecolor(cm.jet(r/10.))

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
    ('help', 0xf128)]


def make_icons():
    for name, ccode in icon_defs:
        fig = make_icon(FONT_PATH, ccode)
        save_icon(fig, name)
    fig = make_matplotlib_icon()
    save_icon(fig, 'matplotlib')


if __name__ == '__main__':
    if not os.path.exists(FONT_PATH):
        print("Download the FontAwesome.otf file and place it in the tools "
              "directory")
    make_icons()
