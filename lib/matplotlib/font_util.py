"""
A bunch of utilities that glue freetypy to matplotlib.

Some of these may be best moved to C, or moved to freetypy itself.

This module is basically temporary.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import freetypy as ft

import numpy as np

from math import ceil


def draw_glyph_to_bitmap(image, x, y, glyph):
    bm = np.asarray(glyph.render())
    x = int(x)
    y = int(y)
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + bm.shape[1], image.shape[1])
    y2 = min(y + bm.shape[0], image.shape[0])
    ox = x1 - x
    oy = y1 - y
    image[y1:y2, x1:x2] |= bm[oy:oy+(y2-y1), ox:ox+(x2-x1)]


def draw_layout_to_bitmap(layout, flags):
    bm = np.zeros(
        (int(ceil(layout.ink_bbox.height + 2)),
         int(ceil(layout.ink_bbox.width + 2))),
        dtype=np.uint8)

    for face, gind, (x, y) in layout.layout:
        glyph = face.load_glyph(gind, flags)
        bbox = glyph.get_cbox(ft.GLYPH_BBOX.SUBPIXELS)
        draw_glyph_to_bitmap(
            bm,
            x - layout.ink_bbox.x_min + bbox.x_min,
            ceil(layout.ink_bbox.y_max - bbox.y_max) - y, glyph)

    return bm
