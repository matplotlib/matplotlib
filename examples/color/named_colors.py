"""
Visualization of named colors.

Simple plot example with the named colors and its visual representation.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


colors_ = list(six.iteritems(colors.cnames))

# Add the single letter colors.
for name, rgb in six.iteritems(colors.ColorConverter.colors):
    hex_ = colors.rgb2hex(rgb)
    colors_.append((name, hex_))

# Transform to hex color values.
hex_ = [color[1] for color in colors_]
# Get the rgb equivalent.
rgb = [colors.hex2color(color) for color in hex_]
# Get the hsv equivalent.
hsv = [colors.rgb_to_hsv(color) for color in rgb]

# Split the hsv values to sort.
hue = [color[0] for color in hsv]
sat = [color[1] for color in hsv]
val = [color[2] for color in hsv]

# Sort by hue, saturation and value.
ind = np.lexsort((val, sat, hue))
sorted_colors = [colors_[i] for i in ind]

n = len(sorted_colors)
ncols = 3
nrows = int(np.ceil(1. * n / ncols))

fig = plt.figure(figsize=(ncols*2.5, nrows*2))
for i, (name, color) in enumerate(sorted_colors):
    ax = fig.add_subplot(nrows, ncols, i + 1)
    ax.text(0.55, 0.5, name, fontsize=12,
            horizontalalignment='left',
            verticalalignment='center')

    # Add extra black line a little bit thicker to make
    # clear colors more visible.
    ax.hlines(0.5, 0, 0.5, color='black', linewidth=10)
    ax.hlines(0.5, 0, 0.5, color=color, linewidth=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

fig.subplots_adjust(left=0.01, right=0.99,
                    top=0.99, bottom=0.01,
                    hspace=1, wspace=0.1)
plt.show()
