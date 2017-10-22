"""
===============
Color reference
===============

All the named colors and what they look like.


Matplotlib contains a number of color palettes. Here is what they look like,
and how they are named.
"""
from __future__ import division

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def plot_colors(colors):
    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted(
        (tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
        for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    n = len(sorted_names)
    if n > 400:
        ncols = 8
    else:
        ncols = 4
    nrows = n // ncols + 1
    figsize = (2.4 * ncols, 0.2 * nrows)
    fig, ax = plt.subplots(figsize=figsize)

    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols

    for i, name in enumerate(sorted_names):
        col = i % ncols
        row = i // ncols
        y = Y - (row * h) - h

        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)

        ax.text(xi_text, y, name, fontsize=(h * 0.5),
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y + h * 0.1, xi_line, xf_line,
                  color=colors[name], linewidth=(h * 0.5))

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1,
                        top=1, bottom=0,
                        hspace=0, wspace=0)


###############################################################################
# Base colors
# ===========
#
# Available in Matplotlib since the age of dawn, the base colors offer the
# user easy access to a large color palette.
colors = dict(mcolors.BASE_COLORS)
plot_colors(colors)


###############################################################################
# Tableau colors
# ==============
#
# The tableau colors are the default the default in Matplotlib since 2.0, and
# can be accessed directly through their names:
colors = dict(mcolors.TABLEAU_COLORS)
plot_colors(colors)

###############################################################################
# Short names
# ===========
#
# For fast access, these 1-letter names cover the basics!
colors = dict(mcolors.CSS4_COLORS)
plot_colors(colors)

###############################################################################
# XKCD colors
# ===========
#
# In case you think all the above aren't enough to suit your needs, the XKCD
# color palette contains an insane number of colors... Check them out.
colors = dict(mcolors.XKCD_COLORS)
plot_colors(colors)
