"""
================
Marker reference
================

Matplotlib supports multiple categories of markers which are selected using
the ``marker`` parameter of plot commands:

- `Unfilled markers`_
- `Filled markers`_
- `Markers created from TeX symbols`_
- Custom markers can be created from paths. See
  :doc:`/gallery/shapes_and_collections/marker_path`.

For a list of all markers see also the `matplotlib.markers` documentation.

For example usages see
:doc:`/gallery/lines_bars_and_markers/scatter_star_poly`.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=12, fontfamily='monospace')
marker_style = dict(linestyle=':', color='0.8', markersize=10,
                    markerfacecolor="tab:blue", markeredgecolor="tab:blue")


def format_axes(ax):
    ax.margins(0.2)
    ax.set_axis_off()
    ax.invert_yaxis()


def split_list(a_list):
    i_half = len(a_list) // 2
    return a_list[:i_half], a_list[i_half:]


###############################################################################
# Unfilled markers
# ================
# Unfilled markers are single-colored.

fig, axs = plt.subplots(ncols=2)
fig.suptitle('Un-filled markers', fontsize=14)

# Filter out filled markers and marker settings that do nothing.
unfilled_markers = [m for m, func in Line2D.markers.items()
                    if func != 'nothing' and m not in Line2D.filled_markers]

for ax, markers in zip(axs, split_list(unfilled_markers)):
    for y, marker in enumerate(markers):
        ax.text(-0.5, y, repr(marker), **text_style)
        ax.plot([y] * 3, marker=marker, **marker_style)
    format_axes(ax)

plt.show()


###############################################################################
# Filled markers
# ==============

fig, axs = plt.subplots(ncols=2)
fig.suptitle('Filled markers', fontsize=14)
for ax, markers in zip(axs, split_list(Line2D.filled_markers)):
    for y, marker in enumerate(markers):
        ax.text(-0.5, y, repr(marker), **text_style)
        ax.plot([y] * 3, marker=marker, **marker_style)
    format_axes(ax)

plt.show()

###############################################################################
# .. _marker_fill_styles:
#
# Marker fill styles
# ------------------
# The edge color and fill color of filled markers can be specified separately.
# Additionally, the ``fillstyle`` can be configured to be unfilled, fully
# filled, or half-filled in various directions. The half-filled styles use
# ``markerfacecoloralt`` as secondary fill color.

fig, ax = plt.subplots()
fig.suptitle('Marker fillstyle', fontsize=14)
fig.subplots_adjust(left=0.4)

filled_marker_style = dict(marker='o', linestyle=':', markersize=15,
                           color='darkgrey',
                           markerfacecolor='tab:blue',
                           markerfacecoloralt='lightsteelblue',
                           markeredgecolor='brown')

for y, fill_style in enumerate(Line2D.fillStyles):
    ax.text(-0.5, y, repr(fill_style), **text_style)
    ax.plot([y] * 3, fillstyle=fill_style, **filled_marker_style)
format_axes(ax)

plt.show()


###############################################################################
# Markers created from TeX symbols
# ================================
#
# Use :doc:`MathText </tutorials/text/mathtext>`, to use custom marker symbols,
# like e.g. ``"$\u266B$"``. For an overview over the STIX font symbols refer
# to the `STIX font table <http://www.stixfonts.org/allGlyphs.html>`_.
# Also see the :doc:`/gallery/text_labels_and_annotations/stix_fonts_demo`.


fig, ax = plt.subplots()
fig.suptitle('Mathtext markers', fontsize=14)
fig.subplots_adjust(left=0.4)

marker_style.update(markeredgecolor="None", markersize=15)
markers = ["$1$", r"$\frac{1}{2}$", "$f$", "$\u266B$", r"$\mathcal{A}$"]

for y, marker in enumerate(markers):
    # Escape dollars so that the text is written "as is", not as mathtext.
    ax.text(-0.5, y, repr(marker).replace("$", r"\$"), **text_style)
    ax.plot([y] * 3, marker=marker, **marker_style)
format_axes(ax)

plt.show()
