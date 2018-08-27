"""
========================
Visualizing named colors
========================

Simple plot example with the named colors and its visual representation.

For more information on colors in matplotlib see

* the :doc:`/tutorials/colors/colors` tutorial;
* the `matplotlib.colors` API;
* the :doc:`/gallery/color/color_demo`.
"""
from matplotlib import colors as mcolors



# First, we define a custom plotting function that accepts a dictionary
# from one of matplotlib's named color palettes.
def plot_colortable(colors, title, sort_colors=True, order="by_row", ncols=4):
    import matplotlib.pyplot as plt
    from matplotlib.colors import rgb_to_hsv, to_rgba

    extra_rows = 2  # additional space for title
    cell_width = 225
    cell_height = 30
    swatch_width = 50

    # Sort colors by hue, saturation, value and name.
    by_hsv = ((tuple(rgb_to_hsv(to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    if sort_colors is True:
        by_hsv = sorted(by_hsv)
    names = [name for hsv, name in by_hsv]

    n = len(names)
    nrows = (n + 1) // ncols
    if n % ncols > 0:
        nrows += 1

    width = cell_width * ncols
    height = cell_height * (nrows + extra_rows)
    dpi = 72
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi),
                           dpi=dpi)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.text(0, cell_height, title, fontsize=20)

    for i, name in enumerate(names):
        if order == "by_row":
            row = i // ncols
            col = i % ncols
        elif order == 'by_column':
            row = i % nrows
            col = i // nrows

        y = (row + extra_rows) * cell_height

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 5

        ax.text(text_pos_x, y, name, fontsize=12,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=colors[name], linewidth=20)

    plt.show()

# Display the 8 base colors in matplotlib.
plot_colortable(mcolors.BASE_COLORS, "Base Colors", sort_colors=False)

# Displays named colors as defined by the CSS specification.
# For more on CSS colors, see https://www.w3.org/TR/css-color-4/
plot_colortable(mcolors.CSS4_COLORS, "CSS Colors")

# The Solarized palette is a 16-color palette designed for screen use.
# For more information, see https://ethanschoonover.com/solarized/
plot_colortable(mcolors.SOLARIZED_COLORS, "Solarized Palette",
                order='by_column', sort_colors=False)

# This displays the classic 10-color default palette in Tableau.
plot_colortable(mcolors.TABLEAU_COLORS, "Tableau Palette")

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib
matplotlib.colors
matplotlib.colors.rgb_to_hsv
matplotlib.colors.to_rgba
matplotlib.axes.Axes.text
matplotlib.axes.Axes.hlines
