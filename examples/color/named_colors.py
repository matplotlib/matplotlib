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
def plot_colors(colors, title, swatch_width=3, swatch_height=1, sort_colors=True, ncols=4):

    import matplotlib.pyplot as plt
    from matplotlib.colors import rgb_to_hsv, to_rgba

    # Sort colors by hue, saturation, value and name.
    by_hsv = ((tuple(rgb_to_hsv(to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    if sort_colors is True:
        by_hsv = sorted(by_hsv)
    names = [name for hsv, name in by_hsv]

    n = len(names)
    nrows = n // ncols
    if n % ncols > 0:
        nrows += 1

    x_inches = swatch_width * ncols
    y_inches = swatch_height * nrows
    fig, ax = plt.subplots(figsize=(x_inches, y_inches))

    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = Y - (row * h) - h

        swatch_start_x = w * (col + 0.05)
        swatch_end_x = w * (col + 0.3)
        text_pos_x = w * (col + 0.35)

        ax.text(text_pos_x, y, name, fontsize=15,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y,
                  swatch_start_x, swatch_end_x,
                  color=colors[name], linewidth=20)

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1,
                        top=1, bottom=0,
                        hspace=0, wspace=0)

    plt.title(title, fontsize=25)

    plt.tight_layout()
    plt.show()


plot_colors(mcolors.BASE_COLORS, "Base Colors")

plot_colors(mcolors.CSS4_COLORS, "CSS Colors")

plot_colors(mcolors.SOLARIZED_COLORS, "Solarized Palette", sort_colors=False)

plot_colors(mcolors.TABLEAU_COLORS, "Tableau Palette")

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
matplotlib.figure.Figure.get_size_inches
matplotlib.figure.Figure.subplots_adjust
matplotlib.axes.Axes.text
matplotlib.axes.Axes.hlines
