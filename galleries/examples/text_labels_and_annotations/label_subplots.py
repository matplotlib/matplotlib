"""
==================
Labelling subplots
==================

Labelling subplots is relatively straightforward, and varies, so Matplotlib
does not have a general method for doing this.

We showcase two methods to position text at a given physical offset (in
fontsize units or in points) away from a corner of the Axes: one using
`~.Axes.annotate`, and one using `.ScaledTranslation`.

For convenience, this example uses `.pyplot.subplot_mosaic` and subplot
labels as keys for the subplots.  However, the approach also works with
`.pyplot.subplots` or keys that are different from what you want to label the
subplot with.
"""

import matplotlib.pyplot as plt

from matplotlib.transforms import ScaledTranslation

# %%
fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              layout='constrained')
for label, ax in axs.items():
    # Use Axes.annotate to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset half-a-fontsize right and half-a-fontsize down
    #   (offset fontsize (+0.5, -0.5)),
    # i.e. just inside the axes.
    ax.annotate(
        label,
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

# %%
fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              layout='constrained')
for label, ax in axs.items():
    # Use ScaledTranslation to put the label
    # - at the top left corner (axes fraction (0, 1)),
    # - offset 20 pixels left and 7 pixels up (offset points (-20, +7)),
    # i.e. just outside the axes.
    ax.text(
        0.0, 1.0, label, transform=(
            ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
        fontsize='medium', va='bottom', fontfamily='serif')

# %%
# If we want it aligned with the title, either incorporate in the title or
# use the *loc* keyword argument:

fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              layout='constrained')
for label, ax in axs.items():
    ax.set_title('Normal Title', fontstyle='italic')
    ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.subplot_mosaic` /
#      `matplotlib.pyplot.subplot_mosaic`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.annotate`
