r"""
==================
Text Rotation Mode
==================

This example illustrates the effect of ``rotation_mode`` on the positioning
of rotated text.

Rotated `.Text`\s are created by passing the parameter ``rotation`` to
the constructor or the axes' method `~.axes.Axes.text`.

The actual positioning depends on the additional parameters
``horizontalalignment``, ``verticalalignment`` and ``rotation_mode``.
``rotation_mode`` determines the order of rotation and alignment:

- ``rotation_mode='default'`` (or None) first rotates the text and then aligns
  the bounding box of the rotated text.
- ``rotation_mode='anchor'`` aligns the unrotated text and then rotates the
  text around the point of alignment.
"""

import matplotlib.pyplot as plt
import numpy as np


def test_rotation_mode(fig, mode, subplot_location):
    ha_list = ["left", "center", "right"]
    va_list = ["top", "center", "baseline", "bottom"]
    axs = np.empty((len(va_list), len(ha_list)), object)
    gs = subplot_location.subgridspec(*axs.shape, hspace=0, wspace=0)
    axs[0, 0] = fig.add_subplot(gs[0, 0])
    for i in range(len(va_list)):
        for j in range(len(ha_list)):
            if (i, j) == (0, 0):
                continue  # Already set.
            axs[i, j] = fig.add_subplot(
                gs[i, j], sharex=axs[0, 0], sharey=axs[0, 0])
    for ax in axs.flat:
        ax.set(aspect=1)

    # labels and title
    for ha, ax in zip(ha_list, axs[-1, :]):
        ax.set_xlabel(ha)
    for va, ax in zip(va_list, axs[:, 0]):
        ax.set_ylabel(va)
    axs[0, 1].set_title(f"rotation_mode='{mode}'", size="large")

    kw = (
        {} if mode == "default" else
        {"bbox": dict(boxstyle="square,pad=0.", ec="none", fc="C1", alpha=0.3)}
    )

    # use a different text alignment in each axes
    for i, va in enumerate(va_list):
        for j, ha in enumerate(ha_list):
            ax = axs[i, j]
            # prepare axes layout
            ax.set(xticks=[], yticks=[])
            ax.axvline(0.5, color="skyblue", zorder=0)
            ax.axhline(0.5, color="skyblue", zorder=0)
            ax.plot(0.5, 0.5, color="C0", marker="o", zorder=1)
            # add text with rotation and alignment settings
            tx = ax.text(0.5, 0.5, "Tpg",
                         size="x-large", rotation=40,
                         horizontalalignment=ha, verticalalignment=va,
                         rotation_mode=mode, **kw)

    if mode == "default":
        # highlight bbox
        fig.canvas.draw()
        for ax in axs.flat:
            text, = ax.texts
            bb = text.get_window_extent().transformed(ax.transData.inverted())
            rect = plt.Rectangle((bb.x0, bb.y0), bb.width, bb.height,
                                 facecolor="C1", alpha=0.3, zorder=2)
            ax.add_patch(rect)


fig = plt.figure(figsize=(8, 5))
gs = fig.add_gridspec(1, 2)
test_rotation_mode(fig, "default", gs[0])
test_rotation_mode(fig, "anchor", gs[1])
plt.show()


#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.text` / `matplotlib.pyplot.text`
