"""
==============
Text alignment
==============

Texts are aligned relative to their anchor point depending on the properties
``horizontalalignment`` and ``verticalalignment``.

.. redirect-from:: /gallery/pyplots/text_layout

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    y = [0.22, 0.34, 0.5, 0.56, 0.78]
    x = [0.17, 0.5, 0.855]
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.spines[:].set_visible(False)
    ax.text(0.5, 0.5, 'plot', fontsize=128, ha='center', va='center', zorder=1)
    ax.hlines(y, x[0], x[-1], color='grey')
    ax.vlines(x, y[0], y[-1], color='grey')
    ax.plot(X.ravel(), Y.ravel(), 'o')
    pad_x = 0.02
    pad_y = 0.04
    ax.text(x[0] - pad_x, y[0], 'bottom', ha='right', va='center')
    ax.text(x[0] - pad_x, y[1], 'baseline', ha='right', va='center')
    ax.text(x[0] - pad_x, y[2], 'center', ha='right', va='center')
    ax.text(x[0] - pad_x, y[3], 'center_baseline', ha='right', va='center')
    ax.text(x[0] - pad_x, y[4], 'top', ha='right', va='center')
    ax.text(x[0], y[0] - pad_y, 'left', ha='center', va='top')
    ax.text(x[1], y[0] - pad_y, 'center', ha='center', va='top')
    ax.text(x[2], y[0] - pad_y, 'right', ha='center', va='top')
    ax.set_xlabel('horizontalalignment', fontsize=14)
    ax.set_ylabel('verticalalignment', fontsize=14, labelpad=35)
    ax.set_title(
        'Relative position of text anchor point depending on alignment')
    plt.show()

"""

#############################################################################
# The following plot uses this to align text relative to a plotted rectangle.

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
p = plt.Rectangle((left, bottom), width, height, fill=False)
p.set_transform(ax.transAxes)
p.set_clip_on(False)
ax.add_patch(p)

ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, 0.5 * (bottom + top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, 0.5 * (bottom + top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(0.5 * (left + right), 0.5 * (bottom + top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)

ax.text(right, 0.5 * (bottom + top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, top, 'rotated\nwith newlines',
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45,
        transform=ax.transAxes)

ax.set_axis_off()

plt.show()
