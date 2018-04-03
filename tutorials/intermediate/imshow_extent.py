"""
*origin* and *extent* in `~.Axes.imshow`
========================================

:meth:`~.Axes.imshow` allows you to render an image (either a 2D array
which will be color-mapped (based on *norm* and *cmap*) or and 3D RGB(A)
array which will be used as-is) to a rectangular region in dataspace.
The orientation of the image in the final rendering is controlled by
the *origin* and *extent* kwargs (and attributes on the resulting
`~.AxesImage` instance) and the data limits of the axes.

The *extent* kwarg controls the bounding box in data coordinates that
the image will fill specified as ``(left, right, bottom, top)`` in
**data coordinates**, the *origin* kwarg controls how the image fills
that bounding box, and the orientation in the final rendered image is
also affected by the axes limits.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def generate_imshow_demo_grid(auto_limits, extents):
    N = len(extents)
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(6, N * (11.25) / 5)
    gs = GridSpec(N, 5, figure=fig)

    columns = {'label': [fig.add_subplot(gs[j, 0]) for j in range(N)],
               'upper': [fig.add_subplot(gs[j, 1:3]) for j in range(N)],
               'lower': [fig.add_subplot(gs[j, 3:5]) for j in range(N)]}

    d = np.arange(42).reshape(6, 7)

    for origin in ['upper', 'lower']:
        for ax, extent in zip(columns[origin], extents):

            im = ax.imshow(d, origin=origin, extent=extent)
            left, right, bottom, top = im.get_extent()
            arrow_style = {'arrowprops': {'arrowstyle': '-|>',
                                          'shrinkA': 0,
                                          'color': '0.5',
                                          'linewidth': 3}}
            ax.annotate('',
                        (left, bottom + 2*np.sign(top - bottom)),
                        (left, bottom),
                        **arrow_style)
            ax.annotate('',
                        (left + 2*np.sign(right - left), bottom),
                        (left, bottom),
                        **arrow_style)

            if auto_limits or top > bottom:
                upper_string, lower_string = 'top', 'bottom'
            else:
                upper_string, lower_string = 'bottom', 'top'

            if auto_limits or left < right:
                port_string, starboard_string = 'left', 'right'
            else:
                port_string, starboard_string = 'right', 'left'

            bbox_kwargs = {'fc': 'w', 'alpha': .75, 'boxstyle': "round4"}
            ann_kwargs = {'xycoords': 'axes fraction',
                          'textcoords': 'offset points',
                          'bbox': bbox_kwargs}

            ax.annotate(upper_string, xy=(.5, 1), xytext=(0, -1),
                        ha='center', va='top', **ann_kwargs)
            ax.annotate(lower_string, xy=(.5, 0), xytext=(0, 1),
                        ha='center', va='bottom', **ann_kwargs)

            ax.annotate(port_string, xy=(0, .5), xytext=(1, 0),
                        ha='left', va='center', rotation=90,
                        **ann_kwargs)
            ax.annotate(starboard_string, xy=(1, .5), xytext=(-1, 0),
                        ha='right', va='center', rotation=-90,
                        **ann_kwargs)

            ax.set_title('origin: {origin}'.format(origin=origin))

            if not auto_limits:
                ax.set_xlim(-1, 7)
                ax.set_ylim(-1, 6)

    for ax, extent in zip(columns['label'], extents):
        text_kwargs = {'ha': 'right',
                       'va': 'center',
                       'xycoords': 'axes fraction',
                       'xy': (1, .5)}
        if extent is None:
            ax.annotate('None', **text_kwargs)
            ax.set_title('`extent=`')
        else:
            left, right, bottom, top = extent
            text = ('left: {left:0.1f}\nright: {right:0.1f}\n' +
                    'bottom: {bottom:0.1f}\ntop: {top:0.1f}\n').format(
                        left=left, right=right, bottom=bottom, top=top)

            ax.annotate(text, **text_kwargs)
        ax.axis('off')


extents = (None,
           (-0.5, 6.5, -0.5, 5.5),
           (-0.5, 6.5, 5.5, -0.5),
           (6.5, -0.5, -0.5, 5.5),
           (6.5, -0.5, 5.5, -0.5))

###############################################################################
#
#
# First, using *extent* we pick a bounding box in dataspace that the
# image will fill and then interpolate/resample the underlying data to
# fill that box.
#
# - If ``origin='lower'`` than the ``[0, 0]`` entry is closest to the
#   ``(left, bottom)`` corner of the bounding box and moving closer to
#   ``(left, top)`` moves along the ``[:, 0]`` axis of the array to
#   higher indexed rows and moving towards ``(right, bottom)`` moves you
#   along the ``[0, :]`` axis of the array to higher indexed columns
#
# - If ``origin='upper'`` then the ``[-1, 0]`` entry is closest to the
#   ``(left, bottom)`` corner of the bounding box and moving towards
#   ``(left, top)`` moves along the ``[:, 0]`` axis of the array to
#   lower index rows and moving towards ``(right, bottom)`` moves you
#   along the ``[-1, :]`` axis of the array to higher indexed columns
#
# To demonstrate this we will plot a linear ramp
# ``np.arange(42).reshape(6, 7)`` with varying parameters.
#

generate_imshow_demo_grid(True, extents[:1])

###############################################################################
#
# If we only specify *origin* we can see why it is so named.  For
# ``origin='upper'`` the ``[0, 0]`` pixel is on the upper left and for
# ``origin='lower'`` the ``[0, 0]`` pixel is in the lower left [#]_.
# The gray arrows are attached to the ``(left, bottom)`` corner of the
# image.  There are two tricky things going on here: first the default
# value of *extent* depends on the value of *origin* and second the x
# and y limits are adjusted to match the extent.  The default *extent*
# is ``(-0.5, numcols-0.5, numrows-0.5, -0.5)`` when ``origin ==
# 'upper'`` and ``(-0.5, numcols-0.5, -0.5, numrows-0.5)`` when ``origin
# == 'lower'`` which puts the pixel centers on integer positions and the
# ``[0, 0]`` pixel at ``(0, 0)`` in dataspace.
#
#
# .. [#] The default value of *origin* is set by :rc:`image.origin`
#    which defaults to ``'upper'`` to match the matrix indexing
#    conventions in math and computer graphics image indexing
#    conventions.

generate_imshow_demo_grid(True, extents[1:])

###############################################################################
#
# If the axes is set to autoscale, then view limits of the axes are set
# to match the *extent* which ensures that the coordinate set by
# ``(left, bottom)`` is at the bottom left of the axes!  However, this
# may invert the axis so they do not increase in the 'natural' direction.
#

generate_imshow_demo_grid(False, extents)

###############################################################################
#
# If we fix the axes limits so ``(0, 0)`` is at the bottom left and
# increases to up and to the right (from the viewer point of view) then
# we can see that:
#
# - The ``(left, bottom)`` anchors the image which then fills the
#   box going towards the ``(right, top)`` point in data space.
# - The first column is always closest to the 'left'.
# - *origin* controls if the first row is closest to 'top' or 'bottom'.
# - The image may be inverted along either direction.
# - The 'left-right' and 'top-bottom' sense of the image is uncoupled from
#   the orientation on the screen.
