"""
========================================
Interactive Adjustment of Colormap Range
========================================

Demonstration of using colorbar, picker, and event functionality to make an
interactively adjustable colorbar widget.

Left clicks and drags inside the colorbar axes adjust the high range of the
color scheme. Likewise, right clicks and drags adjust the low range. The
connected AxesImage immediately updates to reflect the change.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

###############################################################################
# Callback definitions


def on_pick(event):
    adjust_colorbar(event.mouseevent)


def on_move(mouseevent):
    if mouseevent.inaxes is colorbar.ax:
        adjust_colorbar(mouseevent)


def adjust_colorbar(mouseevent):
    if mouseevent.button == MouseButton.LEFT:
        colorbar.norm.vmax = max(mouseevent.ydata, colorbar.norm.vmin)
    elif mouseevent.button == MouseButton.RIGHT:
        colorbar.norm.vmin = min(mouseevent.ydata, colorbar.norm.vmax)
    else:
        # discard all others
        return

    canvas.draw_idle()


###############################################################################
# Generate figure with Axesimage and Colorbar

fig, ax = plt.subplots()
canvas = fig.canvas

delta = 0.1
x = np.arange(-3.0, 4.001, delta)
y = np.arange(-4.0, 3.001, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (0.9*Z1 - 0.5*Z2) * 2

cmap = plt.get_cmap('viridis').with_extremes(
    over='xkcd:orange', under='xkcd:dark red')
axesimage = plt.imshow(Z, cmap=cmap)
colorbar = plt.colorbar(axesimage, ax=ax, use_gridspec=True)

###############################################################################
# Note that axesimage and colorbar share a Normalize object
# so they will stay in sync

assert colorbar.norm is axesimage.norm
colorbar.norm.vmax = 1.5
axesimage.norm.vmin = -0.75

###############################################################################
# Hook Colorbar up to canvas events

# `set_navigate` helps you see what value you are about to set the range
# to, and enables zoom and pan in the colorbar which can be helpful for
# narrow or wide data ranges
colorbar.ax.set_navigate(True)

# React to all motion with left or right mouse buttons held
canvas.mpl_connect("motion_notify_event", on_move)

# React only to left and right clicks
colorbar.ax.set_picker(True)
canvas.mpl_connect("pick_event", on_pick)

###############################################################################
# Display
#
# The colormap will now respond to left and right clicks in the Colorbar axes

plt.show()
