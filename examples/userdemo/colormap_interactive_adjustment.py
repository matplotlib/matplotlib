"""
========================================
Interactive Adjustment of Colormap Range
========================================

Demonstration of using colorbar and picker functionality to make an
interactively adjustable colorbar widget.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


def pick_fn(event):
    adjust_colorbar(event.mouseevent)


def motion_fn(mouseevent):
    if mouseevent.inaxes is colorbar.ax.axes:
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


fig, ax = plt.subplots()
canvas = fig.canvas
arr = np.random.random((100, 100))
axesimage = plt.imshow(arr)
colorbar = plt.colorbar(axesimage, ax=ax, use_gridspec=True)

# helps you see what value you are about to set the range to
colorbar.ax.set_navigate(True)

# React to all motion with left or right mouse buttons held
canvas.mpl_connect("motion_notify_event", motion_fn)

# React only to left and right clicks
colorbar.ax.axes.set_picker(True)
canvas.mpl_connect("pick_event", pick_fn)

plt.show()
