"""
========
Viewlims
========

Creates two identical panels.  Zooming in on the right panel will show
a rectangle in the first panel, denoting the zoomed region.

.. note::
    This example exercises the interactive capabilities of Matplotlib, and this
    will not appear in the static documentation. Please run this code on your
    machine to see the interactivity.

    You can copy and paste individual parts, or download the entire example
    using the link at the bottom of the page.
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector

# A class that will regenerate a fractal set as we zoom in, so that you
# can actually see the increasing detail.  A box in the left panel will show
# the area to which we are zoomed.


class MandelbrotDisplay:
    def __init__(self, h=500, w=500, niter=50, radius=2., power=2):
        self.height = h
        self.width = w
        self.niter = niter
        self.radius = radius
        self.power = power

    def compute_image(self, xstart, xend, ystart, yend):
        self.x = np.linspace(xstart, xend, self.width)
        self.y = np.linspace(ystart, yend, self.height).reshape(-1, 1)
        c = self.x + 1.0j * self.y
        threshold_time = np.zeros((self.height, self.width))
        z = np.zeros(threshold_time.shape, dtype=complex)
        mask = np.ones(threshold_time.shape, dtype=bool)
        for i in range(self.niter):
            z[mask] = z[mask]**self.power + c[mask]
            mask = (np.abs(z) < self.radius)
            threshold_time += mask
        return threshold_time

    def ax_update(self, ax):
        ax.set_autoscale_on(False)  # Otherwise, infinite loop
        # Get the number of points from the number of pixels in the window
        self.width, self.height = \
            np.round(ax.patch.get_window_extent().size).astype(int)
        # Get the range for the new area
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        extent = x0, x1, y0, y1
        # Update the image object with our new data and extent
        im = ax.images[-1]
        im.set_data(self.compute_image(*extent))
        im.set_extent(extent)
        ax.figure.canvas.draw_idle()


def select_callback(eclick, erelease):
    extent = rect_selector.extents

    ax2.set_autoscale_on(False)
    # Zoom the selected part
    # Set xlim, ylim range for plot
    # of rectangle selector box.
    ax2.set_xlim(extent[0], extent[1])
    ax2.set_ylim(extent[2], extent[3])

    # update the right subplot
    md = MandelbrotDisplay()
    md.ax_update(ax2)
    ax2.figure.canvas.draw_idle()

    # update the rectangle box on the left subplot
    rect.set_bounds(*ax2.viewLim.bounds)
    ax1.figure.canvas.draw_idle()

md = MandelbrotDisplay()
Z = md.compute_image(-2., 0.5, -1.25, 1.25)

fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(Z, origin='lower',
           extent=(md.x.min(), md.x.max(), md.y.min(), md.y.max()))
ax2.imshow(Z, origin='lower',
           extent=(md.x.min(), md.x.max(), md.y.min(), md.y.max()))

rect = Rectangle(
    [0, 0], 0, 0, facecolor='none', edgecolor='black', linewidth=1.0)
ax1.add_patch(rect)

rect_selector = RectangleSelector(
        ax2, select_callback,
        useblit=True,
        button=[1],
        spancoords='pixels')

ax2.set_title("Zoom here")
plt.show()
