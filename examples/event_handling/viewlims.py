# Creates two identical panels.  Zooming in on the right panel will show
# a rectangle in the first panel, denoting the zoomed region.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# We just subclass Rectangle so that it can be called with an Axes
# instance, causing the rectangle to update its shape to match the
# bounds of the Axes
class UpdatingRect(Rectangle):
    def __call__(self, ax):
        self.set_bounds(*ax.viewLim.bounds)
        ax.figure.canvas.draw_idle()

x = np.linspace(-3., 3., 20)
y = np.linspace(-3., 3., 20).reshape(-1,1)
Z = (1- x/2 + x**5 + y**3)*np.exp(-x**2-y**2)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.pcolor(x, y, Z)

ax2 = fig.add_subplot(1, 2, 2)
ax2.pcolor(x, y, Z)

rect = UpdatingRect([0, 0], 0, 0, facecolor='None', edgecolor='black')
rect.set_bounds(*ax2.viewLim.bounds)
ax1.add_patch(rect)

# Connect for changing the view limits
ax2.callbacks.connect('xlim_changed', rect)
ax2.callbacks.connect('ylim_changed', rect)

plt.show()
