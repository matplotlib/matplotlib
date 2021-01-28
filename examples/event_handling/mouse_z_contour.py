"""
=========================================
Z-value at mouse position of contour plot
=========================================

This example shows how a motion_notify_event can be used to display the z-value
corresponding to the mouse cursor position.  The z-value is interpolated from
the (x, y, z) data used to generate a contourf plot.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d


def on_mouse_move(event):
    title = ''
    if event.inaxes == ax:
        z = interp(event.xdata, event.ydata)[0]
        if not np.ma.is_masked(z):
            title = f'z = {z:.3f}'
    ax.set_title(title)
    event.canvas.draw()


n = 10
x, y = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))
np.random.seed(42)
x += np.random.normal(scale=0.15/n, size=(n, n))
y += np.random.normal(scale=0.15/n, size=(n, n))
z = np.sin(10*x)*np.cos(10*y)

# Function used to interpolate z-values.
interp = interp2d(x, y, z)

fig, ax = plt.subplots()
cs = ax.contourf(x, y, z)
fig.colorbar(cs)
ax.plot(x, y, x.T, y.T, c='C1', alpha=0.5)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.suptitle('Move mouse to show z-value at cursor position')
plt.show()
