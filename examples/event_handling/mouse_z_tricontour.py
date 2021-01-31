"""
============================================
Z-value at mouse position of tricontour plot
============================================

This example shows how a motion_notify_event can be used to display the z-value
corresponding to the mouse cursor position.  The z-value is interpolated from
the (x, y, z) data used to generate a tricontourf plot.

Note that the tricontour plot is not used to determine the z-value, rather it
is intended to help guide the user's mouse movements.  A tripcolor plot could
be used instead.
"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def on_mouse_move(event):
    title = ''
    if event.inaxes == ax:
        z = interp(event.xdata, event.ydata)
        if not np.ma.is_masked(z):
            title = f'z = {z:.3f}'
    ax.set_title(title)
    event.canvas.draw()


n = 40
np.random.seed(23)
x = np.random.uniform(size=(n,))
y = np.random.uniform(size=(n,))
z = np.sin(10*x)*np.cos(10*y)
triang = mtri.Triangulation(x, y)

# Function used to interpolate z-values.
interp = mtri.LinearTriInterpolator(triang, z)

fig, ax = plt.subplots()
cs = ax.tricontourf(triang, z)
fig.colorbar(cs)
ax.triplot(triang, c='C1', alpha=0.5)
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.suptitle('Move mouse to show z-value at cursor position')
plt.show()
