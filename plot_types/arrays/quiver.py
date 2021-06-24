"""
==================
quiver(X, Y, U, V)
==================

See `~matplotlib.axes.Axes.quiver`.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('mpl_plot_gallery')

# make data
phi = np.linspace(0, 2 * np.pi, 8)
x, y = 4 + 1 * np.cos(phi), 4 + 1 * np.sin(phi)
u, v = 1.5 * np.cos(phi), 1.5 * np.sin(phi)

# plot
fig, ax = plt.subplots()

ax.quiver(x, y, u, v, color="C0", angles='xy',
          scale_units='xy', scale=0.5, width=.05)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
