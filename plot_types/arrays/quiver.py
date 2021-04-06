"""
=========================
quiver([X, Y], U, V, ...)
=========================
"""
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('mpl_plot_gallery')

# make data
T = np.linspace(0, 2*np.pi, 8)
X, Y = 4 + 1 * np.cos(T), 4 + 1 * np.sin(T)
U, V = 1.5 * np.cos(T), 1.5 * np.sin(T)

# plot
fig, ax = plt.subplots()

plt.quiver(X, Y, U, V, color="C0", angles='xy',
            scale_units='xy', scale=0.5, width=.05)

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()
