"""
=========
imshow(Z)
=========

See `~matplotlib.axes.Axes.imshow`.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('mpl_plot_gallery')

# make data
X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
Z = (1 - X/2. + X**5 + Y**3) * np.exp(-X**2 - Y**2)
Z = Z - Z.min()
Z = Z[::16, ::16]

# plot
fig, ax = plt.subplots()
ax.grid(False)

ax.imshow(Z)

plt.show()
