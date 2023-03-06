"""
============
Scatter plot
============

This example showcases a simple scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

x = np.arange(0, 10)
norm = plt.Normalize(0, 10)
cmap = mpl.colormaps['viridis']
cols = cmap(norm(x))
scatter = plt.scatter(x, x, c=x, facecolors='none')

plt.show()
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.scatter` / `matplotlib.pyplot.scatter`