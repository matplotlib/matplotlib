"""
.. _plot_unstructuredgrid:

Unstructured data
=================

Unlike :ref:`structured grids <plot_gridded>`, three-dimensional data is often
collected without the benefit of a grid, with :math:`z = f(x, y)` where
:math:`x` and :math:`y` are one-dimensional arrays of the same length, and
hence :math:`z` is also one-dimensional.  Sometimes it is useful to grid that
data via appropriate interpolation schemes (see `scipy.interpolate.griddata`
for some in python). However, if the data is spaced appropriately, it can sometimes
be plotted directly in Matplotlib.

Matplotlib accomplishes this via a suite of triangulation routines in the
:mod:`matplotlib.tri` module.  The most basic triangulation routine is
`~matplotlib.tri.Triangulation`, which is a simple wrapper around the popular
`Qhull <http://www.qhull.org/>`_ library.  The `~matplotlib.tri.Triangulation`
class is used to create a triangulation of a set of points, and this
triangulation can then be used to create either faceted psuedocolor plots, or
contour plots.

"""

# %%
# triplot
# -------
#
# The most basic plot is the `~.axes.Axes.triplot` method, which plots the
# edges of triangles in a triangulation.  Here we create a simple triangulation
# on a set of points, and plot the edges of the triangles that are created.

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = (5, 4)

# make data:
np.random.seed(19680801)
x = np.random.uniform(-3, 3, 256)
y = np.random.uniform(-3, 3, 256)
z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)

fig, ax = plt.subplots()

# plot the points:
ax.plot(x, y, 'm.')

# plot the triangulation:
ax.triplot(x, y)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('triplot')

# %%
# In advanced usage, one could pass a `~matplotlib.tri.Triangulation` object to
# the `~.axes.Axes.triplot` method.  See :ref:`triplot_demo` for more details.
#
# tripcolor
# ---------
#
# The `~.axes.Axes.tripcolor` method can be used to plot a surface by coloring
# each triangle based on the value of a colormap.  Here we plot the same
# triangulation as above, but color the triangles based on the values of
# :math:`z` on the corners.  The default shading is "flat", which averages the
# values of :math:`z` on each triangle, but `~.axes.Axes.tripcolor` can also perform
# Gouraud shading for a smoother appearance.

fig, axs = plt.subplots(2, 1, figsize=(5, 7))
for ax, shade in zip(axs, ['flat', 'gouraud']):
    ax.plot(x, y, 'm.')
    pc = ax.tripcolor(x, y, z, cmap='RdBu_r', vmin=-1, vmax=1, shading=shade)
    fig.colorbar(pc, ax=ax, label='z')
    ax.set_title(f'tripcolor shading = "{shade}"')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_facecolor('lightgray')

# %%
# It is also possible to set the values on the face of each triangle, but this
# is better done if you have a `~matplotlib.tri.Triangulation` object rather
# than the raw x, y, z arrays.  See :ref:`tripcolor_demo` for more details.
#
# tricontour and tricontourf
# --------------------------
#
# The `~.axes.Axes.tricontour` and `~.axes.Axes.tricontourf` methods can be
# used to plot contours of unstractured data, where the underlying
# triangulation is used to interpolate the contour locations.  Here we plot
# contours of the same data as above.

fig, ax = plt.subplots()
ax.plot(x, y, 'm.')
pc = ax.tricontourf(x, y, z, cmap='RdBu_r', vmin=-1, vmax=1, levels=15)
fig.colorbar(pc, ax=ax, label='z')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_facecolor('lightgray')

# %%
# As with `~.axes.Axes.tripcolor`, a `~matplotlib.tri.Triangulation` object can be
# passed instead of the raw x and y arrays.
#
# .. seealso::
#   For more advanced use of `~.axes.Axes.tricontour` and
#   `~.axes.Axes.tricontourf`, see:
#
#   - :ref:`tricontour_demo`
#   - :ref:`tricontour_smooth_user`
#   - :ref:`tricontour_smooth_delaunay`
#
# .. admonition:: References
#
#   The use of the following functions, methods, classes and modules is shown
#   in this example:
#
#   - `matplotlib.axes.Axes.triplot` / `matplotlib.pyplot.triplot`
#   - `matplotlib.axes.Axes.tripcolor` / `matplotlib.pyplot.tripcolor`
#   - `matplotlib.axes.Axes.tricontour` / `matplotlib.pyplot.tricontour`
#   - `matplotlib.axes.Axes.tricontourf` / `matplotlib.pyplot.tricontourf`
