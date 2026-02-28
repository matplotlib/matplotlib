"""
.. _plot_gridded:

Gridded plots
=============

This section discusses data that mapped onto a two-dimensional grid.  The data
usually has structured coordinates defined by arrays of x and y values, and data
mapped onto those coordinates :math:`Data(x, y)` in a two-dimensional array. The grid
can either be regular, in which case the x and y arrays are 1D and the Data is 2D,
or the grid can be irregular, and X, Y, and Data are all 2D arrays.

"""

import matplotlib.pyplot as plt
import numpy as np

import warnings

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = (5, 4)

# %%
# imshow
# ------
#
# `~.axes.Axes.imshow` is a simple way to plot a 2D array as an image. By
# default, the aspect ratio is set to be equal, so the pixels are square, and
# the origin of the data is the upper-left corner.  The values of the array are
# mapped to colors using a colormap.

# create a square matrix of data:
X, Y = np.meshgrid(np.linspace(-3, 3, 128), np.linspace(-3, 3, 128))
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

fig, ax = plt.subplots()
im = ax.imshow(Z)
fig.colorbar(im, ax=ax)
ax.set_xlabel('pixels in x')
ax.set_ylabel('pixels in y')

# %%
#
# Note how the origin is in the upper left and the x and y axises are in units
# of pixels. These behaviors can be changed with the *origin* and *extent*
# arguments. This is discussed in detail in :ref:`imshow_extent`.
#
# Other colormaps can be chosen through the *cmap* argument (see
# :ref:`colormaps`), and the color limits can be set with the *vmin* and *vmax*
# arguments, or by using the *norm* parameter (see :ref:`colormapnorms`).  An example
# of using *vmin* and *vmax* is shown below:

fig, ax = plt.subplots()
im = ax.imshow(Z, cmap='RdBu_r', vmin=-1, vmax=1)
fig.colorbar(im, ax=ax)
ax.set_xlabel('pixels in x')
ax.set_ylabel('pixels in y')

# %%
#
# There are many options for resampling the data when it is displayed.  See
# :ref:`image_antialiasing` for more information, but note the the default attempts
# to remove aliasing artifacts from data that is "downsampled" (i.e., when the
# number of pixels in the data array is greater than the number of pixels in the
# displayed image).
#
# .. seealso::
#   - :ref:`image_demo`
#   - :ref:`imshow_extent`
#   - :ref:`image_antialiasing`


# %%
# pcolormesh (and pcolor, pcolorfast)
# -----------------------------------
#
# `~.axes.Axes.pcolormesh` is more flexible than `~.axes.Axes.imshow` in that
# the x and y vectors need not be equally spaced (indeed they can be skewed).
# In the example below, the x values are unevenly spaced, and much less finely
# sampled than the y values.  The x and y data are orthogonal, so we can pass
# x and y directly to `~.axes.Axes.pcolormesh`:

x = np.array([-3, -2, -1.6, -1.2, -.8, -.5, -.2, .1, .3, .5, .8, 1.1, 1.5, 1.9, 2.3, 3])
y = np.linspace(-3, 3, 128)
X, Y = np.meshgrid(x, y)
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

fig, ax = plt.subplots()

ax.pcolormesh(x, y, Z, vmin=-0.5, vmax=1.0)

ax.set_xlabel('x')
ax.set_ylabel('y')

# %%
# As noted, `~.axes.Axes.pcolormesh` need not be on an orthogonal grid.  Here
# is an example with the y grid points varying with x.  Note that we need to
# give the grids as 2D arrays for this to work:

# make a 2D array of y values that where the y dimension varies in x:
Yn = Y + 0.3 * np.abs(x)

fig, ax = plt.subplots()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ax.pcolormesh(X, Yn, Z, vmin=-0.5, vmax=1.0)
ax.set_xlabel('x')
ax.set_ylabel('skewed y')

# %%
# Note that the above returns a warning because the y grid is not strictly increasing:
#
#    UserWarning: The input coordinates to pcolormesh are
#    interpreted as cell centers, but are not monotonically
#    increasing or decreasing.
#
# To avoid that warning, we can give `.Axes.pcolormesh` the cell edges rather
# than the cell centers:


def midpoints_plus_ends(x):
    """Given a 1D array, return a new array with the midpoints and the two ends."""
    return np.concatenate(([x[0] - (x[1] - x[0]) / 2],
                           x[:-1] + np.diff(x) / 2,
                           [x[-1] - (x[-1] - x[-2]) / 2]))

yn = midpoints_plus_ends(y)
xn = midpoints_plus_ends(x)
Xn, Yn = np.meshgrid(xn, yn)
Yn = Yn + 0.3 * np.abs(xn)

fig, ax = plt.subplots()
ax.pcolormesh(Xn, Yn, Z, vmin=-0.5, vmax=1.0)

# %%
# Two similar methods are `~.axes.Axes.pcolor` and `~.axes.Axes.pcolorfast`.  There are
# some differences in the way they handle the data, but they are largely the same and
# are not as commonly used as `~.axes.Axes.pcolormesh`. See
# :ref:`Differences between pcolor() and pcolormesh() <differences-pcolor-pcolormesh>`
# for a discussion of the differences.
#
# `~.axes.Axes.pcolorfast` is not as flexible as `~.axes.Axes.pcolormesh`, but can be
# faster for large datasets.

# %%
# .. seealso::
#   - :ref:`pcolormesh_grids`
#   - :ref:`pcolor_demo`

# %%
# contour and contourf
# --------------------
#
# `~.axes.Axes.contour` and `~.axes.Axes.contourf` create contour plots.  They accept
# a 2D array of data, and create a plot with contour lines or filled regions enclosing
# data that is in the same contour level.  The example below shows a simple contour
# plot using the data from above.  Note that as before, the x and y data are not
# necessarily evenly spaced, but in this case they are orthogonal.

fig, axs = plt.subplots(2, 1, figsize=(5, 5))
ax = axs[0]
ax.contour(x, y, Z, levels=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('contour plot')

ax = axs[1]
cf = ax.contourf(x, y, Z, levels=10)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('contourf plot')
fig.colorbar(cf, ax=ax)

# %%
# Note that like for `.Axes.pcolormesh`, the grid need not be orthogonal. In
# the following example the the same data is plotted as above, but with the y
# grid points varying with x. The contour *levels* are also manually specified
# in this example with a list of levels, (see :ref:`contourf_log` for an
# example of using a tick locator instead of a list), and the colormap set with
# the *cmap* argument (see :ref:`colormaps`).

Yn = Y + 0.3 * np.abs(x)

fig, ax = plt.subplots()

cf = ax.contourf(X, Yn, Z, levels=np.arange(-1.3, 1.31, 0.2), cmap='RdBu_r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('contour plot')
fig.colorbar(cf, ax=ax)

# %%
# .. seealso::
#
#    - :ref:`contour_demo`
#    - :ref:`contourf_demo`

# %%
# barbs and quiver
# ----------------
# `~.axes.Axes.barbs` and `~.axes.Axes.quiver` allow us to represent gridded
# vector fields specified at points on an x-y grid by two-dimensional arrays U,
# V. The arrays must be the same shape as x and y and the arrows are placed at
# the corresponding points in the grid.  The main difference between the two
# functions is that `~.axes.Axes.barbs` plots barbs, which are a more
# traditional representation of wind speed and direction, while
# `~.axes.Axes.quiver` plots arrows with a uniform size and the direction is
# given by the vector U, V.

# make data
x = np.linspace(-4, 4, 20)
y = np.linspace(-4, 4, 20)
X, Y = np.meshgrid(x, y)
U = X + Y
V = Y - X

# plot
fig, ax = plt.subplots()

ax.quiver(X, Y, U, V, color="C0", angles='xy',
          scale_units='xy', scale=8, width=.005)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('barbs')

fig, ax = plt.subplots()
ax.barbs(X, Y, U, V, color="C0", barbcolor='C0', flagcolor='C0', length=4,
         linewidth=1.0)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('quiver')

# %%
# streamplot
# ----------
#
# `~.axes.Axes.streamplot` is used to plot the streamlines of a vector field.  It
# makes traces of where a massless particle would go if it followed the vector field
# at each point.  The example below shows the streamlines of the vector field from
# above:

fig, ax = plt.subplots()
ax.streamplot(x, y, U, V)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('streamplot')

# %%
# `~.axes.Axes.streamplot` performs an interpolation and the streamlines are
# not guaranteed to be accurate.  For more accurate streamlines, the data grid
# could be refined.
#
# .. note::
#
#   `~.axes.Axes.streamplot` does not accept 2D arrays for the x and y data:
#   the x and y data must be 1D arrays, and the spacing between the points must
#   be uniform.
#
# .. seealso:: :ref:`Streamplot demo <plot_streamplot>`
#
# .. admonition:: References
#
#   The use of the following functions, methods, classes and modules is shown
#   in this example:
#
#   - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#   - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`
#   - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#   - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
#   - `matplotlib.axes.Axes.barbs` / `matplotlib.pyplot.barbs`
#   - `matplotlib.axes.Axes.quiver` / `matplotlib.pyplot.quiver`
#   - `matplotlib.axes.Axes.streamplot` / `matplotlib.pyplot.streamplot`
