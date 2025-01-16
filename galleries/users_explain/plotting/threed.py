"""
.. _plotting_threed:

3D and volumetric data
======================

Matplotlib has support for three-dimensional plots, including surface plots,
wireframes, and 3D scatter plots.

Note, however, that Matplotlib does not do proper ray-tracing, and hence
multiple surfaces in one visualization will not occlude one another correctly;
the last surface drawn will cover previous surfaces, regardless of their
distance from the viewer.  Either try to combine surfaces into one (which is
possible for simple plots, see :ref:`intersecting_planes` and :ref:`box3d`), or
for true 3D, consider using `Mayavi
<https://docs.enthought.com/mayavi/mayavi/>`_.

Plotting in 3D can be done using the `.mpl_toolkits.mplot3d` toolkit by
declaring an axes is three dimensional, instead of the default two-dimensional.
After that, most of the methods are the same as their two-dimensional
counterparts, except a third array is added to represent the data along the
z-axis.
"""

# %%
# plot and scatter
# ----------------
#
# The most basic 3D plot is a line or collection of scatter plot created from
# sets of (x, y, z) triples.  In analogy with the more common 2D plots, the
# 3D plots are simple to create.  We use the *subplot_kw* argument to create a
# 3D projection for the plot.

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (5, 4)

# make data
n = 100
xs = np.linspace(0, 1, n)
ys = np.sin(xs * 6 * np.pi)
zs = np.cos(xs * 6 * np.pi)

# Make an axes with a 3d projection
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# plot
ax.plot(xs, ys, zs)

# label
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3d plot')


# %%
# For 3D plots, it is possible to rotate the plot by clicking and dragging in
# an interactive backend.  For a static backend, the view can be changed by
# setting the elevation and azimuth angles.  The elevation is the angle above
# the x-y plane, and the azimuth is the rotation around the z-axis.  The
# default view is elevation of 30 degrees and an azimuth of -60 degrees.  The
# azimuth is zero looking along the positive x-axis at the y-z plane, and if
# elevation is positive, then increasing azimuth rotates the view in a
# clockwise direction.

# Make an axes with a 3d projection
fig, axs = plt.subplots(2, 3, figsize=(8.5, 6.5), subplot_kw={"projection": "3d"})

for nn, ax in enumerate(axs.flat):

    # change the view
    ax.view_init(elev=30, azim=((-60 + nn*20)))

    ax.plot([0, 1], [0, 0], -1, 'r')
    ax.plot([1], [0], -1, 'dr')
    ax.plot(xs, ys, zs)
    ax.plot(xs[0], ys[0], zs[0], 'go')
    ax.plot(xs[-1], ys[-1], zs[-1], 'ro')

    ax.set_title(f'view_init(30, {(-60 + nn*20)})')

    # label
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

# %%
# We can also create a 3D scatter plot, analogous to a 2D scatter plot, by
# passing arrays of x, y, and z coordinates to the *scatter* method, and
# specifying *c* and *s* to set the color and size of the
# points, if desired.  Here we set the color to be the same as the x value, and
# the size as the absolute value of the z value.

# make data
c = xs
s = np.abs(zs) * 100

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
sc = ax.scatter(xs, ys, zs, c=c, s=s)
# label
fig.colorbar(sc, ax=ax, location='left', shrink=0.4)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# %%
# plot_wireframe, plot_surface
# ----------------------------
#
# The `~.Axes3D.plot_wireframe` method can be used to plot a wireframe
# version of a surface.  Wireframe spacing is set by the underlying data, but
# can be adjusted by setting the *rstride* (row-stride) and *cstride*
# (column-stride) of the plot if needed.

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

fig, axs = plt.subplots(1, 2, figsize=(6, 3.5), subplot_kw={"projection": "3d"})
axs[0].plot_wireframe(X, Y, Z)
axs[0].set_title('plot_wireframe')

axs[1].plot_wireframe(X, Y, Z, rstride=3, cstride=3, color='C2')
axs[1].set_title('rstride=3, cstride=3')

# %%
# The `.Axes3D.plot_surface` method can be used in the same way to plot a
# surface plot.  If the colormap is not specified then the surface is colored
# with one color, but shading is applied to the surface to give it a 3D
# appearance.  If a colormap via the *cmap* keyword is specified then the
# surface will be colored by the Z value of the surface.  It is possible to
# shade a color-mapped surface by passing the rgba values of the surface colors
# via the *facecolors* keyword (see :ref:`custom_shaded_3d_surface` for
# details).

fig, axs = plt.subplots(1, 2, figsize=(6, 3.5), subplot_kw={"projection": "3d"})
axs[0].plot_surface(X, Y, Z)
axs[0].set_title('plot_surface')

axs[1].plot_surface(X, Y, Z, cmap='viridis')
axs[1].set_title('cmap="viridis"')


# %%
# .. seealso::
#   - :ref:`surface3d`
#   - :ref:`surface3d_2`
#   - :ref:`surface3d_3`
#

# The `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf` method can be used
# just like `~.axes.Axes.tripcolor` when the x and y data is not on a Cartesian
# grid. It makes a triangulation in x and y and plots the surface in 3D.

rng = np.random.default_rng(19680801)
x = rng.uniform(-5, 5, size=128)
y = rng.uniform(-5, 5, size=128)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.plot_trisurf(x, y, z)
ax.set_title('plot_trisurf')

# %%
#
# .. seealso::
#   - :ref:`trisurf3d`
#   - :ref:`trisurf3d_2`
#
# stem and bar3d
# --------------
#
# The `~mpl_toolkits.mplot3d.axes3d.Axes3D.bar3d` and
# `~mpl_toolkits.mplot3d.axes3d.Axes3D.stem` methods can give an idea of
# two-dimensional data counts, similar to `~.axes.Axes.hist2d`.  So for
# instance if we histogram a random array of data:

x = rng.normal(size=5000)
y = rng.normal(size=5000)

H, xedges, yedges, = np.histogram2d(x, y, bins=10)
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

X = X.flatten()
Y = Y.flatten()
H = H.flatten()


fig, axs = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"projection": "3d"})

axs[0].bar3d(X, Y, np.zeros_like(X),
             dx=0.2 * np.ones_like(X), dy=0.2 * np.ones_like(X),
             dz=H.flatten(), shade=True)

axs[0].set_title('bar3d')

axs[1].stem(X.flat, Y.flat, H.flatten())
axs[1].set_title('stem')

# %%
#
# .. seealso::
#  - :ref:`bars3d`
#  - :ref:`stem3d_demo`
#
# quiver
# ------
#
# There is a three-dimensional `.Axes3D.quiver` equivalent of
# `~.axes.Axes.quiver` that is called the same way, but with a third array for
# the z direction.  Here we plot a 3D field of arrows:

# Make data
n = 4
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
z = np.linspace(-1, 1, n)
X, Y, Z = np.meshgrid(x, y, z)
U = (X + Y)/5
V = (Y - X)/5
W = Z*0

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.quiver(X, Y, Z, U, V, W)

# %%
# .. seealso::
#   - :ref:`quiver3d`
#
# .. admonition:: References
#
#   The use of the following functions, methods, classes and modules is shown
#   in this document:
#
#   - `.Axes3D.plot`
#   - `.Axes3D.scatter`
#   - `.Axes3D.plot_wireframe`
#   - `.Axes3D.plot_surface`
#   - `.Axes3D.plot_trisurf`
#   - `.Axes3D.bar3d`
#   - `.Axes3D.stem`
#   - `.Axes3D.quiver`
