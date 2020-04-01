"""
===================
The mplot3d Toolkit
===================

Generating 3D plots using the mplot3d toolkit.

.. currentmodule:: mpl_toolkits.mplot3d

.. contents::
      :backlinks: none

.. _toolkit_mplot3d-tutorial:

Getting started
---------------
3D Axes (of class `.Axes3D`) are created by passing the ``projection="3d"``
keyword argument to `.Figure.add_subplot`::

   import matplotlib.pyplot as plt
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

.. versionchanged:: 1.0.0
   Prior to Matplotlib 1.0.0, `.Axes3D` needed to be directly instantiated with
   ``from mpl_toolkits.mplot3d import Axes3D; ax = Axes3D(fig)``.

.. versionchanged:: 3.2.0
   Prior to Matplotlib 3.2.0, it was necessary to explicitly import the
   :mod:`mpl_toolkits.mplot3d` module to make the '3d' projection to
   `.Figure.add_subplot`.

See the :ref:`toolkit_mplot3d-faq` for more information about the mplot3d
toolkit.

Line plots 3D
-------------

One of the most common transformations is plot3D. If you want to draw a plot, try the following. 
Three-dimensional plots are similar to two-dimensional plots, but different parts exist.::

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D # Changed in version 3.2.0
    import numpy as np 

    fig = plt.figure()
    #fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d') # Axe3D object

    sample_size = 50
    x = np.cumsum(np.random.normal(0, 1, sample_size))
    y = np.cumsum(np.random.normal(0, 1, sample_size))
    z = np.cumsum(np.random.normal(0, 1, sample_size))
    ax.plot(x, y, z, alpha=0.6)
    plt.title("ax.plot")
    plt.show()

If you want to change figure size, you can change. ::

    #fig = plt.figure()
    fig = plt.figure(figsize=(10, 5))

Also, you can draw parametric curve.::

   import numpy as np
   import matplotlib.pyplot as plt

   plt.rcParams['legend.fontsize'] = 10

   fig = plt.figure()
   ax = fig.gca(projection='3d')

   # Prepare arrays x, y, z
   theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
   z = np.linspace(-2, 2, 100)
   r = z**2 + 1
   x = r * np.sin(theta)
   y = r * np.cos(theta)

   ax.plot(x, y, z, label='parametric curve')
   ax.legend()

   plt.show()


Scatter plots
-------------

If you want to draw Scatter plots 3D, you can do like this. 
It's a way to scatter the dots and present them on the 3D screen.
Here, we expressed it in two ways: o shape and ^ shape.::

   import matplotlib.pyplot as plt
   import numpy as np

   # Fixing random state for reproducibility
   np.random.seed(19680801)
   def randrange(n, vmin, vmax):
      '''
      Helper function to make an array of random numbers having shape (n, )
      with each number distributed Uniform(vmin, vmax).
      '''
      return (vmax - vmin)*np.random.rand(n) + vmin

   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   n = 100
   # For each set of style and range settings, plot n random points in the box
   # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
   for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
      xs = randrange(n, 23, 32)
      ys = randrange(n, 0, 100)
      zs = randrange(n, zlow, zhigh)
      ax.scatter(xs, ys, zs, marker=m)

   ax.set_xlabel('X Label')
   ax.set_ylabel('Y Label')
   ax.set_zlabel('Z Label')

   plt.show()


Wireframe plots
---------------
Wireframe plot the entire 3d surface is not shown. Instead, the surface is 
approximated with "wires" laid over the 3D surface. Wire frame 3D 
surface plots can be constructed using `.ax.plot_wireframe()` method.
The general method is below.::

   ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

Wireframe plot is also one of the many ways to express it on the 3d screen.
It's a good way to express something a little more sensuously 
when you're expressing the height of a figure. 
Let's take a look at the example below.::

   from mpl_toolkits.mplot3d import axes3d
   import matplotlib.pyplot as plt


   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   # Grab some test data.
   X, Y, Z = axes3d.get_test_data(0.05)

   # Plot a basic wireframe.
   ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

   plt.show()


Surface plots
-------------
Surface plots are created with `.ax.plot_surface()` method. 
By default, surface plots are a single color. The general format 
of `.ax.plot_surface()` method is below.::

   ax.plot_surface(X, Y, Z)

By default it will be colored in shades of a solid color, 
but it also supports color mapping by supplying the cmap argument.::

   # This import registers the 3D projection, but is otherwise unused.
   from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

   import matplotlib.pyplot as plt
   from matplotlib import cm
   from matplotlib.ticker import LinearLocator, FormatStrFormatter
   import numpy as np


   fig = plt.figure()
   ax = fig.gca(projection='3d')

   # Make data.
   X = np.arange(-5, 5, 0.25)
   Y = np.arange(-5, 5, 0.25)
   X, Y = np.meshgrid(X, Y)
   R = np.sqrt(X**2 + Y**2)
   Z = np.sin(R)

   # Plot the surface.
   surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

   # Customize the z axis.
   ax.set_zlim(-1.01, 1.01)
   ax.zaxis.set_major_locator(LinearLocator(10))
   ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

   # Add a color bar which maps values to colors.
   fig.colorbar(surf, shrink=0.5, aspect=5)

   plt.show()


Tri-Surface plots
-----------------
The (optional) triangulation can be specified in one of two ways; either::

   plot_trisurf(triangulation, ...)

where triangulation is a Triangulation object, or::

   plot_trisurf(X, Y, ...)
   plot_trisurf(X, Y, triangles, ...)
   plot_trisurf(X, Y, triangles=triangles, ...)

Plot a triangulated surface. This import registers the 3D projection, 
but is otherwise unused.::

   from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

   import matplotlib.pyplot as plt
   import numpy as np


   n_radii = 8
   n_angles = 36

   # Make radii and angles spaces (radius r=0 omitted to eliminate duplication).
   radii = np.linspace(0.125, 1.0, n_radii)
   angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]

   # Convert polar (radii, angles) coords to cartesian (x, y) coords.
   # (0, 0) is manually added at this stage,  so there will be no duplicate
   # points in the (x, y) plane.
   x = np.append(0, (radii*np.cos(angles)).flatten())
   y = np.append(0, (radii*np.sin(angles)).flatten())

   # Compute z to make the pringle surface.
   z = np.sin(-x*y)

   fig = plt.figure()
   ax = fig.gca(projection='3d')

   ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)

   plt.show()

Two additional examples of plotting surfaces with triangular mesh.
Try it yourself.::

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.tri as mtri

   # This import registers the 3D projection, but is otherwise unused.
   from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


   fig = plt.figure(figsize=plt.figaspect(0.5))

   #============
   # First plot
   #============

   # Make a mesh in the space of parameterisation variables u and v
   u = np.linspace(0, 2.0 * np.pi, endpoint=True, num=50)
   v = np.linspace(-0.5, 0.5, endpoint=True, num=10)
   u, v = np.meshgrid(u, v)
   u, v = u.flatten(), v.flatten()

   # This is the Mobius mapping, taking a u, v pair and returning an x, y, z
   # triple
   x = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
   y = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
   z = 0.5 * v * np.sin(u / 2.0)

   # Triangulate parameter space to determine the triangles
   tri = mtri.Triangulation(u, v)

   # Plot the surface.  The triangles in parameter space determine which x, y, z
   # points are connected by an edge.
   ax = fig.add_subplot(1, 2, 1, projection='3d')
   ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
   ax.set_zlim(-1, 1)


   #============
   # Second plot
   #============

   # Make parameter spaces radii and angles.
   n_angles = 36
   n_radii = 8
   min_radius = 0.25
   radii = np.linspace(min_radius, 0.95, n_radii)

   angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
   angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
   angles[:, 1::2] += np.pi/n_angles

   # Map radius, angle pairs to x, y, z points.
   x = (radii*np.cos(angles)).flatten()
   y = (radii*np.sin(angles)).flatten()
   z = (np.cos(radii)*np.cos(3*angles)).flatten()

   # Create the Triangulation; no triangles so Delaunay triangulation created.
   triang = mtri.Triangulation(x, y)

   # Mask off unwanted triangles.
   xmid = x[triang.triangles].mean(axis=1)
   ymid = y[triang.triangles].mean(axis=1)
   mask = xmid**2 + ymid**2 < min_radius**2
   triang.set_mask(mask)

   # Plot the surface.
   ax = fig.add_subplot(1, 2, 2, projection='3d')
   ax.plot_trisurf(triang, z, cmap=plt.cm.CMRmap)


   plt.show()

Contour plots
-------------
The ax.contour() function creates three-dimensional contour plot. 
It requires all the input data to be in the form of two-dimensional 
regular grids, with the Z-data evaluated at each point. Here, we will 
show a three-dimensional contour diagram of a three-dimensional sinusoidal function.::

   from mpl_toolkits.mplot3d import axes3d
   import matplotlib.pyplot as plt
   from matplotlib import cm

   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')
   X, Y, Z = axes3d.get_test_data(0.05)
   cset = ax.contour(X, Y, Z, cmap=cm.coolwarm)
   ax.clabel(cset, fontsize=9, inline=1)

   plt.show()

Filled contour plots
--------------------
It's a contour similar to the Contour but filled. 
Examples of use are very similar to contours.::

   from mpl_toolkits.mplot3d import axes3d
   import matplotlib.pyplot as plt
   from matplotlib import cm

   fig = plt.figure()
   ax = fig.gca(projection='3d')
   X, Y, Z = axes3d.get_test_data(0.05)

   cset = ax.contourf(X, Y, Z, cmap=cm.coolwarm)

   ax.clabel(cset, fontsize=9, inline=1)

   plt.show()

Polygon plots
-------------
Demonstrate how to create polygons which fill the space under a line graph. 
In this example polygons are semi-transparent, creating a sort of 
'jagged stained glass' effect.::

   from matplotlib.collections import PolyCollection
   import matplotlib.pyplot as plt
   from matplotlib import colors as mcolors
   import numpy as np

   # Fixing random state for reproducibility
   np.random.seed(19680801)


   def polygon_under_graph(xlist, ylist):
      '''
      Construct the vertex list which defines the polygon filling the space under
      the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
      '''
      return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]


   fig = plt.figure()
   ax = fig.gca(projection='3d')

   # Make verts a list such that verts[i] is a list of (x, y) pairs defining
   # polygon i.
   verts = []

   # Set up the x sequence
   xs = np.linspace(0., 10., 26)

   # The ith polygon will appear on the plane y = zs[i]
   zs = range(4)

   for i in zs:
      ys = np.random.rand(len(xs))
      verts.append(polygon_under_graph(xs, ys))

   poly = PolyCollection(verts, facecolors=['r', 'g', 'b', 'y'], alpha=.6)
   ax.add_collection3d(poly, zs=zs, zdir='y')

   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_xlim(0, 10)
   ax.set_ylim(-1, 4)
   ax.set_zlim(0, 1)

   plt.show()

Bar plots
---------
Demonstrates making a 3D plot which has 2D bar 
graphs projected onto planes y=0, y=1, etc.::

   import matplotlib.pyplot as plt
   import numpy as np

   # Fixing random state for reproducibility
   np.random.seed(19680801)


   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   colors = ['r', 'g', 'b', 'y']
   yticks = [3, 2, 1, 0]
   for c, k in zip(colors, yticks):
      # Generate the random data for the y=k 'layer'.
      xs = np.arange(20)
      ys = np.random.rand(20)

      # You can provide either a single color or an array with the same length as
      # xs and ys. To demonstrate this, we color the first bar of each set cyan.
      cs = [c] * len(xs)
      cs[0] = 'c'

      # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
      ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')

   # On the y axis let's only label the discrete values that we have data for.
   ax.set_yticks(yticks)

   plt.show()

Quiver
------
Demonstrates plotting directional arrows at points on a 3d meshgrid.::

   import matplotlib.pyplot as plt
   import numpy as np

   fig = plt.figure()
   ax = fig.gca(projection='3d')

   # Make the grid
   x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.2),
                        np.arange(-0.8, 1, 0.8))

   # Make the direction data for the arrows
   u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
   v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
   w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
      np.sin(np.pi * z))

   ax.quiver(x, y, z, u, v, w, length=0.1, normalize=True)

   plt.show()

2D plots in 3D
--------------
Demonstrates using ax.plot's zdir keyword to plot 2D data on selective 
axes of a 3D plot. A plot to transform 2D data into 3D. Let's take a look 
at the example below.::

   import numpy as np
   import matplotlib.pyplot as plt

   fig = plt.figure()
   ax = fig.gca(projection='3d')

   # Plot a sin curve using the x and y axes.
   x = np.linspace(0, 1, 100)
   y = np.sin(x * 2 * np.pi) / 2 + 0.5
   ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')

   # Plot scatterplot data (20 2D points per colour) on the x and z axes.
   colors = ('r', 'g', 'b', 'k')

   # Fixing random state for reproducibility
   np.random.seed(19680801)

   x = np.random.sample(20 * len(colors))
   y = np.random.sample(20 * len(colors))
   c_list = []
   for c in colors:
      c_list.extend([c] * 20)
   # By using zdir='y', the y value of these points is fixed to the zs value 0
   # and the (x, y) points are plotted on the x and z axes.
   ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x, z)')

   # Make legend, set axes limits and labels
   ax.legend()
   ax.set_xlim(0, 1)
   ax.set_ylim(0, 1)
   ax.set_zlim(0, 1)
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')

   # Customize the view angle so it's easier to see that the scatter points lie
   # on the plane y=0
   ax.view_init(elev=20., azim=-35)

   plt.show()

Text
----
Demonstrates the placement of text annotations on a 3D plot.

Functionality shown:

Using the text function with three types of 'zdir' values: None, an axis 
name (ex. 'x'), or a direction tuple (ex. (1, 1, 0)).
Using the text function with the color keyword.
Using the text2D function to place text on a fixed position on the ax object.::

   import matplotlib.pyplot as plt


   fig = plt.figure()
   ax = fig.gca(projection='3d')

   # Demo 1: zdir
   zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
   xs = (1, 4, 4, 9, 4, 1)
   ys = (2, 5, 8, 10, 1, 2)
   zs = (10, 3, 8, 9, 1, 8)

   for zdir, x, y, z in zip(zdirs, xs, ys, zs):
      label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
      ax.text(x, y, z, label, zdir)

   # Demo 2: color
   ax.text(9, 0, 0, "red", color='red')

   # Demo 3: text2D
   # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
   ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)

   # Tweaking display region and labels
   ax.set_xlim(0, 10)
   ax.set_ylim(0, 10)
   ax.set_zlim(0, 10)
   ax.set_xlabel('X axis')
   ax.set_ylabel('Y axis')
   ax.set_zlabel('Z axis')

   plt.show()


Subplotting
-----------
Having multiple 3D plots in a single figure is the same
as it is for 2D plots. Also, you can have both 2D and 3D plots
in the same figure. Demonstrate including 3D plots as subplots.::

   import matplotlib.pyplot as plt
   from matplotlib import cm
   import numpy as np

   from mpl_toolkits.mplot3d.axes3d import get_test_data


   # set up a figure twice as wide as it is tall
   fig = plt.figure(figsize=plt.figaspect(0.5))

   #===============
   #  First subplot
   #===============
   # set up the axes for the first plot
   ax = fig.add_subplot(1, 2, 1, projection='3d')

   # plot a 3D surface like in the example mplot3d/surface3d_demo
   X = np.arange(-5, 5, 0.25)
   Y = np.arange(-5, 5, 0.25)
   X, Y = np.meshgrid(X, Y)
   R = np.sqrt(X**2 + Y**2)
   Z = np.sin(R)
   surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
   ax.set_zlim(-1.01, 1.01)
   fig.colorbar(surf, shrink=0.5, aspect=10)

   #===============
   # Second subplot
   #===============
   # set up the axes for the second plot
   ax = fig.add_subplot(1, 2, 2, projection='3d')

   # plot a 3D wireframe like in the example mplot3d/wire3d_demo
   X, Y, Z = get_test_data(0.05)
   ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

   plt.show()

""" 