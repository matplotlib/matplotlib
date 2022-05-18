New external dependency ContourPy used for quad contour calculations
--------------------------------------------------------------------

Previously Matplotlib shipped its own C++ code for calculating the contours of
quad grids. Now the external library
`ContourPy <https://github.com/contourpy/contourpy>`_ is used instead. There
is a choice of four algorithms to use, controlled by the *algorithm* keyword
argument to the functions `~matplotlib.axes.Axes.contour` and
`~matplotlib.axes.Axes.contourf`. The default behaviour is to use
``algorithm='mpl2014'`` which is the same algorithm that Matplotlib has been
using since 2014.

See the `ContourPy documentation <https://contourpy.readthedocs.io>`_ for
further details of the different algorithms.

.. note::

   Contour lines and polygons produced by ``algorithm='mpl2014'`` will be the
   same as those produced before this change to within floating-point
   tolerance. The exception is for duplicate points, i.e. contours containing
   adjacent (x, y) points that are identical; previously the duplicate points
   were removed, now they are kept. Contours affected by this will produce the
   same visual output, but there will be a greater number of points in the
   contours.

   The locations of contour labels obtained by using
   `~matplotlib.axes.Axes.clabel` may also be different.
