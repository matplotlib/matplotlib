.. _toolkit_mplot3d-tutorial:
.. currentmodule:: mpl_toolkits.mplot3d

****************
mplot3d tutorial
****************
.. contents::
      :backlinks: none

.. _mplot3d_getting_started:

Getting started
===============
An Axes3D object is created just like any other axes using
the projection='3d' keyword.
Create a new :class:`matplotlib.figure.Figure` and
add a new axes to it of type :class:`~mpl_toolkits.mplot3d.Axes3D`::

  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

.. versionadded:: 1.0.0
   This approach is the preferred method of creating a 3D axes.

.. note::
   Prior to version 1.0.0, the method of creating a 3D axes was
   different. For those using older versions of matplotlib, change
   ``ax = fig.add_subplot(111, projection='3d')``
   to ``ax = Axes3D(fig)``.

.. _plot3d:

Line plots
====================
.. automethod:: Axes3D.plot

.. plot:: mpl_examples/mplot3d/lines3d_demo.py

.. _scatter3d:

Scatter plots
=============
.. automethod:: Axes3D.scatter

.. plot:: mpl_examples/mplot3d/scatter3d_demo.py

.. _wireframe:

Wireframe plots
===============
.. automethod:: Axes3D.plot_wireframe

.. plot:: mpl_examples/mplot3d/wire3d_demo.py

.. _surface:

Surface plots
=============
.. automethod:: Axes3D.plot_surface

.. plot:: mpl_examples/mplot3d/surface3d_demo.py
.. plot:: mpl_examples/mplot3d/surface3d_demo2.py
.. plot:: mpl_examples/mplot3d/surface3d_demo3.py

.. _trisurface:

Tri-Surface plots
=================
.. automethod:: Axes3D.plot_trisurf

.. plot:: mpl_examples/mplot3d/trisurf3d_demo.py


.. _contour3d:

Contour plots
=============
.. automethod:: Axes3D.contour

.. plot:: mpl_examples/mplot3d/contour3d_demo.py
.. plot:: mpl_examples/mplot3d/contour3d_demo2.py
.. plot:: mpl_examples/mplot3d/contour3d_demo3.py

.. _contourf3d:

Filled contour plots
====================
.. automethod:: Axes3D.contourf

.. plot:: mpl_examples/mplot3d/contourf3d_demo.py
.. plot:: mpl_examples/mplot3d/contourf3d_demo2.py

.. versionadded:: 1.1.0
   The feature demoed in the second contourf3d example was enabled as a
   result of a bugfix for version 1.1.0.

.. _polygon3d:

Polygon plots
====================
.. automethod:: Axes3D.add_collection3d

.. plot:: mpl_examples/mplot3d/polys3d_demo.py

.. _bar3d:

Bar plots
====================
.. automethod:: Axes3D.bar

.. plot:: mpl_examples/mplot3d/bars3d_demo.py

.. _quiver3d:

Quiver
====================
.. automethod:: Axes3D.quiver

.. plot:: mpl_examples/mplot3d/quiver3d_demo.py

.. _2dcollections3d:

2D plots in 3D
====================
.. plot:: mpl_examples/mplot3d/2dcollections3d_demo.py

.. _text3d:

Text
====================
.. automethod:: Axes3D.text

.. plot:: mpl_examples/mplot3d/text3d_demo.py

.. _3dsubplots:

Subplotting
====================
Having multiple 3D plots in a single figure is the same
as it is for 2D plots. Also, you can have both 2D and 3D plots
in the same figure.

.. versionadded:: 1.0.0
   Subplotting 3D plots was added in v1.0.0.  Earlier version can not
   do this.

.. plot:: mpl_examples/mplot3d/subplot3d_demo.py
.. plot:: mpl_examples/mplot3d/mixed_subplots_demo.py
