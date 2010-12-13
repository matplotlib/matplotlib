.. _toolkit_mplot3d-tutorial:
.. currentmodule:: mpl_toolkits.mplot3d

****************
mplot3d tutorial
****************

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

Line plots
====================
.. automethod:: Axes3D.plot

.. plot:: mpl_examples/mplot3d/lines3d_demo.py

Scatter plots
=============
.. automethod:: Axes3D.scatter

.. plot:: mpl_examples/mplot3d/scatter3d_demo.py

Wireframe plots
===============
.. automethod:: Axes3D.plot_wireframe

.. plot:: mpl_examples/mplot3d/wire3d_demo.py

Surface plots
=============
.. automethod:: Axes3D.plot_surface

.. plot:: mpl_examples/mplot3d/surface3d_demo.py
.. plot:: mpl_examples/mplot3d/surface3d_demo2.py
.. plot:: mpl_examples/mplot3d/surface3d_demo3.py

Contour plots
=============
.. automethod:: Axes3D.contour

.. plot:: mpl_examples/mplot3d/contour3d_demo.py
.. plot:: mpl_examples/mplot3d/contour3d_demo2.py
.. plot:: mpl_examples/mplot3d/contour3d_demo3.py

Filled contour plots
====================
.. automethod:: Axes3D.contourf

.. plot:: mpl_examples/mplot3d/contourf3d_demo.py

Polygon plots
====================
.. automethod:: Axes3D.add_collection3d

.. plot:: mpl_examples/mplot3d/polys3d_demo.py

Bar plots
====================
.. automethod:: Axes3D.bar

.. plot:: mpl_examples/mplot3d/bars3d_demo.py

2D plots in 3D
====================
.. plot:: mpl_examples/mplot3d/2dcollections3d_demo.py

Text
====================
.. automethod:: Axes3D.text

.. plot:: mpl_examples/mplot3d/text3d_demo.py

Subplotting
====================
Having multiple 3D plots in a single figure is the same
as it is for 2D plots. And you can mix 2D and 3D plots
into the same figure.

.. plot:: mpl_examples/mplot3d/subplot3d_demo.py
.. plot:: mpl_examples/mplot3d/mixed_subplots_demo.py
