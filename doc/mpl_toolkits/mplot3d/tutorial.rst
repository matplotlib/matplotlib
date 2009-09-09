.. _toolkit_mplot3d-tutorial:
.. currentmodule:: mpl_toolkits.mplot3d

****************
mplot3d tutorial
****************

Getting started
===============
Create a new :class:`matplotlib.figure.Figure` and an
:class:`~mpl_toolkits.mplot3d.Axes3D` object in it::

  import pylab
  fig = pylab.figure()
  from mpl_toolkits.mplot3d import Axes3D
  ax = Axes3D(fig)

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

Contour plots
=============
.. automethod:: Axes3D.contour

.. plot:: mpl_examples/mplot3d/contour3d_demo.py
.. plot:: mpl_examples/mplot3d/contour3d_demo2.py

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

