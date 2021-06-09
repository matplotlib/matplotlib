"""
===================
The mplot3d Toolkit
===================

Generating 3D plots using the mplot3d toolkit.

.. currentmodule:: mpl_toolkits.mplot3d

.. contents::
   :backlinks: none

3D Axes (of class `.Axes3D`) are created by passing the ``projection="3d"``
keyword argument to `.Figure.add_subplot`::

   import matplotlib.pyplot as plt
   fig = plt.figure()
   ax = fig.add_subplot(projection='3d')

Multiple 3D subplots can be added on the same figure, as for 2D subplots.

.. figure:: ../../gallery/mplot3d/images/sphx_glr_subplot3d_001.png
   :target: ../../gallery/mplot3d/subplot3d.html
   :align: center
   :scale: 50

.. versionchanged:: 1.0.0
   Prior to Matplotlib 1.0.0, only a single `.Axes3D` could be created per
   figure; it needed to be directly instantiated as ``ax = Axes3D(fig)``.

.. versionchanged:: 3.2.0
   Prior to Matplotlib 3.2.0, it was necessary to explicitly import the
   :mod:`mpl_toolkits.mplot3d` module to make the '3d' projection to
   `.Figure.add_subplot`.

See the :ref:`toolkit_mplot3d-faq` for more information about the mplot3d
toolkit.

.. _plot3d:

Line plots
====================
.. automethod:: Axes3D.plot

.. figure:: ../../gallery/mplot3d/images/sphx_glr_lines3d_001.png
   :target: ../../gallery/mplot3d/lines3d.html
   :align: center
   :scale: 50

.. _scatter3d:

Scatter plots
=============
.. automethod:: Axes3D.scatter

.. figure:: ../../gallery/mplot3d/images/sphx_glr_scatter3d_001.png
   :target: ../../gallery/mplot3d/scatter3d.html
   :align: center
   :scale: 50

.. _wireframe:

Wireframe plots
===============
.. automethod:: Axes3D.plot_wireframe

.. figure:: ../../gallery/mplot3d/images/sphx_glr_wire3d_001.png
   :target: ../../gallery/mplot3d/wire3d.html
   :align: center
   :scale: 50

.. _surface:

Surface plots
=============
.. automethod:: Axes3D.plot_surface

.. figure:: ../../gallery/mplot3d/images/sphx_glr_surface3d_001.png
   :target: ../../gallery/mplot3d/surface3d.html
   :align: center
   :scale: 50

.. _trisurface:

Tri-Surface plots
=================
.. automethod:: Axes3D.plot_trisurf

.. figure:: ../../gallery/mplot3d/images/sphx_glr_trisurf3d_001.png
   :target: ../../gallery/mplot3d/trisurf3d.html
   :align: center
   :scale: 50

.. _contour3d:

Contour plots
=============
.. automethod:: Axes3D.contour

.. figure:: ../../gallery/mplot3d/images/sphx_glr_contour3d_001.png
   :target: ../../gallery/mplot3d/contour3d.html
   :align: center
   :scale: 50

.. _contourf3d:

Filled contour plots
====================
.. automethod:: Axes3D.contourf

.. figure:: ../../gallery/mplot3d/images/sphx_glr_contourf3d_001.png
   :target: ../../gallery/mplot3d/contourf3d.html
   :align: center
   :scale: 50

.. versionadded:: 1.1.0
   The feature demoed in the second contourf3d example was enabled as a
   result of a bugfix for version 1.1.0.

.. _polygon3d:

Polygon plots
====================
.. automethod:: Axes3D.add_collection3d

.. figure:: ../../gallery/mplot3d/images/sphx_glr_polys3d_001.png
   :target: ../../gallery/mplot3d/polys3d.html
   :align: center
   :scale: 50

.. _bar3d:

Bar plots
====================
.. automethod:: Axes3D.bar

.. figure:: ../../gallery/mplot3d/images/sphx_glr_bars3d_001.png
   :target: ../../gallery/mplot3d/bars3d.html
   :align: center
   :scale: 50

.. _quiver3d:

Quiver
====================
.. automethod:: Axes3D.quiver

.. figure:: ../../gallery/mplot3d/images/sphx_glr_quiver3d_001.png
   :target: ../../gallery/mplot3d/quiver3d.html
   :align: center
   :scale: 50

.. _2dcollections3d:

2D plots in 3D
====================
.. figure:: ../../gallery/mplot3d/images/sphx_glr_2dcollections3d_001.png
   :target: ../../gallery/mplot3d/2dcollections3d.html
   :align: center
   :scale: 50

.. _text3d:

Text
====================
.. automethod:: Axes3D.text

.. figure:: ../../gallery/mplot3d/images/sphx_glr_text3d_001.png
   :target: ../../gallery/mplot3d/text3d.html
   :align: center
   :scale: 50
"""
