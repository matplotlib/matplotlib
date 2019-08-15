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

.. _plot3d:

Line plots
====================
.. automethod:: Axes3D.plot

.. figure:: ../../gallery/mplot3d/images/sphx_glr_lines3d_001.png
   :target: ../../gallery/mplot3d/lines3d.html
   :align: center
   :scale: 50

   Lines3d

.. _scatter3d:

Scatter plots
=============
.. automethod:: Axes3D.scatter

.. figure:: ../../gallery/mplot3d/images/sphx_glr_scatter3d_001.png
   :target: ../../gallery/mplot3d/scatter3d.html
   :align: center
   :scale: 50

   Scatter3d

.. _wireframe:

Wireframe plots
===============
.. automethod:: Axes3D.plot_wireframe

.. figure:: ../../gallery/mplot3d/images/sphx_glr_wire3d_001.png
   :target: ../../gallery/mplot3d/wire3d.html
   :align: center
   :scale: 50

   Wire3d

.. _surface:

Surface plots
=============
.. automethod:: Axes3D.plot_surface

.. figure:: ../../gallery/mplot3d/images/sphx_glr_surface3d_001.png
   :target: ../../gallery/mplot3d/surface3d.html
   :align: center
   :scale: 50

   Surface3d

   Surface3d 2

   Surface3d 3

.. _trisurface:

Tri-Surface plots
=================
.. automethod:: Axes3D.plot_trisurf

.. figure:: ../../gallery/mplot3d/images/sphx_glr_trisurf3d_001.png
   :target: ../../gallery/mplot3d/trisurf3d.html
   :align: center
   :scale: 50

   Trisurf3d


.. _contour3d:

Contour plots
=============
.. automethod:: Axes3D.contour

.. figure:: ../../gallery/mplot3d/images/sphx_glr_contour3d_001.png
   :target: ../../gallery/mplot3d/contour3d.html
   :align: center
   :scale: 50

   Contour3d

   Contour3d 2

   Contour3d 3

.. _contourf3d:

Filled contour plots
====================
.. automethod:: Axes3D.contourf

.. figure:: ../../gallery/mplot3d/images/sphx_glr_contourf3d_001.png
   :target: ../../gallery/mplot3d/contourf3d.html
   :align: center
   :scale: 50

   Contourf3d

   Contourf3d 2

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

   Polys3d

.. _bar3d:

Bar plots
====================
.. automethod:: Axes3D.bar

.. figure:: ../../gallery/mplot3d/images/sphx_glr_bars3d_001.png
   :target: ../../gallery/mplot3d/bars3d.html
   :align: center
   :scale: 50

   Bars3d

.. _quiver3d:

Quiver
====================
.. automethod:: Axes3D.quiver

.. figure:: ../../gallery/mplot3d/images/sphx_glr_quiver3d_001.png
   :target: ../../gallery/mplot3d/quiver3d.html
   :align: center
   :scale: 50

   Quiver3d

.. _2dcollections3d:

2D plots in 3D
====================
.. figure:: ../../gallery/mplot3d/images/sphx_glr_2dcollections3d_001.png
   :target: ../../gallery/mplot3d/2dcollections3d.html
   :align: center
   :scale: 50

   2dcollections3d

.. _text3d:

Text
====================
.. automethod:: Axes3D.text

.. figure:: ../../gallery/mplot3d/images/sphx_glr_text3d_001.png
   :target: ../../gallery/mplot3d/text3d.html
   :align: center
   :scale: 50

   Text3d

.. _3dsubplots:

Subplotting
====================
Having multiple 3D plots in a single figure is the same
as it is for 2D plots. Also, you can have both 2D and 3D plots
in the same figure.

.. versionadded:: 1.0.0
   Subplotting 3D plots was added in v1.0.0.  Earlier version can not
   do this.

.. figure:: ../../gallery/mplot3d/images/sphx_glr_subplot3d_001.png
   :target: ../../gallery/mplot3d/subplot3d.html
   :align: center
   :scale: 50

   Subplot3d

   Mixed Subplots
"""
