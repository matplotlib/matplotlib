
.. _toolkit_axesgrid1-index:

Matplotlib axes_grid1 Toolkit
=============================

The matplotlib :class:`mpl_toolkits.axes_grid1` toolkit is a collection of
helper classes to ease displaying multiple images in matplotlib.  While the
aspect parameter in matplotlib adjust the position of the single axes,
axesgrid1 toolkit provides a framework to adjust the position of
multiple axes according to their aspects.


.. image:: ../../_static/demo_axes_grid.png

.. note::
   AxesGrid toolkit has been a part of matplotlib since v
   0.99. Originally, the toolkit had a single namespace of
   *axes_grid*. In more recent version, the toolkit
   has divided into two separate namespace (*axes_grid1* and *axisartist*).
   While *axes_grid* namespace is maintained for the backward compatibility,
   use of *axes_grid1* and *axisartist* is recommended.

.. toctree::
   :maxdepth: 2

   overview.rst
   users/index.rst
   api/index.rst
