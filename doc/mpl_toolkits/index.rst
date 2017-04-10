.. _toolkits-index:

.. toctree::
   :hidden:

   axes_grid1/index.rst
   axisartist/index.rst
   mplot3d/index.rst

.. _toolkits:

########
Toolkits
########



Toolkits are collections of application-specific functions that extend
Matplotlib.


.. _toolkit_mplot3d:

mplot3d
=======


:ref:`mpl_toolkits.mplot3d <toolkit_mplot3d-index>` provides some basic 3D
plotting (scatter, surf, line, mesh) tools.  Not the fastest or feature
complete 3D library out there, but ships with Matplotlib and thus may be a
lighter weight solution for some use cases.

.. plot:: examples_gallery/mplot3d/plot_contourf3d_2.py

.. _toolkit_axes_grid1:

axes_grid1
==========


The :ref:`mpl_toolkits.axes_grid1 <toolkit_axesgrid1-index>` toolkit is a
collection of helper classes to ease displaying multiple axes in Matplotlib.



.. image:: /_static/demo_axes_grid.png


.. _toolkit_axisartist:

axisartist
==========


The :ref:`mpl_toolkits.axisartist <toolkit_axisartist-index>` toolkit contains
a custom Axes class that is meant to support for curvilinear grids.
