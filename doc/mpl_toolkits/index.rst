.. _toolkits-index:

.. toctree::
   :hidden:

   axes_grid/index.rst
   mplot3d/index.rst


########
Toolkits
########

.. htmlonly::

   :Release: |version|
   :Date: |today|

.. _toolkits:

Toolkits are collections of application-specific functions that extend matplotlib.

.. _toolkit_basemap:

Basemap (*Not distributed with matplotlib*)
============================================

Plots data on map projections, with continental and political
boundaries, see `basemap <http://matplotlib.org/basemap>`_
docs.

.. image:: http://matplotlib.org/basemap/_images/contour1.png
    :height: 400px



Cartopy  (*Not distributed with matplotlib*)
============================================
An alternative mapping library written for matplotlib ``v1.2`` and beyond.
`Cartopy <http://scitools.org.uk/cartopy/docs/latest>`_ builds on top of
matplotlib to provide object oriented map projection definitions and close
integration with Shapely for powerful yet easy-to-use vector data processing
tools. An example plot from the
`Cartopy gallery <http://scitools.org.uk/cartopy/docs/latest/gallery.html>`_:

.. image:: http://scitools.org.uk/cartopy/docs/latest/_images/hurricane_katrina_01_00.png
    :height: 400px


.. _toolkit_gtk:

GTK Tools
=========

mpl_toolkits.gtktools provides some utilities for working with GTK.
This toolkit ships with matplotlib, but requires `pygtk
<http://www.pygtk.org/>`_.


.. _toolkit_excel:

Excel Tools
===========

mpl_toolkits.exceltools provides some utilities for working with
Excel.  This toolkit ships with matplotlib, but requires
`xlwt <http://pypi.python.org/pypi/xlwt>`_


.. _toolkit_natgrid:

Natgrid (*Not distributed with matplotlib*)
===========================================

mpl_toolkits.natgrid is an interface to natgrid C library for gridding
irregularly spaced data.  This requires a separate installation of the
natgrid toolkit from the sourceforge `download
<http://sourceforge.net/project/showfiles.php?group_id=80706&package_id=142792>`_
page.


.. _toolkit_mplot3d:

mplot3d
===========

:ref:`mpl_toolkits.mplot3d <toolkit_mplot3d-index>` provides some basic 3D plotting (scatter, surf,
line, mesh) tools.  Not the fastest or feature complete 3D library out
there, but ships with matplotlib and thus may be a lighter weight
solution for some use cases.

.. plot:: mpl_examples/mplot3d/contourf3d_demo2.py

.. _toolkit_axes_grid:

AxesGrid
========

The matplotlib :ref:`AxesGrid <toolkit_axesgrid-index>` toolkit is a collection of helper classes to
ease displaying multiple images in matplotlib. The AxesGrid toolkit is
distributed with matplotlib source.

.. image:: /_static/demo_axes_grid.png
