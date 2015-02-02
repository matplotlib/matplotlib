.. _toolkits-index:

.. toctree::
   :hidden:

   axes_grid/index.rst
   mplot3d/index.rst

.. _toolkits:

########
Toolkits
########



Toolkits are collections of application-specific functions that extend matplotlib.


.. _toolkits-mapping:


Mapping Toolkits
****************


.. _toolkit_basemap:

Basemap
=======
(*Not distributed with matplotlib*)

Plots data on map projections, with continental and political
boundaries, see `basemap <http://matplotlib.org/basemap>`_
docs.

.. image:: /_static/basemap_contour1.png
    :height: 400px



Cartopy
=======
(*Not distributed with matplotlib*)

An alternative mapping library written for matplotlib ``v1.2`` and beyond.
`Cartopy <http://scitools.org.uk/cartopy/docs/latest>`_ builds on top of
matplotlib to provide object oriented map projection definitions and close
integration with Shapely for powerful yet easy-to-use vector data processing
tools. An example plot from the
`Cartopy gallery <http://scitools.org.uk/cartopy/docs/latest/gallery.html>`_:

.. image:: /_static/cartopy_hurricane_katrina_01_00.png
    :height: 400px


.. _toolkits-shipped:


General Toolkits
****************

.. _toolkit_mplot3d:

mplot3d
=======
.. toctree::
   :maxdepth: 2

   mplot3d/index


:ref:`mpl_toolkits.mplot3d <toolkit_mplot3d-index>` provides some basic 3D plotting (scatter, surf,
line, mesh) tools.  Not the fastest or feature complete 3D library out
there, but ships with matplotlib and thus may be a lighter weight
solution for some use cases.

.. plot:: mpl_examples/mplot3d/contourf3d_demo2.py

.. _toolkit_axes_grid:

AxesGrid
========
.. toctree::
   :maxdepth: 2

   axes_grid/index


The matplotlib :ref:`AxesGrid <toolkit_axesgrid-index>` toolkit is a collection of helper classes to
ease displaying multiple images in matplotlib. The AxesGrid toolkit is
distributed with matplotlib source.



.. image:: /_static/demo_axes_grid.png


.. _toolkit_mpldatacursor:

MplDataCursor
=============
(*Not distributed with matplotlib*)

`MplDataCursor <https://github.com/joferkington/mpldatacursor>`_ is a
toolkit written by Joe Kington to provide interactive "data cursors"
(clickable annotation boxes) for matplotlib.


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

Natgrid
=======
(*Not distributed with matplotlib*)

mpl_toolkits.natgrid is an interface to natgrid C library for gridding
irregularly spaced data.  This requires a separate installation of the
natgrid toolkit from the sourceforge `download
<http://sourceforge.net/project/showfiles.php?group_id=80706&package_id=142792>`_
page.


.. _hl_plotting:

High-Level Plotting
*******************

Several projects have started to provide a higher-level interface to
matplotlib.  These are independent projects.

.. _toolkit_seaborn:

seaborn
=======
(*Not distributed with matplotlib*)

`seaborn <http://web.stanford.edu/~mwaskom/software/seaborn>`_ is a high
level interface for drawing statistical graphics with matplotlib. It
aims to make visualization a central part of exploring and
understanding complex datasets.

.. _toolkit_ggplot:

ggplot
======
(*Not distributed with matplotlib*)

`ggplot <https://github.com/yhat/ggplot>`_ is a port of the R ggplot2
to python based on matplotlib.


.. _toolkit_prettyplotlib:

prettyplotlib
=============
(*Not distributed with matplotlib*)

`prettyplotlib <https://olgabot.github.io/prettyplotlib>`_ is an extension
to matplotlib which changes many of the defaults to make plots some
consider more attractive.
