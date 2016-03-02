.. _thirdparty-index:

*********************
 Third party packages
*********************

Several external packages that extend or build on Matplotlib functionality
exist. Below we list a number of these. Please submit an issue or pull request
on Github if you have created a package that you would like to have included.
We are also happy to host third party packages within the `Matplotlib Github
Organization <https://github.com/matplotlib>`_.

.. _toolkits-general:

General Toolkits
****************


.. _toolkit_mpldatacursor:

MplDataCursor
=============

`MplDataCursor <https://github.com/joferkington/mpldatacursor>`_ is a
toolkit written by Joe Kington to provide interactive "data cursors"
(clickable annotation boxes) for matplotlib.


.. _toolkit_natgrid:

Natgrid
=======

mpl_toolkits.natgrid is an interface to natgrid C library for gridding
irregularly spaced data.  This requires a separate installation of the
`natgrid toolkit <http://github.com/matplotlib/natgrid>`__.


.. _toolkit_matplotlibvenn:

Matplotlib-Venn
===============

`Matplotlib-Venn <https://github.com/konstantint/matplotlib-venn>`_ provides a set of functions for plotting 2- and 3-set area-weighted (or unweighted) Venn diagrams.

mplstereonet
===============

`mplstereonet <https://github.com/joferkington/mplstereonet>`_ provides stereonets for plotting and analyzing orientation data in Matplotlib.


.. _hl_plotting:

High-Level Plotting
*******************

Several projects have started to provide a higher-level interface to
matplotlib.  These are independent projects.

.. _toolkit_seaborn:

seaborn
=======

`seaborn <http://web.stanford.edu/~mwaskom/software/seaborn>`_ is a high
level interface for drawing statistical graphics with matplotlib. It
aims to make visualization a central part of exploring and
understanding complex datasets.

.. _toolkit_ggplot:

ggplot
======

`ggplot <https://github.com/yhat/ggplot>`_ is a port of the R ggplot2
to python based on matplotlib.


.. _toolkit_prettyplotlib:

prettyplotlib
=============

`prettyplotlib <https://olgabot.github.io/prettyplotlib>`_ is an extension
to matplotlib which changes many of the defaults to make plots some
consider more attractive.

.. _toolkit_probscale:

mpl-probscale
=============
`mpl-probscale <http://phobson.github.io/mpl-probscale/index.html>`_
is a small extension that allows matplotlib users to specify probabilty
scales. Simply importing the ``probscale`` module registers the scale
with matplotlib, making it accessible via e.g.,
``ax.set_xscale('prob')`` or ``plt.yscale('prob')``.

.. image:: /_static/probscale_demo.png

iTerm2 terminal backend
=======================

`matplotlib_iterm2 <https://github.com/oselivanov/matplotlib_iterm2>`_ is an
external matplotlib backend using iTerm2 nightly build inline image display
feature.

.. image:: /_static/matplotlib_iterm2_demo.png


.. _toolkits-mapping:

Mapping Toolkits
****************


.. _toolkit_basemap:

Basemap
=======

Plots data on map projections, with continental and political
boundaries, see `basemap <http://matplotlib.org/basemap>`_
docs.

.. image:: /_static/basemap_contour1.png
    :height: 400px



Cartopy
=======

An alternative mapping library written for matplotlib ``v1.2`` and beyond.
`Cartopy <http://scitools.org.uk/cartopy/docs/latest>`_ builds on top of
matplotlib to provide object oriented map projection definitions and close
integration with Shapely for powerful yet easy-to-use vector data processing
tools. An example plot from the
`Cartopy gallery <http://scitools.org.uk/cartopy/docs/latest/gallery.html>`_:

.. image:: /_static/cartopy_hurricane_katrina_01_00.png
    :height: 400px
