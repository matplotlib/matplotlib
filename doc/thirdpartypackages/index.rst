.. _thirdparty-index:

*********************
 Third party packages
*********************

Several external packages that extend or build on Matplotlib functionality
exist. Below we list a number of these. Note that they are each
maintained and distributed separately from Matplotlib, and will need
to be installed individually.

Please submit an issue or pull request
on Github if you have created a package that you would like to have included.
We are also happy to host third party packages within the `Matplotlib Github
Organization <https://github.com/matplotlib>`_.

.. _hl_plotting:

High-Level Plotting
*******************

Several projects provide higher-level interfaces for creating
matplotlib plots.

.. _toolkit_seaborn:

seaborn
=======

`seaborn <http://web.stanford.edu/~mwaskom/software/seaborn>`_ is a high
level interface for drawing statistical graphics with matplotlib. It
aims to make visualization a central part of exploring and
understanding complex datasets.

.. image:: /_static/seaborn.png
    :height: 157px

.. _toolkit_ggplot:

ggplot
======

`ggplot <https://github.com/yhat/ggplot>`_ is a port of the R ggplot2 package
to python based on matplotlib.

.. image:: /_static/ggplot.png
    :height: 195px

.. _toolkit_holoviews:

holoviews
=========

`holoviews <http://holoviews.org>`_ makes it easier to visualize data
interactively, especially in a `Jupyter notebook
<http://jupyter.org>`_, by providing a set of declarative
plotting objects that store your data and associated metadata.  Your
data is then immediately visualizable alongside or overlaid with other
data, either statically or with automatically provided widgets for
parameter exploration.

.. image:: /_static/holoviews.png
    :height: 354px


.. _toolkits-mapping:

Mapping Toolkits
****************

Two independent mapping toolkits are available.

.. _toolkit_basemap:

Basemap
=======

Plots data on map projections, with continental and political
boundaries. See `basemap <http://matplotlib.org/basemap>`_
docs.

.. image:: /_static/basemap_contour1.png
    :height: 400px



Cartopy
=======
`Cartopy <http://scitools.org.uk/cartopy/docs/latest>`_ builds on top of
matplotlib to provide object oriented map projection definitions and close
integration with Shapely for powerful yet easy-to-use vector data processing
tools. An example plot from the
`Cartopy gallery <http://scitools.org.uk/cartopy/docs/latest/gallery.html>`_:

.. image:: /_static/cartopy_hurricane_katrina_01_00.png
    :height: 400px


.. _toolkits-misc:
.. _toolkits-general:

Miscellaneous Toolkits
**********************

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

pyupset
===============
`pyUpSet <https://github.com/ImSoErgodic/py-upset>`_ is a static Python implementation of the `UpSet suite by Lex et al. <http://www.caleydo.org/tools/upset/>`_ to explore complex intersections of sets and data frames.

Windrose
===============
`Windrose <https://github.com/scls19fr/windrose>`_ is a Python Matplotlib, Numpy library to manage wind data, draw windrose (also known as a polar rose plot), draw probability density function and fit Weibull distribution
