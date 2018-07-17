.. _thirdparty-index:

********************
Third party packages
********************

Several external packages that extend or build on Matplotlib functionality are
listed below.  They are maintained and distributed separately from Matplotlib
and thus need to be installed individually.

Please submit an issue or pull request on Github if you have created
a package that you would like to have included.  We are also happy to
host third party packages within the `Matplotlib Github Organization
<https://github.com/matplotlib>`_.

Mapping toolkits
****************

Basemap
=======
`Basemap <http://matplotlib.org/basemap>`_ plots data on map projections, with
continental and political boundaries.

.. image:: /_static/basemap_contour1.png
    :height: 400px

Cartopy
=======
`Cartopy <http://scitools.org.uk/cartopy/docs/latest>`_ builds on top
of Matplotlib to provide object oriented map projection definitions
and close integration with Shapely for powerful yet easy-to-use vector
data processing tools. An example plot from the `Cartopy gallery
<http://scitools.org.uk/cartopy/docs/latest/gallery.html>`_:

.. image:: /_static/cartopy_hurricane_katrina_01_00.png
    :height: 400px

Declarative libraries
*********************

ggplot
======
`ggplot <https://github.com/yhat/ggplot>`_ is a port of the R ggplot2 package
to python based on Matplotlib.

.. image:: /_static/ggplot.png
    :height: 195px

holoviews
=========
`holoviews <http://holoviews.org>`_ makes it easier to visualize data
interactively, especially in a `Jupyter notebook <http://jupyter.org>`_, by
providing a set of declarative plotting objects that store your data and
associated metadata.  Your data is then immediately visualizable alongside or
overlaid with other data, either statically or with automatically provided
widgets for parameter exploration.

.. image:: /_static/holoviews.png
    :height: 354px

plotnine
========

`plotnine <https://plotnine.readthedocs.io/en/stable/>`_ implements a grammar
of graphics, similar to R's `ggplot2 <http://ggplot2.org/>`_. The grammar allows
users to compose plots by explicitly mapping data to the visual objects that
make up the plot.

.. image:: /_static/plotnine.png

Specialty plots
***************

Broken Axes
===========
`brokenaxes <https://github.com/bendichter/brokenaxes>`_ supplies an axes
class that can have a visual break to indicate a discontinuous range.

.. image:: /_static/brokenaxes.png

DeCiDa
======

`DeCiDa <https://pypi.python.org/pypi/DeCiDa>`_ is a library of functions
and classes for electron device characterization, electronic circuit design and
general data visualization and analysis.

Matplotlib-Venn
===============
`Matplotlib-Venn <https://github.com/konstantint/matplotlib-venn>`_ provides a
set of functions for plotting 2- and 3-set area-weighted (or unweighted) Venn
diagrams.

mpl-probscale
=============
`mpl-probscale <http://matplotlib.org/mpl-probscale/>`_ is a small extension
that allows Matplotlib users to specify probabilty scales. Simply importing the
``probscale`` module registers the scale with Matplotlib, making it accessible
via e.g., ``ax.set_xscale('prob')`` or ``plt.yscale('prob')``.

.. image:: /_static/probscale_demo.png

mpl-scatter-density
===================

`mpl-scatter-density <https://github.com/astrofrog/mpl-scatter-density>`_ is a
small package that makes it easy to make scatter plots of large numbers
of points using a density map. The following example contains around 13 million
points and the plotting (excluding reading in the data) took less than a
second on an average laptop:

.. image:: /_static/mpl-scatter-density.png
    :height: 400px

When used in interactive mode, the density map is downsampled on-the-fly while
panning/zooming in order to provide a smooth interactive experience.

mplstereonet
============
`mplstereonet <https://github.com/joferkington/mplstereonet>`_ provides
stereonets for plotting and analyzing orientation data in Matplotlib.

Natgrid
=======
`mpl_toolkits.natgrid <https://github.com/matplotlib/natgrid>`_ is an interface
to the natgrid C library for gridding irregularly spaced data.

pyUpSet
=======
`pyUpSet <https://github.com/ImSoErgodic/py-upset>`_ is a
static Python implementation of the `UpSet suite by Lex et al.
<http://www.caleydo.org/tools/upset/>`_ to explore complex intersections of
sets and data frames.

seaborn
=======
`seaborn <http://seaborn.pydata.org/>`_ is a high level interface for drawing
statistical graphics with Matplotlib. It aims to make visualization a central
part of exploring and understanding complex datasets.

.. image:: /_static/seaborn.png
    :height: 157px

WCSAxes
=======

The `Astropy <http://www.astropy.org>`_ core package includes a submodule
called WCSAxes (available at `astropy.visualization.wcsaxes
<http://docs.astropy.org/en/stable/visualization/wcsaxes/index.html>`_) which
adds Matplotlib projections for Astronomical image data. The following is an
example of a plot made with WCSAxes which includes the original coordinate
system of the image and an overlay of a different coordinate system:

.. image:: /_static/wcsaxes.jpg
    :height: 400px

Windrose
========
`Windrose <https://github.com/scls19fr/windrose>`_ is a Python Matplotlib,
Numpy library to manage wind data, draw windroses (also known as polar rose
plots), draw probability density functions and fit Weibull distributions.


Interactivity
*************

mplcursors
==========
`mplcursors <https://mplcursors.readthedocs.io>`_ provides interactive data
cursors for Matplotlib.

MplDataCursor
=============
`MplDataCursor <https://github.com/joferkington/mpldatacursor>`_ is a toolkit
written by Joe Kington to provide interactive "data cursors" (clickable
annotation boxes) for Matplotlib.

Miscellaneous
*************

adjustText
==========
`adjustText <https://github.com/Phlya/adjustText>`_ is a small library for
automatically adjusting text position in Matplotlib plots to minimize overlaps
between them, specified points and other objects.

.. image:: /_static/adjustText.png

iTerm2 terminal backend
=======================
`matplotlib_iterm2 <https://github.com/oselivanov/matplotlib_iterm2>`_ is an
external Matplotlib backend using the iTerm2 nightly build inline image display
feature.

.. image:: /_static/matplotlib_iterm2_demo.png

mplcairo
========
`mplcairo <https://github.com/anntzer/mplcairo>`_ is a cairo backend for
Matplotlib, with faster and more accurate marker drawing, support for a wider
selection of font formats and complex text layout, and various other features.

mpl-template
============
`mpl-template <https://austinorr.github.io/mpl-template/index.html>`_ provides
a customizable way to add engineering figure elements such as a title block,
border, and logo.

.. image:: /_static/mpl_template_example.png
    :height: 330px
