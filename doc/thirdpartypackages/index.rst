.. _thirdparty-index:

********************
Third party packages
********************

Several external packages that extend or build on Matplotlib functionality are
listed below.  They are maintained and distributed separately from Matplotlib
and thus need to be installed individually.

Please submit an issue or pull request on GitHub if you have created
a package that you would like to have included.  We are also happy to
host third party packages within the `Matplotlib GitHub Organization
<https://github.com/matplotlib>`_.

Mapping toolkits
****************

Basemap
=======
`Basemap <https://matplotlib.org/basemap/>`_ plots data on map projections,
with continental and political boundaries.

.. image:: /_static/basemap_contour1.png
    :height: 400px

Cartopy
=======
`Cartopy <https://scitools.org.uk/cartopy/docs/latest>`_ builds on top
of Matplotlib to provide object oriented map projection definitions
and close integration with Shapely for powerful yet easy-to-use vector
data processing tools. An example plot from the `Cartopy gallery
<https://scitools.org.uk/cartopy/docs/latest/gallery.html>`_:

.. image:: /_static/cartopy_hurricane_katrina_01_00.png
    :height: 400px

Geoplot
=======
`Geoplot <https://residentmario.github.io/geoplot/index.html>`_ builds on top
of Matplotlib and Cartopy to provide a "standard library" of simple, powerful,
and customizable plot types. An example plot from the `Geoplot gallery
<https://residentmario.github.io/geoplot/index.html>`_:

.. image:: /_static/geoplot_nyc_traffic_tickets.png
    :height: 400px

Ridge Map
=========
`ridge_map <https://github.com/ColCarroll/ridge_map>`_ uses Matplotlib,
SRTM.py, NumPy, and scikit-image to make ridge plots of your favorite
ridges.

.. image:: /_static/ridge_map_white_mountains.png
    :height: 364px

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
interactively, especially in a `Jupyter notebook <https://jupyter.org>`_, by
providing a set of declarative plotting objects that store your data and
associated metadata.  Your data is then immediately visualizable alongside or
overlaid with other data, either statically or with automatically provided
widgets for parameter exploration.

.. image:: /_static/holoviews.png
    :height: 354px

plotnine
========

`plotnine <https://plotnine.readthedocs.io/en/stable/>`_ implements a grammar
of graphics, similar to R's `ggplot2 <https://ggplot2.tidyverse.org/>`_.
The grammar allows users to compose plots by explicitly mapping data to the
visual objects that make up the plot.

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

`DeCiDa <https://pypi.org/project/DeCiDa/>`_ is a library of functions
and classes for electron device characterization, electronic circuit design and
general data visualization and analysis.

matplotlib-scalebar
===================

`matplotlib-scalebar <https://github.com/ppinard/matplotlib-scalebar>`_ provides a new artist to display a scale bar, aka micron bar.
It is particularly useful when displaying calibrated images plotted using ``plt.imshow(...)``.

.. image:: /_static/gold_on_carbon.jpg

Matplotlib-Venn
===============
`Matplotlib-Venn <https://github.com/konstantint/matplotlib-venn>`_ provides a
set of functions for plotting 2- and 3-set area-weighted (or unweighted) Venn
diagrams.

mpl-probscale
=============
`mpl-probscale <https://matplotlib.org/mpl-probscale/>`_ is a small extension
that allows Matplotlib users to specify probability scales. Simply importing the
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

Yellowbrick
===========
`Yellowbrick <https://www.scikit-yb.org/>`_ is a suite of visual diagnostic tools for machine learning that enables human steering of the model selection process. Yellowbrick combines scikit-learn with matplotlib using an estimator-based API called the ``Visualizer``, which wraps both sklearn models and matplotlib Axes. ``Visualizer`` objects fit neatly into the machine learning workflow allowing data scientists to integrate visual diagnostic and model interpretation tools into experimentation without extra steps.

.. image:: /_static/yellowbrick.png
    :height: 400px

Animations
**********

animatplot
==========
`animatplot <https://animatplot.readthedocs.io/>`_ is a library for
producing interactive animated plots with the goal of making production of
animated plots almost as easy as static ones.

.. image:: /_static/animatplot.png

For an animated version of the above picture and more examples, see the
`animatplot gallery. <https://animatplot.readthedocs.io/en/stable/gallery.html>`_

gif
===
`gif <https://github.com/maxhumber/gif/>`_ is an ultra lightweight animated gif API.

.. image:: /_static/gif_attachment_example.png

numpngw
=======

`numpngw <https://pypi.org/project/numpngw/>`_  provides functions for writing
NumPy arrays to PNG and animated PNG files.  It also includes the class
``AnimatedPNGWriter`` that can be used to save a Matplotlib animation as an
animated PNG file.  See the example on the PyPI page or at the ``numpngw``
`github repository <https://github.com/WarrenWeckesser/numpngw>`_.

.. image:: /_static/numpngw_animated_example.png

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

Rendering backends
******************

mplcairo
========
`mplcairo <https://github.com/anntzer/mplcairo>`_ is a cairo backend for
Matplotlib, with faster and more accurate marker drawing, support for a wider
selection of font formats and complex text layout, and various other features.

gr
==
`gr <http://gr-framework.org/>`_ is a framework for cross-platform
visualisation applications, which can be used as a high-performance Matplotlib
backend.

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

mpl-template
============
`mpl-template <https://austinorr.github.io/mpl-template/index.html>`_ provides
a customizable way to add engineering figure elements such as a title block,
border, and logo.

.. image:: /_static/mpl_template_example.png
    :height: 330px

blume
=====

`blume <https://pypi.org/project/blume/>`_ provides a replacement for
the Matplotlib ``table`` module.  It fixes a number of issues with the
existing table. See the `blume github repository
<https://github.com/swfiua/blume>`_ for more details.

.. image:: /_static/blume_table_example.png


DNA Features Viewer
===================

`DNA Features Viewer <https://github.com/Edinburgh-Genome-Foundry/DnaFeaturesViewer>`_
provides methods to plot annotated DNA sequence maps (possibly along other Matplotlib
plots) for Bioinformatics and Synthetic Biology applications.

.. image:: /_static/dna_features_viewer_screenshot.png

GUI applications
****************

sviewgui
========

`sviewgui <https://pypi.org/project/sviewgui/>`_ is a PyQt-based GUI for
visualisation of data from csv files or `pandas.DataFrame`\s. Main features:

- Scatter, line, density, histogram, and box plot types
- Settings for the marker size, line width, number of bins of histogram,
  color map (from cmocean)
- Save figure as editable PDF
- Code of the plotted graph is available so that it can be reused and modified
  outside of sviewgui

.. image:: /_static/sviewgui_sample.png
