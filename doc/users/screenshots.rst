.. _matplotlibscreenshots:

**********************
Screenshots
**********************

Here you will find a host of example figures with the code that
generated them

Simple Plot
===========

The most basic :func:`~matplotlib.pyplot.plot`, with text labels

.. plot:: mpl_examples/pylab_examples/simple_plot.py

.. _screenshots_subplot_demo:

Subplot demo
============

Multiple regular axes (numrows by numcolumns) are created with the
:func:`~matplotlib.pyplot.subplot` command.

.. plot:: mpl_examples/pylab_examples/subplot_demo.py

.. _screenshots_histogram_demo:

Histograms
==========

The :func:`~matplotlib.pyplot.hist` command automatically generates
histograms and will return the bin counts or probabilities

.. plot:: mpl_examples/pylab_examples/histogram_demo.py


.. _screenshots_path_demo:

Path demo
=========

You can add aribitrary paths in matplotlib as of release 0.98.  See
the :mod:`matplotlib.path`.

.. plot:: mpl_examples/api/path_patch_demo.py

.. _screenshots_mplot3d_surface:

mplot3d
=========

The mplot3d toolkit (see :ref:`toolkit_mplot3d-tutorial` and
:ref:`mplot3d-examples-index`) has support for simple 3d graphs
including surface, wireframe, scatter, and bar charts (added in
matlpotlib-0.99).  Thanks to John Porter, Jonathon Taylor and Reinier
Heeres for the mplot3d toolkit.  The toolkit is included with all
standard matplotlib installs.

.. plot:: mpl_examples/mplot3d/surface3d_demo.py

.. _screenshots_ellipse_demo:


Ellipses
========

In support of the
`Phoenix <http://www.jpl.nasa.gov/news/phoenix/main.php>`_ mission to
Mars, which used matplotlib in ground tracking of the spacecraft,
Michael Droettboom built on work by Charlie Moad to provide an
extremely accurate 8-spline approximation to elliptical arcs (see
:class:`~matplotlib.patches.Arc`)  in the viewport.  This
provides a scale free, accurate graph of the arc regardless of zoom
level

.. plot:: mpl_examples/pylab_examples/ellipse_demo.py

.. _screenshots_barchart_demo:

Bar charts
==========

The :func:`~matplotlib.pyplot.bar`
command takes error bars as an optional argument.  You can also use up
and down bars, stacked bars, candlestick bars, etc, ... See
`bar_stacked.py <examples/pylab_examples/bar_stacked.py>`_ for another example.
You can make horizontal bar charts with the
:func:`~matplotlib.pyplot.barh` command.

.. plot:: mpl_examples/pylab_examples/barchart_demo.py

.. _screenshots_pie_demo:


Pie charts
==========

The :func:`~matplotlib.pyplot.pie` command
uses a MATLAB compatible syntax to produce pie charts.  Optional
features include auto-labeling the percentage of area, exploding one
or more wedges out from the center of the pie, and a shadow effect.
Take a close look at the attached code that produced this figure; nine
lines of code.

.. plot:: mpl_examples/pylab_examples/pie_demo.py

.. _screenshots_table_demo:

Table demo
==========

The :func:`~matplotlib.pyplot.table` command will place a text table
on the axes

.. plot:: mpl_examples/pylab_examples/table_demo.py


.. _screenshots_scatter_demo:

Scatter demo
============

The :func:`~matplotlib.pyplot.scatter` command makes a scatter plot
with (optional) size and color arguments.  This example plots changes
in Google stock price from one day to the next with the sizes coding
trading volume and the colors coding price change in day i.  Here the
alpha attribute is used to make semitransparent circle markers with
the Agg backend (see :ref:`what-is-a-backend`)

.. plot:: mpl_examples/pylab_examples/scatter_demo2.py


.. _screenshots_slider_demo:

Slider demo
===========

Matplotlib has basic GUI widgets that are independent of the graphical
user interface you are using, allowing you to write cross GUI figures
and widgets.  See matplotlib.widgets and the widget `examples
<examples/widgets>`

.. plot:: mpl_examples/widgets/slider_demo.py


.. _screenshots_fill_demo:

Fill demo
=========

The :func:`~matplotlib.pyplot.fill` command lets you
plot filled polygons.  Thanks to Andrew Straw for providing this
function

.. plot:: mpl_examples/pylab_examples/fill_demo.py


.. _screenshots_date_demo:

Date demo
=========

You can plot date data with major and minor ticks and custom tick
formatters for both the major and minor ticks; see matplotlib.ticker
and matplotlib.dates for details and usage.

.. plot:: mpl_examples/api/date_demo.py

.. _screenshots_jdh_demo:

Financial charts
================

You can make much more sophisticated financial plots.  This example
emulates one of the `ChartDirector
<http://www.advsofteng.com/gallery_finance.html>`_ financial plots.
Some of the data in the plot, are real financial data, some are random
traces that I used since the goal was to illustrate plotting
techniques, not market analysis!


.. plot:: mpl_examples/pylab_examples/finance_work2.py


.. _screenshots_basemap_demo:

Basemap demo
============

Jeff Whitaker's :ref:`toolkit_basemap` add-on toolkit makes it possible to plot data on many
different map projections.  This example shows how to plot contours, markers and text
on an orthographic projection, with NASA's "blue marble" satellite image as a background.

.. plot:: pyplots/plotmap.py

.. _screenshots_log_demo:

Log plots
=========

The :func:`~matplotlib.pyplot.semilogx`,
:func:`~matplotlib.pyplot.semilogy` and
:func:`~matplotlib.pyplot.loglog` functions generate log scaling on the
respective axes.  The lower subplot uses a base10 log on the xaxis and
a base 4 log on the yaxis.  Thanks to Andrew Straw, Darren Dale and
Gregory Lielens for contributions to the log scaling
infrastructure.



.. plot:: mpl_examples/pylab_examples/log_demo.py

.. _screenshots_polar_demo:

Polar plots
===========

The :func:`~matplotlib.pyplot.polar` command generates polar plots.

.. plot:: mpl_examples/pylab_examples/polar_demo.py

.. _screenshots_legend_demo:

Legends
=======

The :func:`~matplotlib.pyplot.legend` command automatically
generates figure legends, with MATLAB compatible legend placement
commands.  Thanks to Charles Twardy for input on the legend
command

.. plot:: mpl_examples/pylab_examples/legend_demo.py

.. _screenshots_mathtext_examples_demo:

Mathtext_examples
=================

A sampling of the many TeX expressions now supported by matplotlib's
internal mathtext engine.  The mathtext module provides TeX style
mathematical expressions using `freetype2
<http://freetype.sourceforge.net/index2.html>`_ and the BaKoMa
computer modern or `STIX <http://www.stixfonts.org>`_ fonts.  See the
:mod:`matplotlib.mathtext` module for additional.  matplotlib mathtext
is an independent implementation, and does not required TeX or any
external packages installed on your computer.  See the tutorial at
:ref:`mathtext-tutorial`.

.. plot:: mpl_examples/pylab_examples/mathtext_examples.py

.. _screenshots_tex_demo:

Native TeX rendering
====================

Although matplotlib's internal math rendering engine is quite
powerful, sometimes you need TeX, and matplotlib supports external TeX
rendering of strings with the *usetex* option.

.. plot:: pyplots/tex_demo.py

.. _screenshots_eeg_demo:

EEG demo
=========

You can embed matplotlib into pygtk, wxpython, Tk, FLTK or Qt
applications.  Here is a screenshot of an eeg viewer called pbrain
which is part of the NeuroImaging in Python suite `NIPY
<http://neuroimaging.scipy.org>`_.  Pbrain is written in pygtk using
matplotlib.  The lower axes uses :func:`~matplotlib.pyplot.specgram`
to plot the spectrogram of one of the EEG channels.  For an example of
how to use the navigation toolbar in your applications, see
:ref:`user_interfaces-embedding_in_gtk2`.  If you want to use
matplotlib in a wx application, see
:ref:`user_interfaces-embedding_in_wx2`.  If you want to work with
`glade <http://glade.gnome.org>`_, see
:ref:`user_interfaces-mpl_with_glade`.

.. image:: ../_static/eeg_small.png
