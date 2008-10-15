Here you will find a host of example figures with the code that
generated them

Simple Plot
===========

The most basic :func:`~matplotlib.pyplot.plot`, with text labels

.. plot:: ../mpl_examples/pylab_examples/simple_plot.py

.. _screenshots_subplot_demo:

Subplot demo
============

Multiple regular axes (numrows by numcolumns) are created with the
:func:`~matplotlib.pyplot.subplot` command.

.. plot:: ../mpl_examples/pylab_examples/subplot_demo.py

.. _screenshots_histogram_demo:

Histograms
==========

The :func:`~matplotlib.pyplot.hist` command automatically generates
histograms and will return the bin counts or probabilities

.. plot:: ../mpl_examples/pylab_examples/histogram_demo.py


.. _screenshots_path_demo:

Path demo
=========

You can add aribitrary paths in matplotlib as of release 0.98.  See
the :mod:`matplotlib.path`.

.. plot:: ../mpl_examples/api/path_patch_demo.py

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

.. plot:: ../mpl_examples/pylab_examples/ellipse_demo.py

.. _screenshots_barchart_demo:

Bar charts
==========

The :func:`~matplotlib.pyplot.bar`
command takes error bars as an optional argument.  You can also use up
and down bars, stacked bars, candlestic' bars, etc, ... See
`bar_stacked.py <examples/pylab_examples/bar_stacked.py>`_ for another example.
You can make horizontal bar charts with the
:func:`~matplotlib.pyplot.barh` command.

.. plot:: ../mpl_examples/pylab_examples/barchart_demo.py

.. _screenshots_pie_demo:


Pie charts
==========

The :func:`~matplotlib.pyplot.pie` command
uses a matlab(TM) compatible syntax to produce py charts.  Optional
features include auto-labeling the percentage of area, "exploding" one
or more wedges out from the center of the pie, and a shadow effect.
Take a close look at the attached code that produced this figure; nine
lines of code.

.. plot:: ../mpl_examples/pylab_examples/pie_demo.py

.. _screenshots_table_demo:

Table demo
==========

The :func:`~matplotlib.pyplot.table` command will place a text table
on the axes

.. plot:: ../mpl_examples/pylab_examples/table_demo.py


.. _screenshots_scatter_demo:

Scatter demo
============

The :func:`~matplotlib.pyplot.scatter` command makes a scatter plot
with (optional) size and color arguments.  This example plots changes
in Intel's stock price from one day to the next with the sizes coding
trading volume and the colors coding price change in day i.  Here the
alpha attribute is used to make semitransparent circle markers with
the Agg backend (see :ref:`what-is-a-backend`)

.. plot:: ../mpl_examples/pylab_examples/scatter_demo2.py


.. _screenshots_slider_demo:

Slider demo
===========

Matplotlib has basic GUI widgets that are independent of the graphical
user interface you are using, allowing you to write cross GUI figures
and widgets.  See matplotlib.widgets and the widget `examples
<examples/widgets>`

[.. plot:: ../mpl_examples/widgets/slider_demo.py


.. _screenshots_fill_demo:

Fill demo
=========

The :func:`~matplotlib.pyplot.fill` command lets you
plot filled polygons.  Thanks to Andrew Straw for providing this
function

.. plot:: ../mpl_examples/pylab_examples/fill_demo.py


.. _screenshots_date_demo:

Date demo
=========

You can plot date data with major and minor ticks and custom tick
formatters for both the major and minor ticks; see matplotlib.ticker
and matplotlib.dates for details and usage.

.. plot:: ../mpl_examples/api/date_demo.py



