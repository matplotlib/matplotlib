.. _howto-faq:

*****
Howto
*****

.. _howto-subplots-adjust:

How do I move the edge of my axes area over to make room for my tick labels?
============================================================================

For subplots, you can control the default spacing on the left, right,
bottom, and top as well as the horizontal and vertical spacing between
multiple rows and columns using the
:meth:`matplotlib.figure.Figure.subplots_adjust` method (in pyplot it
is :func:`~matplotlib.pyplot.subplots_adjust`).  For example, to move
the bottom of the subplots up to make room for some rotated x tick
labels::

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111)

You can control the defaults for these parameters in your
:file:`matplotlibrc` file; see :ref:`customizing-matplotlib`.  For
example, to make the above setting permanent, you would set::

    figure.subplot.bottom : 0.2   # the bottom of the subplots of the figure

The other parameters you can configure are, with their defaults

*left*  = 0.125
    the left side of the subplots of the figure
*right* = 0.9
    the right side of the subplots of the figure
*bottom* = 0.1
    the bottom of the subplots of the figure
*top* = 0.9
    the top of the subplots of the figure
*wspace* = 0.2
    the amount of width reserved for blank space between subplots
*hspace* = 0.2
    the amount of height reserved for white space between subplots

If you want additional control, you can create an
:class:`~matplotlib.axes.Axes` using the
:func:`~matplotlib.pyplot.axes` command (or equivalently the figure
:meth:`matplotlib.figure.Figure.add_axes` method), which allows you to
specify the location explicitly::

    ax = fig.add_axes([left, bottom, width, height])

where all values are in fractional (0 to 1) coordinates.  See
`axes_demo.py <http://matplotlib.sf.net/examples/axes_demo.py>`_ for
an example of placing axes manually.

.. _howto-ticks:

How do I configure the tick linewidths?
=======================================

In matplotlib, the ticks are *markers*.  All
:class:`~matplotlib.lines.Line2D` objects support a line (solid,
dashed, etc) and a marker (circle, square, tick).  The tick linewidth
is controlled by the "markeredgewidth" property::

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(10))

    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(10)

    plt.show()

The other properties that control the tick marker, and all markers,
are ``markerfacecolor``, ``markeredgecolor``, ``markeredgewidth``,
``markersize``.  For more information on configuring ticks, see
:ref:`axis-container` and :ref:`tick-container`.

.. _howto-webapp:

How do I use matplotlib in a web application server?
====================================================

Many users report initial problems trying to use maptlotlib in web
application servers, because by default matplotlib ships configured to
work with a graphical user interface which may require an X11
connection.  Since many barebones application servers do not have X11
enabled, you may get errors if you don't configure matplotlib for use
in these environments.  Most importantly, you need to decide what
kinds of images you want to generate (PNG, PDF, SVG) and configure the
appropriate default backend.  For 99% of users, this will be the Agg
backend, which uses the C++ `antigrain <http://antigrain.com>`_
rendering engine to make nice PNGs.  The Agg backend is also
configured to recognize requests to generate other output formats
(PDF, PS, EPS, SVG).  The easiest way to configure matplotlib to use
Agg is to call::

    # do this before importing pylab or pyplot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

For more on configuring your backend, see
:ref:`what-is-a-backend`.

Alternatively, you can avoid pylab/pyplot altogeher, which will give
you a little more control, by calling the API directly as shown in
`agg_oo.py <http://matplotlib.sf.net/examples/api/agg_oo.py>`_ .

You can either generate hardcopy on the filesystem by calling savefig::

    # do this before importing pylab or pyplot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    fig.savefig('test.png')

or by saving to a file handle::

    import sys
    fig.savefig(sys.stdout)


How do I use matplotlib with apache?
------------------------------------

TODO

How do I use matplotlib with django?
------------------------------------

TODO

How do I use matplotlib with zope?
----------------------------------

TODO


.. _date-index-plots:

How do I skip dates where there is no data?
===========================================

When plotting time series, eg financial time series, one often wants
to leave out days on which there is no data, eg weekends.  By passing
in dates on the x-xaxis, you get large horizontal gaps on periods when
there is not data. The solution is to pass in some proxy x-data, eg
evenly sampled indicies, and then use a custom formatter to format
these as dates. The example below shows how to use an 'index formatter'
to achieve the desired plot::

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    import matplotlib.ticker as ticker

    r = mlab.csv2rec('../data/aapl.csv')
    r.sort()
    r = r[-30:]  # get the last 30 days

    N = len(r)
    ind = np.arange(N)  # the evenly spaced plot indices

    def format_date(x, pos=None):
	thisind = np.clip(int(x+0.5), 0, N-1)
	return r.date[thisind].strftime('%Y-%m-%d')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ind, r.adj_close, 'o-')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    fig.autofmt_xdate()

    plt.show()
