.. _howto-faq:

*****
HOWTO
*****


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
