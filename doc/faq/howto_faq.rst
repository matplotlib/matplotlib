.. _howto-faq:

*****
Howto
*****

.. contents::


.. _howto-findobj:

How do I find all the objects in my figure of a certain type?
=============================================================

Every matplotlib artist (see :ref:`artist-tutorial`) has a method
called :meth:`~matplotlib.artist.Artist.findobj` that can be used to
recursively search the artist for any artists it may contain that meet
some criteria (eg match all :class:`~matplotlib.lines.Line2D`
instances or match some arbitrary filter function).  For example, the
following snippet finds every object in the figure which has a
`set_color` property and makes the object blue::

    def myfunc(x):
	return hasattr(x, 'set_color')

    for o in fig.findobj(myfunc):
	o.set_color('blue')

You can also filter on class instances::

    import matplotlib.text as text
    for o in fig.findobj(text.Text):
	o.set_fontstyle('italic')


.. _howto-transparent:

How do I save transparent figures?
==================================

The :meth:`~matplotlib.pyplot.savefig` command has a keyword argument
*transparent* which, if True, will make the figure and axes
backgrounds transparent when saving, but will not affect the displayed
image on the screen.  If you need finer grained control, eg you do not
want full transparency or you to affect the screen displayed version
as well, you can set the alpha properties directly.  The figure has a
:class:`matplotlib.patches.Rectangle` instance called *patch*
and the axes has a Rectangle instance called *patch*.  You can set
any property on them directly (*facecolor*, *edgecolor*, *linewidth*,
*linestyle*, *alpha*).  Eg::

    fig = plt.figure()
    fig.patch.set_alpha(0.5)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.5)

If you need *all* the figure elements to be transparent, there is
currently no global alpha setting, but you can set the alpha channel
on individual elements, eg::

   ax.plot(x, y, alpha=0.5)
   ax.set_xlabel('volts', alpha=0.5)


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

.. _howto-auto-adjust:

How do I automatically make room for my tick labels?
====================================================

In most use cases, it is enough to simpy change the subplots adjust
parameters as described in :ref:`howto-subplots-adjust`.  But in some
cases, you don't know ahead of time what your tick labels will be, or
how large they will be (data and labels outside your control may be
being fed into your graphing application), and you may need to
automatically adjust your subplot parameters based on the size of the
tick labels.  Any :class:`matplotlib.text.Text` instance can report
its extent in window coordinates (a negative x coordinate is outside
the window), but there is a rub.

The :class:`matplotlib.backend_bases.RendererBase` instance, which is
used to calculate the text size, is not known until the figure is
drawn (:meth:`matplotlib.figure.Figure.draw`).  After the window is
drawn and the text instance knows its renderer, you can call
:meth:`matplotlib.text.Text.get_window_extent``.  One way to solve
this chicken and egg problem is to wait until the figure is draw by
connecting
(:meth:`matplotlib.backend_bases.FigureCanvasBase.mpl_connect`) to the
"on_draw" signal (:class:`~matplotlib.backend_bases.DrawEvent`) and
get the window extent there, and then do something with it, eg move
the left of the canvas over; see :ref:`event-handling-tutorial`.

Here is that gets a bounding box in relative figure coordinates (0..1)
of each of the labels and uses it to move the left of the subplots
over so that the tick labels fit in the figure

.. plot:: auto_subplots_adjust.py
   :include-source:

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


.. _howto-align-label:

How do I align my ylabels across multiple subplots?
===================================================

If you have multiple subplots over one another, and the y data have
different scales, you can often get ylabels that do not align
vertically across the multiple subplots, which can be unattractive.
By default, matplotlib positions the x location of the ylabel so that
it does not overlap any of the y ticks.  You can override this default
behavior by specifying the coordinates of the label.  The example
below shows the default behavior in the left subplots, and the manual
setting in the right subplots.

.. plot:: align_ylabels.py
   :include-source:

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

.. _point-in-poly:

How do I test whether a point is inside a polygon?
==================================================

The :mod:`matplotlib.nxutils` provides two high performance methods:
for a single point use :func:`~matplotlib.nxutils.pnpoly` and for an
array of points use :func:`~matplotlib.nxutils.points_inside_poly`.
For a discussion of the implementation see `pnpoly
<http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html>`_.

.. sourcecode:: ipython

    In [25]: import numpy as np

    In [26]: import matplotlib.nxutils as nx

    In [27]: verts = np.array([ [0,0], [0, 1], [1, 1], [1,0]], float)

    In [28]: nx.pnpoly( 0.5, 0.5, verts)
    Out[28]: 1

    In [29]: nx.pnpoly( 0.5, 1.5, verts)
    Out[29]: 0

    In [30]: points = np.random.rand(10,2)*2

    In [31]: points
    Out[31]:
    array([[ 1.03597426,  0.61029911],
           [ 1.94061056,  0.65233947],
           [ 1.08593748,  1.16010789],
           [ 0.9255139 ,  1.79098751],
           [ 1.54564936,  1.15604046],
           [ 1.71514397,  1.26147554],
           [ 1.19133536,  0.56787764],
           [ 0.40939549,  0.35190339],
           [ 1.8944715 ,  0.61785408],
           [ 0.03128518,  0.48144145]])

    In [32]: nx.points_inside_poly(points, verts)
    Out[32]: array([False, False, False, False, False, False, False,  True, False, True], dtype=bool)

.. htmlonly::

    For a complete example, see :ref:`event_handling-lasso_demo`.


.. _how-to-submit-patch:

How do I submit a patch?
========================

First obtain a copy of matplotlib svn (see :ref:`install-svn`) and
make your changes to the matplotlib source code or documentation and
apply a `svn diff`.  If it is feasible, do your diff from the top
level directory, the one that contains :file:`setup.py`.  Eg,::

    > cd /path/to/matplotlib/source
    > svn diff > mypatch.diff

and then post your patch to the `matplotlib-devel
<http://sourceforge.net/mail/?group_id=80706>`_ mailing list.  If you
do not get a response within 24 hours, post your patch to the
sourceforge patch `tracker
<http://sourceforge.net/tracker2/?atid=560722&group_id=80706&func=browse>`_,
and follow up on the mailing list with a link to the sourceforge patch
submissions.  If you still do not hear anything within a week (this
shouldn't happen!), send us a kind and gentle reminder on the mailing
list.

If you have made lots of local changes and do not want to a diff
against the entire tree, but rather against a single directory or
file, that is fine, but we do prefer svn diffs against HEAD.

You should check out the guide to developing matplotlib to make sure
your patch abides by our coding conventions
:ref:`developers-guide-index`.


.. _howto-click-maps:

Clickable images for HTML
=========================

Andrew Dalke of `Dalke Scientific <http://www.dalkescientific.com>`_
has written a nice `article
<http://www.dalkescientific.com/writings/diary/archive/2005/04/24/interactive_html.html>`_
on how to make html click maps with matplotlib agg PNGs.  We would
also like to add this functionality to SVG and add a SWF backend to
support these kind of images.  If you are interested in contributing
to these efforts that would be great.

.. _howto-set-zorder:

How do I control the depth of plot elements?
=============================================

Within an axes, the order that the various lines, markers, text,
collections, etc appear is determined by the
:meth:`matplotlib.artist.Artist.set_zorder` property.  The default
order is patches, lines, text, with collections of lines and
collections of patches appearing at the same level as regular lines
and patches, respectively::

    line, = ax.plot(x, y, zorder=10)



.. htmlonly::

    See :ref:`pylab_examples-zorder_demo` for a complete example.

You can also use the Axes property
:meth:`matplotlib.axes.Axes.set_axisbelow` to control whether the grid
lines are placed above or below your other plot elements.

.. _howto-axis-equal:

How to I make the aspect ratio for plots equal?
===============================================

The Axes property :meth:`matplotlib.axes.Axes.set_aspect` controls the
aspect ratio of the axes.  You can set it to be 'auto', 'equal', or
some ratio which controls the ratio::

  ax = fig.add_subplot(111, aspect='equal')



.. htmlonly::

    See :ref:`pylab_examples-equal_aspect_ratio` for a complete example.


.. _howto-movie:

How do I make a movie?
======================


If you want to take an animated plot and turn it into a movie, the
best approach is to save a series of image files (eg PNG) and use an
external tool to convert them to a movie.  You can use ` mencoder
<http://www.mplayerhq.hu/DOCS/HTML/en/mencoder.html>`_,
which is part of the `mplayer <http://www.mplayerhq.hu>`_ suite
for this::


    #fps (frames per second) controls the play speed
    mencoder 'mf://*.png' -mf type=png:fps=10 -ovc \\
       lavc -lavcopts vcodec=wmv2 -oac copy -o animation.avi

The swiss army knife of image tools, ImageMagick's `convert
<http://www.imagemagick.org/script/convert.php>`_ works for this as
well.<p>

Here is a simple example script that saves some PNGs, makes them into
a movie, and then cleans up::

    import os, sys
    import matplotlib.pyplot as plt

    files = []
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    for i in range(50):  # 50 frames
        ax.cla()
        ax.imshow(rand(5,5), interpolation='nearest')
        fname = '_tmp%03d.png'%i
        print 'Saving frame', fname
        fig.savefig(fname)
        files.append(fname)

    print 'Making movie animation.mpg - this make take a while'
    os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 \\
      -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg")

.. htmlonly::

    See :ref:`animation-movie_demo` for a complete example.
