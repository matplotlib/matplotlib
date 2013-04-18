.. _howto-faq:

******
How-To
******

.. contents::
   :backlinks: none


.. _howto-plotting:

Plotting: howto
=================

.. _howto-findobj:

Find all objects in a figure of a certain type
----------------------------------------------

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

Save transparent figures
----------------------------------

The :meth:`~matplotlib.pyplot.savefig` command has a keyword argument
*transparent* which, if 'True', will make the figure and axes
backgrounds transparent when saving, but will not affect the displayed
image on the screen.

If you need finer grained control, eg you do not want full transparency
or you want to affect the screen displayed version as well, you can set
the alpha properties directly.  The figure has a
:class:`~matplotlib.patches.Rectangle` instance called *patch*
and the axes has a Rectangle instance called *patch*.  You can set
any property on them directly (*facecolor*, *edgecolor*, *linewidth*,
*linestyle*, *alpha*).  e.g.::

    fig = plt.figure()
    fig.patch.set_alpha(0.5)
    ax = fig.add_subplot(111)
    ax.patch.set_alpha(0.5)

If you need *all* the figure elements to be transparent, there is
currently no global alpha setting, but you can set the alpha channel
on individual elements, e.g.::

   ax.plot(x, y, alpha=0.5)
   ax.set_xlabel('volts', alpha=0.5)


.. _howto-multipage:

Save multiple plots to one pdf file
-----------------------------------

Many image file formats can only have one image per file, but some
formats support multi-page files. Currently only the pdf backend has
support for this. To make a multi-page pdf file, first initialize the
file::

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('multipage.pdf')

You can give the :class:`~matplotlib.backends.backend_pdf.PdfPages`
object to :func:`~matplotlib.pyplot.savefig`, but you have to specify
the format::

    plt.savefig(pp, format='pdf')

An easier way is to call
:meth:`PdfPages.savefig <matplotlib.backends.backend_pdf.PdfPages.savefig>`::

    pp.savefig()

Finally, the multipage pdf object has to be closed::

    pp.close()


.. _howto-subplots-adjust:

Move the edge of an axes to make room for tick labels
----------------------------------------------------------------------------

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
:meth:`~matplotlib.figure.Figure.add_axes` method), which allows you to
specify the location explicitly::

    ax = fig.add_axes([left, bottom, width, height])

where all values are in fractional (0 to 1) coordinates.  See
:ref:`pylab_examples-axes_demo` for an example of placing axes manually.

.. _howto-auto-adjust:

Automatically make room for tick labels
---------------------------------------

.. note::
   This is now easier to handle than ever before.
   Calling :func:`~matplotlib.pyplot.tight_layout` can fix many common
   layout issues. See the :ref:`plotting-guide-tight-layout`.

   The information below is kept here in case it is useful for other
   purposes.

In most use cases, it is enough to simply change the subplots adjust
parameters as described in :ref:`howto-subplots-adjust`.  But in some
cases, you don't know ahead of time what your tick labels will be, or
how large they will be (data and labels outside your control may be
being fed into your graphing application), and you may need to
automatically adjust your subplot parameters based on the size of the
tick labels.  Any :class:`~matplotlib.text.Text` instance can report
its extent in window coordinates (a negative x coordinate is outside
the window), but there is a rub.

The :class:`~matplotlib.backend_bases.RendererBase` instance, which is
used to calculate the text size, is not known until the figure is
drawn (:meth:`~matplotlib.figure.Figure.draw`).  After the window is
drawn and the text instance knows its renderer, you can call
:meth:`~matplotlib.text.Text.get_window_extent`.  One way to solve
this chicken and egg problem is to wait until the figure is draw by
connecting
(:meth:`~matplotlib.backend_bases.FigureCanvasBase.mpl_connect`) to the
"on_draw" signal (:class:`~matplotlib.backend_bases.DrawEvent`) and
get the window extent there, and then do something with it, eg move
the left of the canvas over; see :ref:`event-handling-tutorial`.

Here is an example that gets a bounding box in relative figure coordinates
(0..1) of each of the labels and uses it to move the left of the subplots
over so that the tick labels fit in the figure

.. plot:: pyplots/auto_subplots_adjust.py
   :include-source:

.. _howto-ticks:

Configure the tick linewidths
-----------------------------

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

Align my ylabels across multiple subplots
-----------------------------------------

If you have multiple subplots over one another, and the y data have
different scales, you can often get ylabels that do not align
vertically across the multiple subplots, which can be unattractive.
By default, matplotlib positions the x location of the ylabel so that
it does not overlap any of the y ticks.  You can override this default
behavior by specifying the coordinates of the label.  The example
below shows the default behavior in the left subplots, and the manual
setting in the right subplots.

.. plot:: pyplots/align_ylabels.py
   :include-source:

.. _date-index-plots:

Skip dates where there is no data
---------------------------------

When plotting time series, eg financial time series, one often wants
to leave out days on which there is no data, eg weekends.  By passing
in dates on the x-xaxis, you get large horizontal gaps on periods when
there is not data. The solution is to pass in some proxy x-data, eg
evenly sampled indices, and then use a custom formatter to format
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

Test whether a point is inside a polygon
----------------------------------------

The :mod:`~matplotlib.nxutils` provides two high-performance methods:
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

.. _howto-set-zorder:

Control the depth of plot elements
----------------------------------


Within an axes, the order that the various lines, markers, text,
collections, etc appear is determined by the
:meth:`~matplotlib.artist.Artist.set_zorder` property.  The default
order is patches, lines, text, with collections of lines and
collections of patches appearing at the same level as regular lines
and patches, respectively::

    line, = ax.plot(x, y, zorder=10)

.. htmlonly::

    See :ref:`pylab_examples-zorder_demo` for a complete example.

You can also use the Axes property
:meth:`~matplotlib.axes.Axes.set_axisbelow` to control whether the grid
lines are placed above or below your other plot elements.

.. _howto-axis-equal:

Make the aspect ratio for plots equal
-------------------------------------

The Axes property :meth:`~matplotlib.axes.Axes.set_aspect` controls the
aspect ratio of the axes.  You can set it to be 'auto', 'equal', or
some ratio which controls the ratio::

  ax = fig.add_subplot(111, aspect='equal')



.. htmlonly::

    See :ref:`pylab_examples-equal_aspect_ratio` for a complete example.


.. _howto-movie:

Make a movie
------------

If you want to take an animated plot and turn it into a movie, the
best approach is to save a series of image files (eg PNG) and use an
external tool to convert them to a movie.  You can use `mencoder
<http://www.mplayerhq.hu/DOCS/HTML/en/mencoder.html>`_,
which is part of the `mplayer <http://www.mplayerhq.hu>`_ suite
for this::

    #fps (frames per second) controls the play speed
    mencoder 'mf://*.png' -mf type=png:fps=10 -ovc \\
       lavc -lavcopts vcodec=wmv2 -oac copy -o animation.avi

The swiss army knife of image tools, ImageMagick's `convert
<http://www.imagemagick.org/script/convert.php>`_ works for this as
well.

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

    Josh Lifton provided this example :ref:`old_animation-movie_demo`, which
    is possibly dated since it was written in 2004.


.. _howto-twoscale:

Multiple y-axis scales
----------------------

A frequent request is to have two scales for the left and right
y-axis, which is possible using :func:`~matplotlib.pyplot.twinx` (more
than two scales are not currently supported, though it is on the wish
list).  This works pretty well, though there are some quirks when you
are trying to interactively pan and zoom, because both scales do not get
the signals.

The approach uses :func:`~matplotlib.pyplot.twinx` (and its sister
:func:`~matplotlib.pyplot.twiny`) to use *2 different axes*,
turning the axes rectangular frame off on the 2nd axes to keep it from
obscuring the first, and manually setting the tick locs and labels as
desired.  You can use separate matplotlib.ticker formatters and
locators as desired because the two axes are independent.

.. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    t = np.arange(0.01, 10.0, 0.01)
    s1 = np.exp(t)
    ax1.plot(t, s1, 'b-')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp')

    ax2 = ax1.twinx()
    s2 = np.sin(2*np.pi*t)
    ax2.plot(t, s2, 'r.')
    ax2.set_ylabel('sin')
    plt.show()


.. htmlonly::

    See :ref:`api-two_scales` for a complete example

.. _howto-batch:

Generate images without having a window appear
----------------------------------------------

The easiest way to do this is use a non-interactive backend (see
:ref:`what-is-a-backend`) such as Agg (for PNGs), PDF, SVG or PS.  In
your figure-generating script, just call the
:func:`matplotlib.use` directive before importing pylab or
pyplot::

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot([1,2,3])
    plt.savefig('myfig')


.. seealso::
    :ref:`howto-webapp` for information about running matplotlib inside
    of a web application.

.. _howto-show:

Use :func:`~matplotlib.pyplot.show`
------------------------------------------

When you want to view your plots on your display,
the user interface backend will need to start the GUI mainloop.
This is what :func:`~matplotlib.pyplot.show` does.  It tells
matplotlib to raise all of the figure windows created so far and start
the mainloop. Because this mainloop is blocking by default (i.e., script
execution is paused), you should only call this once per script, at the end.
Script execution is resumed after the last window is closed. Therefore, if
you are using matplotlib to generate only images and do not want a user
interface window, you do not need to call ``show``  (see :ref:`howto-batch`
and :ref:`what-is-a-backend`).

.. note::
   Because closing a figure window invokes the destruction of its plotting
   elements, you should call :func:`~matplotlib.pyplot.savefig` *before*
   calling ``show`` if you wish to save the figure as well as view it.

.. versionadded:: v1.0.0
   ``show`` now starts the GUI mainloop only if it isn't already running.
   Therefore, multiple calls to ``show`` are now allowed.

Having ``show`` block further execution of the script or the python
interpreter depends on whether matplotlib is set for interactive mode
or not.  In non-interactive mode (the default setting), execution is paused
until the last figure window is closed.  In interactive mode, the execution
is not paused, which allows you to create additional figures (but the script
won't finish until the last figure window is closed).

.. note::
   Support for interactive/non-interactive mode depends upon the backend.
   Until version 1.0.0 (and subsequent fixes for 1.0.1), the behavior of
   the interactive mode was not consistent across backends.
   As of v1.0.1, only the macosx backend differs from other backends
   because it does not support non-interactive mode.


Because it is expensive to draw, you typically will not want matplotlib
to redraw a figure many times in a script such as the following::

    plot([1,2,3])            # draw here ?
    xlabel('time')           # and here ?
    ylabel('volts')          # and here ?
    title('a simple plot')   # and here ?
    show()


However, it is *possible* to force matplotlib to draw after every command,
which might be what you want when working interactively at the
python console (see :ref:`mpl-shell`), but in a script you want to
defer all drawing until the call to ``show``.  This is especially
important for complex figures that take some time to draw.
:func:`~matplotlib.pyplot.show` is designed to tell matplotlib that
you're all done issuing commands and you want to draw the figure now.

.. note::

    :func:`~matplotlib.pyplot.show` should typically only be called at
    most once per script and it should be the last line of your
    script.  At that point, the GUI takes control of the interpreter.
    If you want to force a figure draw, use
    :func:`~matplotlib.pyplot.draw` instead.

Many users are frustrated by ``show`` because they want it to be a
blocking call that raises the figure, pauses the script until they
close the figure, and then allow the script to continue running until
the next figure is created and the next show is made.  Something like
this::

   # WARNING : illustrating how NOT to use show
   for i in range(10):
       # make figure i
       show()

This is not what show does and unfortunately, because doing blocking
calls across user interfaces can be tricky, is currently unsupported,
though we have made significant progress towards supporting blocking events.

.. versionadded:: v1.0.0
   As noted earlier, this restriction has been relaxed to allow multiple
   calls to ``show``.  In *most* backends, you can now expect to be
   able to create new figures and raise them in a subsequent call to
   ``show`` after closing the figures from a previous call to ``show``.


.. _howto-contribute:

Contributing: howto
===================

.. _how-to-submit-patch:

Submit a patch
--------------

See :ref:`making-patches` for information on how to make a patch with git.

If you are posting a patch to fix a code bug, please explain your
patch in words -- what was broken before and how you fixed it.  Also,
even if your patch is particularly simple, just a few lines or a
single function replacement, we encourage people to submit git diffs
against HEAD of the branch they are patching.  It just makes life
easier for us, since we (fortunately) get a lot of contributions, and
want to receive them in a standard format.  If possible, for any
non-trivial change, please include a complete, free-standing example
that the developers can run unmodified which shows the undesired
behavior pre-patch and the desired behavior post-patch, with a clear
verbal description of what to look for.  A developer may
have written the function you are working on years ago, and may no
longer be with the project, so it is quite possible you are the world
expert on the code you are patching and we want to hear as much detail
as you can offer.

When emailing your patch and examples, feel free to paste any code
into the text of the message, indeed we encourage it, but also attach
the patches and examples since many email clients screw up the
formatting of plain text, and we spend lots of needless time trying to
reformat the code to make it usable.

You should check out the guide to developing matplotlib to make sure
your patch abides by our coding conventions
:ref:`developers-guide-index`.


.. _how-to-contribute-docs:

Contribute to matplotlib documentation
--------------------------------------

matplotlib is a big library, which is used in many ways, and the
documentation has only scratched the surface of everything it can
do.  So far, the place most people have learned all these features are
through studying the examples (:ref:`how-to-search-examples`), which is a
recommended and great way to learn, but it would be nice to have more
official narrative documentation guiding people through all the dark
corners.  This is where you come in.

There is a good chance you know more about matplotlib usage in some
areas, the stuff you do every day, than many of the core developers
who wrote most of the documentation.  Just pulled your hair out
compiling matplotlib for windows?  Write a FAQ or a section for the
:ref:`installing-faq` page.  Are you a digital signal processing wizard?
Write a tutorial on the signal analysis plotting functions like
:func:`~matplotlib.pyplot.xcorr`, :func:`~matplotlib.pyplot.psd` and
:func:`~matplotlib.pyplot.specgram`.  Do you use matplotlib with
`django <http://www.djangoproject.com/>`_ or other popular web
application servers?  Write a FAQ or tutorial and we'll find a place
for it in the :ref:`users-guide-index`.  Bundle matplotlib in a
`py2exe <http://www.py2exe.org/>`_ app?  ... I think you get the idea.

matplotlib is documented using the `sphinx
<http://sphinx.pocoo.org/index.html>`_ extensions to restructured text
`(ReST) <http://docutils.sourceforge.net/rst.html>`_.  sphinx is an
extensible python framework for documentation projects which generates
HTML and PDF, and is pretty easy to write; you can see the source for this
document or any page on this site by clicking on the *Show Source* link
at the end of the page in the sidebar (or `here
<../_sources/faq/howto_faq.txt>`_ for this document).

The sphinx website is a good resource for learning sphinx, but we have
put together a cheat-sheet at :ref:`documenting-matplotlib` which
shows you how to get started, and outlines the matplotlib conventions
and extensions, eg for including plots directly from external code in
your documents.

Once your documentation contributions are working (and hopefully
tested by actually *building* the docs) you can submit them as a patch
against git.  See :ref:`install-git` and :ref:`how-to-submit-patch`.
Looking for something to do?  Search for `TODO <../search.html?q=todo>`_.




.. _howto-webapp:

Matplotlib in a web application server
======================================

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

Alternatively, you can avoid pylab/pyplot altogether, which will give
you a little more control, by calling the API directly as shown in
:ref:`api-agg_oo`.

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

Here is an example using the Python Imaging Library (PIL).  First, the figure
is saved to a StringIO object which is then fed to PIL for further
processing::

    import StringIO, Image
    imgdata = StringIO.StringIO()
    fig.savefig(imgdata, format='png')
    imgdata.seek(0)  # rewind the data
    im = Image.open(imgdata)


matplotlib with apache
----------------------

TODO; see :ref:`how-to-contribute-docs`.

matplotlib with django
----------------------

TODO; see :ref:`how-to-contribute-docs`.

matplotlib with zope
--------------------

TODO; see :ref:`how-to-contribute-docs`.

.. _howto-click-maps:

Clickable images for HTML
-------------------------

Andrew Dalke of `Dalke Scientific <http://www.dalkescientific.com>`_
has written a nice `article
<http://www.dalkescientific.com/writings/diary/archive/2005/04/24/interactive_html.html>`_
on how to make html click maps with matplotlib agg PNGs.  We would
also like to add this functionality to SVG and add a SWF backend to
support these kind of images.  If you are interested in contributing
to these efforts that would be great.


.. _how-to-search-examples:

Search examples
===============

The nearly 300 code :ref:`examples-index` included with the matplotlib
source distribution are full-text searchable from the :ref:`search`
page, but sometimes when you search, you get a lot of results from the
:ref:`api-index` or other documentation that you may not be interested
in if you just want to find a complete, free-standing, working piece
of example code.  To facilitate example searches, we have tagged every
code example page with the keyword ``codex`` for *code example* which
shouldn't appear anywhere else on this site except in the FAQ.
So if you want to search for an example that uses an
ellipse, :ref:`search` for ``codex ellipse``.


.. _how-to-cite-mpl:

Cite Matplotlib
===============

If you want to refer to matplotlib in a publication, you can use
"Matplotlib: A 2D Graphics Environment" by J. D. Hunter In Computing
in Science & Engineering, Vol. 9, No. 3. (2007), pp. 90-95 (see `this
reference page <http://dx.doi.org/10.1109/MCSE.2007.55>`_)::

  @article{Hunter:2007,
	  Address = {10662 LOS VAQUEROS CIRCLE, PO BOX 3014, LOS ALAMITOS, CA 90720-1314 USA},
	  Author = {Hunter, John D.},
	  Date-Added = {2010-09-23 12:22:10 -0700},
	  Date-Modified = {2010-09-23 12:22:10 -0700},
	  Isi = {000245668100019},
	  Isi-Recid = {155389429},
	  Journal = {Computing In Science \& Engineering},
	  Month = {May-Jun},
	  Number = {3},
	  Pages = {90--95},
	  Publisher = {IEEE COMPUTER SOC},
	  Times-Cited = {21},
	  Title = {Matplotlib: A 2D graphics environment},
	  Type = {Editorial Material},
	  Volume = {9},
	  Year = {2007},
	  Abstract = {Matplotlib is a 2D graphics package used for Python for application
                      development, interactive scripting, and publication-quality image
                      generation across user interfaces and operating systems.},
	  Bdsk-Url-1 = {http://gateway.isiknowledge.com/gateway/Gateway.cgi?GWVersion=2&SrcAuth=Alerting&SrcApp=Alerting&DestApp=WOS&DestLinkType=FullRecord;KeyUT=000245668100019}}
