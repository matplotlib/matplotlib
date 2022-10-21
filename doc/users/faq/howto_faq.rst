.. _howto-faq:

.. redirect-from:: /faq/howto_faq

******
How-to
******

.. contents::
   :backlinks: none


.. _how-to-too-many-ticks:

Why do I have so many ticks, and/or why are they out of order?
--------------------------------------------------------------

One common cause for unexpected tick behavior is passing a *list of strings
instead of numbers or datetime objects*. This can easily happen without notice
when reading in a comma-delimited text file. Matplotlib treats lists of strings
as *categorical* variables
(:doc:`/gallery/lines_bars_and_markers/categorical_variables`), and by default
puts one tick per category, and plots them in the order in which they are
supplied.

.. plot::
    :include-source:
    :align: center

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(6, 2))

    ax[0].set_title('Ticks seem out of order / misplaced')
    x = ['5', '20', '1', '9']  # strings
    y = [5, 20, 1, 9]
    ax[0].plot(x, y, 'd')
    ax[0].tick_params(axis='x', labelcolor='red', labelsize=14)

    ax[1].set_title('Many ticks')
    x = [str(xx) for xx in np.arange(100)]  # strings
    y = np.arange(100)
    ax[1].plot(x, y)
    ax[1].tick_params(axis='x', labelcolor='red', labelsize=14)

The solution is to convert the list of strings to numbers or
datetime objects (often ``np.asarray(numeric_strings, dtype='float')`` or
``np.asarray(datetime_strings, dtype='datetime64[s]')``).

For more information see :doc:`/gallery/ticks/ticks_too_many`.

.. _howto-determine-artist-extent:

Determine the extent of Artists in the Figure
---------------------------------------------

Sometimes we want to know the extent of an Artist.  Matplotlib `.Artist` objects
have a method `.Artist.get_window_extent` that will usually return the extent of
the artist in pixels.  However, some artists, in particular text, must be
rendered at least once before their extent is known.  Matplotlib supplies
`.Figure.draw_without_rendering`, which should be called before calling
``get_window_extent``.

.. _howto-figure-empty:

Check whether a figure is empty
-------------------------------
Empty can actually mean different things. Does the figure contain any artists?
Does a figure with an empty `~.axes.Axes` still count as empty? Is the figure
empty if it was rendered pure white (there may be artists present, but they
could be outside the drawing area or transparent)?

For the purpose here, we define empty as: "The figure does not contain any
artists except it's background patch." The exception for the background is
necessary, because by default every figure contains a `.Rectangle` as it's
background patch. This definition could be checked via::

    def is_empty(figure):
        """
        Return whether the figure contains no Artists (other than the default
        background patch).
        """
        contained_artists = figure.get_children()
        return len(contained_artists) <= 1

We've decided not to include this as a figure method because this is only one
way of defining empty, and checking the above is only rarely necessary.
Usually the user or program handling the figure know if they have added
something to the figure.

Checking whether a figure would render empty cannot be reliably checked except
by actually rendering the figure and investigating the rendered result.

.. _howto-findobj:

Find all objects in a figure of a certain type
----------------------------------------------

Every Matplotlib artist (see :doc:`/tutorials/intermediate/artists`) has a method
called :meth:`~matplotlib.artist.Artist.findobj` that can be used to
recursively search the artist for any artists it may contain that meet
some criteria (e.g., match all :class:`~matplotlib.lines.Line2D`
instances or match some arbitrary filter function).  For example, the
following snippet finds every object in the figure which has a
``set_color`` property and makes the object blue::

    def myfunc(x):
        return hasattr(x, 'set_color')

    for o in fig.findobj(myfunc):
        o.set_color('blue')

You can also filter on class instances::

    import matplotlib.text as text
    for o in fig.findobj(text.Text):
        o.set_fontstyle('italic')

.. _howto-suppress_offset:

Prevent ticklabels from having an offset
----------------------------------------
The default formatter will use an offset to reduce
the length of the ticklabels.  To turn this feature
off on a per-axis basis::

   ax.get_xaxis().get_major_formatter().set_useOffset(False)

set :rc:`axes.formatter.useoffset`, or use a different
formatter.  See :mod:`~matplotlib.ticker` for details.

.. _howto-transparent:

Save transparent figures
------------------------

The :meth:`~matplotlib.pyplot.savefig` command has a keyword argument
*transparent* which, if 'True', will make the figure and axes
backgrounds transparent when saving, but will not affect the displayed
image on the screen.

If you need finer grained control, e.g., you do not want full transparency
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

Many image file formats can only have one image per file, but some formats
support multi-page files.  Currently, Matplotlib only provides multi-page
output to pdf files, using either the pdf or pgf backends, via the
`.backend_pdf.PdfPages` and `.backend_pgf.PdfPages` classes.

.. _howto-auto-adjust:

Make room for tick labels
-------------------------

By default, Matplotlib uses fixed percentage margins around subplots. This can
lead to labels overlapping or being cut off at the figure boundary. There are
multiple ways to fix this:

- Manually adapt the subplot parameters using `.Figure.subplots_adjust` /
  `.pyplot.subplots_adjust`.
- Use one of the automatic layout mechanisms:

  - constrained layout (:doc:`/tutorials/intermediate/constrainedlayout_guide`)
  - tight layout (:doc:`/tutorials/intermediate/tight_layout_guide`)

- Calculate good values from the size of the plot elements yourself
  (:doc:`/gallery/subplots_axes_and_figures/auto_subplots_adjust`)

.. _howto-align-label:

Align my ylabels across multiple subplots
-----------------------------------------

If you have multiple subplots over one another, and the y data have
different scales, you can often get ylabels that do not align
vertically across the multiple subplots, which can be unattractive.
By default, Matplotlib positions the x location of the ylabel so that
it does not overlap any of the y ticks.  You can override this default
behavior by specifying the coordinates of the label.  The example
below shows the default behavior in the left subplots, and the manual
setting in the right subplots.

.. figure:: ../../gallery/text_labels_and_annotations/images/sphx_glr_align_ylabels_001.png
   :target: ../../gallery/text_labels_and_annotations/align_ylabels.html
   :align: center
   :scale: 50

.. _howto-set-zorder:

Control the draw order of plot elements
---------------------------------------

The draw order of plot elements, and thus which elements will be on top, is
determined by the `~.Artist.set_zorder` property.
See :doc:`/gallery/misc/zorder_demo` for a detailed description.

.. _howto-axis-equal:

Make the aspect ratio for plots equal
-------------------------------------

The Axes property :meth:`~matplotlib.axes.Axes.set_aspect` controls the
aspect ratio of the axes.  You can set it to be 'auto', 'equal', or
some ratio which controls the ratio::

  ax = fig.add_subplot(111, aspect='equal')

.. only:: html

    See :doc:`/gallery/subplots_axes_and_figures/axis_equal_demo` for a
    complete example.

.. _howto-twoscale:

Draw multiple y-axis scales
---------------------------

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
desired.  You can use separate ``matplotlib.ticker`` formatters and
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


.. only:: html

    See :doc:`/gallery/subplots_axes_and_figures/two_scales` for a
    complete example.

.. _howto-batch:

Generate images without having a window appear
----------------------------------------------

Simply do not call `~matplotlib.pyplot.show`, and directly save the figure to
the desired format::

    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3])
    plt.savefig('myfig.png')

.. seealso::

    :doc:`/gallery/user_interfaces/web_application_server_sgskip` for
    information about running matplotlib inside of a web application.

.. _how-to-threads:

Work with threads
-----------------

Matplotlib is not thread-safe: in fact, there are known race conditions
that affect certain artists.  Hence, if you work with threads, it is your
responsibility to set up the proper locks to serialize access to Matplotlib
artists.

You may be able to work on separate figures from separate threads.  However,
you must in that case use a *non-interactive backend* (typically Agg), because
most GUI backends *require* being run from the main thread as well.
