.. _whats-new-3-0-0:

What's new in Matplotlib 3.0 (Sep 18, 2018)
===========================================

Improved default backend selection
----------------------------------

The default backend no longer must be set as part of the build
process.  Instead, at run time, the builtin backends are tried in
sequence until one of them imports.

Headless Linux servers (identified by the DISPLAY environment variable not
being defined) will not select a GUI backend.

Cyclic colormaps
----------------

Two new colormaps named 'twilight' and 'twilight_shifted' have been
added.  These colormaps start and end on the same color, and have two
symmetric halves with equal lightness, but diverging color. Since they
wrap around, they are a good choice for cyclic data such as phase
angles, compass directions, or time of day. Like *viridis* and
*cividis*, *twilight* is perceptually uniform and colorblind friendly.


Ability to scale axis by a fixed order of magnitude
---------------------------------------------------

To scale an axis by a fixed order of magnitude, set the *scilimits* argument of
`.Axes.ticklabel_format` to the same (non-zero) lower and upper limits. Say to scale
the y axis by a million (1e6), use

.. code-block:: python

  ax.ticklabel_format(style='sci', scilimits=(6, 6), axis='y')

The behavior of ``scilimits=(0, 0)`` is unchanged. With this setting, Matplotlib will adjust
the order of magnitude depending on the axis values, rather than keeping it fixed. Previously, setting
``scilimits=(m, m)`` was equivalent to setting ``scilimits=(0, 0)``.


Add ``AnchoredDirectionArrows`` feature to mpl_toolkits
-------------------------------------------------------

A new mpl_toolkits class
:class:`~mpl_toolkits.axes_grid1.anchored_artists.AnchoredDirectionArrows`
draws a pair of orthogonal arrows to indicate directions on a 2D plot. A
minimal working example takes in the transformation object for the coordinate
system (typically ax.transAxes), and arrow labels. There are several optional
parameters that can be used to alter layout. For example, the arrow pairs can
be rotated and the color can be changed. By default the labels and arrows have
the same color, but the class may also pass arguments for customizing arrow
and text layout, these are passed to :class:`matplotlib.textpath.TextPath` and
`matplotlib.patches.FancyArrowPatch`. Location, length and width for both
arrow tail and head can be adjusted, the direction arrows and labels can have a
frame. Padding and separation parameters can be adjusted.


Add ``minorticks_on()/off()`` methods for colorbar
--------------------------------------------------

A new method ``ColorbarBase.minorticks_on`` has been added to
correctly display minor ticks on a colorbar. This method doesn't allow the
minor ticks to extend into the regions beyond vmin and vmax when the *extend*
keyword argument (used while creating the colorbar) is set to 'both', 'max' or
'min'. A complementary method ``ColorbarBase.minorticks_off`` has
also been added to remove the minor ticks on the colorbar.


Colorbar ticks can now be automatic
-----------------------------------

The number of ticks placed on colorbars was previously appropriate for a large
colorbar, but looked bad if the colorbar was made smaller (i.e. via the
*shrink* keyword argument). This has been changed so that the number of ticks
is now responsive to how large the colorbar is.



Don't automatically rename duplicate file names
-----------------------------------------------

Previously, when saving a figure to a file using the GUI's
save dialog box, if the default filename (based on the
figure window title) already existed on disk, Matplotlib
would append a suffix (e.g. ``Figure_1-1.png``), preventing
the dialog from prompting to overwrite the file. This
behaviour has been removed. Now if the file name exists on
disk, the user is prompted whether or not to overwrite it.
This eliminates guesswork, and allows intentional
overwriting, especially when the figure name has been
manually set using ``figure.canvas.set_window_title()``.


Legend now has a *title_fontsize* keyword argument (and rcParam)
----------------------------------------------------------------

The title for a `.Figure.legend` and `.Axes.legend` can now have its font size
set via the *title_fontsize* keyword argument.  There is also a new
:rc:`legend.title_fontsize`.  Both default to ``None``, which means the legend
title will have the same font size as the axes default font size (*not* the
legend font size, set by the *fontsize* keyword argument or
:rc:`legend.fontsize`).


Support for axes.prop_cycle property *markevery* in rcParams
------------------------------------------------------------

The Matplotlib ``rcParams`` settings object now supports configuration
of the attribute :rc:`axes.prop_cycle` with cyclers using the *markevery*
Line2D object property. 

Multi-page PDF support for pgf backend
--------------------------------------

The pgf backend now also supports multi-page PDF files.

.. code-block:: python

    from matplotlib.backends.backend_pgf import PdfPages
    import matplotlib.pyplot as plt

    with PdfPages('multipage.pdf') as pdf:
        # page 1
        plt.plot([2, 1, 3])
        pdf.savefig()

        # page 2
        plt.cla()
        plt.plot([3, 1, 2])
        pdf.savefig()


Pie charts are now circular by default
--------------------------------------
We acknowledge that the majority of people do not like egg-shaped pies.
Therefore, an axes to which a pie chart is plotted will be set to have
equal aspect ratio by default. This ensures that the pie appears circular
independent on the axes size or units. To revert to the previous behaviour
set the axes' aspect ratio to automatic by using ``ax.set_aspect("auto")`` or
``plt.axis("auto")``.

Add ``ax.get_gridspec`` to `.SubplotBase`
-----------------------------------------

New method `.SubplotBase.get_gridspec` is added so that users can
easily get the gridspec that went into making an axes:

  .. code::

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 2)
    gs = axs[0, -1].get_gridspec()

    # remove the last column
    for ax in axs[:,-1].flatten():
      ax.remove()

    # make a subplot in last column that spans rows.
    ax = fig.add_subplot(gs[:, -1])
    plt.show()


Axes titles will no longer overlap xaxis
----------------------------------------

Previously an axes title had to be moved manually if an xaxis overlapped
(usually when the xaxis was put on the top of the axes).  Now, the title
will be automatically moved above the xaxis and its decorators (including
the xlabel) if they are at the top.

If desired, the title can still be placed manually.  There is a slight kludge;
the algorithm checks if the y-position of the title is 1.0 (the default),
and moves if it is.  If the user places the title in the default location
(i.e. ``ax.title.set_position(0.5, 1.0)``), the title will still be moved
above the xaxis.  If the user wants to avoid this, they can
specify a number that is close (i.e. ``ax.title.set_position(0.5, 1.01)``)
and the title will not be moved via this algorithm.



New convenience methods for GridSpec
------------------------------------

There are new convenience methods for `.gridspec.GridSpec` and
`.gridspec.GridSpecFromSubplotSpec`.  Instead of the former we can
now call `.Figure.add_gridspec` and for the latter `.SubplotSpec.subgridspec`.

.. code-block:: python

    import matplotlib.pyplot as plt

    fig = plt.figure()
    gs0 = fig.add_gridspec(3, 1)
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    gssub = gs0[2].subgridspec(1, 3)
    for i in range(3):
        fig.add_subplot(gssub[0, i])


Figure has an `~.figure.Figure.add_artist` method
-------------------------------------------------

A method `~.figure.Figure.add_artist` has been added to the
:class:`~.figure.Figure` class, which allows artists to be added directly
to a figure. E.g. ::

   circ = plt.Circle((.7, .5), .05)
   fig.add_artist(circ)

In case the added artist has no transform set previously, it will be set to
the figure transform (``fig.transFigure``).
This new method may be useful for adding artists to figures without axes or to
easily position static elements in figure coordinates.


``:math:`` directive renamed to ``:mathmpl:``
---------------------------------------------

The ``:math:`` rst role provided by `matplotlib.sphinxext.mathmpl` has been
renamed to ``:mathmpl:`` to avoid conflicting with the ``:math:`` role that
Sphinx 1.8 provides by default.  (``:mathmpl:`` uses Matplotlib to render math
expressions to images embedded in html, whereas Sphinx uses MathJax.)

When using Sphinx<1.8, both names (``:math:`` and ``:mathmpl:``) remain
available for backwards-compatibility.
