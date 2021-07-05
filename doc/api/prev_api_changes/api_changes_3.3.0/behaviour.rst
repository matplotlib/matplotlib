Behaviour changes
-----------------

``Formatter.fix_minus``
~~~~~~~~~~~~~~~~~~~~~~~
`.Formatter.fix_minus` now performs hyphen-to-unicode-minus replacement
whenever :rc:`axes.unicode_minus` is True; i.e. its behavior matches the one
of ``ScalarFormatter.fix_minus`` (`.ScalarFormatter` now just inherits that
implementation).

This replacement is now used by the ``format_data_short`` method of the various
builtin formatter classes, which affects the cursor value in the GUI toolbars.

``FigureCanvasBase`` now always has a ``manager`` attribute, which may be None
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, it did not necessarily have such an attribute.  A check for
``hasattr(figure.canvas, "manager")`` should now be replaced by
``figure.canvas.manager is not None`` (or ``getattr(figure.canvas, "manager", None) is not None``
for back-compatibility).

`.cbook.CallbackRegistry` now propagates exceptions when no GUI event loop is running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`.cbook.CallbackRegistry` now defaults to propagating exceptions thrown by
callbacks when no interactive GUI event loop is running.  If a GUI event loop
*is* running, `.cbook.CallbackRegistry` still defaults to just printing a
traceback, as unhandled exceptions can make the program completely ``abort()``
in that case.

``Axes.locator_params()`` validates ``axis`` parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`.axes.Axes.locator_params` used to accept any value for ``axis`` and silently
did nothing, when passed an unsupported value. It now raises a ``ValueError``.

``Axis.set_tick_params()`` validates ``which`` parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`.Axis.set_tick_params` (and the higher level `.axes.Axes.tick_params` and
`.pyplot.tick_params`) used to accept any value for ``which`` and silently
did nothing, when passed an unsupported value. It now raises a ``ValueError``.

``Axis.set_ticklabels()`` must match ``FixedLocator.locs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If an axis is using a `.ticker.FixedLocator`, typically set by a call to
`.Axis.set_ticks`, then the number of ticklabels supplied must match the
number of locations available (``FixedFormattor.locs``).  If not, a
``ValueError`` is raised.

``backend_pgf.LatexManager.latex``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``backend_pgf.LatexManager.latex`` is now created with ``encoding="utf-8"``, so
its ``stdin``, ``stdout``, and ``stderr`` attributes are utf8-encoded.

``pyplot.xticks()`` and ``pyplot.yticks()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, passing labels without passing the ticks to either `.pyplot.xticks`
and `.pyplot.yticks` would result in

    TypeError: object of type 'NoneType' has no len()

It now raises a ``TypeError`` with a proper description of the error.

Setting the same property under multiple aliases now raises a TypeError
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, calling e.g. ``plot(..., color=somecolor, c=othercolor)`` would
emit a warning because ``color`` and ``c`` actually map to the same Artist
property.  This now raises a TypeError.

`.FileMovieWriter` temporary frames directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`.FileMovieWriter` now defaults to writing temporary frames in a temporary
directory, which is always cleared at exit.  In order to keep the individual
frames saved on the filesystem, pass an explicit *frame_prefix*.

`.Axes.plot` no longer accepts *x* and *y* being both 2D and with different numbers of columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, calling `.Axes.plot` e.g. with *x* of shape ``(n, 3)`` and *y* of
shape ``(n, 2)`` would plot the first column of *x* against the first column
of *y*, the second column of *x* against the second column of *y*, **and** the
first column of *x* against the third column of *y*.  This now raises an error
instead.

`.Text.update_from` now copies usetex state from the source Text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`~.Axes.stem` now defaults to ``use_line_collection=True``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This creates the stem plot as a `.LineCollection` rather than individual
`.Line2D` objects, greatly improving performance.

rcParams color validator is now stricter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, rcParams entries whose values were color-like accepted "spurious"
extra letters or characters in the "middle" of the string, e.g. ``"(0, 1a, '0.5')"``
would be interpreted as ``(0, 1, 0.5)``.  These extra characters (including the
internal quotes) now cause a ValueError to be raised.

`.SymLogNorm` now has a *base* parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, `.SymLogNorm` had no *base* keyword argument, and
defaulted to ``base=np.e`` whereas the documentation said it was
``base=10``.  In preparation to make the default 10, calling
`.SymLogNorm` without the new *base* keyword argument emits a
deprecation warning.


`~.Axes.errorbar` now color cycles when only errorbar color is set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously setting the *ecolor* would turn off automatic color cycling for the plot, leading to the
the lines and markers defaulting to whatever the first color in the color cycle was in the case of
multiple plot calls.

`.rcsetup.validate_color_for_prop_cycle` now always raises TypeError for bytes input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It previously raised `TypeError`, **except** when the input was of the form
``b"C[number]"`` in which case it raised a ValueError.

`.FigureCanvasPS.print_ps` and `.FigureCanvasPS.print_eps` no longer apply edgecolor and facecolor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods now assume that the figure edge and facecolor have been correctly
applied by `.FigureCanvasBase.print_figure`, as they are normally called
through it.

This behavior is consistent with other figure saving methods
(`.FigureCanvasAgg.print_png`, `.FigureCanvasPdf.print_pdf`,
`.FigureCanvasSVG.print_svg`).

`.pyplot.subplot()` now raises TypeError when given an incorrect number of arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is consistent with other signature mismatch errors.  Previously a
ValueError was raised.

Shortcut for closing all figures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Shortcuts for closing all figures now also work for the classic toolbar.
There is no default shortcut any more because unintentionally closing all figures by a key press
might happen too easily. You can configure the shortcut yourself
using :rc:`keymap.quit_all`.

Autoscale for arrow
~~~~~~~~~~~~~~~~~~~
Calling ax.arrow() will now autoscale the axes.

``set_tick_params(label1On=False)`` now also makes the offset text (if any) invisible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... because the offset text can rarely be interpreted without tick labels
anyways.

`.Axes.annotate` and `.pyplot.annotate` parameter name changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The parameter ``s`` to `.Axes.annotate` and  `.pyplot.annotate` is renamed to
``text``, matching `.Annotation`.

The old parameter name remains supported, but
support for it will be dropped in a future Matplotlib release.

`.font_manager.json_dump` now locks the font manager dump file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... to prevent multiple processes from writing to it at the same time.

`.pyplot.rgrids` and `.pyplot.thetagrids` now act as setters also when called with only kwargs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, keyword arguments were silently ignored when no positional
arguments were given.

`.Axis.get_minorticklabels` and `.Axis.get_majorticklabels` now returns plain list
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Previously, `.Axis.get_minorticklabels` and `.Axis.get_majorticklabels` returns
silent_list. Their return type is now changed to normal list.
`.get_xminorticklabels`, `.get_yminorticklabels`, `.get_zminorticklabels`,
`.Axis.get_ticklabels`, `.get_xmajorticklabels`, `.get_ymajorticklabels` and
`.get_zmajorticklabels` methods will be affected by this change.

Default slider formatter
~~~~~~~~~~~~~~~~~~~~~~~~
The default method used to format `.Slider` values has been changed to use a
`.ScalarFormatter` adapted the slider values limits.  This should ensure that
values are displayed with an appropriate number of significant digits even if
they are much smaller or much bigger than 1.  To restore the old behavior,
explicitly pass a "%1.2f" as the *valfmt* parameter to `.Slider`.

Add *normalize*  keyword argument to ``Axes.pie``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``pie()`` used to draw a partial pie if the sum of the values was < 1. This behavior
is deprecated and will change to always normalizing the values to a full pie by default.
If you want to draw a partial pie, please pass ``normalize=False`` explicitly.

``table.CustomCell`` is now an alias for `.table.Cell`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All the functionality of ``CustomCell`` has been moved to its base class
`~.table.Cell`.

wx Timer interval
~~~~~~~~~~~~~~~~~
Setting the timer interval on a not-yet-started ``TimerWx`` won't start it
anymore.

"step"-type histograms default to the zorder of `.Line2D`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This ensures that they go above gridlines by default.  The old ``zorder`` can
be kept by passing it as a keyword argument to `.Axes.hist`.

`.Legend` and `.OffsetBox` visibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`.Legend` and `.OffsetBox` subclasses (`.PaddedBox`, `.AnchoredOffsetbox`, and
`.AnnotationBbox`) no longer directly keep track of the visibility of their
underlying `.Patch` artist, but instead pass that flag down to the `.Patch`.

`.Legend` and `.Table` no longer allow invalid locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This affects legends produced on an Axes (`.Axes.legend` and `.pyplot.legend`)
and on a Figure (`.Figure.legend` and `.pyplot.figlegend`).  Figure legends also
no longer accept the unsupported ``'best'`` location.  Previously, invalid Axes
locations would use ``'best'`` and invalid Figure locations would used ``'upper
right'``.

Passing Line2D's *drawstyle* together with *linestyle* is removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of ``plt.plot(..., linestyle="steps--")``, use ``plt.plot(...,
linestyle="--", drawstyle="steps")``. ``ds`` is also an alias for
``drawstyle``.

Upper case color strings
~~~~~~~~~~~~~~~~~~~~~~~~

Support for passing single-letter colors (one of "rgbcmykw") as UPPERCASE
characters is removed; these colors are now case-sensitive (lowercase).

tight/constrained_layout no longer worry about titles that are too wide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*tight_layout* and *constrained_layout* shrink axes to accommodate
"decorations" on the axes.  However, if an xlabel or title is too long in the
x direction, making the axes smaller in the x-direction doesn't help.  The
behavior of both has been changed to ignore the width of the title and
xlabel and the height of the ylabel in the layout logic.

This also means there is a new keyword argument for `.axes.Axes.get_tightbbox`
and `.axis.Axis.get_tightbbox`: ``for_layout_only``, which defaults to *False*,
but if *True* returns a bounding box using the rules above.

:rc:`savefig.facecolor` and :rc:`savefig.edgecolor` now default to "auto"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This newly allowed value for :rc:`savefig.facecolor` and :rc:`savefig.edgecolor`,
as well as the *facecolor* and *edgecolor* parameters to `.Figure.savefig`, means
"use whatever facecolor and edgecolor the figure current has".

When using a single dataset, `.Axes.hist` no longer wraps the added artist in a `.silent_list`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When `.Axes.hist` is called with a single dataset, it adds to the axes either
a `.BarContainer` object (when ``histtype="bar"`` or ``"barstacked"``), or a
`.Polygon` object (when ``histype="step"`` or ``"stepfilled"``) -- the latter
being wrapped in a list-of-one-element.  Previously, either artist would be
wrapped in a `.silent_list`.  This is no longer the case: the `.BarContainer` is
now returned as is (this is an API breaking change if you were directly relying
on the concrete `list` API; however, `.BarContainer` inherits from `tuple` so
most common operations remain available), and the list-of-one `.Polygon` is
returned as is.  This makes the `repr` of the returned artist more accurate: it
is now ::

    <BarContainer object of 10 artists>  # "bar", "barstacked"
    [<matplotlib.patches.Polygon object at 0xdeadbeef>]  # "step", "stepfilled"

instead of ::

    <a list of 10 Patch objects>  # "bar", "barstacked"
    <a list of 1 Patch objects>  # "step", "stepfilled"

When `.Axes.hist` is called with multiple artists, it still wraps its return
value in a `.silent_list`, but uses more accurate type information ::

    <a list of 3 BarContainer objects>  # "bar", "barstacked"
    <a list of 3 List[Polygon] objects>  # "step", "stepfilled"

instead of ::

    <a list of 3 Lists of Patches objects>  # "bar", "barstacked"
    <a list of 3 Lists of Patches objects>  # "step", "stepfilled"

Qt and wx backends no longer create a status bar by default
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The coordinates information is now displayed in the toolbar, consistently with
the other backends.  This is intended to simplify embedding of Matplotlib in
larger GUIs, where Matplotlib may control the toolbar but not the status bar.

:rc:`text.hinting` now supports names mapping to FreeType flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:rc:`text.hinting` now supports the values "default", "no_autohint",
"force_autohint", and "no_hinting", which directly map to the FreeType flags
FT_LOAD_DEFAULT, etc.  The old synonyms (respectively "either", "native",
"auto", and "none") are still supported, but their use is discouraged.  To get
normalized values, use `.backend_agg.get_hinting_flag`, which returns integer
flag values.

`.cbook.get_sample_data` auto-loads numpy arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When `.cbook.get_sample_data` is used to load a npy or npz file and the
keyword-only parameter ``np_load`` is True, the file is automatically loaded
using `numpy.load`.  ``np_load`` defaults to False for backwards compatibility,
but will become True in a later release.

``get_text_width_height_descent`` now checks ``ismath`` rather than :rc:`text.usetex`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... to determine whether a string should be passed to the usetex machinery or
not.  This allows single strings to be marked as not-usetex even when the
rcParam is True.

`.Axes.vlines`, `.Axes.hlines`, `.pyplot.vlines` and `.pyplot.hlines` *colors* parameter default change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *colors* parameter will now default to :rc:`lines.color`, while previously it defaulted to 'k'.

Aggressively autoscale clim in ``ScalerMappable`` classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Previously some plotting methods would defer autoscaling until the
first draw if only one of the *vmin* or *vmax* keyword arguments were
passed (`.Axes.scatter`, `.Axes.hexbin`, `.Axes.imshow`,
`.Axes.pcolorfast`) but would scale based on the passed data if
neither was passed (independent of the *norm* keyword arguments).
Other methods (`.Axes.pcolor`, `.Axes.pcolormesh`) always autoscaled
base on the initial data.

All of the plotting methods now resolve the unset *vmin* or *vmax*
at the initial call time using the data passed in.

If you were relying on exactly one of the *vmin* or *vmax* remaining
unset between the time when the method is called and the first time
the figure is rendered you get back the old behavior by manually setting
the relevant limit back to `None` ::

  cm_obj.norm.vmin = None
  # or
  cm_obj.norm.vmax = None

which will be resolved during the draw process.
