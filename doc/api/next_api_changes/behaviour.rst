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

Previously, `.SymLogNorm` had no *base* kwarg, and defaulted to ``base=np.e``
whereas the documentation said it was ``base=10``.  In preparation to make
the default 10, calling `.SymLogNorm` without the new *base* kwarg emits a
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
