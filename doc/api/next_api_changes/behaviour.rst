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
