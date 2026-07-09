Removals
--------


``matplotlib.cm.get_cmap``
~~~~~~~~~~~~~~~~~~~~~~~~~~

Colormaps are now available through the `.ColormapRegistry` accessible via
`matplotlib.colormaps` or `matplotlib.pyplot.colormaps`.

If you have the name of a colormap as a string, you can use a direct lookup,
``matplotlib.colormaps[name]`` or ``matplotlib.pyplot.colormaps[name]``. Alternatively,
``matplotlib.colormaps.get_cmap`` will maintain the existing behavior of additionally
passing through `.Colormap` instances and converting ``None`` to the default colormap.
`matplotlib.pyplot.get_cmap` will stay as a shortcut to
``matplotlib.colormaps.get_cmap``.

``boxplot`` tick labels
^^^^^^^^^^^^^^^^^^^^^^^

The parameter *labels* has been removed in favour of *tick_labels* for clarity and
consistency with `~.Axes.bar`.

``plot_date``
~~~~~~~~~~~~~

Use of ``plot_date`` has been discouraged since Matplotlib 3.5 and deprecated since 3.9.
The ``plot_date`` function has now been removed.

- ``datetime``-like data should directly be plotted using `~.Axes.plot`.
- If you need to plot plain numeric data as :ref:`date-format` or need to set a
  timezone, call ``ax.xaxis.axis_date`` / ``ax.yaxis.axis_date`` before `~.Axes.plot`.
  See `.Axis.axis_date`.

``GridHelperCurveLinear.get_tick_iterator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is removed with no replacement.

*nth_coord* parameter to axisartist helpers for fixed axis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Helper APIs in `.axisartist` for generating a "fixed" axis on rectilinear axes
(`.FixedAxisArtistHelperRectilinear`) no longer take a *nth_coord* parameter.
That parameter is entirely inferred from the (required) *loc* parameter.

For curvilinear axes, the *nth_coord* parameter remains supported (it affects
the *ticks*, not the axis position itself), but it is now keyword-only.

``rcsetup.interactive_bk``, ``rcsetup.non_interactive_bk`` and ``rcsetup.all_backends``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are removed and replaced by ``matplotlib.backends.backend_registry.list_builtin``
with the following arguments

- ``matplotlib.backends.BackendFilter.INTERACTIVE``
- ``matplotlib.backends.BackendFilter.NON_INTERACTIVE``
- ``None``

*interval* parameter of ``TimerBase.start``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The timer interval parameter can no longer be set while starting it. The interval can be
specified instead in the timer constructor, or by setting the timer.interval attribute.

``TransformNode.is_bbox``
~~~~~~~~~~~~~~~~~~~~~~~~~

... is removed. Instead check the object using ``isinstance(..., BboxBase)``.

``BboxTransformToMaxOnly``
~~~~~~~~~~~~~~~~~~~~~~~~~~

... is removed. It can be replaced by ``BboxTransformTo(LockableBbox(bbox, x0=0, y0=0))``.

Image path semantics of toolmanager-based tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously, MEP22 ("toolmanager-based") Tools would try to load their icon
(``tool.image``) relative to the current working directory, or, as a fallback, from
Matplotlib's own image directory. Because both approaches are problematic for
third-party tools (the end-user may change the current working directory at any time,
and third-parties cannot add new icons in Matplotlib's image directory), this behavior
has been removed; instead, ``tool.image`` is now interpreted relative to the directory
containing the source file where the ``Tool.image`` class attribute is defined.
(Defining ``tool.image`` as an absolute path also works and is compatible with both the
old and the new semantics.)
