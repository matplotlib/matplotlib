Behaviour Changes
-----------------

plot() shorthand format interprets "Cn" (n>9) as a color-cycle color
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously, ``plot(..., "-C11")`` would be interpreted as requesting a plot using
linestyle "-", color "C1" (color #1 of the color cycle), and marker "1" ("tri-down").
It is now interpreted as requesting linestyle "-" and color "C11" (color #11 of the
color cycle).

It is recommended to pass ambiguous markers (such as "1") explicitly using the *marker*
keyword argument. If the shorthand form is desired, such markers can also be
unambiguously set by putting them *before* the color string.

Legend labels for ``plot``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously if a sequence was passed to the *label* parameter of `~.Axes.plot` when
plotting a single dataset, the sequence was automatically cast to string for the legend
label. Now, if the sequence has only one element, that element will be the legend label.
To keep the old behavior, cast the sequence to string before passing.

Boxplots now ignore masked data points
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`~matplotlib.axes.Axes.boxplot` and `~matplotlib.cbook.boxplot_stats` now ignore any
masked points in the input data.

``axhspan`` and ``axvspan`` now return ``Rectangle``\s, not ``Polygon``\s
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This change allows using `~.Axes.axhspan` to draw an annulus on polar axes.

This change also affects other elements built via `~.Axes.axhspan` and `~.Axes.axvspan`,
such as ``Slider.poly``.

Improved handling of pan/zoom events of overlapping Axes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The forwarding of pan/zoom events is now determined by the visibility of the
background-patch (e.g. ``ax.patch.get_visible()``) and by the ``zorder`` of the axes.

- Axes with a visible patch capture the event and do not pass it on to axes below. Only
  the Axes with the highest ``zorder`` that contains the event is triggered (if there
  are multiple Axes with the same ``zorder``, the last added Axes counts)
- Axes with an invisible patch are also invisible to events and they are passed on to
  the axes below.

To override the default behavior and explicitly set whether an Axes should forward
navigation events, use `.Axes.set_forward_navigation_events`.

``loc='best'`` for ``legend`` now considers ``Text`` and ``PolyCollections``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The location selection ``legend`` now considers the existence of ``Text`` and
``PolyCollections`` in the ``badness`` calculation.

Note: The ``best`` option can already be quite slow for plots with large amounts of
data. For ``PolyCollections``, it only considers the ``Path`` of ``PolyCollections`` and
not the enclosed area when checking for overlap to reduce additional latency. However,
it can still be quite slow when there are large amounts of ``PolyCollections`` in the
plot to check for.

Exception when not passing a Bbox to BboxTransform*-classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exception when not passing a Bbox to BboxTransform*-classes that expect one, e.g.,
`~matplotlib.transforms.BboxTransform` has changed from ``ValueError`` to ``TypeError``.

*loc* parameter of ``Cell`` no longer accepts ``None``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default value of the *loc* parameter has been changed from ``None`` to ``right``,
which already was the default location. The behavior of `.Cell` didn't change when
called without an explicit *loc* parameter.

``ContourLabeler.add_label`` now respects *use_clabeltext*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

... and sets `.Text.set_transform_rotates_text` accordingly.

``Line2D``
^^^^^^^^^^

When creating a Line2D or using `.Line2D.set_xdata` and `.Line2D.set_ydata`,
passing x/y data as non sequence is now an error.

``ScalarMappable``\s auto-scale their norm when an array is set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Collections previously deferred auto-scaling of the norm until draw time. This has been
changed to scale the norm whenever the first array is set to align with the docstring
and reduce unexpected behavior when accessing the norm before drawing.

``SubplotParams`` moved from ``matplotlib.figure`` to ``matplotlib.gridspec``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is still importable from ``matplotlib.figure``, so does not require any changes to
existing code.

``PowerNorm`` no longer clips values below vmin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``clip=False`` is set (the default) on `~matplotlib.colors.PowerNorm`, values below
``vmin`` are now linearly normalised. Previously they were clipped to zero. This fixes
issues with the display of colorbars associated with a power norm.

Image path semantics of toolmanager-based tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously, MEP22 ("toolmanager-based") Tools would try to load their icon
(``tool.image``) relative to the current working directory, or, as a fallback, from
Matplotlib's own image directory. Because both approaches are problematic for
third-party tools (the end-user may change the current working directory at any time,
and third-parties cannot add new icons in Matplotlib's image directory), this behavior
is deprecated; instead, ``tool.image`` is now interpreted relative to the directory
containing the source file where the ``Tool.image`` class attribute is defined.
(Defining ``tool.image`` as an absolute path also works and is compatible with both the
old and the new semantics.)
