Removals
--------

Top-level cmap registration and access functions in ``mpl.cm``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As part of the `multi-step refactoring of colormap registration
<https://github.com/matplotlib/matplotlib/issues/20853>`_, the following functions have
been removed:

- ``matplotlib.cm.get_cmap``; use ``matplotlib.colormaps[name]`` instead if you have a
  `str`.

  Use `matplotlib.cm.ColormapRegistry.get_cmap` if you have a `str`, `None` or a
  `matplotlib.colors.Colormap` object that you want to convert to a `.Colormap` object.
- ``matplotlib.cm.register_cmap``; use `matplotlib.colormaps.register
  <.ColormapRegistry.register>` instead.
- ``matplotlib.cm.unregister_cmap``; use `matplotlib.colormaps.unregister
  <.ColormapRegistry.unregister>` instead.
- ``matplotlib.pyplot.register_cmap``; use `matplotlib.colormaps.register
  <.ColormapRegistry.register>` instead.

The `matplotlib.pyplot.get_cmap` function will stay available for backward
compatibility.

Contour labels
^^^^^^^^^^^^^^

``contour.ClabelText`` and ``ContourLabeler.set_label_props`` are removed. Use
``Text(..., transform_rotates_text=True)`` as a replacement for
``contour.ClabelText(...)`` and ``text.set(text=text, color=color,
fontproperties=labeler.labelFontProps, clip_box=labeler.axes.bbox)`` as a replacement
for the ``ContourLabeler.set_label_props(label, text, color)``.

The ``labelFontProps``, ``labelFontSizeList``, and ``labelTextsList`` attributes of
`.ContourLabeler` have been removed.  Use the ``labelTexts`` attribute and the font
properties of the corresponding text objects instead.

``num2julian``, ``julian2num`` and ``JULIAN_OFFSET``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

... of the `.dates` module are removed without replacements. These were undocumented and
not exported.

Julian dates in Matplotlib were calculated from a Julian date epoch: ``jdate = (date -
np.datetime64(EPOCH)) / np.timedelta64(1, 'D')``.  Conversely, a Julian date was
converted to datetime as ``date = np.timedelta64(int(jdate * 24 * 3600), 's') +
np.datetime64(EPOCH)``. Matplotlib was using ``EPOCH='-4713-11-24T12:00'`` so that
2000-01-01 at 12:00 is 2_451_545.0 (see https://en.wikipedia.org/wiki/Julian_day).

``offsetbox`` methods
^^^^^^^^^^^^^^^^^^^^^

``offsetbox.bbox_artist`` is removed. This was just a wrapper to call
`.patches.bbox_artist` if a flag is set in the file, so use that directly if you need
the behavior.

``OffsetBox.get_extent_offsets`` and ``OffsetBox.get_extent`` are removed; these methods
are also removed on all subclasses of `.OffsetBox`. To get the offsetbox extents,
instead of ``get_extent``, use `.OffsetBox.get_bbox`, which directly returns a `.Bbox`
instance. To also get the child offsets, instead of ``get_extent_offsets``, separately
call `~.OffsetBox.get_offset` on each children after triggering a draw.

``parse_fontconfig_pattern`` raises on unknown constant names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously, in a fontconfig pattern like ``DejaVu Sans:foo``, the unknown ``foo``
constant name would be silently ignored.  This now raises an error.

``tri`` submodules
^^^^^^^^^^^^^^^^^^

The ``matplotlib.tri.*`` submodules are removed.  All functionality is available in
``matplotlib.tri`` directly and should be imported from there.

Widget API
^^^^^^^^^^

- ``CheckButtons.rectangles`` and ``CheckButtons.lines`` are removed; `.CheckButtons`
  now draws itself using `~.Axes.scatter`.
- ``RadioButtons.circles`` is removed; `.RadioButtons` now draws itself using
  `~.Axes.scatter`.
- ``MultiCursor.needclear`` is removed with no replacement.
- The unused parameter *x* to ``TextBox.begin_typing`` was a required argument, and is
  now removed.

Most arguments to widgets have been made keyword-only
"""""""""""""""""""""""""""""""""""""""""""""""""""""

Passing all but the very few first arguments positionally in the constructors of Widgets
is now keyword-only. In general, all optional arguments are keyword-only.

``Axes3D`` API
^^^^^^^^^^^^^^

- ``Axes3D.unit_cube``, ``Axes3D.tunit_cube``, and ``Axes3D.tunit_edges`` are removed
  without replacement.
- ``axes3d.vvec``, ``axes3d.eye``, ``axes3d.sx``, and ``axes3d.sy`` are removed without
  replacement.

Inconsistent *nth_coord* and *loc* passed to ``_FixedAxisArtistHelperBase``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value of the *nth_coord* parameter of ``_FixedAxisArtistHelperBase`` and its
subclasses is now inferred from the value of *loc*; passing inconsistent values (e.g.,
requesting a "top y axis" or a "left x axis") has no more effect.

Passing undefined *label_mode* to ``Grid``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

... is no longer allowed. This includes `mpl_toolkits.axes_grid1.axes_grid.Grid`,
`mpl_toolkits.axes_grid1.axes_grid.AxesGrid`, and
`mpl_toolkits.axes_grid1.axes_grid.ImageGrid` as well as the corresponding classes
imported from `mpl_toolkits.axisartist.axes_grid`.

Pass ``label_mode='keep'`` instead to get the previous behavior of not modifying labels.

``draw_gouraud_triangle``
^^^^^^^^^^^^^^^^^^^^^^^^^

... is removed. Use `~.RendererBase.draw_gouraud_triangles` instead.

A ``draw_gouraud_triangle`` call in a custom `~matplotlib.artist.Artist` can readily be
replaced as::

    self.draw_gouraud_triangles(gc, points.reshape((1, 3, 2)),
                                colors.reshape((1, 3, 4)), trans)

A `~.RendererBase.draw_gouraud_triangles` method can be implemented from an
existing ``draw_gouraud_triangle`` method as::

    transform = transform.frozen()
    for tri, col in zip(triangles_array, colors_array):
        self.draw_gouraud_triangle(gc, tri, col, transform)

Miscellaneous removals
^^^^^^^^^^^^^^^^^^^^^^

The following items have previously been replaced, and are now removed:

- *ticklabels* parameter of ``matplotlib.axis.Axis.set_ticklabels`` has been renamed to
  *labels*.
- ``Barbs.barbs_doc`` and ``Quiver.quiver_doc`` are removed. These are the doc-strings
  and should not be accessible as a named class member, but as normal doc-strings would.
- ``collections.PolyCollection.span_where`` and ``collections.BrokenBarHCollection``;
  use ``fill_between`` instead.
- ``Legend.legendHandles`` was undocumented and has been renamed to ``legend_handles``.

The following items have been removed without replacements:

- The attributes ``repeat`` of `.TimedAnimation` and subclasses and ``save_count`` of
  `.FuncAnimation` are considered private and removed.
- ``matplotlib.backend.backend_agg.BufferRegion.to_string``
- ``matplotlib.backend.backend_agg.BufferRegion.to_string_argb``
- ``matplotlib.backends.backend_ps.PsBackendHelper``
- ``matplotlib.backends.backend_webagg.ServerThread``
- *raw* parameter of `.GridSpecBase.get_grid_positions`
- ``matplotlib.patches.ConnectionStyle._Base.SimpleEvent``
- ``passthru_pt`` attribute of ``mpl_toolkits.axisartist.AxisArtistHelper``
