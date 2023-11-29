Deprecations
------------

``Axes`` subclasses should override ``clear`` instead of ``cla``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For clarity, `.axes.Axes.clear` is now preferred over `.Axes.cla`. However, for
backwards compatibility, the latter will remain as an alias for the former.

For additional compatibility with third-party libraries, Matplotlib will
continue to call the ``cla`` method of any `~.axes.Axes` subclasses if they
define it. In the future, this will no longer occur, and Matplotlib will only
call the ``clear`` method in `~.axes.Axes` subclasses.

It is recommended to define only the ``clear`` method when on Matplotlib 3.6,
and only ``cla`` for older versions.

rcParams type
~~~~~~~~~~~~~

Relying on ``rcParams`` being a ``dict`` subclass is deprecated.

Nothing will change for regular users because ``rcParams`` will continue to
be dict-like (technically fulfill the ``MutableMapping`` interface).

The `.RcParams` class does validation checking on calls to
``.RcParams.__getitem__`` and ``.RcParams.__setitem__``.  However, there are rare
cases where we want to circumvent the validation logic and directly access the
underlying data values.   Previously, this could be accomplished via  a call to
the parent methods  ``dict.__getitem__(rcParams, key)`` and
``dict.__setitem__(rcParams, key, val)``.

Matplotlib 3.7 introduces ``rcParams._set(key, val)`` and
``rcParams._get(key)`` as a replacement to calling the parent methods. They are
intentionally marked private to discourage external use; However, if direct
`.RcParams` data access is needed, please switch from the dict functions to the
new ``_get()`` and ``_set()``. Even though marked private, we guarantee API
stability for these methods and they are subject to Matplotlib's API and
deprecation policy.

Please notify the Matplotlib developers if you rely on ``rcParams`` being a
dict subclass in any other way, for which there is no migration path yet.

Deprecation aliases in cbook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module ``matplotlib.cbook.deprecation`` was previously deprecated in
Matplotlib 3.4, along with deprecation-related API in ``matplotlib.cbook``. Due
to technical issues, ``matplotlib.cbook.MatplotlibDeprecationWarning`` and
``matplotlib.cbook.mplDeprecation`` did not raise deprecation warnings on use.
Changes in Python have now made it possible to warn when these aliases are
being used.

In order to avoid downstream breakage, these aliases will now warn, and their
removal has been pushed from 3.6 to 3.8 to give time to notice said warnings.
As replacement, please use `matplotlib.MatplotlibDeprecationWarning`.

``draw_gouraud_triangle``
~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated as in most backends this is a redundant call. Use
`~.RendererBase.draw_gouraud_triangles` instead. A ``draw_gouraud_triangle``
call in a custom `~matplotlib.artist.Artist` can readily be replaced as::

    self.draw_gouraud_triangles(gc, points.reshape((1, 3, 2)),
                                colors.reshape((1, 3, 4)), trans)

A `~.RendererBase.draw_gouraud_triangles` method can be implemented from an
existing ``draw_gouraud_triangle`` method as::

    transform = transform.frozen()
    for tri, col in zip(triangles_array, colors_array):
        self.draw_gouraud_triangle(gc, tri, col, transform)

``matplotlib.pyplot.get_plot_commands``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is a pending deprecation. This is considered internal and no end-user
should need it.

``matplotlib.tri`` submodules are deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``matplotlib.tri.*`` submodules are deprecated.  All functionality is
available in ``matplotlib.tri`` directly and should be imported from there.

Passing undefined *label_mode* to ``Grid``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated. This includes `mpl_toolkits.axes_grid1.axes_grid.Grid`,
`mpl_toolkits.axes_grid1.axes_grid.AxesGrid`, and
`mpl_toolkits.axes_grid1.axes_grid.ImageGrid` as well as the corresponding
classes imported from `mpl_toolkits.axisartist.axes_grid`.

Pass ``label_mode='keep'`` instead to get the previous behavior of not modifying labels.

Colorbars for orphaned mappables are deprecated, but no longer raise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before 3.6.0, Colorbars for mappables that do not have a parent axes would
steal space from the current Axes.  3.6.0 raised an error on this, but without
a deprecation cycle.  For 3.6.1 this is reverted, the current axes is used,
but a deprecation warning is shown instead.  In this undetermined case users
and libraries should explicitly specify what axes they want space to be stolen
from: ``fig.colorbar(mappable, ax=plt.gca())``.

``Animation`` attributes
~~~~~~~~~~~~~~~~~~~~~~~~

The attributes ``repeat`` of `.TimedAnimation` and subclasses and
``save_count`` of `.FuncAnimation` are considered private and deprecated.

``contour.ClabelText`` and ``ContourLabeler.set_label_props``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... are deprecated.

Use ``Text(..., transform_rotates_text=True)`` as a replacement for
``contour.ClabelText(...)`` and ``text.set(text=text, color=color,
fontproperties=labeler.labelFontProps, clip_box=labeler.axes.bbox)`` as a
replacement for the ``ContourLabeler.set_label_props(label, text, color)``.

``ContourLabeler`` attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``labelFontProps``, ``labelFontSizeList``, and ``labelTextsList``
attributes of `.ContourLabeler` have been deprecated.  Use the ``labelTexts``
attribute and the font properties of the corresponding text objects instead.

``backend_ps.PsBackendHelper`` and ``backend_ps.ps_backend_helper``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are deprecated with no replacement.

``backend_webagg.ServerThread`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... with no replacement.

``parse_fontconfig_pattern`` will no longer ignore unknown constant names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, in a fontconfig pattern like ``DejaVu Sans:foo``, the unknown
``foo`` constant name would be silently ignored.  This now raises a warning,
and will become an error in the future.

``BufferRegion.to_string`` and ``BufferRegion.to_string_argb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are deprecated.  Use ``np.asarray(buffer_region)`` to get an array view on
a buffer region without making a copy; to convert that view from RGBA (the
default) to ARGB, use ``np.take(..., [2, 1, 0, 3], axis=2)``.

``num2julian``, ``julian2num`` and ``JULIAN_OFFSET``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... of the `.dates` module are deprecated without replacements. These are
undocumented and not exported. If you rely on these, please make a local copy.

``unit_cube``, ``tunit_cube``, and ``tunit_edges``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... of `.Axes3D` are deprecated without replacements. If you rely on them,
please copy the code of the corresponding private function (name starting
with ``_``).

Most arguments to widgets have been made keyword-only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing all but the very few first arguments positionally in the constructors
of Widgets is deprecated. Most arguments will become keyword-only in a future
version.

``SimpleEvent``
~~~~~~~~~~~~~~~

The ``SimpleEvent`` nested class (previously accessible via the public
subclasses of ``ConnectionStyle._Base``, such as `.ConnectionStyle.Arc`, has
been deprecated.

``RadioButtons.circles``
~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated.  (RadioButtons now draws itself using `~.Axes.scatter`.)

``CheckButtons.rectangles`` and ``CheckButtons.lines``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CheckButtons.rectangles`` and ``CheckButtons.lines`` are deprecated.
(``CheckButtons`` now draws itself using `~.Axes.scatter`.)

``OffsetBox.get_extent_offsets`` and ``OffsetBox.get_extent``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are deprecated; these methods are also deprecated on all subclasses of
`.OffsetBox`.

To get the offsetbox extents, instead of ``get_extent``, use
`.OffsetBox.get_bbox`, which directly returns a `.Bbox` instance.

To also get the child offsets, instead of ``get_extent_offsets``, separately
call `~.OffsetBox.get_offset` on each children after triggering a draw.

``legend.legendHandles``
~~~~~~~~~~~~~~~~~~~~~~~~

... was undocumented and has been renamed to ``legend_handles``. Using ``legendHandles`` is deprecated.

``ticklabels`` parameter of `.Axis.set_ticklabels` renamed to ``labels``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``offsetbox.bbox_artist``
~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated. This is just a wrapper to call `.patches.bbox_artist` if a
flag is set in the file, so use that directly if you need the behavior.

``Quiver.quiver_doc`` and ``Barbs.barbs_doc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are deprecated. These are the doc-string and should not be accessible as
a named class member.

Deprecate unused parameter *x* to ``TextBox.begin_typing``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This parameter was unused in the method, but was a required argument.

Deprecation of top-level cmap registration and access functions in ``mpl.cm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As part of a `multi-step process
<https://github.com/matplotlib/matplotlib/issues/20853>`_ we are refactoring
the global state for managing the registered colormaps.

In Matplotlib 3.5 we added a `.ColormapRegistry` class and exposed an instance
at the top level as ``matplotlib.colormaps``. The existing top level functions
in `matplotlib.cm` (``get_cmap``, ``register_cmap``, ``unregister_cmap``) were
changed to be aliases around the same instance. In Matplotlib 3.6 we have
marked those top level functions as pending deprecation.

In Matplotlib 3.7, the following functions have been marked for deprecation:

- ``matplotlib.cm.get_cmap``; use ``matplotlib.colormaps[name]`` instead if you
  have a `str`.

  **Added 3.6.1** Use `matplotlib.cm.ColormapRegistry.get_cmap` if you
  have a string, `None` or a `matplotlib.colors.Colormap` object that you want
  to convert to a `matplotlib.colors.Colormap` instance.
- ``matplotlib.cm.register_cmap``; use `matplotlib.colormaps.register
  <.ColormapRegistry.register>` instead
- ``matplotlib.cm.unregister_cmap``; use `matplotlib.colormaps.unregister
  <.ColormapRegistry.unregister>` instead
- ``matplotlib.pyplot.register_cmap``; use `matplotlib.colormaps.register
  <.ColormapRegistry.register>` instead

The `matplotlib.pyplot.get_cmap` function will stay available for backward
compatibility.

``BrokenBarHCollection`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It was just a thin wrapper inheriting from `.PolyCollection`;
`~.Axes.broken_barh` has now been changed to return a `.PolyCollection`
instead.

The ``BrokenBarHCollection.span_where`` helper is likewise deprecated; for the
duration of the deprecation it has been moved to the parent `.PolyCollection`
class.  Use `~.Axes.fill_between` as a replacement; see
:doc:`/gallery/lines_bars_and_markers/span_regions` for an example.

Passing inconsistent ``loc`` and ``nth_coord`` to axisartist helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trying to construct for example a "top y-axis" or a "left x-axis" is now
deprecated.

``passthru_pt``
~~~~~~~~~~~~~~~

This attribute of ``AxisArtistHelper``\s is deprecated.

``axes3d.vvec``, ``axes3d.eye``, ``axes3d.sx``, and ``axes3d.sy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are deprecated without replacement.

``Line2D``
~~~~~~~~~~

When creating a Line2D or using `.Line2D.set_xdata` and `.Line2D.set_ydata`,
passing x/y data as non sequence is deprecated.
