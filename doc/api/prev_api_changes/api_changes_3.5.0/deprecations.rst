Deprecations
------------

Discouraged: ``Figure`` parameters *tight_layout* and *constrained_layout*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Figure`` parameters *tight_layout* and *constrained_layout* are
triggering competing layout mechanisms and thus should not be used together.

To make the API clearer, we've merged them under the new parameter *layout*
with values 'constrained' (equal to ``constrained_layout=True``), 'tight'
(equal to ``tight_layout=True``). If given, *layout* takes precedence.

The use of *tight_layout* and *constrained_layout* is discouraged in favor of
*layout*. However, these parameters will stay available for backward
compatibility.

Modification of ``Axes`` children sublists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`Behavioural API Changes 3.5 - Axes children combined` for more
information; modification of the following sublists is deprecated:

* ``Axes.artists``
* ``Axes.collections``
* ``Axes.images``
* ``Axes.lines``
* ``Axes.patches``
* ``Axes.tables``
* ``Axes.texts``

To remove an Artist, use its `.Artist.remove` method. To add an Artist, use the
corresponding ``Axes.add_*`` method.

Passing incorrect types to ``Axes.add_*`` methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following ``Axes.add_*`` methods will now warn if passed an unexpected
type. See their documentation for the types they expect.

- `.Axes.add_collection`
- `.Axes.add_image`
- `.Axes.add_line`
- `.Axes.add_patch`
- `.Axes.add_table`

Discouraged: ``plot_date``
~~~~~~~~~~~~~~~~~~~~~~~~~~

The use of `~.Axes.plot_date` is discouraged. This method exists for historic
reasons and may be deprecated in the future.

- ``datetime``-like data should directly be plotted using `~.Axes.plot`.
- If you need to plot plain numeric data as :ref:`date-format` or
  need to set a timezone, call ``ax.xaxis.axis_date`` / ``ax.yaxis.axis_date``
  before `~.Axes.plot`. See `.Axis.axis_date`.

``epoch2num`` and ``num2epoch`` are deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods convert from unix timestamps to matplotlib floats, but are not
used internally to matplotlib, and should not be needed by end users. To
convert a unix timestamp to datetime, simply use
`datetime.datetime.utcfromtimestamp`, or to use NumPy `~numpy.datetime64`
``dt = np.datetime64(e*1e6, 'us')``.

Auto-removal of grids by `~.Axes.pcolor` and `~.Axes.pcolormesh`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`~.Axes.pcolor` and `~.Axes.pcolormesh` currently remove any visible axes major
grid. This behavior is deprecated; please explicitly call ``ax.grid(False)`` to
remove the grid.

The first parameter of ``Axes.grid`` and ``Axis.grid`` has been renamed to *visible*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter was previously named *b*. This deprecation only matters if that
parameter was passed using a keyword argument, e.g. ``grid(b=False)``.

Unification and cleanup of Selector widget API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The API for Selector widgets has been unified to use:

- *props* for the properties of the Artist representing the selection.
- *handle_props* for the Artists representing handles for modifying the
  selection.
- *grab_range* for the maximal tolerance to grab a handle with the mouse.

Additionally, several internal parameters and attribute have been deprecated
with the intention of keeping them private.

RectangleSelector and EllipseSelector
.....................................

The *drawtype* keyword argument to `~matplotlib.widgets.RectangleSelector` is
deprecated. In the future the only behaviour will be the default behaviour of
``drawtype='box'``.

Support for ``drawtype=line`` will be removed altogether as it is not clear
which points are within and outside a selector that is just a line. As a
result, the *lineprops* keyword argument to
`~matplotlib.widgets.RectangleSelector` is also deprecated.

To retain the behaviour of ``drawtype='none'``, use ``rectprops={'visible':
False}`` to make the drawn `~matplotlib.patches.Rectangle` invisible.

Cleaned up attributes and arguments are:

- The ``active_handle`` attribute has been privatized and deprecated.
- The ``drawtype`` attribute has been privatized and deprecated.
- The ``eventpress`` attribute has been privatized and deprecated.
- The ``eventrelease`` attribute has been privatized and deprecated.
- The ``interactive`` attribute has been privatized and deprecated.
- The *marker_props* argument is deprecated, use *handle_props* instead.
- The *maxdist* argument is deprecated, use *grab_range* instead.
- The *rectprops* argument is deprecated, use *props* instead.
- The ``rectprops`` attribute has been privatized and deprecated.
- The ``state`` attribute has been privatized and deprecated.
- The ``to_draw`` attribute has been privatized and deprecated.

PolygonSelector
...............

- The *line* attribute is deprecated. If you want to change the selector artist
  properties, use the ``set_props`` or ``set_handle_props`` methods.
- The *lineprops* argument is deprecated, use *props* instead.
- The *markerprops* argument is deprecated, use *handle_props* instead.
- The *maxdist* argument and attribute is deprecated, use *grab_range* instead.
- The *vertex_select_radius* argument and attribute is deprecated, use
  *grab_range* instead.

SpanSelector
............

- The ``active_handle`` attribute has been privatized and deprecated.
- The ``eventpress`` attribute has been privatized and deprecated.
- The ``eventrelease`` attribute has been privatized and deprecated.
- The *maxdist* argument and attribute is deprecated, use *grab_range* instead.
- The ``pressv`` attribute has been privatized and deprecated.
- The ``prev`` attribute has been privatized and deprecated.
- The ``rect`` attribute has been privatized and deprecated.
- The *rectprops* argument is deprecated, use *props* instead.
- The ``rectprops`` attribute has been privatized and deprecated.
- The *span_stays* argument is deprecated, use the *interactive* argument
  instead.
- The ``span_stays`` attribute has been privatized and deprecated.
- The ``state`` attribute has been privatized and deprecated.

LassoSelector
.............

- The *lineprops* argument is deprecated, use *props* instead.
- The ``onpress`` and ``onrelease`` methods are deprecated. They are straight
  aliases for ``press`` and ``release``.

``ConversionInterface.convert`` no longer needs to accept unitless values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, custom subclasses of `.units.ConversionInterface` needed to
implement a ``convert`` method that not only accepted instances of the unit,
but also unitless values (which are passed through as is). This is no longer
the case (``convert`` is never called with a unitless value), and such support
in `.StrCategoryConverter` is deprecated. Likewise, the
``.ConversionInterface.is_numlike`` helper is deprecated.

Consider calling `.Axis.convert_units` instead, which still supports unitless
values.

Locator and Formatter wrapper methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``set_view_interval``, ``set_data_interval`` and ``set_bounds`` methods of
`.Locator`\s and `.Formatter`\s (and their common base class, TickHelper) are
deprecated. Directly manipulate the view and data intervals on the underlying
axis instead.

Unused positional parameters to ``print_<fmt>`` methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

None of the ``print_<fmt>`` methods implemented by canvas subclasses used
positional arguments other that the first (the output filename or file-like),
so these extra parameters are deprecated.

``QuadMesh`` signature
~~~~~~~~~~~~~~~~~~~~~~

The `.QuadMesh` signature ::

    def __init__(meshWidth, meshHeight, coordinates,
                 antialiased=True, shading='flat', **kwargs)

is deprecated and replaced by the new signature ::

    def __init__(coordinates, *, antialiased=True, shading='flat', **kwargs)

In particular:

- The *coordinates* argument must now be a (M, N, 2) array-like. Previously,
  the grid shape was separately specified as (*meshHeight* + 1, *meshWidth* +
  1) and *coordinates* could be an array-like of any shape with M * N * 2
  elements.
- All parameters except *coordinates* are keyword-only now.

rcParams will no longer cast inputs to str
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a deprecation period, rcParams that expect a (non-pathlike) str will no
longer cast non-str inputs using `str`. This will avoid confusing errors in
subsequent code if e.g. a list input gets implicitly cast to a str.

Case-insensitive scales
~~~~~~~~~~~~~~~~~~~~~~~

Previously, scales could be set case-insensitively (e.g.,
``set_xscale("LoG")``). This is deprecated; all builtin scales use lowercase
names.

Interactive cursor details
~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting a mouse cursor on a window has been moved from the toolbar to the
canvas. Consequently, several implementation details on toolbars and within
backends have been deprecated.

``NavigationToolbar2.set_cursor`` and ``backend_tools.SetCursorBase.set_cursor``
................................................................................

Instead, use the `.FigureCanvasBase.set_cursor` method on the canvas (available
as the ``canvas`` attribute on the toolbar or the Figure.)

``backend_tools.SetCursorBase`` and subclasses
..............................................

``backend_tools.SetCursorBase`` was subclassed to provide backend-specific
implementations of ``set_cursor``. As that is now deprecated, the subclassing
is no longer necessary. Consequently, the following subclasses are also
deprecated:

- ``matplotlib.backends.backend_gtk3.SetCursorGTK3``
- ``matplotlib.backends.backend_qt5.SetCursorQt``
- ``matplotlib.backends._backend_tk.SetCursorTk``
- ``matplotlib.backends.backend_wx.SetCursorWx``

Instead, use the `.backend_tools.ToolSetCursor` class.

``cursord`` in GTK, Qt, and wx backends
.......................................

The ``backend_gtk3.cursord``, ``backend_qt.cursord``, and
``backend_wx.cursord`` dictionaries are deprecated. This makes the GTK module
importable on headless environments.

Miscellaneous deprecations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``is_url`` and ``URL_REGEX`` are deprecated. (They were previously defined in
  the toplevel :mod:`matplotlib` module.)
- The ``ArrowStyle.beginarrow`` and ``ArrowStyle.endarrow`` attributes are
  deprecated; use the ``arrow`` attribute to define the desired heads and tails
  of the arrow.
- ``backend_pgf.LatexManager.str_cache`` is deprecated.
- ``backends.qt_compat.ETS`` and ``backends.qt_compat.QT_RC_MAJOR_VERSION`` are
  deprecated, with no replacement.
- The ``blocking_input`` module has been deprecated. Instead, use
  ``canvas.start_event_loop()`` and ``canvas.stop_event_loop()`` while
  connecting event callbacks as needed.
- ``cbook.report_memory`` is deprecated; use ``psutil.virtual_memory`` instead.
- ``cm.LUTSIZE`` is deprecated. Use :rc:`image.lut` instead. This value only
  affects colormap quantization levels for default colormaps generated at
  module import time.
- ``Collection.__init__`` previously ignored *transOffset* without *offsets* also
  being specified. In the future, *transOffset* will begin having an effect
  regardless of *offsets*. In the meantime, if you wish to set *transOffset*,
  call `.Collection.set_offset_transform` explicitly.
- ``Colorbar.patch`` is deprecated; this attribute is not correctly updated
  anymore.
- ``ContourLabeler.get_label_width`` is deprecated.
- ``dviread.PsfontsMap`` now raises LookupError instead of KeyError for missing
  fonts.
- ``Dvi.baseline`` is deprecated (with no replacement).
- The *format* parameter of ``dviread.find_tex_file`` is deprecated (with no
  replacement).
- ``FancyArrowPatch.get_path_in_displaycoord`` and
  ``ConnectionPath.get_path_in_displaycoord`` are deprecated. The path in
  display coordinates can still be obtained, as for other patches, using
  ``patch.get_transform().transform_path(patch.get_path())``.
- The ``font_manager.win32InstalledFonts`` and
  ``font_manager.get_fontconfig_fonts`` helper functions have been deprecated.
- All parameters of ``imshow`` starting from *aspect* will become keyword-only.
- ``QuadMesh.convert_mesh_to_paths`` and ``QuadMesh.convert_mesh_to_triangles``
  are deprecated. ``QuadMesh.get_paths()`` can be used as an alternative for
  the former; there is no replacement for the latter.
- ``ScalarMappable.callbacksSM`` is deprecated. Use
  ``ScalarMappable.callbacks`` instead.
- ``streamplot.get_integrator`` is deprecated.
- ``style.core.STYLE_FILE_PATTERN``, ``style.core.load_base_library``, and
  ``style.core.iter_user_libraries`` are deprecated.
- ``SubplotParams.validate`` is deprecated. Use `.SubplotParams.update` to
  change `.SubplotParams` while always keeping it in a valid state.
- The ``grey_arrayd``, ``font_family``, ``font_families``, and ``font_info``
  attributes of `.TexManager` are deprecated.
- ``Text.get_prop_tup`` is deprecated with no replacements (because the `.Text`
  class cannot know whether a backend needs to update cache e.g. when the
  text's color changes).
- ``Tick.apply_tickdir`` didn't actually update the tick markers on the
  existing Line2D objects used to draw the ticks and is deprecated; use
  `.Axis.set_tick_params` instead.
- ``tight_layout.auto_adjust_subplotpars`` is deprecated.

- The ``grid_info`` attribute of ``axisartist`` classes has been deprecated.
- ``axisartist.clip_path`` is deprecated with no replacement.
- ``axes_grid1.axes_grid.CbarAxes`` and ``axes_grid1.axisartist.CbarAxes`` are
  deprecated (they are now dynamically generated based on the owning axes
  class).
- The ``axes_grid1.Divider.get_vsize_hsize`` and
  ``axes_grid1.Grid.get_vsize_hsize`` methods are deprecated. Copy their
  implementations if needed.
- ``AxesDivider.append_axes(..., add_to_figure=False)`` is deprecated. Use
  ``ax.remove()`` to remove the Axes from the figure if needed.
- ``FixedAxisArtistHelper.change_tick_coord`` is deprecated with no
  replacement.
- ``floating_axes.GridHelperCurveLinear.get_boundary`` is deprecated, with no
  replacement.
- ``ParasiteAxesBase.get_images_artists`` has been deprecated.

- The "units finalize" signal (previously emitted by Axis instances) is
  deprecated. Connect to "units" instead.
- Passing formatting parameters positionally to ``stem()`` is deprecated

``plot_directive`` deprecations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``:encoding:`` option to ``.. plot`` directive has had no effect since
Matplotlib 1.3.1, and is now deprecated.

The following helpers in `matplotlib.sphinxext.plot_directive` are deprecated:

- ``unescape_doctest`` (use `doctest.script_from_examples` instead),
- ``split_code_at_show``, 
- ``run_code``.

Testing support
~~~~~~~~~~~~~~~

``matplotlib.test()`` is deprecated
...................................

Run tests using ``pytest`` from the commandline instead. The variable
``matplotlib.default_test_modules`` is only used for ``matplotlib.test()`` and
is thus deprecated as well.

To test an installed copy, be sure to specify both ``matplotlib`` and
``mpl_toolkits`` with ``--pyargs``::

    pytest --pyargs matplotlib.tests mpl_toolkits.tests

See :ref:`testing` for more details.

Unused pytest fixtures and markers
..................................

The fixture ``matplotlib.testing.conftest.mpl_image_comparison_parameters`` is
not used internally by Matplotlib. If you use this please copy it into your
code base.

The ``@pytest.mark.style`` marker is deprecated; use ``@mpl.style.context``,
which has the same effect.

Support for ``nx1 = None`` or ``ny1 = None`` in ``AxesLocator`` and ``Divider.locate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In `.axes_grid1.axes_divider`, various internal APIs will stop supporting
passing ``nx1 = None`` or ``ny1 = None`` to mean ``nx + 1`` or ``ny + 1``, in
preparation for a possible future API which allows indexing and slicing of
dividers (possibly ``divider[a:b] == divider.new_locator(a, b)``, but also
``divider[a:] == divider.new_locator(a, <end>)``). The user-facing
`.Divider.new_locator` API is unaffected -- it correctly normalizes ``nx1 =
None`` and ``ny1 = None`` as needed.
