Removals
--------

``epoch2num`` and ``num2epoch`` are removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods convert from unix timestamps to matplotlib floats, but are not
used internally to Matplotlib, and should not be needed by end users. To
convert a unix timestamp to datetime, simply use
`datetime.datetime.fromtimestamp`, or to use NumPy `~numpy.datetime64`
``dt = np.datetime64(e*1e6, 'us')``.

Locator and Formatter wrapper methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``set_view_interval``, ``set_data_interval`` and ``set_bounds`` methods of
`.Locator`\s and `.Formatter`\s (and their common base class, TickHelper) are
removed. Directly manipulate the view and data intervals on the underlying
axis instead.

Interactive cursor details
~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting a mouse cursor on a window has been moved from the toolbar to the
canvas. Consequently, several implementation details on toolbars and within
backends have been removed.

``NavigationToolbar2.set_cursor`` and ``backend_tools.SetCursorBase.set_cursor``
................................................................................

Instead, use the `.FigureCanvasBase.set_cursor` method on the canvas (available
as the ``canvas`` attribute on the toolbar or the Figure.)

``backend_tools.SetCursorBase`` and subclasses
..............................................

``backend_tools.SetCursorBase`` was subclassed to provide backend-specific
implementations of ``set_cursor``. As that is now removed, the subclassing
is no longer necessary. Consequently, the following subclasses are also
removed:

- ``matplotlib.backends.backend_gtk3.SetCursorGTK3``
- ``matplotlib.backends.backend_qt5.SetCursorQt``
- ``matplotlib.backends._backend_tk.SetCursorTk``
- ``matplotlib.backends.backend_wx.SetCursorWx``

Instead, use the `.backend_tools.ToolSetCursor` class.

``cursord`` in GTK and wx backends
..................................

The ``backend_gtk3.cursord`` and ``backend_wx.cursord`` dictionaries are
removed. This makes the GTK module importable on headless environments.

``auto_add_to_figure=True`` for ``Axes3D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is no longer supported. Instead use ``fig.add_axes(ax)``.

The first parameter of ``Axes.grid`` and ``Axis.grid`` has been renamed to *visible*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter was previously named *b*. This name change only matters if that
parameter was passed using a keyword argument, e.g. ``grid(b=False)``.

Removal of deprecations in the Selector widget API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RectangleSelector and EllipseSelector
.....................................

The *drawtype* keyword argument to `~matplotlib.widgets.RectangleSelector` is
removed. From now on, the only behaviour will be ``drawtype='box'``.

Support for ``drawtype=line`` is removed altogether. As a
result, the *lineprops* keyword argument to
`~matplotlib.widgets.RectangleSelector` is also removed.

To retain the behaviour of ``drawtype='none'``, use ``rectprops={'visible':
False}`` to make the drawn `~matplotlib.patches.Rectangle` invisible.

Cleaned up attributes and arguments are:

- The ``active_handle`` attribute has been privatized and removed.
- The ``drawtype`` attribute has been privatized and removed.
- The ``eventpress`` attribute has been privatized and removed.
- The ``eventrelease`` attribute has been privatized and removed.
- The ``interactive`` attribute has been privatized and removed.
- The *marker_props* argument is removed, use *handle_props* instead.
- The *maxdist* argument is removed, use *grab_range* instead.
- The *rectprops* argument is removed, use *props* instead.
- The ``rectprops`` attribute has been privatized and removed.
- The ``state`` attribute has been privatized and removed.
- The ``to_draw`` attribute has been privatized and removed.

PolygonSelector
...............

- The *line* attribute is removed. If you want to change the selector artist
  properties, use the ``set_props`` or ``set_handle_props`` methods.
- The *lineprops* argument is removed, use *props* instead.
- The *markerprops* argument is removed, use *handle_props* instead.
- The *maxdist* argument and attribute is removed, use *grab_range* instead.
- The *vertex_select_radius* argument and attribute is removed, use
  *grab_range* instead.

SpanSelector
............

- The ``active_handle`` attribute has been privatized and removed.
- The ``eventpress`` attribute has been privatized and removed.
- The ``eventrelease`` attribute has been privatized and removed.
- The ``pressv`` attribute has been privatized and removed.
- The ``prev`` attribute has been privatized and removed.
- The ``rect`` attribute has been privatized and removed.
- The *rectprops* parameter has been renamed to *props*.
- The ``rectprops`` attribute has been privatized and removed.
- The *span_stays* parameter has been renamed to *interactive*.
- The ``span_stays`` attribute has been privatized and removed.
- The ``state`` attribute has been privatized and removed.

LassoSelector
.............

- The *lineprops* argument is removed, use *props* instead.
- The ``onpress`` and ``onrelease`` methods are removed. They are straight
  aliases for ``press`` and ``release``.
- The ``matplotlib.widgets.TextBox.DIST_FROM_LEFT`` attribute has been
  removed.  It was marked as private in 3.5.
  
``backend_template.show``
~~~~~~~~~~~~~~~~~~~~~~~~~
... has been removed, in order to better demonstrate the new backend definition
API.

Unused positional parameters to ``print_<fmt>`` methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

None of the ``print_<fmt>`` methods implemented by canvas subclasses used
positional arguments other that the first (the output filename or file-like),
so these extra parameters are removed.

``QuadMesh`` signature
~~~~~~~~~~~~~~~~~~~~~~

The `.QuadMesh` signature ::

    def __init__(meshWidth, meshHeight, coordinates,
                 antialiased=True, shading='flat', **kwargs)

is removed and replaced by the new signature ::

    def __init__(coordinates, *, antialiased=True, shading='flat', **kwargs)

In particular:

- The *coordinates* argument must now be a (M, N, 2) array-like. Previously,
  the grid shape was separately specified as (*meshHeight* + 1, *meshWidth* +
  1) and *coordinates* could be an array-like of any shape with M * N * 2
  elements.
- All parameters except *coordinates* are keyword-only now.
  
Expiration of ``FancyBboxPatch`` deprecations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `.FancyBboxPatch` constructor no longer accepts the *bbox_transmuter*
parameter, nor can the *boxstyle* parameter be set to "custom" -- instead,
directly set *boxstyle* to the relevant boxstyle instance.  The
*mutation_scale* and *mutation_aspect* parameters have also become
keyword-only.

The *mutation_aspect* parameter is now handled internally and no longer passed
to the boxstyle callables when mutating the patch path.

Testing support
~~~~~~~~~~~~~~~

``matplotlib.test()`` has been removed
......................................

Run tests using ``pytest`` from the commandline instead. The variable
``matplotlib.default_test_modules`` was only used for ``matplotlib.test()`` and
is thus removed as well.

To test an installed copy, be sure to specify both ``matplotlib`` and
``mpl_toolkits`` with ``--pyargs``::

    pytest --pyargs matplotlib.tests mpl_toolkits.tests

See :ref:`testing` for more details.

Auto-removal of grids by `~.Axes.pcolor` and `~.Axes.pcolormesh`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`~.Axes.pcolor` and `~.Axes.pcolormesh` previously remove any visible axes
major grid. This behavior is removed; please explicitly call ``ax.grid(False)``
to remove the grid.

Modification of ``Axes`` children sublists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`Behavioural API Changes 3.5 - Axes children combined` for more
information; modification of the following sublists is no longer supported:

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

The following ``Axes.add_*`` methods will now raise if passed an unexpected
type. See their documentation for the types they expect.

- `.Axes.add_collection`
- `.Axes.add_image`
- `.Axes.add_line`
- `.Axes.add_patch`
- `.Axes.add_table`


``ConversionInterface.convert`` no longer accepts unitless values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, custom subclasses of `.units.ConversionInterface` needed to
implement a ``convert`` method that not only accepted instances of the unit,
but also unitless values (which are passed through as is). This is no longer
the case (``convert`` is never called with a unitless value), and such support
in ``.StrCategoryConverter`` is removed. Likewise, the
``.ConversionInterface.is_numlike`` helper is removed.

Consider calling `.Axis.convert_units` instead, which still supports unitless
values.


Normal list of `.Artist` objects now returned by `.HandlerLine2D.create_artists`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Matplotlib 3.5 and 3.6 a proxy list was returned that simulated the return
of `.HandlerLine2DCompound.create_artists`. Now a list containing only the
single artist is return.


rcParams will no longer cast inputs to str
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rcParams that expect a (non-pathlike) str no longer cast non-str inputs using
`str`. This will avoid confusing errors in subsequent code if e.g. a list input
gets implicitly cast to a str.

Case-insensitive scales
~~~~~~~~~~~~~~~~~~~~~~~

Previously, scales could be set case-insensitively (e.g.,
``set_xscale("LoG")``).  Now all builtin scales use lowercase names.

Support for ``nx1 = None`` or ``ny1 = None`` in ``AxesLocator`` and ``Divider.locate``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In `.axes_grid1.axes_divider`, various internal APIs no longer supports
passing ``nx1 = None`` or ``ny1 = None`` to mean ``nx + 1`` or ``ny + 1``, in
preparation for a possible future API which allows indexing and slicing of
dividers (possibly ``divider[a:b] == divider.new_locator(a, b)``, but also
``divider[a:] == divider.new_locator(a, <end>)``). The user-facing
`.Divider.new_locator` API is unaffected -- it correctly normalizes ``nx1 =
None`` and ``ny1 = None`` as needed.


change signature of ``.FigureCanvasBase.enter_notify_event``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *xy* parameter is now required and keyword only.  This was deprecated in
3.0 and originally slated to be removed in 3.5.

``Colorbar`` tick update parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *update_ticks* parameter of `.Colorbar.set_ticks` and
`.Colorbar.set_ticklabels` was ignored since 3.5 and has been removed.

plot directive removals
~~~~~~~~~~~~~~~~~~~~~~~

The public methods:

- ``matplotlib.sphinxext.split_code_at_show``
- ``matplotlib.sphinxext.unescape_doctest``
- ``matplotlib.sphinxext.run_code``

have been removed.

The deprecated *encoding* option to the plot directive has been removed.

Miscellaneous removals
~~~~~~~~~~~~~~~~~~~~~~

- ``is_url`` and ``URL_REGEX`` are removed. (They were previously defined in
  the toplevel :mod:`matplotlib` module.)
- The ``ArrowStyle.beginarrow`` and ``ArrowStyle.endarrow`` attributes are
  removed; use the ``arrow`` attribute to define the desired heads and tails
  of the arrow.
- ``backend_pgf.LatexManager.str_cache`` is removed.
- ``backends.qt_compat.ETS`` and ``backends.qt_compat.QT_RC_MAJOR_VERSION`` are
  removed, with no replacement.
- The ``blocking_input`` module is removed. Instead, use
  ``canvas.start_event_loop()`` and ``canvas.stop_event_loop()`` while
  connecting event callbacks as needed.
- ``cbook.report_memory`` is removed; use ``psutil.virtual_memory`` instead.
- ``cm.LUTSIZE`` is removed. Use :rc:`image.lut` instead. This value only
  affects colormap quantization levels for default colormaps generated at
  module import time.
- ``Colorbar.patch`` is removed; this attribute was not correctly updated
  anymore.
- ``ContourLabeler.get_label_width`` is removed.
- ``Dvi.baseline`` is removed (with no replacement).
- The *format* parameter of ``dviread.find_tex_file`` is removed (with no
  replacement).
- ``FancyArrowPatch.get_path_in_displaycoord`` and
  ``ConnectionPath.get_path_in_displaycoord`` are removed. The path in
  display coordinates can still be obtained, as for other patches, using
  ``patch.get_transform().transform_path(patch.get_path())``.
- The ``font_manager.win32InstalledFonts`` and
  ``font_manager.get_fontconfig_fonts`` helper functions are removed.
- All parameters of ``imshow`` starting from *aspect* are keyword-only.
- ``QuadMesh.convert_mesh_to_paths`` and ``QuadMesh.convert_mesh_to_triangles``
  are removed. ``QuadMesh.get_paths()`` can be used as an alternative for the
  former; there is no replacement for the latter.
- ``ScalarMappable.callbacksSM`` is removed. Use
  ``ScalarMappable.callbacks`` instead.
- ``streamplot.get_integrator`` is removed.
- ``style.core.STYLE_FILE_PATTERN``, ``style.core.load_base_library``, and
  ``style.core.iter_user_libraries`` are removed.
- ``SubplotParams.validate`` is removed. Use `.SubplotParams.update` to
  change `.SubplotParams` while always keeping it in a valid state.
- The ``grey_arrayd``, ``font_family``, ``font_families``, and ``font_info``
  attributes of `.TexManager` are removed.
- ``Text.get_prop_tup`` is removed with no replacements (because the `.Text`
  class cannot know whether a backend needs to update cache e.g. when the
  text's color changes).
- ``Tick.apply_tickdir`` didn't actually update the tick markers on the
  existing Line2D objects used to draw the ticks and is removed; use
  `.Axis.set_tick_params` instead.
- ``tight_layout.auto_adjust_subplotpars`` is removed.
- The ``grid_info`` attribute of ``axisartist`` classes has been removed.
- ``axes_grid1.axes_grid.CbarAxes`` and ``axisartist.axes_grid.CbarAxes`` are
  removed (they are now dynamically generated based on the owning axes
  class).
- The ``axes_grid1.Divider.get_vsize_hsize`` and
  ``axes_grid1.Grid.get_vsize_hsize`` methods are removed.
- ``AxesDivider.append_axes(..., add_to_figure=False)`` is removed. Use
  ``ax.remove()`` to remove the Axes from the figure if needed.
- ``FixedAxisArtistHelper.change_tick_coord`` is removed with no
  replacement.
- ``floating_axes.GridHelperCurveLinear.get_boundary`` is removed with no
  replacement.
- ``ParasiteAxesBase.get_images_artists`` is removed.
- The "units finalize" signal (previously emitted by Axis instances) is
  removed. Connect to "units" instead.
- Passing formatting parameters positionally to ``stem()`` is no longer
  possible.
- ``axisartist.clip_path`` is removed with no replacement.

