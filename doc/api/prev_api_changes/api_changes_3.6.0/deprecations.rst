Deprecations
------------

Parameters to ``plt.figure()`` and the ``Figure`` constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All parameters to `.pyplot.figure` and the `.Figure` constructor, other than
*num*, *figsize*, and *dpi*, will become keyword-only after a deprecation
period.

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

Pending deprecation top-level cmap registration and access functions in ``mpl.cm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As part of a `multi-step process
<https://github.com/matplotlib/matplotlib/issues/20853>`_ we are refactoring
the global state for managing the registered colormaps.

In Matplotlib 3.5 we added a `.ColormapRegistry` class and exposed an instance
at the top level as ``matplotlib.colormaps``. The existing top level functions
in `matplotlib.cm` (``get_cmap``, ``register_cmap``, ``unregister_cmap``) were
changed to be aliases around the same instance.

In Matplotlib 3.6 we have marked those top level functions as pending
deprecation with the intention of deprecation in Matplotlib 3.7. The following
functions have been marked for pending deprecation:

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

Pending deprecation of layout methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The methods `~.Figure.set_tight_layout`, `~.Figure.set_constrained_layout`, are
discouraged, and now emit a `PendingDeprecationWarning` in favor of explicitly
referencing the layout engine via ``figure.set_layout_engine('tight')`` and
``figure.set_layout_engine('constrained')``. End users should not see the
warning, but library authors should adjust.

The methods `~.Figure.set_constrained_layout_pads` and
`~.Figure.get_constrained_layout_pads` are will be deprecated in favor of
``figure.get_layout_engine().set()`` and ``figure.get_layout_engine().get()``,
and currently emit a `PendingDeprecationWarning`.

seaborn styles renamed
~~~~~~~~~~~~~~~~~~~~~~

Matplotlib currently ships many style files inspired from the seaborn library
("seaborn", "seaborn-bright", "seaborn-colorblind", etc.) but they have gone
out of sync with the library itself since the release of seaborn 0.9. To
prevent confusion, the style files have been renamed "seaborn-v0_8",
"seaborn-v0_8-bright", "seaborn-v0_8-colorblind", etc. Users are encouraged to
directly use seaborn to access the up-to-date styles.

Auto-removal of overlapping Axes by ``plt.subplot`` and ``plt.subplot2grid``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, `.pyplot.subplot` and `.pyplot.subplot2grid` would automatically
remove preexisting Axes that overlap with the newly added Axes. This behavior
was deemed confusing, and is now deprecated. Explicitly call ``ax.remove()`` on
Axes that need to be removed.

Passing *linefmt* positionally to ``stem`` is undeprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Positional use of all formatting parameters in `~.Axes.stem` has been
deprecated since Matplotlib 3.5. This deprecation is relaxed so that one can
still pass *linefmt* positionally, i.e. ``stem(x, y, 'r')``.

``stem(..., use_line_collection=False)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated with no replacement. This was a compatibility fallback to a
former more inefficient representation of the stem lines.

Positional / keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing all but the very few first arguments positionally in the constructors
of Artists is deprecated. Most arguments will become keyword-only in a future
version.

Passing too many positional arguments to ``tripcolor`` is now deprecated (extra
arguments were previously silently ignored).

Passing *emit* and *auto* parameters of ``set_xlim``, ``set_ylim``,
``set_zlim``, ``set_rlim`` positionally is deprecated; they will become
keyword-only in a future release.

The *transOffset* parameter of `.Collection.set_offset_transform` and the
various ``create_collection`` methods of legend handlers has been renamed to
*offset_transform* (consistently with the property name).

Calling ``MarkerStyle()`` with no arguments or ``MarkerStyle(None)`` is
deprecated; use ``MarkerStyle("")`` to construct an empty marker style.

``Axes.get_window_extent`` / ``Figure.get_window_extent`` accept only
*renderer*. This aligns the API with the general `.Artist.get_window_extent`
API. All other parameters were ignored anyway.

The *cleared* parameter of ``get_renderer``, which only existed for AGG-based
backends, has been deprecated. Use ``renderer.clear()`` instead to explicitly
clear the renderer buffer.

Methods to set parameters in ``LogLocator`` and ``LogFormatter*``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In `~.LogFormatter` and derived subclasses, the methods ``base`` and
``label_minor`` for setting the respective parameter are deprecated and
replaced by ``set_base`` and ``set_label_minor``, respectively.

In `~.LogLocator`, the methods ``base`` and ``subs`` for setting the respective
parameter are deprecated. Instead, use ``set_params(base=..., subs=...)``.

``Axes.get_renderer_cache``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canvas now takes care of the renderer and whether to cache it or not. The
alternative is to call ``axes.figure.canvas.get_renderer()``.

Groupers from ``get_shared_x_axes`` / ``get_shared_y_axes`` will be immutable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modifications to the Groupers returned by ``get_shared_x_axes`` and
``get_shared_y_axes`` are deprecated. In the future, these methods will return
immutable views on the grouper structures. Note that previously, calling e.g.
``join()`` would already fail to set up the correct structures for sharing
axes; use `.Axes.sharex` or `.Axes.sharey` instead.

Unused methods in ``Axis``, ``Tick``, ``XAxis``, and ``YAxis``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Tick.label`` has been pending deprecation since 3.1 and is now deprecated.
Use ``Tick.label1`` instead.

The following methods are no longer used and deprecated without a replacement:

- ``Axis.get_ticklabel_extents``
- ``Tick.get_pad_pixels``
- ``XAxis.get_text_heights``
- ``YAxis.get_text_widths``

``mlab.stride_windows``
~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated. Use ``np.lib.stride_tricks.sliding_window_view`` instead (or
``np.lib.stride_tricks.as_strided`` on NumPy < 1.20).

Event handlers
~~~~~~~~~~~~~~

The ``draw_event``, ``resize_event``, ``close_event``, ``key_press_event``,
``key_release_event``, ``pick_event``, ``scroll_event``,
``button_press_event``, ``button_release_event``, ``motion_notify_event``,
``enter_notify_event`` and ``leave_notify_event`` methods of
`.FigureCanvasBase` are deprecated. They had inconsistent signatures across
backends, and made it difficult to improve event metadata.

In order to trigger an event on a canvas, directly construct an `.Event` object
of the correct class and call ``canvas.callbacks.process(event.name, event)``.

Widgets
~~~~~~~

All parameters to ``MultiCursor`` starting from *useblit* are becoming
keyword-only (passing them positionally is deprecated).

The ``canvas`` and ``background`` attributes of ``MultiCursor`` are deprecated
with no replacement.

The *visible* attribute of Selector widgets has been deprecated; use
``set_visible`` or ``get_visible`` instead.

The *state_modifier_keys* attribute of Selector widgets has been privatized and
the modifier keys must be set when creating the widget.

``Axes3D.dist``
~~~~~~~~~~~~~~~

... has been privatized. Use the *zoom* keyword argument in
`.Axes3D.set_box_aspect` instead.

3D Axis
~~~~~~~

The previous constructor of `.axis3d.Axis`, with signature ``(self, adir,
v_intervalx, d_intervalx, axes, *args, rotate_label=None, **kwargs)`` is
deprecated in favor of a new signature closer to the one of 2D Axis; it is now
``(self, axes, *, rotate_label=None, **kwargs)`` where ``kwargs`` are forwarded
to the 2D Axis constructor. The axis direction is now inferred from the axis
class' ``axis_name`` attribute (as in the 2D case); the ``adir`` attribute is
deprecated.

The ``init3d`` method of 3D Axis is also deprecated; all the relevant
initialization is done as part of the constructor.

The ``d_interval`` and ``v_interval`` attributes of 3D Axis are deprecated; use
``get_data_interval`` and ``get_view_interval`` instead.

The ``w_xaxis``, ``w_yaxis``, and ``w_zaxis`` attributes of ``Axis3D`` have
been pending deprecation since 3.1. They are now deprecated. Instead use
``xaxis``, ``yaxis``, and ``zaxis``.

``mplot3d.axis3d.Axis.set_pane_pos`` is deprecated. This is an internal method
where the provided values are overwritten during drawing. Hence, it does not
serve any purpose to be directly accessible.

The two helper functions ``mplot3d.axis3d.move_from_center`` and
``mplot3d.axis3d.tick_update_position`` are considered internal and deprecated.
If these are required, please vendor the code from the corresponding private
methods ``_move_from_center`` and ``_tick_update_position``.

``Figure.callbacks`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Figure ``callbacks`` property is deprecated. The only signal was
"dpi_changed", which can be replaced by connecting to the "resize_event" on the
canvas ``figure.canvas.mpl_connect("resize_event", func)`` instead.

``FigureCanvas`` without a ``required_interactive_framework`` attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Support for such canvas classes is deprecated. Note that canvas classes which
inherit from ``FigureCanvasBase`` always have such an attribute.

Backend-specific deprecations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``backend_gtk3.FigureManagerGTK3Agg`` and
  ``backend_gtk4.FigureManagerGTK4Agg``; directly use
  ``backend_gtk3.FigureManagerGTK3`` and ``backend_gtk4.FigureManagerGTK4``
  instead.
- The *window* parameter to ``backend_gtk3.NavigationToolbar2GTK3`` had no
  effect, and is now deprecated.
- ``backend_gtk3.NavigationToolbar2GTK3.win``
- ``backend_gtk3.RendererGTK3Cairo`` and ``backend_gtk4.RendererGTK4Cairo``;
  use `.RendererCairo` instead, which has gained the ``set_context`` method,
  which also auto-infers the size of the underlying surface.
- ``backend_cairo.RendererCairo.set_ctx_from_surface`` and
  ``backend_cairo.RendererCairo.set_width_height`` in favor of
  `.RendererCairo.set_context`.
- ``backend_gtk3.error_msg_gtk``
- ``backend_gtk3.icon_filename`` and ``backend_gtk3.window_icon``
- ``backend_macosx.NavigationToolbar2Mac.prepare_configure_subplots`` has been
  replaced by ``configure_subplots()``.
- ``backend_pdf.Name.hexify``
- ``backend_pdf.Operator`` and ``backend_pdf.Op.op`` are deprecated in favor of
  a single standard `enum.Enum` interface on `.backend_pdf.Op`.
- ``backend_pdf.fill``; vendor the code of the similarly named private
  functions if you rely on these functions.
- ``backend_pgf.LatexManager.texcommand`` and
  ``backend_pgf.LatexManager.latex_header``
- ``backend_pgf.NO_ESCAPE``
- ``backend_pgf.common_texification``
- ``backend_pgf.get_fontspec``
- ``backend_pgf.get_preamble``
- ``backend_pgf.re_mathsep``
- ``backend_pgf.writeln``
- ``backend_ps.convert_psfrags``
- ``backend_ps.quote_ps_string``; vendor the code of the similarly named
  private functions if you rely on it.
- ``backend_qt.qApp``; use ``QtWidgets.QApplication.instance()`` instead.
- ``backend_svg.escape_attrib``; vendor the code of the similarly named private
  functions if you rely on it.
- ``backend_svg.escape_cdata``; vendor the code of the similarly named private
  functions if you rely on it.
- ``backend_svg.escape_comment``; vendor the code of the similarly named
  private functions if you rely on it.
- ``backend_svg.short_float_fmt``; vendor the code of the similarly named
  private functions if you rely on it.
- ``backend_svg.generate_transform`` and ``backend_svg.generate_css``
- ``backend_tk.NavigationToolbar2Tk.lastrect`` and
  ``backend_tk.RubberbandTk.lastrect``
- ``backend_tk.NavigationToolbar2Tk.window``; use ``toolbar.master`` instead.
- ``backend_tools.ToolBase.destroy``; To run code upon tool removal, connect to
  the ``tool_removed_event`` event.
- ``backend_wx.RendererWx.offset_text_height``
- ``backend_wx.error_msg_wx``

- ``FigureCanvasBase.pick``; directly call `.Figure.pick`, which has taken over
  the responsibility of checking the canvas widget lock as well.
- ``FigureCanvasBase.resize``, which has no effect; use
  ``FigureManagerBase.resize`` instead.

- ``FigureManagerMac.close``

- ``FigureFrameWx.sizer``; use ``frame.GetSizer()`` instead.
- ``FigureFrameWx.figmgr`` and ``FigureFrameWx.get_figure_manager``; use
  ``frame.canvas.manager`` instead.
- ``FigureFrameWx.num``; use ``frame.canvas.manager.num`` instead.
- ``FigureFrameWx.toolbar``; use ``frame.GetToolBar()`` instead.
- ``FigureFrameWx.toolmanager``; use ``frame.canvas.manager.toolmanager``
  instead.

Modules
~~~~~~~

The modules ``matplotlib.afm``, ``matplotlib.docstring``,
``matplotlib.fontconfig_pattern``, ``matplotlib.tight_bbox``,
``matplotlib.tight_layout``, and ``matplotlib.type1font`` are considered
internal and public access is deprecated.

``checkdep_usetex`` deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method was only intended to disable tests in case no latex install was
found. As such, it is considered to be private and for internal use only.

Please vendor the code if you need this.

``date_ticker_factory`` deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``date_ticker_factory`` method in the `matplotlib.dates` module is
deprecated. Instead use `~.AutoDateLocator` and `~.AutoDateFormatter` for a
more flexible and scalable locator and formatter.

If you need the exact ``date_ticker_factory`` behavior, please copy the code.

``dviread.find_tex_file`` will raise ``FileNotFoundError``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the future, ``dviread.find_tex_file`` will raise a `FileNotFoundError` for
missing files. Previously, it would return an empty string in such cases.
Raising an exception allows attaching a user-friendly message instead. During
the transition period, a warning is raised.

``transforms.Affine2D.identity()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated in favor of directly calling the `.Affine2D` constructor with
no arguments.

Deprecations in ``testing.decorators``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unused class ``CleanupTestCase`` and decorator ``cleanup`` are deprecated
and will be removed. Vendor the code, including the private function
``_cleanup_cm``.

The function ``check_freetype_version`` is considered internal and deprecated.
Vendor the code of the private function ``_check_freetype_version``.

``text.get_rotation()``
~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated with no replacement. Copy the original implementation if
needed.

Miscellaneous internals
~~~~~~~~~~~~~~~~~~~~~~~

- ``axes_grid1.axes_size.AddList``; use ``sum(sizes, start=Fixed(0))`` (for
  example) to sum multiple size objects.
- ``axes_size.Padded``; use ``size + pad`` instead
- ``axes_size.SizeFromFunc``, ``axes_size.GetExtentHelper``
- ``AxisArtistHelper.delta1`` and ``AxisArtistHelper.delta2``
- ``axislines.GridHelperBase.new_gridlines`` and
  ``axislines.Axes.new_gridlines``
- ``cbook.maxdict``; use the standard library ``functools.lru_cache`` instead.
- ``_DummyAxis.dataLim`` and ``_DummyAxis.viewLim``; use
  ``get_data_interval()``, ``set_data_interval()``, ``get_view_interval()``,
  and ``set_view_interval()`` instead.
- ``GridSpecBase.get_grid_positions(..., raw=True)``
- ``ImageMagickBase.delay`` and ``ImageMagickBase.output_args``
- ``MathtextBackend``, ``MathtextBackendAgg``, ``MathtextBackendPath``,
  ``MathTextWarning``
- ``TexManager.get_font_config``; it previously returned an internal hashed key
  for used for caching purposes.
- ``TextToPath.get_texmanager``; directly construct a `.texmanager.TexManager`
  instead.
- ``ticker.is_close_to_int``; use ``math.isclose(x, round(x))`` instead.
- ``ticker.is_decade``; use ``y = numpy.log(x)/numpy.log(base);
  numpy.isclose(y, numpy.round(y))`` instead.
