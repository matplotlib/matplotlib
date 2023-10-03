Removals
--------

cbook removals
~~~~~~~~~~~~~~

- ``matplotlib.cbook.MatplotlibDeprecationWarning`` and
  ``matplotlib.cbook.mplDeprecation`` are removed; use
  `matplotlib.MatplotlibDeprecationWarning` instead.
- ``cbook.maxdict``; use the standard library ``functools.lru_cache`` instead.

Groupers from ``get_shared_x_axes`` / ``get_shared_y_axes`` are immutable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modifications to the Groupers returned by ``get_shared_x_axes`` and
``get_shared_y_axes`` are no longer allowed. Note that previously, calling e.g.
``join()`` would already fail to set up the correct structures for sharing
axes; use `.Axes.sharex` or `.Axes.sharey` instead.

Deprecated modules removed
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following deprecated modules are removed:

* ``afm``
* ``docstring``
* ``fontconfig_pattern``
* ``tight_bbox``
* ``tight_layout``
* ``type1font``

Parameters to ``plt.figure()`` and the ``Figure`` constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All parameters to `.pyplot.figure` and the `.Figure` constructor, other than
*num*, *figsize*, and *dpi*, are now keyword-only.

``stem(..., use_line_collection=False)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is no longer supported. This was a compatibility fallback to a
former more inefficient representation of the stem lines.

Positional / keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing all but the very few first arguments positionally in the constructors
of Artists is no longer possible. Most arguments are now keyword-only.

The *emit* and *auto* parameters of ``set_xlim``, ``set_ylim``,
``set_zlim``, ``set_rlim`` are now keyword-only.

The *transOffset* parameter of `.Collection.set_offset_transform` and the
various ``create_collection`` methods of legend handlers has been renamed to
*offset_transform* (consistently with the property name).

``Axes.get_window_extent`` / ``Figure.get_window_extent`` accept only
*renderer*. This aligns the API with the general `.Artist.get_window_extent`
API. All other parameters were ignored anyway.

Methods to set parameters in ``LogLocator`` and ``LogFormatter*``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In `~.LogFormatter` and derived subclasses, the methods ``base`` and
``label_minor`` for setting the respective parameter are removed and
replaced by ``set_base`` and ``set_label_minor``, respectively.

In `~.LogLocator`, the methods ``base`` and ``subs`` for setting the respective
parameter are removed. Instead, use ``set_params(base=..., subs=...)``.

``Axes.get_renderer_cache``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The canvas now takes care of the renderer and whether to cache it or not,
so the ``Axes.get_renderer_cache`` method is removed. The
alternative is to call ``axes.figure.canvas.get_renderer()``.

Unused methods in ``Axis``, ``Tick``, ``XAxis``, and ``YAxis``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Tick.label`` is now removed. Use ``Tick.label1`` instead.

The following methods are no longer used and removed without a replacement:

- ``Axis.get_ticklabel_extents``
- ``Tick.get_pad_pixels``
- ``XAxis.get_text_heights``
- ``YAxis.get_text_widths``

``mlab.stride_windows``
~~~~~~~~~~~~~~~~~~~~~~~

... is removed. Use ``numpy.lib.stride_tricks.sliding_window_view`` instead.

``Axes3D``
~~~~~~~~~~

The ``dist`` attribute has been privatized. Use the *zoom* keyword argument in
`.Axes3D.set_box_aspect` instead.

The ``w_xaxis``, ``w_yaxis``, and ``w_zaxis`` attributes are now removed.
Instead use ``xaxis``, ``yaxis``, and ``zaxis``.

3D Axis
~~~~~~~

``mplot3d.axis3d.Axis.set_pane_pos`` is removed. This is an internal method
where the provided values are overwritten during drawing. Hence, it does not
serve any purpose to be directly accessible.

The two helper functions ``mplot3d.axis3d.move_from_center`` and
``mplot3d.axis3d.tick_update_position`` are considered internal and deprecated.
If these are required, please vendor the code from the corresponding private
methods ``_move_from_center`` and ``_tick_update_position``.

``checkdep_usetex`` removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method was only intended to disable tests in case no latex install was
found. As such, it is considered to be private and for internal use only.

Please vendor the code from a previous version if you need this.

``date_ticker_factory`` removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``date_ticker_factory`` method in the `matplotlib.dates` module is
removed. Instead use `~.AutoDateLocator` and `~.AutoDateFormatter` for a
more flexible and scalable locator and formatter.

If you need the exact ``date_ticker_factory`` behavior, please copy the code
from a previous version.

``transforms.Affine2D.identity()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is removed in favor of directly calling the `.Affine2D` constructor with
no arguments.

Removals in ``testing.decorators``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The unused class ``CleanupTestCase`` and decorator ``cleanup`` are removed.
The function ``check_freetype_version`` is considered internal and removed.
Vendor the code from a previous version.

``text.get_rotation()``
~~~~~~~~~~~~~~~~~~~~~~~

... is removed with no replacement. Copy the previous implementation if
needed.
``Figure.callbacks`` is removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Figure ``callbacks`` property has been removed. The only signal was
"dpi_changed", which can be replaced by connecting to the "resize_event" on the
canvas ``figure.canvas.mpl_connect("resize_event", func)`` instead.


Passing too many positional arguments to ``tripcolor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... raises ``TypeError`` (extra arguments were previously ignored).


The *filled* argument to ``Colorbar`` is removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This behavior was already governed by the underlying ``ScalarMappable``.


Widgets
~~~~~~~

The *visible* attribute setter of Selector widgets has been removed; use ``set_visible``
The associated getter is also deprecated, but not yet expired.

``Axes3D.set_frame_on`` and ``Axes3D.get_frame_on`` removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Axes3D.set_frame_on`` is documented as "Set whether the 3D axes panels are
drawn.". However, it has no effect on 3D axes and is being removed in
favor of ``Axes3D.set_axis_on`` and ``Axes3D.set_axis_off``.

Miscellaneous internals
~~~~~~~~~~~~~~~~~~~~~~~

- ``axes_grid1.axes_size.AddList``; use ``sum(sizes, start=Fixed(0))`` (for
  example) to sum multiple size objects.
- ``axes_size.Padded``; use ``size + pad`` instead
- ``axes_size.SizeFromFunc``, ``axes_size.GetExtentHelper``
- ``AxisArtistHelper.delta1`` and ``AxisArtistHelper.delta2``
- ``axislines.GridHelperBase.new_gridlines`` and
  ``axislines.Axes.new_gridlines``
- ``_DummyAxis.dataLim`` and ``_DummyAxis.viewLim``; use
  ``get_data_interval()``, ``set_data_interval()``, ``get_view_interval()``,
  and ``set_view_interval()`` instead.
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


Backend-specific removals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``backend_pdf.Name.hexify``
- ``backend_pdf.Operator`` and ``backend_pdf.Op.op`` are removed in favor of
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
- ``backend_svg.escape_attrib``; vendor the code of the similarly named private
  functions if you rely on it.
- ``backend_svg.escape_cdata``; vendor the code of the similarly named private
  functions if you rely on it.
- ``backend_svg.escape_comment``; vendor the code of the similarly named
  private functions if you rely on it.
- ``backend_svg.short_float_fmt``; vendor the code of the similarly named
  private functions if you rely on it.
- ``backend_svg.generate_transform`` and ``backend_svg.generate_css``

Removal of deprecated APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following deprecated APIs have been removed.  Unless a replacement is stated, please
vendor the previous implementation if needed.

- The following methods of `.FigureCanvasBase`: ``pick`` (use ``Figure.pick`` instead),
  ``resize``, ``draw_event``, ``resize_event``, ``close_event``, ``key_press_event``,
  ``key_release_event``, ``pick_event``, ``scroll_event``, ``button_press_event``,
  ``button_release_event``, ``motion_notify_event``, ``leave_notify_event``,
  ``enter_notify_event`` (for all the ``foo_event`` methods, construct the relevant
  `.Event` object and call ``canvas.callbacks.process(event.name, event)`` instead).
- ``ToolBase.destroy`` (connect to ``tool_removed_event`` instead).
- The *cleared* parameter to `.FigureCanvasAgg.get_renderer` (call ``renderer.clear()``
  instead).
- The following methods of `.RendererCairo`: ``set_ctx_from_surface`` and
  ``set_width_height`` (use ``set_context`` instead, which automatically infers the
  canvas size).
- The ``window`` or ``win`` parameters and/or attributes of ``NavigationToolbar2Tk``,
  ``NavigationToolbar2GTK3``, and ``NavigationToolbar2GTK4``, and the ``lastrect``
  attribute of ``NavigationToolbar2Tk``
- The ``error_msg_gtk`` function and the ``icon_filename`` and ``window_icon`` globals
  in ``backend_gtk3``; the ``error_msg_wx`` function in ``backend_wx``.
- ``FigureManagerGTK3Agg`` and ``FigureManagerGTK4Agg`` (use ``FigureManagerGTK3``
  instead); ``RendererGTK3Cairo`` and ``RendererGTK4Cairo``.
- ``NavigationToolbar2Mac.prepare_configure_subplots`` (use
  `~.NavigationToolbar2.configure_subplots` instead).
- ``FigureManagerMac.close``.
- The ``qApp`` global in `.backend_qt` (use ``QtWidgets.QApplication.instance()``
  instead).
- The ``offset_text_height`` method of ``RendererWx``; the ``sizer``, ``figmgr``,
  ``num``, ``toolbar``, ``toolmanager``, ``get_canvas``, and ``get_figure_manager``
  attributes or methods of ``FigureFrameWx`` (use ``frame.GetSizer()``,
  ``frame.canvas.manager``, ``frame.canvas.manager.num``, ``frame.GetToolBar()``,
  ``frame.canvas.manager.toolmanager``, the *canvas_class* constructor parameter, and
  ``frame.canvas.manager``, respectively, instead).
- ``FigureFrameWxAgg`` and ``FigureFrameWxCairo`` (use
  ``FigureFrameWx(..., canvas_class=FigureCanvasWxAgg)`` and
  ``FigureFrameWx(..., canvas_class=FigureCanvasWxCairo)``, respectively, instead).
- The ``filled`` attribute and the ``draw_all`` method of `.Colorbar` (instead of
  ``draw_all``, use ``figure.draw_without_rendering``).
- Calling `.MarkerStyle` without setting the *marker* parameter or setting it to None
  (use ``MarkerStyle("")`` instead).
- Support for third-party canvas classes without a ``required_interactive_framework``
  attribute (this can only occur if the canvas class does not inherit from
  `.FigureCanvasBase`).
- The ``canvas`` and ``background`` attributes of `.MultiCursor`; the
  ``state_modifier_keys`` attribute of selector widgets.
- Passing *useblit*, *horizOn*, or *vertOn* positionally to `.MultiCursor`.
- Support for the ``seaborn-<foo>`` styles; use ``seaborn-v0_8-<foo>`` instead, or
  directly use the seaborn API.
