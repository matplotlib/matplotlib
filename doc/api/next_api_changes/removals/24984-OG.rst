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
