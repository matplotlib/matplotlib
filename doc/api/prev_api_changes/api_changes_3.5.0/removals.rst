Removals
--------

The following deprecated APIs have been removed:

Removed behaviour
~~~~~~~~~~~~~~~~~

Stricter validation of function parameters
..........................................

- Calling `.Figure.add_axes` with no arguments will raise an error. Adding a
  free-floating axes needs a position rectangle. If you want a figure-filling
  single axes, use `.Figure.add_subplot` instead.
- `.Figure.add_subplot` validates its inputs; in particular, for
  ``add_subplot(rows, cols, index)``, all parameters must be integral.
  Previously strings and floats were accepted and converted to int.
- Passing *None* as the *which* argument to ``autofmt_xdate`` is no longer
  supported; use its more explicit synonym, ``which="major"``, instead.
- Setting the *orientation* of an ``eventplot()`` or `.EventCollection` to
  "none" or *None* is no longer supported; set it to "horizontal" instead.
  Moreover, the two orientations ("horizontal" and "vertical") are now
  case-sensitive.
- Passing parameters *norm* and *vmin*/*vmax* simultaneously to functions using
  colormapping such as ``scatter()`` and ``imshow()`` is no longer supported.
  Instead of ``norm=LogNorm(), vmin=min_val, vmax=max_val`` pass
  ``norm=LogNorm(min_val, max_val)``. *vmin* and *vmax* should only be used
  without setting *norm*.
- Passing *None* as either the *radius* or *startangle* arguments of an
  `.Axes.pie` is no longer accepted; use the explicit defaults of 1 and 0,
  respectively, instead.
- Passing *None* as the *normalize* argument of `.Axes.pie` (the former
  default) is no longer accepted, and the pie will always be normalized by
  default. If you wish to plot an incomplete pie, explicitly pass
  ``normalize=False``.
- Support for passing *None* to ``subplot_class_factory`` has been removed.
  Explicitly pass in the base `~matplotlib.axes.Axes` class instead.
- Passing multiple keys as a single comma-separated string or multiple
  arguments to `.ToolManager.update_keymap` is no longer supported; pass keys
  as a list of strings instead.
- Passing the dash offset as *None* is no longer accepted, as this was never
  universally implemented, e.g. for vector output. Set the offset to 0 instead.
- Setting a custom method overriding `.Artist.contains` using
  ``Artist.set_contains`` has been removed, as has ``Artist.get_contains``.
  There is no replacement, but you may still customize pick events using
  `.Artist.set_picker`.
- `~.Axes.semilogx`, `~.Axes.semilogy`, `~.Axes.loglog`, `.LogScale`, and
  `.SymmetricalLogScale` used to take keyword arguments that depends on the
  axis orientation ("basex" vs "basey", "subsx" vs "subsy", "nonposx" vs
  "nonposy"); these parameter names have been removed in favor of "base",
  "subs", "nonpositive". This removal also affects e.g. ``ax.set_yscale("log",
  basey=...)`` which must now be spelled ``ax.set_yscale("log", base=...)``.

  The change from "nonpos" to "nonpositive" also affects
  `~.scale.LogTransform`, `~.scale.InvertedLogTransform`,
  `~.scale.SymmetricalLogTransform`, etc.

  To use *different* bases for the x-axis and y-axis of a `~.Axes.loglog` plot,
  use e.g. ``ax.set_xscale("log", base=10); ax.set_yscale("log", base=2)``.
- Passing *None*, or no argument, to ``parasite_axes_class_factory``,
  ``parasite_axes_auxtrans_class_factory``, ``host_axes_class_factory`` is no
  longer accepted; pass an explicit base class instead.

Case-sensitivity is now enforced more
......................................

- Upper or mixed-case property names are no longer normalized to lowercase in
  `.Artist.set` and `.Artist.update`. This allows one to pass names such as
  *patchA* or *UVC*.
- Case-insensitive capstyles and joinstyles are no longer lower-cased; please
  pass capstyles ("miter", "round", "bevel") and joinstyles ("butt", "round",
  "projecting") as lowercase.
- Saving metadata in PDF with the PGF backend no longer changes keys to
  lowercase. Only the canonically cased keys listed in the PDF specification
  (and the `~.backend_pgf.PdfPages` documentation) are accepted.

No implicit initialization of ``Tick`` attributes
.................................................

The `.Tick` constructor no longer initializes the attributes ``tick1line``,
``tick2line``, ``gridline``, ``label1``, and ``label2`` via ``_get_tick1line``,
``_get_tick2line``, ``_get_gridline``, ``_get_text1``, and ``_get_text2``.
Please directly set the attribute in the subclass' ``__init__`` instead.

``NavigationToolbar2`` subclass changes
.......................................

Overriding the ``_init_toolbar`` method of `.NavigationToolbar2` to initialize
third-party toolbars is no longer supported. Instead, the toolbar should be
initialized in the ``__init__`` method of the subclass (which should call the
base-class' ``__init__`` as appropriate).

The ``press`` and ``release`` methods of `.NavigationToolbar2` were called when
pressing or releasing a mouse button, but *only* when an interactive pan or
zoom was occurring (contrary to what the docs stated). They are no longer
called; if you write a backend which needs to customize such events, please
directly override ``press_pan``/``press_zoom``/``release_pan``/``release_zoom``
instead.

Removal of old file mode flag
.............................

Flags containing "U" passed to `.cbook.to_filehandle` and `.cbook.open_file_cm`
are no longer accepted. This is consistent with their removal from `open` in
Python 3.9.

Keymaps toggling ``Axes.get_navigate`` have been removed
........................................................

This includes numeric key events and rcParams.

The ``TTFPATH`` and ``AFMPATH`` environment variables
.....................................................

Support for the (undocumented) ``TTFPATH`` and ``AFMPATH`` environment
variables has been removed. Register additional fonts using
``matplotlib.font_manager.fontManager.addfont()``.

Modules
~~~~~~~

- ``matplotlib.backends.qt_editor.formsubplottool``; use
  ``matplotlib.backends.backend_qt.SubplotToolQt`` instead.
- ``matplotlib.compat``
- ``matplotlib.ttconv``
- The Qt4-based backends, ``qt4agg`` and ``qt4cairo``, have been removed. Qt4
  has reached its end-of-life in 2015 and there are no releases of either PyQt4
  or PySide for recent versions of Python. Please use one of the Qt5 or Qt6
  backends.

Classes, methods and attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following module-level classes/variables have been removed:

- ``backend_bases.StatusbarBase`` and all its subclasses, and ``StatusBarWx``;
  messages are displayed in the toolbar
- ``backend_pgf.GraphicsContextPgf``
- ``MODIFIER_KEYS``, ``SUPER``, ``ALT``, ``CTRL``, and ``SHIFT`` of
  `matplotlib.backends.backend_qt5agg` and
  `matplotlib.backends.backend_qt5cairo`
- ``backend_wx.DEBUG_MSG``
- ``dviread.Encoding``
- ``Fil``, ``Fill``, ``Filll``, ``NegFil``, ``NegFill``, ``NegFilll``, and
  ``SsGlue`` from `.mathtext`; directly construct glue instances with
  ``Glue("fil")``, etc.
- ``mathtext.GlueSpec``
- ``OldScalarFormatter``, ``IndexFormatter`` and ``IndexDateFormatter``; use
  `.FuncFormatter` instead
- ``OldAutoLocator``
- ``AVConvBase``, ``AVConvWriter`` and ``AVConvFileWriter``. Debian 8 (2015,
  EOL 06/2020) and Ubuntu 14.04 (EOL 04/2019) were the last versions of Debian
  and Ubuntu to ship avconv. It remains possible to force the use of avconv by
  using the FFmpeg-based writers with :rc:`animation.ffmpeg_path` set to
  "avconv".
- ``matplotlib.axes._subplots._subplot_classes``
- ``axes_grid1.axes_rgb.RGBAxesBase``; use ``RGBAxes`` instead

The following class attributes have been removed:

- ``backend_pgf.LatexManager.latex_stdin_utf8``
- ``backend_pgf.PdfPages.metadata``
- ``ContourSet.ax`` and ``Quiver.ax``; use ``ContourSet.axes`` or
  ``Quiver.axes`` as with other artists
- ``DateFormatter.illegal_s``
- ``dates.YearLocator.replaced``; `.YearLocator` is now a subclass of
  `.RRuleLocator`, and the attribute ``YearLocator.replaced`` has been removed.
  For tick locations that required modifying this, a custom rrule and
  `.RRuleLocator` can be used instead.
- ``FigureManagerBase.statusbar``; messages are displayed in the toolbar
- ``FileMovieWriter.clear_temp``
- ``mathtext.Glue.glue_subtype``
- ``MovieWriter.args_key``, ``MovieWriter.exec_key``, and
  ``HTMLWriter.args_key``
- ``NavigationToolbar2QT.basedir``; the base directory to the icons is
  ``os.path.join(mpl.get_data_path(), "images")``
- ``NavigationToolbar2QT.ctx``
- ``NavigationToolbar2QT.parent``; to access the parent window, use
  ``toolbar.canvas.parent()`` or ``toolbar.parent()``
- ``prevZoomRect``, ``retinaFix``, ``savedRetinaImage``, ``wxoverlay``,
  ``zoomAxes``, ``zoomStartX``, and ``zoomStartY`` attributes of
  ``NavigationToolbar2Wx``
- ``NonUniformImage.is_grayscale``, ``PcolorImage.is_grayscale``, for
  consistency with ``AxesImage.is_grayscale``. (Note that previously, these
  attributes were only available *after rendering the image*).
- ``RendererCairo.fontweights``, ``RendererCairo.fontangles``
- ``used_characters`` of `.RendererPdf`, `.PdfFile`, and `.RendererPS`
- ``LogScale.LogTransform``, ``LogScale.InvertedLogTransform``,
  ``SymmetricalScale.SymmetricalTransform``, and
  ``SymmetricalScale.InvertedSymmetricalTransform``; directly access the
  transform classes from `matplotlib.scale`
- ``cachedir``, ``rgba_arrayd``, ``serif``, ``sans_serif``, ``cursive``, and
  ``monospace`` attributes of `.TexManager`
- ``axleft``, ``axright``, ``axbottom``, ``axtop``, ``axwspace``, and
  ``axhspace`` attributes of `.widgets.SubplotTool`; access the ``ax``
  attribute of the corresponding slider
- ``widgets.TextBox.params_to_disable``
- ``angle_helper.LocatorBase.den``; it has been renamed to *nbins*
- ``axes_grid.CbarAxesBase.cbid`` and ``axes_grid.CbarAxesBase.locator``; use
  ``mappable.colorbar_cid`` or ``colorbar.locator`` instead

The following class methods have been removed:

- ``Axes.update_datalim_bounds``; use ``ax.dataLim.set(Bbox.union([ax.dataLim,
  bounds]))``
- ``pan`` and ``zoom`` methods of `~.axis.Axis` and `~.ticker.Locator` have
  been removed; panning and zooming are now implemented using the
  ``start_pan``, ``drag_pan``, and ``end_pan`` methods of `~.axes.Axes`
- ``.BboxBase.inverse_transformed``; call `.BboxBase.transformed` on the
  `~.Transform.inverted()` transform
- ``Collection.set_offset_position`` and ``Collection.get_offset_position``
  have been removed; the ``offset_position`` of the `.Collection` class is now
  "screen"
- ``Colorbar.on_mappable_changed`` and ``Colorbar.update_bruteforce``; use
  ``Colorbar.update_normal()`` instead
- ``docstring.Substitution.from_params`` has been removed; directly assign to
  ``params`` of `.Substitution` instead
- ``DraggableBase.artist_picker``; set the artist's picker instead
- ``DraggableBase.on_motion_blit``; use `.DraggableBase.on_motion` instead
- ``FigureCanvasGTK3._renderer_init``
- ``Locator.refresh()`` and the associated helper methods
  ``NavigationToolbar2.draw()`` and ``ToolViewsPositions.refresh_locators()``
- ``track_characters`` and ``merge_used_characters`` of `.RendererPdf`,
  `.PdfFile`, and `.RendererPS`
- ``RendererWx.get_gc``
- ``SubplotSpec.get_rows_columns``; use the ``GridSpec.nrows``,
  ``GridSpec.ncols``, ``SubplotSpec.rowspan``, and ``SubplotSpec.colspan``
  properties instead.
- ``ScalarMappable.update_dict``, ``ScalarMappable.add_checker()``, and
  ``ScalarMappable.check_update()``; register a callback in
  ``ScalarMappable.callbacks`` to be notified of updates
- ``TexManager.make_tex_preview`` and ``TexManager.make_dvi_preview``
- ``funcleft``, ``funcright``, ``funcbottom``, ``functop``, ``funcwspace``, and
  ``funchspace`` methods of `.widgets.SubplotTool`

- ``axes_grid1.axes_rgb.RGBAxes.add_RGB_to_figure``
- ``axisartist.axis_artist.AxisArtist.dpi_transform``
- ``axisartist.grid_finder.MaxNLocator.set_factor`` and
  ``axisartist.grid_finder.FixedLocator.set_factor``; the factor is always 1
  now

Functions
~~~~~~~~~

- ``bezier.make_path_regular`` has been removed; use ``Path.cleaned()`` (or
  ``Path.cleaned(curves=True)``, etc.) instead, but note that these methods add
  a ``STOP`` code at the end of the path.
- ``bezier.concatenate_paths`` has been removed; use
  ``Path.make_compound_path()`` instead.
- ``cbook.local_over_kwdict`` has been removed; use `.cbook.normalize_kwargs`
  instead.
- ``qt_compat.is_pyqt5`` has been removed due to the release of PyQt6. The Qt
  version can be checked using ``QtCore.qVersion()``.
- ``testing.compare.make_external_conversion_command`` has been removed.
- ``axes_grid1.axes_rgb.imshow_rgb`` has been removed; use
  ``imshow(np.dstack([r, g, b]))`` instead.

Arguments
~~~~~~~~~

- The *s* parameter to `.Axes.annotate` and  `.pyplot.annotate` is no longer
  supported; use the new name *text*.
- The *inframe* parameter to `.Axes.draw` has been removed; use
  `.Axes.redraw_in_frame` instead.
- The *required*, *forbidden* and *allowed* parameters of
  `.cbook.normalize_kwargs` have been removed.
- The *ismath* parameter of the ``draw_tex`` method of all renderer classes has
  been removed (as a call to ``draw_tex`` — not to be confused with
  ``draw_text``! — means that the entire string should be passed to the
  ``usetex`` machinery anyways). Likewise, the text machinery will no longer
  pass the *ismath* parameter when calling ``draw_tex`` (this should only
  matter for backend implementers).
- The *quality*, *optimize*, and *progressive* parameters of `.Figure.savefig`
  (which only affected JPEG output) have been removed, as well as from the
  corresponding ``print_jpg`` methods. JPEG output options can be set by
  directly passing the relevant parameters in *pil_kwargs*.
- The *clear_temp* parameter of `.FileMovieWriter` has been removed; files
  placed in a temporary directory (using ``frame_prefix=None``, the default)
  will be cleared; files placed elsewhere will not.
- The *copy* parameter of ``mathtext.Glue`` has been removed.
- The *quantize* parameter of `.Path.cleaned()` has been removed.
- The *dummy* parameter of `.RendererPgf` has been removed.
- The *props* parameter of `.Shadow` has been removed; use keyword arguments
  instead.
- The *recursionlimit* parameter of `matplotlib.test` has been removed.
- The *label* parameter of `.Tick` has no effect and has been removed.
- `~.ticker.MaxNLocator` no longer accepts a positional parameter and the
  keyword argument *nbins* simultaneously because they specify the same
  quantity.
- The *add_all* parameter to ``axes_grid.Grid``, ``axes_grid.ImageGrid``,
  ``axes_rgb.make_rgb_axes``, and ``axes_rgb.RGBAxes`` have been removed; the
  APIs always behave as if ``add_all=True``.
- The *den* parameter of ``axisartist.angle_helper.LocatorBase`` has been
  removed; use *nbins* instead.

- The *s* keyword argument to `.AnnotationBbox.get_fontsize` has no effect and
  has been removed.
- The *offset_position* keyword argument of the `.Collection` class has been
  removed; the ``offset_position`` now "screen".
- Arbitrary keyword arguments to ``StreamplotSet`` have no effect and have been
  removed.

- The *fontdict* and *minor* parameters of `.Axes.set_xticklabels` /
  `.Axes.set_yticklabels` are now keyword-only.
- All parameters of `.Figure.subplots` except *nrows* and *ncols* are now
  keyword-only; this avoids typing e.g. ``subplots(1, 1, 1)`` when meaning
  ``subplot(1, 1, 1)``, but actually getting ``subplots(1, 1, sharex=1)``.
- All parameters of `.pyplot.tight_layout` are now keyword-only, to be
  consistent with `.Figure.tight_layout`.
- ``ColorbarBase`` only takes a single positional argument now, the ``Axes`` to
  create it in, with all other options required to be keyword arguments. The
  warning for keyword arguments that were overridden by the mappable is now
  removed.

- Omitting the *renderer* parameter to `.Axes.draw` is no longer supported; use
  ``axes.draw_artist(axes)`` instead.
- Passing ``ismath="TeX!"`` to `.RendererAgg.get_text_width_height_descent` is
  no longer supported; pass ``ismath="TeX"`` instead,
- Changes to the signature of the `.Axes.draw` method make it consistent with
  all other artists; thus additional parameters to `.Artist.draw` have also
  been removed.

rcParams
~~~~~~~~

- The ``animation.avconv_path`` and ``animation.avconv_args`` rcParams have
  been removed.
- The ``animation.html_args`` rcParam has been removed.
- The ``keymap.all_axes`` rcParam has been removed.
- The ``mathtext.fallback_to_cm`` rcParam has been removed. Use
  :rc:`mathtext.fallback` instead.
- The ``savefig.jpeg_quality`` rcParam has been removed.
- The ``text.latex.preview`` rcParam has been removed.
- The following deprecated rcParams validators, defined in `.rcsetup`, have
  been removed:

  - ``validate_alignment``
  - ``validate_axes_titlelocation``
  - ``validate_axis_locator``
  - ``validate_bool_maybe_none``
  - ``validate_fontset``
  - ``validate_grid_axis``
  - ``validate_hinting``
  - ``validate_legend_loc``
  - ``validate_mathtext_default``
  - ``validate_movie_frame_fmt``
  - ``validate_movie_html_fmt``
  - ``validate_movie_writer``
  - ``validate_nseq_float``
  - ``validate_nseq_int``
  - ``validate_orientation``
  - ``validate_pgf_texsystem``
  - ``validate_ps_papersize``
  - ``validate_svg_fontset``
  - ``validate_toolbar``
  - ``validate_webagg_address``

- Some rcParam validation has become stricter:

  - :rc:`axes.axisbelow` no longer accepts strings starting with "line"
    (case-insensitive) as "line"; use "line" (case-sensitive) instead.
  - :rc:`text.latex.preamble` and :rc:`pdf.preamble` no longer accept
    non-string values.
  - All ``*.linestyle`` rcParams no longer accept ``offset = None``; set the
    offset to 0 instead.
