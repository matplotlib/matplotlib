Removals
--------
The ``matplotlib.testing.determinism`` module, which exposes no public API, has
been deleted.

The following API elements have been removed:

- ``backend_gtk3.PIXELS_PER_INCH``
- ``backend_pgf.re_escapetext``, ``backend_pgf.re_mathdefault``.
- the ``matplotlib.backends.tkagg``, ``matplotlib.backends.windowing``,
  ``matplotlib.backends.wx_compat``, and ``matplotlib.compat.subprocess``
  modules
- ``RcParams.msg_depr``, ``RcParams.msg_depr_ignore``,
  ``RcParams.msg_depr_set``, ``RcParams.msg_obsolete``,
  ``RcParams.msg_backend_obsolete``
- ``afm.parse_afm`` (use ``afm.AFM instead``)
- ``axes.Axes.mouseover_set``
- ``backend_cairo.ArrayWrapper``, ``backend_cairo.RendererCairo.convert_path``
- ``backend_gtk3.FileChooserDialog.sorted_filetypes`` (use
  ``sorted(self.filetypes.items())`` instead)
- ``backend_pgf.get_texcommand``
- ``backend_pdf.PdfFile.texFontMap``
- ``backend_ps.get_bbox``
- ``backend_qt.FigureCanvasQt.keyAutoRepeat`` (use
  ``event.guiEvent.isAutoRepeat`` instead), ``backend_qt.error_msg_qt``,
  ``backend_qt.exception_handler``
- ``backend_wx.FigureCanvasWx.macros``
- ``backends.pylab_setup``
- ``cbook.Bunch`` (use ``types.SimpleNamespace`` instead), ``cbook.Locked``,
  ``cbook.unicode_safe``, ``cbook.is_numlike`` (use
  ``isinstance(..., numbers.Number)`` instead), ``cbook.mkdirs`` (use
  ``os.makedirs(..., exist_ok=True)`` instead), ``cbook.GetRealpathAndStat``
  (use ``cbook.get_realpath_and_stat`` instead),
  ``cbook.listFiles``
- ``container.Container.set_remove_method``
- ``contour.ContourLabeler.cl``, ``contour.ContourLabeler.cl_xy``,
  ``contour.ContourLabeler.cl_cvalues`` (use ``labelTexts``, ``labelXYs``,
  ``labelCValues`` instead)
- ``dates.DateFormatter.strftime``, ``dates.DateFormatter.strftime_pre_1900``
- ``font_manager.TempCache``, ``font_manager.FontManager.ttffiles``,
  ``font_manager.FontManager.afmfiles``
- ``mathtext.unichr_safe`` (use ``chr`` instead)
- ``patches.YAArrow`` (use ``patches.FancyArrowPatch`` instead)
- ``sphinxext.plot_directive.remove_coding``
- ``table.Table.get_child_artists``
- ``testing.compare.compare_float``, ``testing.decorators.CleanupTest``,
  ``testing.decorators.ImageComparisonTest``,
  ``testing.decorators.skip_if_command_unavailable``,
  support for nose-based tests
- ``text.Annotation.arrow`` (use ``text.Annotation.arrow_patch`` instead)
- ``textpath.TextToPath.tex_font_map``
- ``ticker.Base``, ``ticker.closeto``, ``ticker.nearest_long``
- ``axes_grid1.axes_divider.LocatableAxesBase``,
  ``axes_grid1.axes_divider.locatable_axes_factory``,
  ``axes_grid1.axes_divider.Axes`` (use ``axes_grid1.mpl_axes.Axes`` instead),
  ``axes_grid1.axes_divider.LocatableAxes`` (use ``axes_grid1.mpl_axes.Axes``
  instead)
- ``axisartist.axes_divider.Axes``, ``axisartist.axes_divider.LocatableAxes``
  (use ``axisartist.axislines.Axes`` instead)
- the *normed* keyword argument to ``hist`` (use *density* instead)
- passing ``(verts, 0)`` or ``(..., 3)`` when specifying a marker to specify a
  path or a circle, respectively (instead, use ``verts`` or ``"o"``,
  respectively)
- :rc:`examples.directory`

The following members of ``matplotlib.backends.backend_pdf.PdfFile`` were removed:

- ``nextObject``
- ``nextFont``
- ``nextAlphaState``
- ``nextHatch``
- ``nextImage``
- ``alphaStateObject``

The ``required_interactive_framework`` attribute of backend modules introduced
in Matplotlib 3.0 has been moved to the ``FigureCanvas`` class, in order to
let it be inherited by third-party canvas subclasses and to make it easier to
know what interactive framework is required by a canvas class.

``backend_qt4.FigureCanvasQT5``, which is an alias for
``backend_qt5.FigureCanvasQT`` (but only exists under that name in
``backend_qt4``), has been removed.

