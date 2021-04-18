Removals
--------
The following deprecated APIs have been removed:

Removed behaviour
~~~~~~~~~~~~~~~~~

- The "smart bounds" functionality on `~.axis.Axis` and `.Spine` has been
  deleted, and the related methods have been removed.
- Converting a string with single color characters (e.g. ``'cymk'``) in
  `~.colors.to_rgba_array` is no longer supported. Instead, the colors can be
  passed individually in a list (e.g. ``['c', 'y', 'm', 'k']``).
- Returning a factor equal to ``None`` from ``mpl_toolkits.axisartist``
  Locators (which are **not** the same as "standard" tick Locators), or passing
  a factor equal to ``None`` to axisartist Formatters (which are **not** the
  same as "standard" tick Formatters) is no longer supported. Pass a factor
  equal to 1 instead.

Modules
~~~~~~~

- The entire ``matplotlib.testing.disable_internet`` module has been removed.
  The `pytest-remotedata package
  <https://github.com/astropy/pytest-remotedata>`_ can be used instead.
- The ``mpl_toolkits.axes_grid1.colorbar`` module and its colorbar
  implementation have been removed in favor of `matplotlib.colorbar`.

Classes, methods and attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The `.animation.MovieWriterRegistry` methods ``.set_dirty()``,
  ``.ensure_not_dirty()``, and ``.reset_available_writers()`` do nothing and
  have been removed.  The ``.avail()`` method has been removed; use ``.list()``
  instead to get a list of available writers.
- The ``matplotlib.artist.Artist.eventson`` and
  ``matplotlib.container.Container.eventson`` attributes have no effect and
  have been removed.
- ``matplotlib.axes.Axes.get_data_ratio_log`` has been removed.
- ``matplotlib.axes.SubplotBase.rowNum``; use
  ``ax.get_subplotspec().rowspan.start`` instead.
- ``matplotlib.axes.SubplotBase.colNum``; use
  ``ax.get_subplotspec().colspan.start`` instead.
- ``matplotlib.axis.Axis.set_smart_bounds`` and
  ``matplotlib.axis.Axis.get_smart_bounds`` have been removed.
- ``matplotlib.colors.DivergingNorm`` has been renamed to
  `~matplotlib.colors.TwoSlopeNorm`.
- ``matplotlib.figure.AxesStack`` has been removed.
- ``matplotlib.font_manager.JSONEncoder`` has been removed; use
  `.font_manager.json_dump` to dump a `.FontManager` instance.
- The ``matplotlib.ft2font.FT2Image`` methods ``.as_array()``,
  ``.as_rgba_str()``, ``.as_str()``, ``.get_height()`` and ``.get_width()``
  have been removed. Convert the ``FT2Image`` to a NumPy array with
  ``np.asarray`` before processing it.
- ``matplotlib.quiver.QuiverKey.quiverkey_doc`` has been removed; use
  ``matplotlib.quiver.QuiverKey.__init__.__doc__`` instead.
- ``matplotlib.spines.Spine.set_smart_bounds`` and
  ``matplotlib.spines.Spine.get_smart_bounds`` have been removed.
- ``matplotlib.testing.jpl_units.UnitDbl.checkUnits`` has been removed; use
  ``units not in self.allowed`` instead.
- The unused ``matplotlib.ticker.Locator.autoscale`` method has been removed
  (pass the axis limits to `.Locator.view_limits` instead). The derived methods
  ``Locator.autoscale``, ``AutoDateLocator.autoscale``,
  ``RRuleLocator.autoscale``, ``RadialLocator.autoscale``,
  ``ThetaLocator.autoscale``, and ``YearLocator.autoscale`` have also been
  removed.
- ``matplotlib.transforms.BboxBase.is_unit`` has been removed; check the
  `.Bbox` extents if needed.
- ``matplotlib.transforms.Affine2DBase.matrix_from_values(...)`` has been
  removed; use (for example) ``Affine2D.from_values(...).get_matrix()``
  instead.

* ``matplotlib.backend_bases.FigureCanvasBase.draw_cursor`` has been removed.
* ``matplotlib.backends.backend_gtk.ConfigureSubplotsGTK3.destroy`` and
  ``matplotlib.backends.backend_gtk.ConfigureSubplotsGTK3.init_window`` methods
  have been removed.
* ``matplotlib.backends.backend_gtk.ConfigureSubplotsGTK3.window`` property has
  been removed.
* ``matplotlib.backends.backend_macosx.FigureCanvasMac.invalidate`` has been
  removed.
* ``matplotlib.backends.backend_pgf.RendererPgf.latexManager`` has been removed.
* ``matplotlib.backends.backend_wx.FigureFrameWx.statusbar``,
  ``matplotlib.backends.backend_wx.NavigationToolbar2Wx.set_status_bar``, and
  ``matplotlib.backends.backend_wx.NavigationToolbar2Wx.statbar`` have been
  removed. The status bar can be retrieved by calling standard wx methods
  (``frame.GetStatusBar()`` and
  ``toolbar.GetTopLevelParent().GetStatusBar()``).
* ``matplotlib.backends.backend_wx.ConfigureSubplotsWx.configure_subplots`` and
  ``matplotlib.backends.backend_wx.ConfigureSubplotsWx.get_canvas`` have been
  removed.


- ``mpl_toolkits.axisartist.grid_finder.GridFinderBase`` has been removed; use
  `.GridFinder` instead.
- ``mpl_toolkits.axisartist.axis_artist.BezierPath`` has been removed; use
  `.patches.PathPatch` instead.

Functions
~~~~~~~~~

- ``matplotlib.backends.backend_pgf.repl_escapetext`` and
  ``matplotlib.backends.backend_pgf.repl_mathdefault`` have been removed.
- ``matplotlib.checkdep_ps_distiller`` has been removed.
- ``matplotlib.cm.revcmap`` has been removed; use `.Colormap.reversed`
  instead.
- ``matplotlib.colors.makeMappingArray`` has been removed.
- ``matplotlib.compare_versions`` has been removed; use comparison of
  ``distutils.version.LooseVersion``\s instead.
- ``matplotlib.dates.mx2num`` has been removed.
- ``matplotlib.font_manager.createFontList`` has been removed;
  `.font_manager.FontManager.addfont` is now available to register a font at a
  given path.
- ``matplotlib.get_home`` has been removed; use standard library instead.
- ``matplotlib.mlab.apply_window`` and ``matplotlib.mlab.stride_repeat`` have
  been removed.
- ``matplotlib.rcsetup.update_savefig_format`` has been removed; this just
  replaced ``'auto'`` with ``'png'``, so do the same.
- ``matplotlib.rcsetup.validate_animation_writer_path`` has been removed.
- ``matplotlib.rcsetup.validate_path_exists`` has been removed; use
  `os.path.exists` or `pathlib.Path.exists` instead.
- ``matplotlib.style.core.is_style_file`` and
  ``matplotlib.style.core.iter_style_files`` have been removed.
- ``matplotlib.testing.is_called_from_pytest`` has been removed.
- ``mpl_toolkits.mplot3d.axes3d.unit_bbox`` has been removed; use `.Bbox.unit`
  instead.


Arguments
~~~~~~~~~

- Passing more than one positional argument to `.axes.Axes.axis` will now
  raise an error.
- Passing ``"range"`` to the *whis* parameter of `.Axes.boxplot` and
  `.cbook.boxplot_stats` to mean "the whole data range" is  no longer
  supported.
- Passing scalars to the *where* parameter in `.axes.Axes.fill_between` and
  `.axes.Axes.fill_betweenx` is no longer accepted and non-matching sizes now
  raise a `ValueError`.
- The *verts* parameter to `.Axes.scatter` has been removed; use *marker* instead.
- The *minor* parameter in `.Axis.set_ticks` and ``SecondaryAxis.set_ticks`` is
  now keyword-only.
- `.scale.ScaleBase`, `.scale.LinearScale` and `.scale.SymmetricalLogScale` now
  error if any unexpected keyword arguments are passed to their constructors.
- The *renderer* parameter to `.Figure.tight_layout` has been removed; this
  method now always uses the renderer instance cached on the `.Figure`.
- The *locator* parameter to
  `mpl_toolkits.axes_grid1.axes_grid.CbarAxesBase.colorbar` has been removed in
  favor of its synonym *ticks* (which already existed previously,
  and is consistent with :mod:`matplotlib.colorbar`).
- The *switch_backend_warn* parameter to ``matplotlib.test`` has no effect and
  has been removed.
- The *dryrun* parameter to the various ``FigureCanvas*.print_*`` methods has
  been removed.

rcParams
~~~~~~~~

- The ``datapath`` rcParam has been removed. Use `matplotlib.get_data_path`
  instead.
- The ``mpl_toolkits.legacy_colorbar`` rcParam has no effect and has been
  removed.
- Setting :rc:`boxplot.whiskers` to ``"range"`` is no longer valid; set it to
  ``0, 100`` instead.
- Setting :rc:`savefig.format` to ``"auto"`` is no longer valid; use ``"png"``
  instead.
- Setting :rc:`text.hinting` to `False` or `True` is no longer valid; set it to
  ``"auto"`` or ``"none"`` respectively.

sample_data removals
~~~~~~~~~~~~~~~~~~~~
The sample datasets listed below have been removed.  Suggested replacements for
demonstration purposes are listed in parentheses.

- ``None_vs_nearest-pdf.png``,
- ``aapl.npz`` (use ``goog.npz``),
- ``ada.png``, ``grace_hopper.png`` (use ``grace_hopper.jpg``),
- ``ct.raw.gz`` (use ``s1045.ima.gz``),
- ``damodata.csv`` (use ``msft.csv``).
