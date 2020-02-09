Removals
--------
The following deprecated APIs have been removed:

Modules
~~~~~~~
- ``backends.qt_editor.formlayout`` (use the formlayout module available on
  PyPI instead).

Classes and methods
~~~~~~~~~~~~~~~~~~~
- ``backend_bases.RendererBase.strip_math()``
  (use ``cbook.strip_math()`` instead)

- ``backend_wx.debug_on_error()`` (no replacement)
- ``backend_wx.raise_msg_to_str()`` (no replacement)
- ``backend_wx.fake_stderr`` (no replacement)
- ``backend_wx.MenuButtonWx`` (no replacement)
- ``backend_wx.PrintoutWx`` (no replacement)
- ``_backend_tk.NavigationToolbar2Tk.set_active()`` (no replacement)

- ``backend_ps.PsBackendHelper.gs_exe`` property (no replacement)
- ``backend_ps.PsBackendHelper.gs_version`` property (no replacement)
- ``backend_ps.PsBackendHelper.supports_ps2write`` property (no replacement)
- ``backend_ps.RendererPS.afmfontd`` property (no replacement)
- ``backend_ps.GraphicsContextPS.shouldstroke`` property (no replacement)

- ``backend_gtk3.FileChooserDialog`` (no replacement)
- ``backend_gtk3.SaveFigureGTK3.get_filechooser()`` (no replacement)
- ``backend_gtk3.NavigationToolbar2GTK3.get_filechooser()`` (no replacement)

- ``backend_gtk3cairo.FigureManagerGTK3Cairo``
  (use ``backend_gtk3.FigureManagerGTK3`` instead)

- ``backend_pdf.RendererPdf.afm_font_cache`` property (no replacement)

- ``backend_pgf.LatexManagerFactory`` (no replacement)

- ``backend_qt5.NavigationToolbar2QT.buttons`` property (no replacement)
- ``backend_qt5.NavigationToolbar2QT.adj_window`` property (no replacement)

- ``matplotlib.checkdep_dvipng`` (no replacement)
- ``matplotlib.checkdep_ghostscript`` (no replacement)
- ``matplotlib.checkdep_pdftops`` (no replacement)
- ``matplotlib.checkdep_inkscape`` (no replacement)
- ``matplotlib.get_py2exe_datafiles`` (no replacement)
- ``matplotlib.tk_window_focus`` (use ``rcParams['tk.window_focus']`` instead)

- ``pyplot.plotfile()`` (Instead, load the data using
  `pandas.read_csv` or `numpy.loadtxt` or similar and use regular pyplot
  functions to plot the loaded data.)
- ``rcsetup.validate_qt4()`` (no replacement)
- ``rcsetup.validate_qt5()`` (no replacement)
- ``rcsetup.validate_verbose()`` (no replacement)
- ``rcsetup.ValidateInterval`` (no replacement)

- ``sphinxext.plot_directive.plot_directive()``
  (use the class ``PlotDirective`` instead)

- ``Artist.aname`` property (no replacement)
- ``Axis.iter_ticks`` (no replacement)

- ``image.BboxImage.interp_at_native`` property (no replacement)
- ``lines.Line2D.verticalOffset`` property (no replacement)
- ``bezier.find_r_to_boundary_of_closedpath()`` (no relacement)

- ``quiver.Quiver.color()`` (use ``Quiver.get_facecolor()`` instead)
- ``quiver.Quiver.keyvec`` property (no replacement)
- ``quiver.Quiver.keytext`` property (no replacement)

- ``colorbar.ColorbarBase.get_cmap`` (use ``ScalarMappable.get_cmap`` instead)
- ``colorbar.ColorbarBase.set_cmap`` (use ``ScalarMappable.set_cmap`` instead)
- ``colorbar.ColorbarBase.get_clim`` (use ``ScalarMappable.get_clim`` instead)
- ``colorbar.ColorbarBase.set_clim`` (use ``ScalarMappable.set_clim`` instead)
- ``colorbar.ColorbarBase.set_norm`` (use ``ScalarMappable.set_norm`` instead)

- ``text.TextWithDash`` (use ``text.Annotation`` instead)

- ``mplot3d.proj3d.line2d()`` (no replacement)
- ``mplot3d.proj3d.line2d_dist()`` (no replacement)
- ``mplot3d.proj3d.line2d_seg_dist()`` (no replacement)
- ``mplot3d.proj3d.mod()`` (use `numpy.linalg.norm` instead)
- ``mplot3d.proj3d.proj_transform_vec()`` (no replacement)
- ``mplot3d.proj3d.proj_transform_vec_clip()`` (no replacement)
- ``mplot3d.proj3d.vec_pad_ones()`` (no replacement)
- ``mplot3d.proj3d.proj_trans_clip_points()`` (no replacement)

- ``mplot3d.art3d.norm_angle()`` (no replacement)
- ``mplot3d.art3d.norm_text_angle()`` (no replacement)
- ``mplot3d.art3d.path_to_3d_segment()`` (no replacement)
- ``mplot3d.art3d.paths_to_3d_segments()`` (no replacement)
- ``mplot3d.art3d.path_to_3d_segment_with_codes()`` (no replacement)
- ``mplot3d.art3d.paths_to_3d_segments_with_codes()`` (no replacement)
- ``mplot3d.art3d.get_patch_verts()`` (no replacement)
- ``mplot3d.art3d.get_colors()`` (no replacement)
- ``mplot3d.art3d.zalpha()`` (no replacement)

- ``mplot3d.axis3d.get_flip_min_max()`` (no replacement)
- ``mplot3d.axis3d.Axis.get_tick_positions()`` (no replacement)

- ``axisartist.axis_artist.UnimplementedException`` (no replacement)
- ``axisartist.axislines.SimpleChainedObjects``
  (use ``axis_grid1.mpl_axes.SimpleChainedObjects`` instead)
- ``axisartist.axislines.Axes.AxisDict``
  (use ``axis_grid1.mpl_axes.Axes.AxisDict`` instead)

Arguments
~~~~~~~~~
- ``Axes.text()`` / ``pyplot.text()`` do not support the parameter ``withdash``
  anymore. Use ``Axes.annotate()`` and ``pyplot.annotate()`` instead.
- The first parameter of `matplotlib.use` has been renamed from ``arg`` to
  ``backend`` (only relevant if you pass by keyword).
- The parameter ``warn`` of `matplotlib.use` has been removed. A failure to
  switch the backend will now always raise an ``ImportError`` if ``force`` is
  set; catch that error if necessary.
- All parameters of `matplotlib.use` except the first one are now keyword-only.
- The unused parameters ``shape`` and ``imlim`` of `~.axes.Axes.imshow()` are
  now removed. All parameters beyond ``extent`` are now keyword-only.
- The unused parameter ``interp_at_native`` of `.BboxImage` has been removed.
- The parameter ``usetex`` of `.TextToPath.get_text_path` has been removed.
  Use ``ismath='TeX'`` instead.
- The parameter ``block`` of ``show()`` is now keyword-only.
- The parameter ``frameon`` of `.Figure.savefig` has been removed.  Use
  ``facecolor="none"`` to get a transparent background.

rcParams
~~~~~~~~
- The ``text.latex.unicode`` rcParam has been removed, with no replacement.
  Matplotlib now always supports unicode in usetex.
- The ``savefig.frameon`` rcParam has been removed.  Set
  :rc:`savefig.facecolor` to "none" to get a transparent background.
- The ``pgf.debug``, ``verbose.fileo`` and ``verbose.verbose.level`` rcParams,
  which had no effect, have been removed.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
- ``MATPLOTLIBDATA`` (no replacement).

mathtext
~~~~~~~~
- The ``\stackrel`` command (which behaved differently from its LaTeX version)
  has been removed.  Use ``\genfrac`` instead.
