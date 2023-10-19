
Deprecations
------------

`matplotlib.use`
~~~~~~~~~~~~~~~~
The ``warn`` parameter to `matplotlib.use()` is deprecated (catch the
`ImportError` emitted on backend switch failure and reemit a warning yourself
if so desired).

plotfile
~~~~~~~~
``.pyplot.plotfile`` is deprecated in favor of separately loading and plotting
the data.  Use pandas or NumPy to load data, and pandas or matplotlib to plot
the resulting data.

axes and axis
~~~~~~~~~~~~~
Setting ``Axis.major.locator``, ``Axis.minor.locator``, ``Axis.major.formatter``
or ``Axis.minor.formatter`` to an object that is not a subclass of `.Locator` or
`.Formatter` (respectively) is deprecated.  Note that these attributes should
usually be set using `.Axis.set_major_locator`, `.Axis.set_minor_locator`, etc.
which already raise an exception when an object of the wrong class is passed.

Passing more than one positional argument or unsupported keyword arguments to
`~matplotlib.axes.Axes.axis()` is deprecated (such arguments used to be
silently ignored).

``minor`` argument will become keyword-only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Using the parameter ``minor`` to ``get_*ticks()`` / ``set_*ticks()`` as a
positional parameter is deprecated. It will become keyword-only in future
versions.

``axes_grid1``
~~~~~~~~~~~~~~
The ``mpl_toolkits.axes_grid1.colorbar`` module and its colorbar implementation
are deprecated in favor of :mod:`matplotlib.colorbar`, as the former is
essentially abandoned and the latter is a more featureful replacement with a
nearly compatible API (for example, the following additional keywords are
supported: ``panchor``, ``extendfrac``, ``extendrect``).

The main differences are:

- Setting the ticks on the colorbar is done by calling ``colorbar.set_ticks``
  rather than ``colorbar.cbar_axis.set_xticks`` or
  ``colorbar.cbar_axis.set_yticks``; the ``locator`` parameter to ``colorbar()``
  is deprecated in favor of its synonym ``ticks`` (which already existed
  previously, and is consistent with :mod:`matplotlib.colorbar`).
- The colorbar's long axis is accessed with ``colorbar.xaxis`` or
  ``colorbar.yaxis`` depending on the orientation, rather than
  ``colorbar.cbar_axis``.
- The default ticker is no longer ``MaxNLocator(5)``, but a
  ``_ColorbarAutoLocator``.
- Overdrawing multiple colorbars on top of one another in a single Axes (e.g.
  when using the ``cax`` attribute of `~.axes_grid1.axes_grid.ImageGrid`
  elements) is not supported; if you previously relied on the second colorbar
  being drawn over the first, you can call ``cax.cla()`` to clear the axes
  before drawing the second colorbar.

During the deprecation period, the ``mpl_toolkits.legacy_colorbar``
rcParam can be set to True to use ``mpl_toolkits.axes_grid1.colorbar`` in
:mod:`mpl_toolkits.axes_grid1` code with a deprecation warning (the default),
or to False to use ``matplotlib.colorbar``.

Passing a ``pad`` size of ``None`` (the default) as a synonym for zero to
the ``append_axes``, ``new_horizontal`` and ``new_vertical`` methods of
`.axes_grid1.axes_divider.AxesDivider` is deprecated.  In a future release, the
default value of ``None`` will mean "use :rc:`figure.subplot.wspace` or
:rc:`figure.subplot.hspace`" (depending on the orientation).  Explicitly pass
``pad=0`` to keep the old behavior.

Axes3D
~~~~~~
``mplot3d.axis3d.get_flip_min_max`` is deprecated.

``axes3d.unit_bbox`` is deprecated (use ``Bbox.unit`` instead).

``axes3d.Axes3D.w_xaxis``, ``.w_yaxis``, and ``.w_zaxis`` are deprecated (use
``.xaxis``, ``.yaxis``, and ``.zaxis`` instead).

`matplotlib.cm`
~~~~~~~~~~~~~~~
``cm.revcmap`` is deprecated.  Use `.Colormap.reversed` to reverse a colormap.

``cm.datad`` no longer contains entries for reversed colormaps in their
"unconverted" form.

axisartist
~~~~~~~~~~
``mpl_toolkits.axisartist.grid_finder.GridFinderBase`` is deprecated (its
only use is to be inherited by the `.GridFinder` class which just provides
more defaults in the constructor and directly sets the transforms, so
``GridFinderBase``'s methods were just moved to `.GridFinder`).

``axisartist.axis_artist.BezierPath`` is deprecated (use `.patches.PathPatch`
to draw arbitrary Paths).

``AxisArtist.line`` is now a `.patches.PathPatch` instance instead of a
``BezierPath`` instance.

Returning a factor equal to None from axisartist Locators (which are **not**
the same as "standard" tick Locators), or passing a factor equal to None
to axisartist Formatters (which are **not** the same as "standard" tick
Formatters) is deprecated.  Pass a factor equal to 1 instead.

For the `mpl_toolkits.axisartist.axis_artist.AttributeCopier` class, the
constructor and the ``set_ref_artist`` method, and the *default_value*
parameter of ``get_attribute_from_ref_artist``, are deprecated.

Deprecation of the constructor means that classes inheriting from
`.AttributeCopier` should no longer call its constructor.

Locators
~~~~~~~~
The unused ``Locator.autoscale`` method is deprecated (pass the axis limits to
`.Locator.view_limits` instead).

Animation
~~~~~~~~~
The following methods and attributes of the `.MovieWriterRegistry` class are
deprecated: ``set_dirty``, ``ensure_not_dirty``, ``reset_available_writers``,
``avail``.

``smart_bounds()``
~~~~~~~~~~~~~~~~~~
The "smart_bounds" functionality is deprecated.  This includes
``Axis.set_smart_bounds()``, ``Axis.get_smart_bounds()``,
``Spine.set_smart_bounds()``, and ``Spine.get_smart_bounds()``.

``boxplot()``
~~~~~~~~~~~~~
Setting the ``whis`` parameter of `.Axes.boxplot` and `.cbook.boxplot_stats` to
"range" to mean "the whole data range" is deprecated; set it to (0, 100) (which
gets interpreted as percentiles) to achieve the same effect.

``fill_between()``
~~~~~~~~~~~~~~~~~~
Passing scalars to parameter *where* in ``fill_between()`` and
``fill_betweenx()`` is deprecated. While the documentation already states that
*where* must be of the same size as *x* (or *y*), scalars were accepted and
broadcasted to the size of *x*. Non-matching sizes will raise a ``ValueError``
in the future.

``scatter()``
~~~~~~~~~~~~~
Passing the *verts* parameter to `.axes.Axes.scatter` is deprecated; use the
*marker* parameter instead.

``tight_layout()``
~~~~~~~~~~~~~~~~~~
The ``renderer`` parameter to `.Figure.tight_layout` is deprecated; this method
now always uses the renderer instance cached on the `.Figure`.

rcParams
~~~~~~~~
The ``rcsetup.validate_animation_writer_path`` function is deprecated.

Setting :rc:`savefig.format` to "auto" is deprecated; use its synonym "png" instead.

Setting :rc:`text.hinting` to True or False is deprecated; use their synonyms
"auto" or "none" instead.

``rcsetup.update_savefig_format`` is deprecated.

``rcsetup.validate_path_exists`` is deprecated (use ``os.path.exists`` to check
whether a path exists).

``rcsetup.ValidateInterval`` is deprecated.

Dates
~~~~~
``dates.mx2num`` is deprecated.

TK
~~
``NavigationToolbar2Tk.set_active`` is deprecated, as it has no (observable)
effect.

WX
~~
``FigureFrameWx.statusbar`` and ``NavigationToolbar2Wx.statbar`` are deprecated.
The status bar can be retrieved by calling standard wx methods
(``frame.GetStatusBar()`` and ``toolbar.GetTopLevelParent().GetStatusBar()``).

``backend_wx.ConfigureSubplotsWx.configure_subplots`` and
``backend_wx.ConfigureSubplotsWx.get_canvas`` are deprecated.

PGF
~~~
``backend_pgf.repl_escapetext`` and ``backend_pgf.repl_mathdefault`` are
deprecated.

``RendererPgf.latexManager`` is deprecated.

FigureCanvas
~~~~~~~~~~~~
``FigureCanvasBase.draw_cursor`` (which has never done anything and has never
been overridden in any backend) is deprecated.

``FigureCanvasMac.invalidate`` is deprecated in favor of its synonym,
``FigureCanvasMac.draw_idle``.

The ``dryrun`` parameter to the various ``FigureCanvasFoo.print_foo`` methods
is deprecated.


QuiverKey doc
~~~~~~~~~~~~~
``quiver.QuiverKey.quiverkey_doc`` is deprecated; use
``quiver.QuiverKey.__init__.__doc__`` instead.

`matplotlib.mlab`
~~~~~~~~~~~~~~~~~
``mlab.apply_window`` and ``mlab.stride_repeat`` are deprecated.

Fonts
~~~~~
``font_manager.JSONEncoder`` is deprecated.  Use `.font_manager.json_dump` to
dump a `.FontManager` instance.

``font_manager.createFontList`` is deprecated.  `.font_manager.FontManager.addfont`
is now available to register a font at a given path.

The ``as_str``, ``as_rgba_str``, ``as_array``, ``get_width`` and ``get_height``
methods of ``matplotlib.ft2font.FT2Image`` are deprecated.  Convert the ``FT2Image``
to a NumPy array with ``np.asarray`` before processing it.

Colors
~~~~~~
The function ``matplotlib.colors.makeMappingArray`` is not considered part of
the public API any longer. Thus, it's deprecated.

Using a string of single-character colors as a color sequence (e.g. "rgb") is
deprecated. Use an explicit list instead.

Scales
~~~~~~
Passing unsupported keyword arguments to `.ScaleBase`, and its subclasses
`.LinearScale` and `.SymmetricalLogScale`, is deprecated and will raise a
`TypeError` in 3.3.

If extra keyword arguments are passed to `.LogScale`, `TypeError` will now be
raised instead of `ValueError`.

Testing
~~~~~~~
The ``matplotlib.testing.disable_internet`` module is deprecated.  Use (for
example) pytest-remotedata_ instead.

.. _pytest-remotedata: https://pypi.org/project/pytest-remotedata/

Support in `matplotlib.testing` for nose-based tests is deprecated (a
deprecation is emitted if using e.g. the decorators from that module while
both 1) matplotlib's conftests have not been called and 2) nose is in
``sys.modules``).

``testing.is_called_from_pytest`` is deprecated.

During the deprecation period, to force the generation of nose base tests,
import nose first.

The *switch_backend_warn* parameter to ``matplotlib.test`` has no effect and is
deprecated.

``testing.jpl_units.UnitDbl.UnitDbl.checkUnits`` is deprecated.

``DivergingNorm`` renamed to ``TwoSlopeNorm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DivergingNorm`` was a misleading name; although the norm was
developed with the idea that it would likely be used with diverging
colormaps, the word 'diverging' does not describe or evoke the norm's
mapping function.  Since that function is monotonic, continuous, and
piece-wise linear with two segments, the norm has been renamed to
`.TwoSlopeNorm`

Misc
~~~~
``matplotlib.get_home`` is deprecated (use e.g. ``os.path.expanduser("~")``)
instead.

``matplotlib.compare_versions`` is deprecated (use comparison of
``distutils.version.LooseVersion``\s instead).

``matplotlib.checkdep_ps_distiller`` is deprecated.

``matplotlib.figure.AxesStack`` is considered private API and will be removed
from the public API in future versions.

``BboxBase.is_unit`` is deprecated (check the Bbox extents if needed).

``Affine2DBase.matrix_from_values(...)`` is deprecated.  Use (for example)
``Affine2D.from_values(...).get_matrix()`` instead.

``style.core.is_style_file`` and ``style.core.iter_style_files``
are deprecated.

The ``datapath`` rcParam
~~~~~~~~~~~~~~~~~~~~~~~~
Use `.get_data_path` instead.  (The rcParam is deprecated because it cannot be
meaningfully set by an end user.)  The rcParam had no effect from 3.2.0, but
was deprecated only in 3.2.1.  In 3.2.1+ if ``'datapath'`` is set in a
``matplotlibrc`` file it will be respected, but this behavior will be removed in 3.3.
