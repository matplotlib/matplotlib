Deprecations
------------

``figure.add_axes()`` without arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Calling ``fig.add_axes()`` with no arguments currently does nothing. This call
will raise an error in the future. Adding a free-floating axes needs a position
rectangle. If you want a figure-filling single axes, use ``add_subplot()``
instead.

``backend_wx.DEBUG_MSG``
~~~~~~~~~~~~~~~~~~~~~~~~
``backend_wx.DEBUG_MSG`` is deprecated.  The wx backends now use regular
logging.

``Colorbar.config_axis()``
~~~~~~~~~~~~~~~~~~~~~~~~~~
``Colorbar.config_axis()`` is considered internal. Its use is deprecated.

``NonUniformImage.is_grayscale`` and ``PcolorImage.is_grayscale``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These attributes are deprecated, for consistency with ``AxesImage.is_grayscale``,
which was removed back in Matplotlib 2.0.0.  (Note that previously, these
attributes were only available *after rendering the image*).

``den`` parameter and attribute to :mod:`mpl_toolkits.axisartist.angle_helper`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For all locator classes defined in :mod:`mpl_toolkits.axisartist.angle_helper`,
the ``den`` parameter has been renamed to ``nbins``, and the ``den`` attribute
deprecated in favor of its (preexisting) synonym ``nbins``, for consistency
with locator classes defined in :mod:`matplotlib.ticker`.

``backend_pgf.LatexManager.latex_stdin_utf8``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``backend_pgf.LatexManager.latex`` is now created with ``encoding="utf-8"``, so
its ``stdin`` attribute is already utf8-encoded; the ``latex_stdin_utf8``
attribute is thus deprecated.

Flags containing "U" passed to `.cbook.to_filehandle` and `.cbook.open_file_cm`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please remove "U" from flags passed to `.cbook.to_filehandle` and
`.cbook.open_file_cm`.  This is consistent with their removal from `open` in
Python 3.9.

PDF and PS character tracking internals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``used_characters`` attribute and ``track_characters`` and
``merge_used_characters`` methods of `.RendererPdf`, `.PdfFile`, and
`.RendererPS` are deprecated.

Case-insensitive capstyles and joinstyles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please pass capstyles ("miter", "round", "bevel") and joinstyles ("butt",
"round", "projecting") as lowercase.

Passing raw data to ``register_cmap()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Passing raw data via parameters *data* and *lut* to `.register_cmap()` is
deprecated. Instead, explicitly create a `.LinearSegmentedColormap` and pass
it via the *cmap* parameter:
``register_cmap(cmap=LinearSegmentedColormap(name, data, lut))``.

``DateFormatter.illegal_s``
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This attribute is unused and deprecated.

``widgets.TextBox.params_to_disable``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This attribute is deprecated.

Revert deprecation \*min, \*max keyword arguments to ``set_x/y/zlim_3d()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These keyword arguments  were deprecated in 3.0, alongside with the respective
parameters in ``set_xlim()`` / ``set_ylim()``. The deprecations of the 2D
versions were already reverted in in 3.1.

``cbook.local_over_kwdict``
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This function is deprecated.  Use `.cbook.normalize_kwargs` instead.

Passing both singular and plural *colors*, *linewidths*, *linestyles* to `.Axes.eventplot`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Passing e.g. both *linewidth* and *linewidths* will raise a TypeError in the
future.

Setting :rc:`text.latex.preamble` or :rc:`pdf.preamble` to non-strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These rcParams should be set to string values.  Support for None (meaning the
empty string) and lists of strings (implicitly joined with newlines) is
deprecated.

Parameters *norm* and *vmin*/*vmax* should not be used simultaneously
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Passing parameters *norm* and *vmin*/*vmax* simultaneously to functions using
colormapping such as ``scatter()`` and ``imshow()`` is deprecated.
Inestead of ``norm=LogNorm(), vmin=min_val, vmax=max_val`` pass
``norm=LogNorm(min_val, max_val)``. *vmin* and *vmax* should only be used
without setting *norm*.

Effectless parameters of `.Figure.colorbar` and `matplotlib.colorbar.Colorbar`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The *cmap* and *norm* parameters of `.Figure.colorbar` and
`matplotlib.colorbar.Colorbar` have no effect because they are always
overridden by the mappable's colormap and norm; they are thus deprecated.
Likewise, passing the *alpha*, *boundaries*, *values*, *extend*, or *filled*
parameters with a `.ContourSet` mappable, or the *alpha* parameter with an
`.Artist` mappable, is deprecated, as the mappable would likewise override
them.

``args_key`` and ``exec_key`` attributes of builtin `.MovieWriter`\s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These attributes are deprecated.

Unused parameters
~~~~~~~~~~~~~~~~~
The following parameters do not have any effect and are deprecated:

- arbitrary keyword arguments to ``StreamplotSet``
- parameter *quantize* of `.Path.cleaned()`
- parameter *s* of `.AnnotationBbox.get_fontsize()`
- parameter *label* of `.Tick`

Passing *props* to `.Shadow`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The parameter *props* of `.Shadow` is deprecated. Use keyword arguments
instead.

``Axes.update_datalim_bounds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This method is deprecated.  Use
``ax.dataLim.set(Bbox.union([ax.dataLim, bounds]))`` instead.

``{,Symmetrical}LogScale.{,Inverted}LogTransform``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``LogScale.LogTransform``, ``LogScale.InvertedLogTransform``,
``SymmetricalScale.SymmetricalTransform`` and
``SymmetricalScale.InvertedSymmetricalTransform`` are deprecated.  Directly
access the transform classes from the :mod:`.scale` module.

``TexManager.cachedir``
~~~~~~~~~~~~~~~~~~~~~~~
Use `matplotlib.get_cachedir()` instead.

Setting `.Line2D`\'s pickradius via `.Line2D.set_picker`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setting a `.Line2D`\'s pickradius (i.e. the tolerance for pick events
and containment checks) via `.Line2D.set_picker` is deprecated.  Use
`.Line2D.set_pickradius` instead.

`.Line2D.set_picker` no longer sets the artist's custom-contain() check.

``Artist.set_contains``, ``Artist.get_contains``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setting a custom method overriding `.Artist.contains` is deprecated.
There is no replacement, but you may still customize pick events using
`.Artist.set_picker`.

`~matplotlib.colorbar.Colorbar` methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``on_mappable_changed`` and ``update_bruteforce`` methods of
`~matplotlib.colorbar.Colorbar` are deprecated; both can be replaced by calls
to `~matplotlib.colorbar.Colorbar.update_normal`.

``OldScalarFormatter``, ``IndexFormatter`` and ``DateIndexFormatter``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These formatters are deprecated.  Their functionality can be implemented using
e.g. `.FuncFormatter`.

``OldAutoLocator``
~~~~~~~~~~~~~~~~~~
This ticker is deprecated.

*required*, *forbidden* and *allowed* parameters of `.cbook.normalize_kwargs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These parameters are deprecated.

The ``TTFPATH`` and ``AFMPATH`` environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Support for the (undocumented) ``TTFPATH`` and ``AFMPATH`` environment
variables is deprecated.  Additional fonts may be registered using
``matplotlib.font_manager.fontManager.addfont()``.

``matplotlib.compat``
~~~~~~~~~~~~~~~~~~~~~
This module is deprecated.

``matplotlib.backends.qt_editor.formsubplottool``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module is deprecated.  Use ``matplotlib.backends.backend_qt5.SubplotToolQt``
instead.

AVConv animation writer deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``AVConvBase``, ``AVConvWriter`` and ``AVConvFileWriter`` classes, and the
associated ``animation.avconv_path`` and ``animation.avconv_args`` rcParams are
deprecated.

Debian 8 (2015, EOL 06/2020) and Ubuntu 14.04 (EOL 04/2019) were the
last versions of Debian and Ubuntu to ship avconv.  It remains possible
to force the use of avconv by using the ffmpeg-based writers with
:rc:`animation.ffmpeg_path` set to "avconv".

log/symlog scale base, ticks, and nonpos specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`~.Axes.semilogx`, `~.Axes.semilogy`, `~.Axes.loglog`, `.LogScale`, and
`.SymmetricalLogScale` used to take keyword arguments that depends on the axis
orientation ("basex" vs "basey", "subsx" vs "subsy", "nonposx" vs "nonposy");
these parameter names are now deprecated in favor of "base", "subs",
"nonpositive".  This deprecation also affects e.g. ``ax.set_yscale("log",
basey=...)`` which must now be spelled ``ax.set_yscale("log", base=...)``.

The change from "nonpos" to "nonpositive" also affects `~.scale.LogTransform`,
`~.scale.InvertedLogTransform`, `~.scale.SymmetricalLogTransform`, etc.

To use *different* bases for the x-axis and y-axis of a `~.Axes.loglog` plot,
use e.g. ``ax.set_xscale("log", base=10); ax.set_yscale("log", base=2)``.

``DraggableBase.artist_picker``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This method is deprecated.  If you previously reimplemented it in a subclass,
set the artist's picker instead with `.Artist.set_picker`.

*clear_temp* parameter and attribute of `.FileMovieWriter`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The *clear_temp* parameter and attribute of `.FileMovieWriter` is
deprecated.  In the future, files placed in a temporary directory (using
``frame_prefix=None``, the default) will be cleared; files placed elsewhere
will not.

Deprecated rcParams validators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following validators, defined in `.rcsetup`, are deprecated:
``validate_fontset``, ``validate_mathtext_default``, ``validate_alignment``,
``validate_svg_fontset``, ``validate_pgf_texsystem``,
``validate_movie_frame_fmt``, ``validate_axis_locator``,
``validate_movie_html_fmt``, ``validate_grid_axis``,
``validate_axes_titlelocation``, ``validate_toolbar``,
``validate_ps_papersize``, ``validate_legend_loc``,
``validate_bool_maybe_none``, ``validate_hinting``,
``validate_movie_writers``, ``validate_webagg_address``.
To test whether an rcParam value would be acceptable, one can test e.g. ``rc =
RcParams(); rc[k] = v`` raises an exception.

Stricter rcParam validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:rc:`axes.axisbelow` currently normalizes all strings starting with "line"
(case-insensitive) to the option "line".  This is deprecated; in a future
version only the exact string "line" (case-sensitive) will be supported.

``add_subplot()`` validates its inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In particular, for ``add_subplot(rows, cols, index)``, all parameters must
be integral. Previously strings and floats were accepted and converted to
int. This will now emit a deprecation warning.

Toggling axes navigation from the keyboard using "a" and digit keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Axes navigation can still be toggled programmatically using
`.Axes.set_navigate`.

The following related APIs are also deprecated:
``backend_tools.ToolEnableAllNavigation``,
``backend_tools.ToolEnableNavigation``, and ``rcParams["keymap.all_axes"]``.

``matplotlib.test(recursionlimit=...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The *recursionlimit* parameter of ``matplotlib.test`` is deprecated.

mathtext glues
~~~~~~~~~~~~~~
The *copy* parameter of ``mathtext.Glue`` is deprecated (the underlying glue
spec is now immutable).  ``mathtext.GlueSpec`` is deprecated.

Signatures of `.Artist.draw` and `.Axes.draw`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The *inframe* parameter to `.Axes.draw` is deprecated.  Use
`.Axes.redraw_in_frame` instead.

Not passing the *renderer* parameter to `.Axes.draw` is deprecated.  Use
``axes.draw_artist(axes)`` instead.

These changes make the signature of the ``draw`` (``artist.draw(renderer)``)
method consistent across all artists; thus, additional parameters to
`.Artist.draw` are deprecated.

``DraggableBase.on_motion_blit``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This method is deprecated.  `.DraggableBase.on_motion` now handles both the
blitting and the non-blitting cases.

Passing the dash offset as None
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fine control of dash patterns can be achieved by passing an ``(offset,
(on-length, off-length, on-length, off-length, ...))`` pair as the linestyle
property of `.Line2D` and `.LineCollection`.  Previously, certain APIs would
accept ``offset = None`` as a synonym for ``offset = 0``, but this was never
universally implemented, e.g. for vector output.  Support for ``offset = None``
is deprecated, set the offset to 0 instead.

``RendererCairo.fontweights``, ``RendererCairo.fontangles``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... are deprecated.

``autofmt_xdate(which=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is deprecated, use its more explicit synonym, ``which="major"``, instead.

JPEG options
~~~~~~~~~~~~
The *quality*, *optimize*, and *progressive* keyword arguments to
`~.Figure.savefig`, which were only used when saving to JPEG, are deprecated.
:rc:`savefig.jpeg_quality` is likewise deprecated.

Such options should now be directly passed to Pillow using
``savefig(..., pil_kwargs={"quality": ..., "optimize": ..., "progressive": ...})``.

``dviread.Encoding``
~~~~~~~~~~~~~~~~~~~~
This class was (mostly) broken and is deprecated.

Axis and Locator ``pan`` and ``zoom``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The unused ``pan`` and ``zoom`` methods of `~.axis.Axis` and `~.ticker.Locator`
are deprecated.  Panning and zooming are now implemented using the
``start_pan``, ``drag_pan``, and ``end_pan`` methods of `~.axes.Axes`.

Passing None to various Axes subclass factories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Support for passing ``None`` as base class to `.axes.subplot_class_factory`,
``axes_grid1.parasite_axes.host_axes_class_factory``,
``axes_grid1.parasite_axes.host_subplot_class_factory``,
``axes_grid1.parasite_axes.parasite_axes_class_factory``, and
``axes_grid1.parasite_axes.parasite_axes_auxtrans_class_factory`` is deprecated.
Explicitly pass the correct base ``Axes`` class instead.

``axes_rgb``
~~~~~~~~~~~~
In :mod:`mpl_toolkits.axes_grid1.axes_rgb`, ``imshow_rgb`` is deprecated (use
``ax.imshow(np.dstack([r, g, b]))`` instead); ``RGBAxesBase`` is deprecated
(use ``RGBAxes`` instead); ``RGBAxes.add_RGB_to_figure`` is deprecated (it was
an internal helper).

``Substitution.from_params``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This method is deprecated.  If needed, directly assign to the ``params``
attribute of the Substitution object.

PGF backend cleanups
~~~~~~~~~~~~~~~~~~~~
The *dummy* parameter of `.RendererPgf` is deprecated.

`.GraphicsContextPgf` is deprecated (use `.GraphicsContextBase` instead).

``set_factor`` method of :mod:`mpl_toolkits.axisartist` locators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``set_factor`` method of :mod:`mpl_toolkits.axisartist` locators (which are
different from "standard" Matplotlib tick locators) is deprecated.

`.widgets.SubplotTool` callbacks and axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``funcleft``, ``funcright``, ``funcbottom``, ``functop``, ``funcwspace``,
and ``funchspace`` methods of `.widgets.SubplotTool` are deprecated.

The ``axleft``, ``axright``, ``axbottom``, ``axtop``, ``axwspace``, and
``axhspace`` attributes of `.widgets.SubplotTool` are deprecated.  Access the
``ax`` attribute of the corresponding slider, if needed.

mathtext ``Glue`` helper classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``Fil``, ``Fill``, ``Filll``, ``NegFil``, ``NegFill``, ``NegFilll``, and
``SsGlue`` classes in the :mod:`matplotlib.mathtext` module are deprecated.
As an alternative, directly construct glue instances with ``Glue("fil")``, etc.

NavigationToolbar2._init_toolbar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overriding this method to initialize third-party toolbars is deprecated.
Instead, the toolbar should be initialized in the ``__init__`` method of the
subclass (which should call the base-class' ``__init__`` as appropriate).  To
keep back-compatibility with earlier versions of Matplotlib (which *required*
``_init_toolbar`` to be overridden), a fully empty implementation (``def
_init_toolbar(self): pass``) may be kept and will not trigger the deprecation
warning.

NavigationToolbar2QT.parent and .basedir
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These attributes are deprecated.  In order to access the parent window, use
``toolbar.canvas.parent()``.  Once the deprecation period is elapsed, it will
also be accessible as ``toolbar.parent()``.  The base directory to the icons
is ``os.path.join(mpl.get_data_path(), "images")``.

NavigationToolbar2QT.ctx
~~~~~~~~~~~~~~~~~~~~~~~~
This attribute is deprecated.

NavigationToolbar2.press and .release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These methods were called when pressing or releasing a mouse button,
but *only* when an interactive pan or zoom was occurring (contrary to
what the docs stated).  They are deprecated; if you write a backend
which needs to customize such events, please directly override
``press_pan``/``press_zoom``/``release_pan``/``release_zoom`` instead.

Path helpers in :mod:`.bezier`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``bezier.make_path_regular`` is deprecated.  Use ``Path.cleaned()`` (or
``Path.cleaned(curves=True)``, etc.) instead (but note that these methods add a
``STOP`` code at the end of the path).

``bezier.concatenate_paths`` is deprecated.  Use ``Path.make_compound_path()``
instead.

``animation.html_args`` rcParam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The unused ``animation.html_args`` rcParam and ``animation.HTMLWriter.args_key``
attribute are deprecated.

``text.latex.preview`` rcParam
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This rcParam, which controlled the use of the preview.sty LaTeX package to
align TeX string baselines, is deprecated, as Matplotlib's own dvi parser now
computes baselines just as well as preview.sty.

``SubplotSpec.get_rows_columns``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This method is deprecated.  Use the ``GridSpec.nrows``, ``GridSpec.ncols``,
``SubplotSpec.rowspan``, and ``SubplotSpec.colspan`` properties instead.

Qt4-based backends
~~~~~~~~~~~~~~~~~~
The qt4agg and qt4cairo backends are deprecated.

*fontdict* and *minor* parameters of `.Axes.set_xticklabels` and `.Axes.set_yticklabels` will become keyword-only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``RendererWx.get_gc``
~~~~~~~~~~~~~~~~~~~~~
This method is deprecated.  Access the ``gc`` attribute directly instead.

*add_all* parameter in ``axes_grid``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The *add_all* parameter of `.axes_grid1.axes_grid.Grid`,
`.axes_grid1.axes_grid.ImageGrid`, `.axes_grid1.axes_rgb.make_rgb_axes` and
`.axes_grid1.axes_rgb.RGBAxes` is deprecated.  Axes are now always added to the
parent figure, though they can be later removed with ``ax.remove()``.

``BboxBase.inverse_transformed``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``.BboxBase.inverse_transformed`` is deprecated (call `.BboxBase.transformed`
on the `~.Transform.inverted()` transform instead).

*orientation* of ``eventplot()`` and `.EventCollection`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Setting the *orientation* of an ``eventplot()`` or `.EventCollection` to "none"
or None is deprecated; set it to "horizontal" instead.  Moreover, the two
orientations ("horizontal" and "vertical") will become case-sensitive in the
future.

*minor* kwarg to `.Axis.get_ticklocs` will become keyword-only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Passing this argument positionally is deprecated.

Case-insensitive properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Normalization of upper or mixed-case property names to lowercase in
`.Artist.set` and `.Artist.update` is deprecated.  In the future, property
names will be passed as is, allowing one to pass names such as *patchA* or
*UVC*.

``ContourSet.ax``, ``Quiver.ax``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These attributes are deprecated in favor of ``ContourSet.axes`` and
``Quiver.axes``, for consistency with other artists.

``Locator.refresh()`` and associated methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``Locator.refresh()`` is deprecated.  This method was called at certain places
to let locators update their internal state, typically based on the axis
limits.  Locators should now always consult the axis limits when called, if
needed.

The associated helper methods ``NavigationToolbar2.draw()`` and
``ToolViewsPositions.refresh_locators()`` are deprecated, and should be
replaced by calls to ``draw_idle()`` on the corresponding canvas.

`.ScalarMappable` checkers
~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``add_checker`` and ``check_update`` methods and ``update_dict`` attribute
of `.ScalarMappable` are deprecated.

`.pyplot.tight_layout` and ``ColorbarBase`` parameters will become keyword-only
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All parameters of `.pyplot.tight_layout` and all parameters of ``ColorbarBase``
except for the first (*ax*) will become keyword-only, consistently with
`.Figure.tight_layout` and ``Colorbar``, respectively.

`.Axes.pie` radius and startangle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Passing ``None`` as either the ``radius`` or ``startangle`` of an `.Axes.pie`
is deprecated; use the explicit defaults of 1 and 0, respectively, instead.

``AxisArtist.dpi_transform``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated.  Scale ``Figure.dpi_scale_trans`` by 1/72 to achieve the
same effect.

``offset_position`` property of `.Collection`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``offset_position`` property of `.Collection` is deprecated.  In the
future, `.Collection`\s will always behave as if ``offset_position`` is set to
"screen" (the default).

Support for passing ``offset_position="data"`` to the ``draw_path_collection``
of all renderer classes is deprecated.

`.transforms.AffineDeltaTransform` can be used as a replacement.  This API is
experimental and may change in the future.

``testing.compare.make_external_conversion_command``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... is deprecated.

`.epoch2num` and `.num2epoch` are deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These are unused and can be easily reproduced by other date tools.
`.get_epoch` will return Matplotlib's epoch.

``axes_grid1.CbarAxes`` attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``cbid`` and ``locator`` attribute are deprecated.  Use
``mappable.colorbar_cid`` and ``colorbar.locator``, as for standard colorbars.
