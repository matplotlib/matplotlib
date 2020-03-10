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
Setting a custom method overridding `.Artist.contains` is deprecated.
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
``validate_movie_writers``.
To test whether an rcParam value would be acceptable, one can test e.g. ``rc =
RcParams(); rc[k] = v`` raises an exception.

Stricter rcParam validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:rc:`axes.axisbelow` currently normalizes all strings starting with "line"
(case-insensitive) to the option "line".  This is deprecated; in a future
version only the exact string "line" (case-sensitive) will be supported.

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
