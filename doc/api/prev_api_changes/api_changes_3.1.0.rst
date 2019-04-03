API Changes for 3.1.0
=====================

API removals
------------

The following API elements have been removed:

- ``font_manager.USE_FONTCONFIG``
- ``font_manager.cachedir``
- ``__version__numpy__``

Deprecations
------------

- ``Spine.is_frame_like``

This has not been used in the codebase since its addition in 2009.

- ``axis3d.Axis.get_tick_positions``

This has never been used internally, there is no equivalent method exists on
the 2D Axis classes, and despite the similar name, it has a completely
different behavior from the 2D Axis' `axis.Axis.get_ticks_position` method.

- ``backend_pgf.LatexManagerFactory``
- ``mpl_toolkits.axisartist.axislines.SimpleChainedObjects``
- ``mpl_toolkits.Axes.AxisDict``
- ``checkdep_dvipng``
- ``checkdep_ghostscript``
- ``checkdep_pdftops``
- ``checkdep_inkscape``
- ``ticker.decade_up``
- ``ticker.decade_down``
- ``backend_pdf.RendererPdf.afm_font_cache``
- ``backend_ps.RendererPS.afmfontd``
- ``projections.process_projection_requirements``
- ``dates.seconds()``
- ``dates.minutes()``
- ``dates.hours()``
- ``dates.weeks()``


`Text` now has a ``c`` alias for the ``color`` property
-------------------------------------------------------

For consistency with `Line2D`, the `Text` class has gained the ``c``
alias for the ``color`` property. For example, one can now write
``ax.text(.5, .5, "foo", c="red")``.

Changes in AFM parsing
----------------------

In accordance with the AFM spec, the AFM parser no longer truncates the
``UnderlinePosition`` and ``UnderlineThickness`` fields to integers.

The ``Notice`` field (which can only be publicly accessed by the deprecated
``afm.parse_afm`` API) is no longer decoded to a `str`, but instead kept as
`bytes`, to support non-conformant AFM files that use non-ASCII characters in
that field.

Axes.tick_params argument checking
----------------------------------

Previously `Axes.tick_params` silently did nothing when an invalid *axis*
parameter was supplied. This behavior has been changed to raise a ValueError
instead.

xmin, xmax, ymin, ymax kwargs un-deprecated
-------------------------------------------

xmin, xmax kwargs to set_xlim and ymin, ymax kwargs to set_ylim have been
un-deprecated. These were deprecated in 3.0, but have been re-instated due to
user feedback.

matplotlib.backends.qt_editor.formlayout module is deprecated
-------------------------------------------------------------

This module is a vendored, modified version of the official formlayout_ module
available on PyPI. Install that module separately if you need it.

.. _formlayout: https://pypi.org/project/formlayout/

Setting the same artist property multiple time via aliases is deprecated
------------------------------------------------------------------------

Previously, code such as ``plt.plot([0, 1], c="red", color="blue")`` would
emit a warning indicating that ``c`` and ``color`` are aliases of one another,
and only keep the ``color`` kwarg.  This behavior has been deprecated; in a
future version, this will raise a TypeError, similar to Python's behavior when
a keyword argument is passed twice (``plt.plot([0, 1], c="red", c="blue")``).

This warning is raised by `~.cbook.normalize_kwargs`.

Artist.set keyword normalisation
--------------------------------

`Artist.set` now normalizes keywords before sorting them.

Previously it sorted its keyword arguments in reverse alphabetical
order (with a special-case to put ``color`` at the end) before applying them.

It now normalizes aliases (and, as above, emits a warning on duplicate
properties) before doing the sorting (so ``c`` goes to the end too).

Verbose deprecations
--------------------

The ``verbose.fileo`` and ``verbose.level`` rcParams have been deprecated.
These have had no effect since the switch from Matplotlib's old custom Verbose
logging to the stdlib's `logging` module. In addition the
``rcsetup.validate_verbose`` function is deprecated.

mathtext deprecations
---------------------

The ``\stackrel`` mathtext command is deprecated (it behaved differently
from LaTeX's ``\stackrel``.  To stack two mathtext expressions, use
``\genfrac{left-delim}{right-delim}{fraction-bar-thickness}{}{top}{bottom}``.

Undeprecations
--------------

The ``obj_type`` kwarg to the ``cbook.deprecated`` decorator is undeprecated.

Additional positional arguments to `Axis.set_ticklabels`
--------------------------------------------------------
Passing positional arguments to `Axis.set_ticklabels` beyond `ticklabels`
itself has no effect, and support for them is deprecated.

Boxplot tick methods
--------------------

The ``manage_xticks`` parameter of `~Axes.boxplot` and `~Axes.bxp` has been
renamed (with a deprecation period) to ``manage_ticks``, to take into account
the fact that it manages either x or y ticks depending on the ``vert``
parameter.

When ``manage_ticks`` is True (the default), these methods now attempt to take
previously drawn boxplots into account when setting the axis limits, ticks, and
tick labels.

get_window_extents changes
--------------------------

`.matplotlib.axes.Axes.get_window_extent` used to return a bounding box
that was slightly larger than the axes, presumably to take into account
the ticks that may be on a spine.  However, it was not scaling the tick sizes
according to the dpi of the canvas, and it did not check if the ticks were
visible, or on the spine.

Now  `.matplotlib.axes.Axes.get_window_extent` just returns the axes extent
with no padding for ticks.

This affects `.matplotlib.axes.Axes.get_tightbbox` in cases where there are
outward ticks with no tick labels, and it also removes the (small) pad around
axes in that case.

`.spines.get_window_extent` now takes into account ticks that are on the
spine.

`matplotlib.use` parameter change
---------------------------------

The first parameter of `matplotlib.use` has been renamed from *arg* to
*backend*. This will only affect cases where that parameter has been set
as a keyword argument. The common usage pattern as a positional argument
``matplotlib.use('Qt5Agg')`` is not affected.

``~`` now interpreted as space in mathtext
------------------------------------------

In constructs such as ``"$1~2$"``, mathtext now interprets the tilde as a
space, consistently with TeX (this was previously a parse error).

Exception on failing animations changed
---------------------------------------

Previously, subprocess failures in the animation framework would raise either
in a `RuntimeError` or a `ValueError` depending on when the error occurred.
They now raise a `subprocess.CalledProcessError` with attributes set as
documented by the exception class.

MouseEvents now include the event name in their ``str()``
---------------------------------------------------------

... instead of the prefix "MPL MouseEvent".

API changes
-----------

The arguments of `matplotlib.testing.compare.calculate_rms` have been renamed
from ``expectedImage, actualImage``, to ``expected_image, actual_image``.

Deprecations
------------

The ``matplotlib.testing.decorators.switch_backend`` decorator is deprecated.
Test functions should use ``pytest.mark.backend(...)``, and the mark will be
picked up by the ``matplotlib.testing.conftest.mpl_test_settings`` fixture.

Matplotlib.use now has an ImportError for interactive backend
-------------------------------------------------------------

Switching backends via `matplotlib.use` is now allowed by default,
regardless of whether `matplotlib.pyplot` has been imported. If the user
tries to switch from an already-started interactive backend to a different
interactive backend, an ImportError will be raised.

Deprecations
------------

``Text.is_math_text`` is deprecated.

``TextPath.is_math_text`` and ``TextPath.text_get_vertices_codes`` are
deprecated.  As an alternative to the latter, construct a new ``TextPath``
object.

The ``usetex`` parameter of ``TextToPath.get_text_path`` is deprecated and
folded into the ``ismath`` parameter, which can now take the values False,
True, and "TeX", consistently with other low-level text processing functions.

Behavior changes
----------------

Previously, if :rc:`text.usetex` was True, then constructing a `TextPath` on
a non-mathtext string with ``usetex=False`` would rely on the mathtext parser
(but not on usetex support!) to parse the string.  The mathtext parser is not
invoked anymore, which may cause slight changes in glyph positioning.

Deprecations
------------

The ``.color`` attribute of `Quiver` objects is deprecated.  Instead, use (as
for any `Collection`) the ``get_facecolor`` method.  Note that setting to the
``.color`` attribute did not update the quiver artist, whereas calling
``set_facecolor`` does.

Deprecations
------------

The function ``matplotlib.scale.get_scale_docs()`` and its alias
``matplotlib.pyplot.get_scale_docs()`` are considered internal and will be
removed from the public API in a future version.

Passing a single string as *labels* to `Sankey.add`
---------------------------------------------------

Previously, `Sankey.add` would only accept a single string as the *labels*
argument if its length is equal to the number of flows, in which case it would
use one character of the string for each flow.

The behavior has been changed to match the documented one: when a single string
is passed, it is used to label all the flows.

API deprecations
----------------

The following API elements are deprecated:

- ``get_py2exe_datafiles``, ``tk_window_focus``,
- ``backend_gtk3.FileChooserDialog``,
  ``backend_gtk3.NavigationToolbar2GTK3.get_filechooser``,
  ``backend_gtk3.SaveFigureGTK3.get_filechooser``,
- ``backend_ps.PsBackendHelper``, ``backend_ps.ps_backend_helper``,
- ``cbook.iterable``,
- ``cbook.get_label``, ``cbook.iterable``,
- ``font_manager.OSXInstalledFonts``,
- ``mlab.demean``,

The following environment variables are deprecated:
- ``MATPLOTLIBDATA``,

Deprecations
------------

The ``Path.has_nonfinite`` attribute is deprecated (use ``not
np.isfinite(path.vertices).all()`` instead).

Deprecations
------------
￼
``TextToPath.glyph_to_path`` is deprecated (call ``font.get_path()`` and
manually transform the path instead).

Deprecations
------------

The (unused and always None) ``NavigationToolbar2QT.adj_window`` attribute is
deprecated.

``OldScalarFormatter.pprint_val``, ``ScalarFormatter.pprint_val``, and ``LogFormatter.pprint_val`` are deprecated
-----------------------------------------------------------------------------------------------------------------

They are helper methods that do not even have a consistent signatures across formatter classes.

Deprecations
------------

``dates.strpdate2num`` and ``dates.bytespdate2num`` are brittle in the
presence of locale changes, and are deprecated.  Use standard datetime
parsers such as `time.strptime` or `dateutil.parser.parse`, and additionally
call `matplotlib.dates.date2num` if you insist on converting to Matplotlib's
internal datetime representation; or use ``dates.datestr2num``.

Text alignment fixes made
-------------------------

Text alignment was incorrect, in particular for multiline text objects
with large descenders (i.e. subscripts) and rotated text.  These have been
fixed and made more consistent, but could make old code that has compensated
no longer have the correct alignment.

Deprecations
------------

``mpl_toolkits.axisartist.axis_artist.UnimplementedException`` is deprecated.

mplot3d is always registered by default
---------------------------------------

It is not necessary to import mplot3d anymore to create 3d axes with
``fig.add_subplot(111, projection="3d")``.

Path code types like ``Path.MOVETO`` are now ``np.uint8`` instead of ``int``
----------------------------------------------------------------------------

``Path.STOP``, ``Path.MOVETO``, ``Path.LINETO``, ``Path.CURVE3``,
``Path.CURVE4`` and ``Path.CLOSEPOLY`` are now of the type ``Path.code_type``
(``np.uint8`` by default) instead of plain ``int``. This makes their type
match the array value type of the ``Path.codes`` array.

Changes regarding the text.latex.unicode rcParam
------------------------------------------------

The rcParam now defaults to True and is deprecated (i.e., in future versions
of Matplotlib, unicode input will always be supported).

Moreover, the underlying implementation now uses ``\usepackage[utf8]{inputenc}``
instead of ``\usepackage{ucs}\usepackage[utf8x]{inputenc}``.

Changes in handling of degenerate bounds passed to `set_xlim`
-------------------------------------------------------------

When bounds passed to `set_xlim` (`set_xlim`, etc.) are degenerate (i.e. the
lower and upper value are equal), the method used to "expand" the bounds now
matches the expansion behavior of autoscaling when the plot contains a single
x-value, and should in particular produce nicer limits for non-linear scales.

Deprecations
------------

The following keyword arguments are deprecated:

- Passing ``shade=None`` to
  `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface` is deprecated. This was
  an unintended implementation detail with the same semantics as
  ``shade=False``. Please use the latter code instead.API removals


The following deprecated APIs were removed:

Classes and methods
-------------------
- ``Verbose`` (replaced by python logging library)
- ``artist.Artist.hitlist`` (no replacement)
- ``artist.Artist.is_figure_set`` (use ``artist.figure is not None`` instead)
- ``axis.Axis.unit_data`` (use ``axis.Axis.units`` instead)
- ``backend_bases.FigureCanvasBase.onRemove`` (no replacement)
  ``backend_bases.FigureManagerBase.show_popup`` (this never did anything)
- ``backend_wx.SubplotToolWx`` (no replacement)
- ``backend_wx.Toolbar`` (use ``backend_wx.NavigationToolbar2Wx`` instead)
- ``cbook.align_iterators`` (no replacment)
- ``contour.ContourLabeler.get_real_label_width`` (no replacement)
- ``legend.Legend.draggable`` (use `legend.Legend.set_draggable()` instead)
- ``texmanager.TexManager.postscriptd``, ``texmanager.TexManager.pscnt``,
  ``texmanager.TexManager.make_ps``, ``texmanager.TexManager.get_ps_bbox``
  (no replacements)

Arguments
---------
- The ``fig`` kwarg to ``GridSpec.get_subplot_params`` and
  ``GridSpecFromSubplotSpec.get_subplot_params`` (use the argument
  ``figure`` instead)
- Passing 'box-forced' to `axes.Axes.set_adjustable` (use 'box' instead)
- Support for the strings 'on'/'true'/'off'/'false' to mean
  ``True``/``False`` (directly use ``True``/``False`` instead).
  The following functions are affected: `Axes.grid`, `Axes3D.grid`
  `Axis.set_tick_params`, `pyplot.box`.
- Using `pyplot.axes` with an `axes.Axes` type argument
  (use `pyplot.sca` instead)

Other
-----
- svgfont support (in :rc:`svg.fonttype`) has been removed,
- Logging is now done with the standard python ``logging`` library.
  ``matplotlib.verbose`` and the command line switches ``--verbose-LEVEL`` are
  removed.

  To control the logging output use::

    import logging
    logger = logging.getLogger('matplotlib')
    logger.set_level(logging.INFO)

Deprecations
------------

Support for passing single-letter colors (one of "rgbcmykw") as UPPERCASE
characters is deprecated; these colors will become case-sensitive (lowercase)
after the deprecation period has passed.

The goal is to decrease the number of ambiguous cases when using the ``data``
keyword to plotting methods; e.g. ``plot("X", "Y", data={"X": ..., "Y": ...})``
will not warn about "Y" possibly being a color anymore after the deprecation
period has passed.

Allow "real" LaTeX code for ``pgf.preamble`` and ``text.latex.preamble`` in matplotlib rc file
----------------------------------------------------------------------------------------------

Previously, the rc file keys ``pgf.preamble`` and ``text.latex.preamble`` were parsed using commmas as separators. This would break valid LaTeX code, such as::

\usepackage[protrusion=true, expansion=false]{microtype}

The parsing has been modified to pass the complete line to the LaTeX system,
keeping all commas.

Passing a list of strings from within a Python script still works as it used to.

Passing a list containing non-strings now fails, instead of coercing the results to strings.

Equal aspect axes for 3D plots disabled
---------------------------------------

Setting the aspect on 3D axes previously returned non-sensical
results (e.g. see https://github.com/matplotlib/matplotlib/issues/1077).
Calling ``ax.set_aspect('equal')`` or ``ax.set_aspect(num)``
on a 3D axes now raises a ``NotImplementedError``.

Deprecations
------------

The class variable ``matplotlib.ticker.MaxNLocator.default_params`` is
deprecated and will be removed in a future version. The defaults are not
supposed to be user-configurable.

``matplotlib.ticker.MaxNLocator`` and its ``set_params`` method will issue
a warning on unknown keyword arguments instead of silently ignoring them.
Future versions will raise an error.

Behavior changes
----------------

Calling `.Figure.add_subplot()` with no positional arguments used to do
nothing; this now is equivalent to calling ``add_subplot(111)`` instead.

Numpy minimum version bump
--------------------------

Matplotlib 3.1 requires numpy>=1.11.

Deprecations
------------

The following (unused and never updated) attributes are deprecated:
``NavigationToolbar2QT.buttons``, ``Line2D.verticalOffset``, ``Quiver.keytext``,
``Quiver.keyvec``, ``SpanSelector.buttonDown``.

The ``interp_at_native`` parameter to ``BboxImage``, which has no effect since
Matplotlib 2.0, is deprecated.

Deprecations
------------

``cbook.is_hashable`` is deprecated (use
``isinstance(..., collections.abc.Hashable)`` instead).

Deprecations
------------

``cbook.safezip`` is deprecated (manually check the lengths of the inputs
instead, or rely on numpy to do it).

``Axes3D.voxels`` now shades the resulting voxels
-------------------------------------------------

See Whats new for details.  The previous behavior can be achieved by passing
``shade=False``.

Deprecations
------------

- The ``LogTransformBase``, ``Log10Transform``, ``Log2Transform``,
  ``NaturalLogTransformLog``, ``InvertedLogTransformBase``,
  ``InvertedLog10Transform``, ``InvertedLog2Transform``, and
  ``InvertedNaturalLogTransform`` classes (all defined in
  :mod:`matplotlib.scales`) are deprecated.  As a replacement, use the general
  `LogTransform` and `InvertedLogTransform` classes, whose constructors take a
  *base* argument.

Changes to search paths for FreeType and libpng
-----------------------------------------------

The ``MPLBASEDIRLIST`` environment variables and ``basedirlist`` entry in
``setup.cfg`` have no effect anymore.  Instead, if building in situations where
FreeType or libpng are not in the compiler or linker's default path, set the
standard environment variables ``CFLAGS``/``LDFLAGS`` on Linux or OSX, or
``CL``/``LINK`` on Windows, to indicate the relevant paths.

See details in :file:`INSTALL.rst`.

Deprecations
------------

The `\mathcircled` mathtext command (which is not a real TeX command)
is deprecated.  Directly use unicode characters (e.g.
``"\N{CIRCLED LATIN CAPITAL LETTER A}"`` or ``"\u24b6"``) instead.

Support for setting :rc:`mathtext.default` to circled is deprecated.

matplotlib.font_manager.win32InstalledFonts return value
--------------------------------------------------------

`matplotlib.font_manager.win32InstalledFonts` returns an empty list instead
of None if no fonts are found.

The HTMLWriter constructor is more strict
-----------------------------------------

It no longer normalizes unknown values of *default_mode* to 'loop', but errors
out instead.

New `Formatter.format_ticks` method
-----------------------------------

The `Formatter` class gained a new `~Formatter.format_ticks` method, which
takes the list of all tick locations as a single argument and returns the list
of all formatted values.  It is called by the axis tick handling code and, by
default, first calls `~Formatter.set_locs` with all locations, then repeatedly
calls `~Formatter.__call__` for each location.

Tick-handling code in the codebase that previously performed this sequence
(`~Formatter.set_locs` followed by repeated `~Formatter.__call__`) have been
updated to use `~Formatter.format_ticks`.

`~Formatter.format_ticks` is intended to be overridden by `Formatter`
subclasses for which the formatting of a tick value depends on other tick
values, such as `ConciseDateFormatter`.

`matplotlib.colorbar.ColorbarBase` is no longer a subclass of `.ScalarMappable`
-------------------------------------------------------------------------------

This inheritance lead to a confusing situation where the
`ScalarMappable` passed to `matplotlib.colorbar.Colorbar` (`~.Figure.colorbar`)
had a ``set_norm`` method, as did the colorbar.  The colorbar is now purely a
follower to the `ScalarMappable` norm and colormap, and the old inherited methods
`~matplotlib.colorbar.ColorbarBase.set_norm`,
`~matplotlib.colorbar.ColorbarBase.set_cmap`,
`~matplotlib.colorbar.ColorbarBase.set_clim` are deprecated, as are the
getter versions of those calls.  To set the norm associated with a colorbar do
``colorbar.mappable.set_norm()`` etc.

`StemContainer` now stores `LineCollection`
-------------------------------------------

`StemContainer` objects can now store a `LineCollection` object instead of a
list of `Line2D` objects for stem lines plotted using `ax.stem`. This gives a
very large performance boost to displaying and moving `ax.stem` plots.

This will become the default behaviour in Matplotlib 3.3. To use it now, the
``use_line_collection`` keyword argument to ~`.axes.stem` can be set to
``True``.

Individual line segments can be extracted from the `LineCollection` using
`LineCollection.get_segements()`. See the `LineCollection` documentation for
other methods to retrieve the collection properties.

Modify output of Axes.hist when input consists of multiple empty lists
----------------------------------------------------------------------

Input that consists of multiple empty lists will now return a list of histogram
values for each one of the lists. For example, an input of ``[[],[]]`` will
return 2 lists of histogram values. Previously, a single list was returned.

Deprecations
------------

The ``backend_wx.IDLE_DELAY`` global, being unused and only relevant to the now
removed wx "idling" code, is deprecated (note that as it is a module-level
global, no deprecation warning is emitted when accessing it).

`FontManager.score_weight` is more strict with its inputs
---------------------------------------------------------

Previously, when a weight string was passed to `FontManager.score_weight`,

- if the weight was the string representation of an integer, it would be
  converted to that integer,
- otherwise, if the weight was not a standard weight name, it would be silently
  replaced by a value of 500 ("normal" weight).

`FontManager.score_weight` now raises an exception on such inputs.

Removal of deprecated :mod:`matplotlib.mlab` code
-------------------------------------------------

Lots of code inside the :mod:`matplotlib.mlab` module which was deprecated
in Matplotlib 2.2 has been removed. See below for a list:

- ``mlab.exp_safe`` (use `numpy.exp` instead)
- ``mlab.amap``
- ``mlab.logspace`` (use `numpy.logspace` instead)
- ``mlab.rms_flat``
- ``mlab.l1norm`` (use ``numpy.linalg.norm(a, ord=1)`` instead)
- ``mlab.l2norm`` (use ``numpy.linalg.norm(a, ord=2)`` instead)
- ``mlab.norm_flat`` (use ``numpy.linalg.norm(a.flat, ord=2)`` instead)
- ``mlab.frange`` (use `numpy.arange` instead)
- ``mlab.identity`` (use `numpy.identity` instead)
- ``mlab.base_repr``
- ``mlab.binary_repr``
- ``mlab.ispower2``
- ``mlab.log2`` (use `numpy.log2` instead)
- ``mlab.isvector``
- ``mlab.movavg``
- ``mlab.safe_isinf`` (use `numpy.isinf` instead)
- ``mlab.safe_isnan`` (use `numpy.isnan` instead)
- ``mlab.cohere_pairs`` (use `scipy.signal.coherence` instead)
- ``mlab.entropy`` (use `scipy.stats.entropy` instead)
- ``mlab.normpdf`` (use `scipy.stats.norm.pdf` instead)
- ``mlab.find`` (use ``np.nonzero(np.ravel(condition))`` instead)
- ``mlab.longest_contiguous_ones``
- ``mlab.longest_ones``
- ``mlab.PCA``
- ``mlab.prctile`` (use `numpy.percentile` instead)
- ``mlab.prctile_rank``
- ``mlab.center_matrix``
- ``mlab.rk4`` (use `scipy.integrate.ode` instead)
- ``mlab.bivariate_normal``
- ``mlab.get_xyz_where``
- ``mlab.get_sparse_matrix``
- ``mlab.dist`` (use `numpy.hypot` instead)
- ``mlab.dist_point_to_segment``
- ``mlab.griddata`` (use `scipy.interpolate.griddata`)
- ``mlab.less_simple_linear_interpolation`` (use `numpy.interp`)
- ``mlab.slopes``
- ``mlab.stineman_interp``
- ``mlab.segments_intersect``
- ``mlab.fftsurr``
- ``mlab.offset_line``
- ``mlab.quad2cubic``
- ``mlab.vector_lengths``
- ``mlab.distances_along_curve``
- ``mlab.path_length``
- ``mlab.cross_from_above``
- ``mlab.cross_from_below``
- ``mlab.contiguous_regions`` (use `.cbook.contiguous_regions` instead)
- ``mlab.is_closed_polygon``
- ``mlab.poly_between``
- ``mlab.poly_below``
- ``mlab.inside_poly``
- ``mlab.csv2rec``
- ``mlab.rec2csv`` (use `numpy.recarray.tofile` instead)
- ``mlab.rec2text`` (use `numpy.recarray.tofile` instead)
- ``mlab.rec_summarize``
- ``mlab.rec_join``
- ``mlab.recs_join``
- ``mlab.rec_groupby``
- ``mlab.rec_keep_fields``
- ``mlab.rec_drop_fields``
- ``mlab.rec_append_fields``
- ``mlab.csvformat_factory``
- ``mlab.get_formatd``
- ``mlab.FormatDatetime`` (use `datetime.datetime.strftime` instead)
- ``mlab.FormatDate`` (use `datetime.date.strftime` instead)
- ``mlab.FormatMillions``, ``mlab.FormatThousands``, ``mlab.FormatPercent``,
  ``mlab.FormatBool``, ``mlab.FormatInt``, ``mlab.FormatFloat``,
  ``mlab.FormatFormatStr``, ``mlab.FormatString``, ``mlab.FormatObj``
- ``mlab.donothing_callback``

``Cn`` colors now support ``n>=10``
-----------------------------------

It is now possible to go beyond the tenth color in the property cycle using
``Cn`` syntax, e.g. ``plt.plot([1, 2], color="C11")`` now uses the 12th color
in the cycle.

Note that previously, a construct such as ``plt.plot([1, 2], "C11")`` would be
interpreted as a request to use color ``C1`` and marker ``1`` (an "inverted Y").
To obtain such a plot, one should now use ``plt.plot([1, 2], "1C1")`` (so that
the first "1" gets correctly interpreted as a marker specification), or, more
explicitly, ``plt.plot([1, 2], marker="1", color="C1")``.

Changes in parameter names
--------------------------

- The ``arg`` parameter to `matplotlib.use` has been renamed to ``backend``.
- The ``normed`` parameter to `Axes.hist2d` has been renamed to ``density``.
- The ``s`` parameter to `Annotation` (and indirectly `Axes.annotation`) has
  been renamed to ``text``.
- The ``tolerence`` parameter to
  `bezier.find_bezier_t_intersecting_with_closedpath`,
  `bezier.split_bezier_intersecting_with_closedpath`,
  `bezier.find_r_to_boundary_of_closedpath`,
  `bezier.split_path_inout` and `bezier.check_if_parallel` has been renamed to
  ``tolerance``.

In each case, the old parameter name remains supported (it cannot be used
simultaneously with the new name), but suppport for it will be dropped in
Matplotlib 3.3.

- The unused ``shape`` and ``imlim`` parameters to `Axes.imshow` are
  deprecated.  To avoid triggering the deprecation warning, the ``filternorm``,
  ``filterrad``, ``resample``, and ``url`` arguments should be passed by
  keyword.

Deprecations
------------

The ``bezier.find_r_to_boundary_of_closedpath`` function is deprecated (it has
always returned None instead of the requested radius).

Poly3DCollection.set_zsort
--------------------------

`Poly3DCollection.set_zsort` no longer silently ignores invalid inputs, or
False (which was always broken).  Passing True to mean "average" is deprecated.

Timer.remove_callback won't consider ``*args, **kwargs`` in a future version
----------------------------------------------------------------------------

Currently, ``Timer.remove_callback(func, *args, **kwargs)`` removes a callback
previously added by ``Timer.add_callback(func, *args, **kwargs)``, but if
``*args, **kwargs`` is not passed in (``Timer.remove_callback(func)``), then
the first callback with a matching ``func`` is removed, regardless of whether
it was added with or without ``*args, **kwargs``.

In a future version, ``Timer.remove_callback`` will always use the latter
behavior (not consider ``*args, **kwargs``); to specifically consider them, add
the callback as a `functools.partial` object ::

   cb = timer.add_callback(functools.partial(func, *args, **kwargs))
   # ...
   # later
   timer.remove_callback(cb)

``Timer.add_callback`` was modified to return *func* to simplify the above
usage (previously it returned None); this also allows using it as a decorator.

The new API is modelled after `atexit.register` / `atexit.unregister`.

Changes to the internal tick handling API
-----------------------------------------

``Axis.iter_ticks`` (which only served as a helper to the private
``Axis._update_ticks``) is deprecated.

The signature of the (private) ``Axis._update_ticks`` has been changed to not
take the renderer as argument anymore (that argument is unused).

Deprecations
------------

The ``text.TextWithDash`` class and the ``withdash`` keyword argument to
``text()`` is deprecated.  Consider using ``annotate()`` instead.

`~Axes.bxp` now respects :rc:`boxplot.boxprops.linewidth` even when *patch_artist* is set
-----------------------------------------------------------------------------------------

Previously, when the *patch_artist* parameter was set, `~Axes.bxp` would ignore
:rc:`boxplot.boxprops.linewidth`.  This was an oversight -- in particular,
`~Axes.boxplot` did not ignore it.

This oversight is now fixed.

Deprecations
------------

Support for custom backends that do not provide a ``set_hatch_color`` method is
deprecated.  We suggest that custom backends let their ``GraphicsContext``
class inherit from `GraphicsContextBase`, to at least provide stubs for all
required methods.

The fields ``Artist.aname`` and ``Axes.aname`` are deprecated. Please use

``isinstance()`` or ``__class__.__name__`` checks instead.``Axes.fmt_xdata`` and ``Axes.fmt_ydata`` no longer ignore TypeErrors raised by a user-provided formatter
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Previously, if the user provided a ``fmt_xdata`` or ``fmt_ydata`` function that
raised a TypeError (or set them to a non-callable), the exception would be
silently ignored and the default formatter be used instead.  This is no longer
the case; the exception is now propagated out.

Axes methods now raise TypeError instead of RuntimeError on mismatched calls
----------------------------------------------------------------------------

In certain cases, Axes methods (and pyplot functions) used to raise a
RuntimeError if they were called with a ``data`` kwarg and otherwise mismatched
arguments.  They now raise a ``TypeError`` instead.

Removed rcParams
----------------

The following deprecated rcParams are removed:

- ``text.dvipnghack``,
- ``nbagg.transparent`` (use :rc:`figure.facecolor` instead),
- ``plugins.directory``,
- ``axes.hold``,
- ``backend.qt4`` and ``backend.qt5`` (set the :envvar:`QT_API` environment
  variable instead).

The associated validator functions ``rcsetup.validate_qt4`` and
``validate_qt5`` are deprecated.

Passing 'normal' to `Axes.axis()` is deprecated
-----------------------------------------------

Use ``axis('auto')`` instead.


The image comparison test decorators now skip (rather than xfail) the test for uncomparable formats
---------------------------------------------------------------------------------------------------

The affected decorators are `~.image_comparison` and `~.check_figures_equal`.
The deprecated `~.ImageComparisonTest` class is likewise changed.
:orphan:

Drop support for ``pgi`` in the GTK3 backends
---------------------------------------------
``pgi``, an alternative implementation to PyGObject, is no longer supported in
the GTK3 backends. PyGObject should be used instead.

API changes
-----------

The ``--no-network`` flag to ``tests.py`` has been removed (no test requires
internet access anymore).  If it is desired to disable internet access both for
old and new versions of Matplotlib, use ``tests.py -m 'not network'`` (which is
now a no-op).

Minor ticks that collide with major ticks are always hidden
-----------------------------------------------------------

Previously, certain locator classes (`LogLocator`, `AutoMinorLocator`)
contained custom logic to avoid emitting tick locations that collided with
major ticks when they were used as minor locators.

This logic has now moved to the Axis class, and is used *regardless of the
ticker class*.  ``xaxis.minor.locator()`` now includes positions that collide
with ``xaxis.major.locator()``, but ``xaxis.get_minorticklocs()`` does not.

If you were relying on both the major and minor tick labels to appear on the
same tick, you may need to update your code.  For example, the following
snippet labeled days using major ticks, and hours and minutes using minor
ticks::

    import numpy as np
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    t = np.arange("2018-11-03", "2018-11-06", dtype="datetime64")
    x = np.random.rand(len(t))

    fig, ax = plt.subplots()
    ax.plot(t, x)
    ax.xaxis.set(
        major_locator=mdates.DayLocator(),
        major_formatter=mdates.DateFormatter("\n%a"),
        minor_locator=mdates.HourLocator((0, 6, 12, 18)),
        minor_formatter=mdates.DateFormatter("%H:%M"),
    )

    plt.show()

and added a newline to the major ticks labels to avoid them crashing into the
minor tick labels.

With the API change, the major tick labels should also include hours and
minutes, as the minor ticks are gone, so the ``major_formatter`` should be
``mdates.DateFormatter("%H:%M\n%a")``.

Removals
--------

The ``collections.CIRCLE_AREA_FACTOR`` constant has been removed.

Deprecations
------------

``GraphicsContextPS.shouldstroke`` is deprecated.

Stricter `~.Axes.plot` format string parsing
--------------------------------------------

In certain cases, `~.Axes.plot` would previously accept format strings
specifying more than one linestyle (e.g. ``"---."`` which specifies both
``"--"`` and ``"-."``); only use one of them would be used.

This now raises a ValueError instead.

Deprecations
------------

``path.get_paths_extents`` is deprecated; use
`~.path.get_path_collection_extents` instead.

API changes
-----------

Passing the ``block`` argument of ``plt.show`` positionally is deprecated; it
should be passed by keyword.

When using the nbagg backend, ``plt.show`` used to silently accept and ignore
all combinations of positional and keyword arguments.  This behavior is
deprecated.

Deprecations
------------

Passing a non-1D (typically, (n, 1)-shaped) input to `Axes.pie` is deprecated.
Pass a 1D array instead.

Deprecations
------------

The `TextPath` constructor used to silently drop ignored arguments; this
behavior is deprecated.

Deprecations
------------

The ``frameon`` kwarg to ``savefig`` and the ``savefig.frameon`` rcParam
are deprecated.  To emulate ``frameon = False``, set ``facecolor`` to fully
transparent (``"none"``, or ``(0, 0, 0, 0)``).

Deprecations
------------

``cbook.dedent``, ``docstring.Appender``, ``docstring.dedent``, and
``docstring.copy_dedent`` are deprecated (use the standard library's docstring
manipulation tools, such as `inspect.cleandoc` and `inspect.getdoc` instead).

Changes to `matplotlib.axes.Axes.spy`
-------------------------------------

The method `matplotlib.axes.Axes.spy` now raises a TypeError for the keyword
arguments 'interpolation' and 'linestyle' instead of silently ignoring them.

Furthermore, `matplotlib.axes.Axes.spy` spy does now allow for an 'extent'
argument (was silently ignored so far).

A bug with `spy(..., origin='lower') is fixed: So far this flipped the
data but not the y-axis resulting in a mismatch between axes labels and
actual data indices. Now, `origin='lower'` flips both the data and the y-axis
labels.

Deprecations
------------

The ``matplotlib.sphinxext.mathmpl`` and
``matplotlib.sphinxext.plot_directive`` interfaces have changed from the
(Sphinx-)deprecated function-based interface to a class-based interface.  This
should not affect end users, but the
``matplotlib.sphinxext.mathmpl.math_directive`` and
``matplotlib.sphinxext.plot_directive.plot_directive`` functions are now
deprecated.

Log-scaled axes avoid having zero or only one tick
--------------------------------------------------

When the default `LogLocator` would generate no ticks for an axis (e.g., an
axis with limits from 0.31 to 0.39) or only a single tick, it now instead falls
back on the linear `AutoLocator` to pick reasonable tick positions.

Invalid inputs
--------------

Passing invalid locations to `legend` and `table` used to fallback on a default
location.  This behavior is deprecated and will throw an exception in a future
version.

`offsetbox.AnchoredText` is unable to handle the ``horizontalalignment`` or
``verticalalignment`` kwargs, and used to ignore them with a warning.  This
behavior is deprecated and will throw an exception in a future version.

Passing steps less than 1 or greater than 10 to `MaxNLocator` used to result in
undefined behavior.  It now throws a ValueError.

Changes to the signatures of `cbook.deprecated` and `cbook.warn_deprecated`
---------------------------------------------------------------------------

All arguments to the `cbook.deprecated` decorator and `cbook.warn_deprecated`
function, except the first one (the version where the deprecation occurred),
are now keyword-only.  The goal is to avoid accidentally setting the "message"
argument when the "name" (or "alternative") argument was intended, as this has
repeatedly occurred in the past.
Deprecations
````````````

Support for passing (n, 1)-shaped error arrays to errorbar(), which was not
documented and did not work for ``n = 2``, is deprecated (pass a 1D array
instead).


Removal of deprecated :mod:`matplotlib.pylab` code
--------------------------------------------------

Lots of code inside the :mod:`matplotlib.mlab` module which was deprecated
in Matplotlib 2.2 has been removed. This means the following functions are
no longer available in the `matplotlib.pylab` module:

  - ``amap``
  - ``base_repr``
  - ``binary_repr``
  - ``bivariate_normal``
  - ``center_matrix``
  - ``csv2rec`` (use `numpy.recarray.tofile` instead)
  - ``dist`` (use `numpy.hypot` instead)
  - ``dist_point_to_segment``
  - ``distances_along_curve``
  - ``entropy`` (use `scipy.stats.entropy` instead)
  - ``exp_safe`` (use `numpy.exp` instead)
  - ``fftsurr``
  - ``find`` (use ``np.nonzero(np.ravel(condition))`` instead)
  - ``frange`` (use `numpy.arange` instead)
  - ``get_sparse_matrix``
  - ``get_xyz_where``
  - ``griddata`` (use `scipy.interpolate.griddata` instead)
  - ``identity`` (use `numpy.identity` instead)
  - ``inside_poly``
  - ``is_closed_polygon``
  - ``ispower2``
  - ``isvector``
  - ``l1norm`` (use ``numpy.linalg.norm(a, ord=1)`` instead)
  - ``l2norm`` (use ``numpy.linalg.norm(a, ord=2)`` instead)
  - ``log2`` (use `numpy.log2` instead)
  - ``longest_contiguous_ones``
  - ``longest_ones``
  - ``movavg``
  - ``norm_flat`` (use ``numpy.linalg.norm(a.flat, ord=2)`` instead)
  - ``normpdf`` (use `scipy.stats.norm.pdf` instead)
  - ``path_length``
  - ``poly_below``
  - ``poly_between``
  - ``prctile`` (use `numpy.percentile` instead)
  - ``prctile_rank``
  - ``rec2csv`` (use `numpy.recarray.tofile` instead)
  - ``rec_append_fields``
  - ``rec_drop_fields``
  - ``rec_join``
  - ``rk4`` (use `scipy.integrate.ode` instead)
  - ``rms_flat``
  - ``segments_intersect``
  - ``slopes``
  - ``stineman_interp``
  - ``vector_lengths``

Deprecations
------------

``axes3d.Axes3D.w_xaxis``, ``.w_yaxis``, and ``.w_zaxis`` are deprecated (use
``.xaxis``, ``.yaxis``, and ``.zaxis`` instead).


Autoscaling changes
-------------------

On log-axes where a single value is plotted at a "full" decade (1, 10, 100,
etc.), the autoscaling now expands the axis symmetrically around that point,
instead of adding a decade only to the right.

`FigureCanvasAgg.buffer_rgba` and `RendererAgg.buffer_rgba` now return a memoryview
-----------------------------------------------------------------------------------

The ``buffer_rgba`` method now allows direct access to the renderer's
underlying buffer (as a ``(m, n, 4)``-shape memoryview) rather than copying the
data to a new bytestring.  This is consistent with the behavior on Py2, where a
buffer object was returned.

PathCollections created with `~.Axes.scatter` now keep track of invalid points
------------------------------------------------------------------------------

Previously, points with nonfinite (infinite or nan) coordinates would not be
included in the offsets (as returned by `PathCollection.get_offsets`) of a
`PathCollection` created by `~.Axes.scatter`, and points with nonfinite values
(as specified by the *c* kwarg) would not be included in the array (as returned
by `PathCollection.get_array`)

Such points are now included, but masked out by returning a masked array.

If the *plotnonfinite* kwarg to `~.Axes.scatter` is set, then points with
nonfinite values are plotted using the bad color of the `PathCollection`\ 's
colormap (as set by `Colormap.set_bad`).

Changed exceptions
------------------

- `mpl_toolkits.axes_grid1.axes_size.GetExtentHelper` now raises `ValueError`
  for invalid directions instead of `KeyError`.

ghostscript 8 support removed
-----------------------------

Support for ghostscript 8.60 (released in 2007) has been removed.  The oldest
supported version of ghostscript is now 9.0 (released in 2010).
