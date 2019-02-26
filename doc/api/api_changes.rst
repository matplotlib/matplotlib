
=============
 API Changes
=============

A log of changes to the most recent version of Matplotlib that affect the
outward-facing API. If updating Matplotlib breaks your scripts, this list may
help you figure out what caused the breakage and how to fix it by updating
your code. For API changes in older versions see :doc:`api_changes_old`.

For new features that were added to Matplotlib, see :ref:`whats-new`.

This pages lists API changes for the most recent version of Matplotlib.

.. toctree::
   :maxdepth: 1

   api_changes_old

..

   .. note::

     The list below is a table of contents of individual files from the 'next_api_changes' folder.
     When a release is made

       - The full text list below should be moved into its own file in
         'prev_api_changes' for minor and major versions, add sections at
         the top for bug-fix releases.
       - All the files in 'next_api_changes' should be moved to the bottom of this page
       - This note, and the toctree below should be commented out


      .. toctree::
         :glob:
         :maxdepth: 1

         next_api_changes/*

API Changes for 3.0.3
=====================

matplotlib.font_manager.win32InstalledFonts return value
--------------------------------------------------------

`matplotlib.font_manager.win32InstalledFonts` returns an empty list instead
of None if no fonts are found.


Matplotlib.use now has an ImportError for interactive backend
-------------------------------------------------------------

Switching backends via `matplotlib.use` is now allowed by default,
regardless of whether `matplotlib.pyplot` has been imported. If the user
tries to switch from an already-started interactive backend to a different
interactive backend, an ImportError will be raised.

API Changes for 3.0.1
=====================

`.tight_layout.auto_adjust_subplotpars` can return ``None`` now if the new
subplotparams will collapse axes to zero width or height.  This prevents
``tight_layout`` from being executed.  Similarly
`.tight_layout.get_tight_layout_figure` will return None.

API Changes for 3.0.0
=====================

Drop support for python 2
-------------------------

Matplotlib 3 only supports python 3.5 and higher.


Changes to backend loading
--------------------------

Failure to load backend modules (``macosx`` on non-framework builds and
``gtk3`` when running headless) now raises `ImportError` (instead of
`RuntimeError` and `TypeError`, respectively).

Third-party backends that integrate with an interactive framework are now
encouraged to define the ``required_interactive_framework`` global value to one
of the following values: "qt5", "qt4", "gtk3", "wx", "tk", or "macosx". This
information will be used to determine whether it is possible to switch from a
backend to another (specifically, whether they use the same interactive
framework).



`.Axes.hist2d` now uses `~.Axes.pcolormesh` instead of `~.Axes.pcolorfast`
--------------------------------------------------------------------------

`.Axes.hist2d` now uses `~.Axes.pcolormesh` instead of `~.Axes.pcolorfast`,
which will improve the handling of log-axes.  Note that the
returned *image* now is of type `~.matplotlib.collections.QuadMesh`
instead of `~.matplotlib.image.AxesImage`.

`.matplotlib.axes.Axes.get_tightbbox` now includes all artists
--------------------------------------------------------------

For Matplotlib 3.0, *all* artists are now included in the bounding box
returned by `.matplotlib.axes.Axes.get_tightbbox`.

`.matplotlib.axes.Axes.get_tightbbox` adds a new kwarg ``bbox_extra_artists``
to manually specify the list of artists on the axes to include in the
tight bounding box calculation.

Layout tools like `.Figure.tight_layout`, ``constrained_layout``,
and ``fig.savefig('fname.png', bbox_inches="tight")`` use
`.matplotlib.axes.Axes.get_tightbbox` to determine the bounds of each axes on
a figure and adjust spacing between axes.

In Matplotlib 2.2 ``get_tightbbox`` started to include legends made on the
axes, but still excluded some other artists, like text that may overspill an
axes.  This has been expanded to include *all* artists.

This new default may be overridden in either of three ways:

1. Make the artist to be excluded a child of the figure, not the axes. E.g.,
   call ``fig.legend()`` instead of ``ax.legend()`` (perhaps using
   `~.matplotlib.axes.Axes.get_legend_handles_labels` to gather handles and
   labels from the parent axes).
2. If the artist is a child of the axes, set the artist property
   ``artist.set_in_layout(False)``.
3. Manually specify a list of artists in the new kwarg ``bbox_extra_artists``.


`Text.set_text` with string argument ``None`` sets string to empty
------------------------------------------------------------------

`Text.set_text` when passed a string value of ``None`` would set the
string to ``"None"``, so subsequent calls to `Text.get_text` would return
the ambiguous ``"None"`` string.

This change sets text objects passed ``None`` to have empty strings, so that
`Text.get_text` returns an empty string.




``Axes3D.get_xlim``, ``get_ylim`` and ``get_zlim`` now return a tuple
---------------------------------------------------------------------

They previously returned an array.  Returning a tuple is consistent with the
behavior for 2D axes.




``font_manager.list_fonts`` now follows the platform's casefolding semantics
----------------------------------------------------------------------------

i.e., it behaves case-insensitively on Windows only.


``bar`` / ``barh`` no longer accepts ``left`` / ``bottom`` as first named argument
----------------------------------------------------------------------------------

These arguments were renamed in 2.0 to ``x`` / ``y`` following the change of the
default alignment from ``edge`` to ``center``.


Different exception types for undocumented options
--------------------------------------------------

- Passing ``style='comma'`` to :meth:`~matplotlib.axes.Axes.ticklabel_format`
  was never supported.  It now raises ``ValueError`` like all other
  unsupported styles, rather than ``NotImplementedError``.

- Passing the undocumented ``xmin`` or ``xmax`` arguments to
  :meth:`~matplotlib.axes.Axes.set_xlim` would silently override the ``left``
  and ``right`` arguments.  :meth:`~matplotlib.axes.Axes.set_ylim` and the
  3D equivalents (e.g. :meth:`~mpl_toolkits.axes.Axes3D.set_zlim3d`) had a
  corresponding problem.
  The ``_min`` and ``_max`` arguments are now deprecated, and a ``TypeError``
  will be raised if they would override the earlier limit arguments.


Improved call signature for ``Axes.margins``
--------------------------------------------

:meth:`matplotlib.axes.Axes.margins` and :meth:`mpl_toolkits.mplot3d.Axes3D.margins`
no longer accept arbitrary keywords. ``TypeError`` will therefore be raised
if unknown kwargs are passed; previously they would be silently ignored.

If too many positional arguments are passed, ``TypeError`` will be raised
instead of ``ValueError``, for consistency with other call-signature violations.

``Axes3D.margins`` now raises ``TypeError`` instead of emitting a deprecation
warning if only two positional arguments are passed.  To supply only ``x`` and
``y`` margins, use keyword arguments.



Explicit arguments instead of \*args, \*\*kwargs
------------------------------------------------

:PEP:`3102` describes keyword-only arguments, which allow Matplotlib
to provide explicit call signatures - where we previously used
``*args, **kwargs`` and ``kwargs.pop``, we can now expose named
arguments.  In some places, unknown kwargs were previously ignored but
now raise ``TypeError`` because ``**kwargs`` has been removed.

- :meth:`matplotlib.axes.Axes.stem` no longer accepts unknown keywords,
  and raises ``TypeError`` instead of emitting a deprecation.
- :meth:`matplotlib.axex.Axes.stem` now raises TypeError when passed
  unhandled positional arguments.  If two or more arguments are passed
  (ie X, Y, [linefmt], ...) and Y cannot be cast to an array, an error
  will be raised instead of treating X as Y and Y as linefmt.
- :meth:`mpl_toolkits.axes_grid1.axes_divider.SubPlotDivider` raises
  ``TypeError`` instead of ``Exception`` when passed unknown kwargs.



Cleanup decorators and test classes no longer destroy warnings filter on exit
-----------------------------------------------------------------------------

The decorators and classes in matplotlib.testing.decorators no longer
destroy the warnings filter on exit. Instead, they restore the warnings
filter that existed before the test started using ``warnings.catch_warnings``.


Non-interactive FigureManager classes are now aliases of FigureManagerBase
--------------------------------------------------------------------------

The `FigureManagerPdf`, `FigureManagerPS`, and `FigureManagerSVG` classes,
which were previously empty subclasses of `FigureManagerBase` (i.e., not
adding or overriding any attribute or method), are now direct aliases for
`FigureManagerBase`.


Change to the output of `.image.thumbnail`
------------------------------------------

When called with ``preview=False``, `.image.thumbnail` previously returned an
figure whose canvas class was set according to the output file extension.  It
now returns a figure whose canvas class is the base `FigureCanvasBase` (and
relies on `FigureCanvasBase.print_figure`) to handle the canvas switching
properly).

As a side effect of this change, `.image.thumbnail` now also supports .ps, .eps,
and .svgz output.



`.FuncAnimation` now draws artists according to their zorder when blitting
--------------------------------------------------------------------------

`.FuncAnimation` now draws artists returned by the user-
function according to their zorder when using blitting,
instead of using the order in which they are being passed.
However, note that only zorder of passed artists will be
respected, as they are drawn on top of any existing artists
(see `#11369 <https://github.com/matplotlib/matplotlib/issues/11369>`_).


Contour color autoscaling improvements
--------------------------------------

Selection of contour levels is now the same for contour and
contourf; previously, for contour, levels outside the data range were
deleted.  (Exception: if no contour levels are found within the
data range, the `levels` attribute is replaced with a list holding
only the minimum of the data range.)

When contour is called with levels specified as a target number rather
than a list, and the 'extend' kwarg is used, the levels are now chosen
such that some data typically will fall in the extended range.

When contour is called with a `LogNorm` or a `LogLocator`, it will now
select colors using the geometric mean rather than the arithmetic mean
of the contour levels.


Streamplot last row and column fixed
------------------------------------

A bug was fixed where the last row and column of data in
`~.Axes.axes.streamplot` were being dropped.


Changed default `AutoDateLocator` kwarg *interval_multiples* to ``True``
------------------------------------------------------------------------

The default value of the tick locator for dates, `.dates.AutoDateLocator`
kwarg *interval_multiples* was set to ``False`` which leads to not-nice
looking automatic ticks in many instances.  The much nicer
``interval_multiples=True`` is the new default.  See below to get the
old behavior back:

  .. plot::

    import matplotlib.pyplot as plt
    import datetime
    import matplotlib.dates as mdates

    t0 = datetime.datetime(2009, 8, 20, 1, 10, 12)
    tf = datetime.datetime(2009, 8, 20, 1, 42, 11)


    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    ax = axs[0]
    ax.axhspan(t0, tf, facecolor="blue", alpha=0.25)
    ax.set_ylim(t0 - datetime.timedelta(minutes=3),
                tf + datetime.timedelta(minutes=3))
    ax.set_title('NEW DEFAULT')

    ax = axs[1]
    ax.axhspan(t0, tf, facecolor="blue", alpha=0.25)
    ax.set_ylim(t0 - datetime.timedelta(minutes=3),
                tf + datetime.timedelta(minutes=3))
    # old behavior
    locator = mdates.AutoDateLocator(interval_multiples=False, )
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(mdates.AutoDateFormatter(locator))

    ax.set_title('OLD')
    plt.show()


`.Axes.get_position` now returns actual position if aspect changed
------------------------------------------------------------------

`.Axes.get_position` used to return the original position unless a
draw had been triggered or `.Axes.apply_aspect` had been called, even
if the kwarg *original* was set to ``False``.   Now `.Axes.apply_aspect`
is called so ``ax.get_position()`` will return the new modified position.
To get the old behavior use ``ax.get_position(original=True)``.


The ticks for colorbar now adjust for the size of the colorbar
--------------------------------------------------------------

Colorbar ticks now adjust for the size of the colorbar if the
colorbar is made from a mappable that is not a contour or
doesn't have a BoundaryNorm, or boundaries are not specified.
If boundaries, etc are specified, the colorbar maintains the
original behavior.


Colorbar for log-scaled hexbin
------------------------------

When using `hexbin` and plotting with a logarithmic color scale, the colorbar
ticks are now correctly log scaled. Previously the tick values were linear
scaled log(number of counts).

PGF backend now explicitly makes black text black
-------------------------------------------------

Previous behavior with the pgf backend was for text specified as black to
actually be the default color of whatever was rendering the pgf file (which was
of course usually black). The new behavior is that black text is black,
regardless of the default color. However, this means that there is no way to
fall back on the default color of the renderer.


Blacklisted rcparams no longer updated by `rcdefaults`, `rc_file_defaults`, `rc_file`
-------------------------------------------------------------------------------------

The rc modifier functions `rcdefaults`, `rc_file_defaults` and `rc_file`
now ignore rcParams in the `matplotlib.style.core.STYLE_BLACKLIST` set.  In
particular, this prevents the ``backend`` and ``interactive`` rcParams from
being incorrectly modified by these functions.



`CallbackRegistry` now stores callbacks using stdlib's `WeakMethod`\s
---------------------------------------------------------------------

In particular, this implies that ``CallbackRegistry.callbacks[signal]`` is now
a mapping of callback ids to `WeakMethod`\s (i.e., they need to be first called
with no arguments to retrieve the method itself).


Changes regarding the text.latex.unicode rcParam
------------------------------------------------

The rcParam now defaults to True and is deprecated (i.e., in future versions
of Maplotlib, unicode input will always be supported).

Moreover, the underlying implementation now uses ``\usepackage[utf8]{inputenc}``
instead of ``\usepackage{ucs}\usepackage[utf8x]{inputenc}``.


Return type of ArtistInspector.get_aliases changed
--------------------------------------------------

`ArtistInspector.get_aliases` previously returned the set of aliases as
``{fullname: {alias1: None, alias2: None, ...}}``.  The dict-to-None mapping
was used to simulate a set in earlier versions of Python.  It has now been
replaced by a set, i.e. ``{fullname: {alias1, alias2, ...}}``.

This value is also stored in `ArtistInspector.aliasd`, which has likewise
changed.


Removed ``pytz`` as a dependency
--------------------------------

Since ``dateutil`` and ``pytz`` both provide time zones, and
matplotlib already depends on ``dateutil``, matplotlib will now use
``dateutil`` time zones internally and drop the redundant dependency
on ``pytz``. While ``dateutil`` time zones are preferred (and
currently recommended in the Python documentation), the explicit use
of ``pytz`` zones is still supported.

Deprecations
------------

Modules
```````
The following modules are deprecated:

- :mod:`matplotlib.compat.subprocess`. This was a python 2 workaround, but all
  the functionality can now be found in the python 3 standard library
  :mod:`subprocess`.
- :mod:`matplotlib.backends.wx_compat`. Python 3 is only compatible with
  wxPython 4, so support for wxPython 3 or earlier can be dropped.

Classes, methods, functions, and attributes
```````````````````````````````````````````

The following classes, methods, functions, and attributes are deprecated:

- ``RcParams.msg_depr``, ``RcParams.msg_depr_ignore``,
  ``RcParams.msg_depr_set``, ``RcParams.msg_obsolete``,
  ``RcParams.msg_backend_obsolete``
- ``afm.parse_afm``
- ``backend_pdf.PdfFile.texFontMap``
- ``backend_pgf.get_texcommand``
- ``backend_ps.get_bbox``
- ``backend_qt5.FigureCanvasQT.keyAutoRepeat`` (directly check
  ``event.guiEvent.isAutoRepeat()`` in the event handler to decide whether to
  handle autorepeated key presses).
- ``backend_qt5.error_msg_qt``, ``backend_qt5.exception_handler``
- ``backend_wx.FigureCanvasWx.macros``
- ``backends.pylab_setup``
- ``cbook.GetRealpathAndStat``, ``cbook.Locked``
- ``cbook.is_numlike`` (use ``isinstance(..., numbers.Number)`` instead),
  ``cbook.listFiles``, ``cbook.unicode_safe``
- ``container.Container.set_remove_method``,
- ``contour.ContourLabeler.cl``, ``.cl_xy``, and ``.cl_cvalues``
- ``dates.DateFormatter.strftime_pre_1900``, ``dates.DateFormatter.strftime``
- ``font_manager.TempCache``
- ``image._ImageBase.iterpnames``, use the ``interpolation_names`` property
  instead. (this affects classes that inherit from ``_ImageBase`` including
  :class:`FigureImage`, :class:`BboxImage`, and :class:`AxesImage`)
- ``mathtext.unichr_safe`` (use ``chr`` instead)
- ``patches.Polygon.xy``
- ``table.Table.get_child_artists`` (use ``get_children`` instead)
- ``testing.compare.ImageComparisonTest``, ``testing.compare.compare_float``
- ``testing.decorators.CleanupTest``,
  ``testing.decorators.skip_if_command_unavailable``
- ``FigureCanvasQT.keyAutoRepeat`` (directly check
  ``event.guiEvent.isAutoRepeat()`` in the event handler to decide whether to
  handle autorepeated key presses)
- ``FigureCanvasWx.macros``
- ``_ImageBase.iterpnames``, use the ``interpolation_names`` property instead.
  (this affects classes that inherit from ``_ImageBase`` including
  :class:`FigureImage`, :class:`BboxImage`, and :class:`AxesImage`)
- ``patches.Polygon.xy``
- ``texmanager.dvipng_hack_alpha``
- ``text.Annotation.arrow``
- `.Legend.draggable()`, in favor of `.Legend.set_draggable()`
   (``Legend.draggable`` may be reintroduced as a property in future releases)
- ``textpath.TextToPath.tex_font_map``
- :class:`matplotlib.cbook.deprecation.mplDeprecation` will be removed
  in future versions. It is just an alias for
  :class:`matplotlib.cbook.deprecation.MatplotlibDeprecationWarning`.
  Please use the
  :class:`~matplotlib.cbook.MatplotlibDeprecationWarning` directly if
  neccessary.
- The ``matplotlib.cbook.Bunch`` class has been deprecated. Instead, use
  `types.SimpleNamespace` from the standard library which provides the same
  functionality.
- ``Axes.mouseover_set`` is now a frozenset, and deprecated.  Directly
  manipulate the artist's ``.mouseover`` attribute to change their mouseover
  status.

The following keyword arguments are deprecated:

- passing ``verts`` to ``Axes.scatter`` (use ``marker`` instead)
- passing ``obj_type`` to ``cbook.deprecated``

The following call signatures are deprecated:

- passing a ``wx.EvtHandler`` as first argument to ``backend_wx.TimerWx``


rcParams
````````

The following rcParams are deprecated:

- ``examples.directory`` (use ``datapath`` instead)
- ``pgf.debug`` (the pgf backend relies on logging)
- ``text.latex.unicode`` (always True now)


marker styles
`````````````
- Using ``(n, 3)`` as marker style to specify a circle marker is deprecated.  Use
  ``"o"`` instead.
- Using ``([(x0, y0), (x1, y1), ...], 0)`` as marker style to specify a custom
  marker path is deprecated.  Use ``[(x0, y0), (x1, y1), ...]`` instead.


Deprecation of ``LocatableAxes`` in toolkits
````````````````````````````````````````````

The ``LocatableAxes`` classes in toolkits have been deprecated. The base `Axes`
classes provide the same functionality to all subclasses, thus these mixins are
no longer necessary. Related functions have also been deprecated. Specifically:

* ``mpl_toolkits.axes_grid1.axes_divider.LocatableAxesBase``: no specific
  replacement; use any other ``Axes``-derived class directly instead.
* ``mpl_toolkits.axes_grid1.axes_divider.locatable_axes_factory``: no specific
  replacement; use any other ``Axes``-derived class directly instead.
* ``mpl_toolkits.axes_grid1.axes_divider.Axes``: use
  `mpl_toolkits.axes_grid1.mpl_axes.Axes` directly.
* ``mpl_toolkits.axes_grid1.axes_divider.LocatableAxes``: use
  `mpl_toolkits.axes_grid1.mpl_axes.Axes` directly.
* ``mpl_toolkits.axisartist.axes_divider.Axes``: use
  `mpl_toolkits.axisartist.axislines.Axes` directly.
* ``mpl_toolkits.axisartist.axes_divider.LocatableAxes``: use
  `mpl_toolkits.axisartist.axislines.Axes` directly.

Removals
--------

Hold machinery
``````````````

Setting or unsetting ``hold`` (:ref:`deprecated in version 2.0<v200_deprecate_hold>`) has now
been completely removed. Matplotlib now always behaves as if ``hold=True``.
To clear an axes you can manually use :meth:`~.axes.Axes.cla()`,
or to clear an entire figure use :meth:`~.figure.Figure.clf()`.


Removal of deprecated backends
``````````````````````````````

Deprecated backends have been removed:

- GTKAgg
- GTKCairo
- GTK
- GDK


Deprecated APIs
```````````````

The following deprecated API elements have been removed:

- The deprecated methods ``knownfailureif`` and ``remove_text`` have been removed
  from :mod:`matplotlib.testing.decorators`.
- The entire contents of ``testing.noseclasses`` have also been removed.
- ``matplotlib.checkdep_tex``, ``matplotlib.checkdep_xmllint``
- ``backend_bases.IdleEvent``
- ``cbook.converter``, ``cbook.tostr``, ``cbook.todatetime``, ``cbook.todate``,
  ``cbook.tofloat``, ``cbook.toint``, ``cbook.unique``,
  ``cbook.is_string_like``, ``cbook.is_sequence_of_strings``,
  ``cbook.is_scalar``, ``cbook.soundex``, ``cbook.dict_delall``,
  ``cbook.get_split_ind``, ``cbook.wrap``, ``cbook.get_recursive_filelist``,
  ``cbook.pieces``, ``cbook.exception_to_str``, ``cbook.allequal``,
  ``cbook.alltrue``, ``cbook.onetrue``, ``cbook.allpairs``, ``cbook.finddir``,
  ``cbook.reverse_dict``, ``cbook.restrict_dict``, ``cbook.issubclass_safe``,
  ``cbook.recursive_remove``, ``cbook.unmasked_index_ranges``,
  ``cbook.Null``, ``cbook.RingBuffer``, ``cbook.Sorter``, ``cbook.Xlator``,
- ``font_manager.weight_as_number``, ``font_manager.ttfdict_to_fnames``
- ``pyplot.colors``, ``pyplot.spectral``
- ``rcsetup.validate_negative_linestyle``,
  ``rcsetup.validate_negative_linestyle_legacy``,
- ``testing.compare.verifiers``, ``testing.compare.verify``
- ``testing.decorators.knownfailureif``,
  ``testing.decorators.ImageComparisonTest.remove_text``
- ``tests.assert_str_equal``, ``tests.test_tinypages.file_same``
- ``texmanager.dvipng_hack_alpha``,
- ``_AxesBase.axesPatch``, ``_AxesBase.set_color_cycle``,
  ``_AxesBase.get_cursor_props``, ``_AxesBase.set_cursor_props``
- ``_ImageBase.iterpnames``
- ``FigureCanvasBase.start_event_loop_default``;
- ``FigureCanvasBase.stop_event_loop_default``;
- ``Figure.figurePatch``,
- ``FigureCanvasBase.dynamic_update``, ``FigureCanvasBase.idle_event``,
  ``FigureCanvasBase.get_linestyle``, ``FigureCanvasBase.set_linestyle``
- ``FigureCanvasQTAggBase``
- ``FigureCanvasQTAgg.blitbox``
- ``FigureCanvasTk.show`` (alternative: ``FigureCanvasTk.draw``)
- ``FigureManagerTkAgg`` (alternative: ``FigureManagerTk``)
- ``NavigationToolbar2TkAgg`` (alternative: ``NavigationToolbar2Tk``)
- ``backend_wxagg.Toolbar`` (alternative: ``backend_wxagg.NavigationToolbar2WxAgg``)
- ``RendererAgg.debug()``
- passing non-numbers to ``EngFormatter.format_eng``
- passing ``frac`` to ``PolarAxes.set_theta_grids``
- any mention of idle events

The following API elements have been removed:

- ``backend_cairo.HAS_CAIRO_CFFI``
- ``sphinxext.sphinx_version``


Proprietary sphinx directives
`````````````````````````````

The matplotlib documentation used the proprietary sphinx directives
`.. htmlonly::`, and `.. latexonly::`. These have been replaced with the
standard sphinx directives `.. only:: html` and `.. only:: latex`. This
change will not affect any users. Only downstream package maintainers, who
have used the proprietary directives in their docs, will have to switch to the
sphinx directives.


lib/mpl_examples symlink
````````````````````````

The symlink from lib/mpl_examples to ../examples has been removed.
This is not installed as an importable package and should not affect
end users, however this may require down-stream packagers to adjust.
The content is still available top-level examples directory.
