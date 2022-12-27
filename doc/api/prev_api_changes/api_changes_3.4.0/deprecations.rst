Deprecations
------------

Extra parameters to Axes constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters of the Axes constructor other than *fig* and *rect* will become
keyword-only in a future version.

``pyplot.gca`` and ``Figure.gca`` keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing keyword arguments to `.pyplot.gca` or `.figure.Figure.gca` will not be
supported in a future release.

``Axis.cla``, ``RadialAxis.cla``, ``ThetaAxis.cla`` and ``Spine.cla``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods are deprecated in favor of the respective ``clear()`` methods.

Invalid hatch pattern characters are no longer ignored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When specifying hatching patterns, characters that are not recognized will
raise a deprecation warning. In the future, this will become a hard error.

``imread`` reading from URLs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing a URL to `~.pyplot.imread()` is deprecated. Please open the URL for
reading and directly use the Pillow API
(``PIL.Image.open(urllib.request.urlopen(url))``, or
``PIL.Image.open(io.BytesIO(requests.get(url).content))``) instead.

Subplot-related attributes and methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some ``SubplotBase`` methods and attributes have been deprecated and/or moved
to `.SubplotSpec`:

- ``get_geometry`` (use ``SubplotBase.get_subplotspec`` instead),
- ``change_geometry`` (use ``SubplotBase.set_subplotspec`` instead),
- ``is_first_row``, ``is_last_row``, ``is_first_col``, ``is_last_col`` (use the
  corresponding methods on the `.SubplotSpec` instance instead),
- ``update_params`` (now a no-op),
- ``figbox`` (use ``ax.get_subplotspec().get_geometry(ax.figure)`` instead to
  recompute the geometry, or ``ax.get_position()`` to read its current value),
- ``numRows``, ``numCols`` (use the ``nrows`` and ``ncols`` attribute on the
  `.GridSpec` instead).

Likewise, the ``get_geometry``, ``change_geometry``, ``update_params``, and
``figbox`` methods/attributes of `.SubplotDivider` have been deprecated, with
similar replacements.

``is_url`` and ``URL_REGEX``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are deprecated. (They were previously defined in the toplevel
:mod:`matplotlib` module.)

``matplotlib.style.core`` deprecations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``STYLE_FILE_PATTERN``, ``load_base_library``, and ``iter_user_libraries`` are
deprecated.

``dpi_cor`` property of `.FancyArrowPatch`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This parameter is considered internal and deprecated.

Passing ``boxstyle="custom", bbox_transmuter=...`` to ``FancyBboxPatch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to use a custom boxstyle, directly pass it as the *boxstyle* argument
to `.FancyBboxPatch`. This was previously already possible, and is consistent
with custom arrow styles and connection styles.

BoxStyles are now called without passing the *mutation_aspect* parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mutation aspect is now handled by the artist itself. Hence the
*mutation_aspect* parameter of ``BoxStyle._Base.__call__`` is deprecated, and
custom boxstyles should be implemented to not require this parameter (it can be
left as a parameter defaulting to 1 for back-compatibility).

``ContourLabeler.get_label_coords`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is considered an internal helper.

Line2D and Patch no longer duplicate ``validJoin`` and ``validCap``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validation of joinstyle and capstyles is now centralized in ``rcsetup``.

Setting a Line2D's pickradius via ``set_picker`` is undeprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This cancels the deprecation introduced in Matplotlib 3.3.0.

``MarkerStyle`` is considered immutable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``MarkerStyle.set_fillstyle()`` and ``MarkerStyle.set_marker()`` are
deprecated. Create a new ``MarkerStyle`` with the respective parameters
instead.

``MovieWriter.cleanup`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cleanup logic is now fully implemented in `.MovieWriter.finish`. Third-party
movie writers should likewise move the relevant cleanup logic there, as
overridden ``cleanup``\s will no longer be called in the future.

*minimumdescent* parameter/property of ``TextArea``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.offsetbox.TextArea` has behaved as if *minimumdescent* was always True
(regardless of the value to which it was set) since Matplotlib 1.3, so the
parameter/property is deprecated.

``colorbar`` now warns when the mappable's Axes is different from the current Axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, `.Figure.colorbar` and `.pyplot.colorbar` steal space by default
from the current Axes to place the colorbar. In a future version, they will
steal space from the mappable's Axes instead. In preparation for this change,
`.Figure.colorbar` and `.pyplot.colorbar` now emits a warning when the current
Axes is not the same as the mappable's Axes.

Colorbar docstrings
~~~~~~~~~~~~~~~~~~~

The following globals in :mod:`matplotlib.colorbar` are deprecated:
``colorbar_doc``, ``colormap_kw_doc``, ``make_axes_kw_doc``.

``ColorbarPatch`` and ``colorbar_factory`` are deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All the relevant functionality has been moved to the
`~matplotlib.colorbar.Colorbar` class.

Backend deprecations
~~~~~~~~~~~~~~~~~~~~

- ``FigureCanvasBase.get_window_title`` and
  ``FigureCanvasBase.set_window_title`` are deprecated. Use the corresponding
  methods on the FigureManager if using pyplot, or GUI-specific methods if
  embedding.
- The *resize_callback* parameter to ``FigureCanvasTk`` was never used
  internally and is deprecated. Tk-level custom event handlers for resize
  events can be added to a ``FigureCanvasTk`` using e.g.
  ``get_tk_widget().bind('<Configure>', ..., True)``.
- The ``key_press`` and ``button_press`` methods of `.FigureManagerBase`, which
  incorrectly did nothing when using ``toolmanager``, are deprecated in favor
  of directly passing the event to the `.CallbackRegistry` via
  ``self.canvas.callbacks.process(event.name, event)``.
- ``RendererAgg.get_content_extents`` and
  ``RendererAgg.tostring_rgba_minimized`` are deprecated.
- ``backend_pgf.TmpDirCleaner`` is deprecated, with no replacement.
- ``GraphicsContextPS`` is deprecated. The PostScript backend now uses
  `.GraphicsContextBase`.

wx backend cleanups
~~~~~~~~~~~~~~~~~~~

The *origin* parameter to ``_FigureCanvasWxBase.gui_repaint`` is deprecated
with no replacement; ``gui_repaint`` now automatically detects the case where
it is used with the wx renderer.

The ``NavigationToolbar2Wx.get_canvas`` method is deprecated; directly
instantiate a canvas (``FigureCanvasWxAgg(frame, -1, figure)``) if needed.

Unused positional parameters to ``print_<fmt>`` methods are deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

None of the ``print_<fmt>`` methods implemented by canvas subclasses used
positional arguments other that the first (the output filename or file-like),
so these extra parameters are deprecated.

The *dpi* parameter of ``FigureCanvas.print_foo`` printers is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `~.Figure.savefig` machinery already took care of setting the figure DPI
to the desired value, so ``print_foo`` can directly read it from there. Not
passing *dpi* to ``print_foo`` allows clearer detection of unused parameters
passed to `~.Figure.savefig`.

Passing `bytes` to ``FT2Font.set_text``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated, pass `str` instead.

``ps.useafm`` deprecated for mathtext
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Outputting mathtext using only standard PostScript fonts has likely been broken
for a while (issue `#18722
<https://github.com/matplotlib/matplotlib/issues/18722>`_). In Matplotlib 3.5,
the setting :rc:`ps.useafm` will have no effect on mathtext.

``MathTextParser("bitmap")`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The associated APIs ``MathtextBackendBitmap``, ``MathTextParser.to_mask``,
``MathTextParser.to_rgba``, ``MathTextParser.to_png``, and
``MathTextParser.get_depth`` are likewise deprecated.

To convert a text string to an image, either directly draw the text to an
empty `.Figure` and save the figure using a tight bbox, as demonstrated in
:doc:`/gallery/text_labels_and_annotations/mathtext_asarray`, or use
`.mathtext.math_to_image`.

When using `.math_to_image`, text color can be set with e.g.::

    with plt.rc_context({"text.color": "tab:blue"}):
        mathtext.math_to_image(text, filename)

and an RGBA array can be obtained with e.g.::

    from io import BytesIO
    buf = BytesIO()
    mathtext.math_to_image(text, buf, format="png")
    buf.seek(0)
    rgba = plt.imread(buf)

Deprecation of mathtext internals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following API elements previously exposed by the :mod:`.mathtext` module
are considered to be implementation details and public access to them is
deprecated:

- ``Fonts`` and all its subclasses,
- ``FontConstantsBase`` and all its subclasses,
- ``Node`` and all its subclasses,
- ``Ship``, ``ship``,
- ``Error``,
- ``Parser``,
- ``SHRINK_FACTOR``, ``GROW_FACTOR``,
- ``NUM_SIZE_LEVELS``,
- ``latex_to_bakoma``, ``latex_to_cmex``, ``latex_to_standard``,
- ``stix_virtual_fonts``,
- ``tex2uni``.

Deprecation of various mathtext helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MathtextBackendPdf``, ``MathtextBackendPs``, ``MathtextBackendSvg``,
and ``MathtextBackendCairo`` classes from the :mod:`.mathtext` module, as
well as the corresponding ``.mathtext_parser`` attributes on ``RendererPdf``,
``RendererPS``, ``RendererSVG``, and ``RendererCairo``, are deprecated. The
``MathtextBackendPath`` class can be used to obtain a list of glyphs and
rectangles in a mathtext expression, and renderer-specific logic should be
directly implemented in the renderer.

``StandardPsFonts.pswriter`` is unused and deprecated.

Widget class internals
~~~~~~~~~~~~~~~~~~~~~~

Several `.widgets.Widget` class internals have been privatized and deprecated:

- ``AxesWidget.cids``
- ``Button.cnt`` and ``Button.observers``
- ``CheckButtons.cnt`` and ``CheckButtons.observers``
- ``RadioButtons.cnt`` and ``RadioButtons.observers``
- ``Slider.cnt`` and ``Slider.observers``
- ``TextBox.cnt``, ``TextBox.change_observers`` and
  ``TextBox.submit_observers``

3D properties on renderers
~~~~~~~~~~~~~~~~~~~~~~~~~~

The properties of the 3D Axes that were placed on the Renderer during draw are
now deprecated:

- ``renderer.M``
- ``renderer.eye``
- ``renderer.vvec``
- ``renderer.get_axis_position``

These attributes are all available via `.Axes3D`, which can be accessed via
``self.axes`` on all `.Artist`\s.

*renderer* argument of ``do_3d_projection`` method for ``Collection3D``/``Patch3D``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *renderer* argument for the ``do_3d_projection`` method on ``Collection3D``
and ``Patch3D`` is no longer necessary, and passing it during draw is
deprecated.

*project* argument of ``draw`` method for ``Line3DCollection``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *project* argument for the ``draw`` method on ``Line3DCollection`` is
deprecated. Call `.Line3DCollection.do_3d_projection` explicitly instead.

Extra positional parameters to ``plot_surface`` and ``plot_wireframe``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Positional parameters to `~.axes3d.Axes3D.plot_surface` and
`~.axes3d.Axes3D.plot_wireframe` other than ``X``, ``Y``, and ``Z`` are
deprecated. Pass additional artist properties as keyword arguments instead.

``ParasiteAxesAuxTransBase`` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The functionality of that mixin class has been moved to the base
``ParasiteAxesBase`` class. Thus, ``ParasiteAxesAuxTransBase``,
``ParasiteAxesAuxTrans``, and ``parasite_axes_auxtrans_class_factory`` are
deprecated.

In general, it is suggested to use ``HostAxes.get_aux_axes`` to create
parasite Axes, as this saves the need of manually appending the parasite
to ``host.parasites`` and makes sure that their ``remove()`` method works
properly.

``AxisArtist.ZORDER`` attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``AxisArtist.zorder`` instead.

``GridHelperBase`` invalidation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``GridHelperBase.invalidate``, ``GridHelperBase.valid``, and
``axislines.Axes.invalidate_grid_helper`` methods are considered internal
and deprecated.

``sphinext.plot_directive.align``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated. Use ``docutils.parsers.rst.directives.images.Image.align``
instead.

Deprecation-related functionality is considered internal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module ``matplotlib.cbook.deprecation`` is considered internal and will be
removed from the public API. This also holds for deprecation-related re-imports
in ``matplotlib.cbook``, i.e. ``matplotlib.cbook.deprecated()``,
``matplotlib.cbook.warn_deprecated()``,
``matplotlib.cbook.MatplotlibDeprecationWarning`` and
``matplotlib.cbook.mplDeprecation``.

If needed, external users may import ``MatplotlibDeprecationWarning`` directly
from the ``matplotlib`` namespace. ``mplDeprecation`` is only an alias of
``MatplotlibDeprecationWarning`` and should not be used anymore.
