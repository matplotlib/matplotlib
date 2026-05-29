Deprecations
------------


In-place modification of colormaps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Colormaps are planned to become immutable in the long term.

As a first step, in-place modifications of colormaps are now pending-deprecated. This
affects the following methods of `.Colormap`:

- `.Colormap.set_bad` - use ``cmap.with_extremes(bad=...)`` instead
- `.Colormap.set_under` - use ``cmap.with_extremes(under=...)`` instead
- `.Colormap.set_over` - use ``cmap.with_extremes(over=...)`` instead
- `.Colormap.set_extremes` - use ``cmap.with_extremes(...)`` instead

Use the respective `.Colormap.with_extremes` and appropriate keyword arguments instead
which returns a copy of the colormap (available since Matplotlib 3.4). Alternatively, if
you create the colormap yourself, you can also pass the respective arguments to the
constructor (available since Matplotlib 3.11).

Contour labelling on filled contours
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using `~.Axes.clabel` to label filled contours created with `~.Axes.contourf` is
deprecated. ``clabel()`` is designed to label contour lines (`.Axes.contour`), and using
it with filled contours can lead to inconsistent plots. If you want to add labels to
filled contours, the recommended approach is to first create the filled contours with
`~.Axes.contourf`, then overlay contour lines using `~.Axes.contour`, and finally apply
`~.Axes.clabel` to those contour lines for labeling. For an example see
:doc:`/gallery/images_contours_and_fields/contourf_demo`.

``boxplot`` and ``bxp`` *vert* parameter, and ``rcParams["boxplot.vertical"]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter *vert: bool* has been deprecated on `~.Axes.boxplot` and `~.Axes.bxp`. It
is replaced by *orientation: {"vertical", "horizontal"}* for API consistency.

``rcParams["boxplot.vertical"]``, which controlled the orientation of ``boxplot``, is
deprecated without replacement.

``violinplot`` and ``violin`` *vert* parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter *vert: bool* has been deprecated on `~.Axes.violinplot` and
`~.Axes.violin`. It will be replaced by *orientation: {"vertical", "horizontal"}* for
API consistency.

Arbitrary code in ``axes.prop_cycle`` rcParam strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``axes.prop_cycle`` rcParam accepts Python expressions that are evaluated in a
limited context. The evaluation context has been further limited and some expressions
that previously worked (list comprehensions, for example) no longer will. This change is
made without a deprecation period to improve security. The previously documented cycler
operations at https://matplotlib.org/cycler/ are still supported.

Capitalization of None in matplotlibrc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In :file:`matplotlibrc` config files every capitalization of None was accepted for
denoting the Python constant `None`. This is now deprecated, and the only accepted
capitalization is None, i.e., starting with a capital letter and all other letters in
lowercase.

Third-party scales no longer need to have an *axis* parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since Matplotlib 3.1 `PR 12831 <https://github.com/matplotlib/matplotlib/pull/12831>`_
scale objects should be reusable and therefore independent of any particular Axis.
Therefore, the use of the *axis* parameter in the ``__init__`` had been discouraged.
However, having that parameter in the signature was still necessary for API
backwards-compatibility. This is no longer the case.

`.register_scale` now accepts scale classes with or without this parameter.

The *axis* parameter is pending-deprecated. It will be deprecated in Matplotlib 3.13,
and removed in Matplotlib 3.15.

Third-party scales are recommended to remove the *axis* parameter now if they can afford
to restrict compatibility to Matplotlib >= 3.11 already. Otherwise, they may keep the
*axis* parameter and remove it in time for Matplotlib 3.13.

``matplotlib.style.core``
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``matplotlib.style.core`` module is deprecated. All APIs intended for public use are
now available in `matplotlib.style` directly (including ``USER_LIBRARY_PATHS``, which
was previously not reexported).

The following APIs of ``matplotlib.style.core`` have been deprecated with no
replacement: ``BASE_LIBRARY_PATH``, ``STYLE_EXTENSION``, ``STYLE_BLACKLIST``,
``update_user_library``, ``read_style_directory``, ``update_nested_dict``.

Font hinting and kerning factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Due to internal changes to support complex text rendering, the hinting factor and
kerning factor on fonts are no longer used. Setting the ``text.hinting_factor`` or
``text.kerning_factor`` rcParams (the latter of which existed only for
backwards-compatibility) to any value other than None is deprecated, and they will be
removed in the future.

Likewise, passing the ``hinting_factor`` argument to the `.FT2Font` constructor is
deprecated.

``FT2Image`` image buffer
~~~~~~~~~~~~~~~~~~~~~~~~~

Use 2D uint8 ndarrays instead. In particular:

- The ``FT2Image`` constructor took ``width, height`` as separate parameters but the
  ndarray constructor takes ``(height, width)`` as single tuple parameter.
- `.FT2Font.draw_glyph_to_bitmap` now (also) takes 2D uint8 arrays as input.
- ``FT2Image.draw_rect_filled`` should be replaced by directly setting pixel values to
  black.
- The ``image`` attribute of the object returned by ``MathTextParser("agg").parse`` is
  now a 2D uint8 array.

``DviFont.widths``
~~~~~~~~~~~~~~~~~~

... is deprecated with no replacement.

``PdfFile`` internals
~~~~~~~~~~~~~~~~~~~~~

The ``PdfFile.dviFontInfo``, ``PdfFile.fontNames``, ``PdfFile.multi_byte_charprocs``,
and ``PdfFile.type1Descriptors`` attributes are deprecated with no replacement.

The *fontfile* parameter of ``PdfFile.createType1Descriptor`` is deprecated; all
relevant pieces of information are now directly extracted from the *t1font* argument.

``Tfm``'s internal metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct access to ``Tfm``'s ``widths``, ``heights``, ``depths`` dicts is deprecated;
access a glyph's metrics with `.Tfm.get_metrics` instead.

``font_manager.is_opentype_cff_font`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is no replacement.

``Axes.set_navigate_mode`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... with no replacement.

Parameters ``Axes3D.set_aspect(..., anchor=..., share=...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameters *anchor* and *share* of `.Axes3D.set_aspect` are deprecated. They had no
effect on 3D axes and will be removed in a future version.

``BezierSegment.point_at_t``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated. Instead, it is possible to call the BezierSegment with an argument.

Formatter attributes
~~~~~~~~~~~~~~~~~~~~

These following attributes are considered internal and users should not have a need to
access them:

- `.ScalarFormatter`: ``orderOfMagnitude`` and ``format``
- `.ConciseDateFormatter`: ``offset_format``
- `.Formatter`: ``locs``

Parameter ``ListedColormap(..., N=...)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing the parameter *N* to `.ListedColormap` is deprecated. Please preprocess the list
colors yourself if needed.

``kw``, ``fontproperties``, ``labelcolor``, and ``verts`` attributes of ``QuiverKey``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These attributes are deprecated (note that modifying ``fontproperties``, ``labelcolor``,
or ``verts`` after the first draw had no effect previously). Directly access the
relevant attributes on the sub-artists ``QuiverKey.vector`` and ``QuiverKey.text``,
instead.

``apply_theta_transforms`` option in ``PolarTransform``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applying theta transforms in `~matplotlib.projections.polar.PolarTransform` and
`~matplotlib.projections.polar.InvertedPolarTransform` has been removed, and the
*apply_theta_transforms* keyword argument is deprecated for both classes.

If you need to retain the behaviour where theta values are transformed, chain the
``PolarTransform`` with a `~matplotlib.transforms.Affine2D` transform that performs the
theta shift and/or sign shift.

*axes* parameter of ``RadialLocator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated. `~.polar.RadialLocator` now fetches the relevant information from the
Axis' parent Axes.

Transform helper functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following functions in the `.transforms` module are deprecated, because they are
considerer internal functionality and should not be used by end users:

- ``matplotlib.transforms.nonsingular``
- ``matplotlib.transforms.interval_contains``
- ``matplotlib.transforms.interval_contains_open``

``InvertedSymmetricalLogTransform.invlinthresh``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``invlinthresh`` attribute of `.InvertedSymmetricalLogTransform` is deprecated. Use
the ``.inverted().transform(linthresh)`` method instead.

:mod:`.axisartist` now uses more standard tick direction controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, the position of :mod:`.axisartist` ticks (inside or outside the axes) were
set using ``set_tick_out(bool)``. They are now set using ``set_tick_direction("in")``
(or "out", or "inout"), and respect :rc:`xtick.direction` and :rc:`ytick.direction`. In
particular, they default to pointing outwards, consistently with the rest of the
library.

The *tick_out* parameter of `.Ticks` has been deprecated (use *tick_direction* instead).
The ``Ticks.get_tick_out`` method is deprecated (use `.Ticks.get_tick_direction`
instead).

The unused ``locs_angles_labels`` attribute of `.Ticks` and `.LabelBase` has also been
deprecated.

``GridFinder.get_grid_info`` now takes a single bbox as parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Passing ``x1, y1, x2, y2`` as separate parameters is deprecated.

``GridFinder.transform_xy`` and ``GridFinder.inv_transform_xy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are deprecated. Directly use the standard transform returned by
`.GridFinder.get_transform` instead.

``axes_grid.Grid.ngrids``
~~~~~~~~~~~~~~~~~~~~~~~~~

This attribute has been deprecated and renamed ``n_axes``, consistently with the new
name of the `~.axes_grid.Grid` constructor parameter that allows setting the actual
number of axes in the grid (the old parameter, ``ngrids``, did not actually work since
Matplotlib 3.3).

The same change has been made in ``axes_grid.ImageGrid``.

*canvas* parameter to ``MultiCursor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... is deprecated. It has been unused for a while already.

Please remove the parameter and change the call from ``MultiCursor(canvas, axes)`` to
``MultiCursor(axes)``. Both calls are valid throughout the deprecation period.

``CallbackRegistry.disconnect`` *cid* parameter renamed to *cid_or_func*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *cid* parameter of `.CallbackRegistry.disconnect` has been renamed to *cid_or_func*.
The method now also accepts a callable, which will disconnect that callback from all
signals or from a specific signal if the *signal* keyword argument is provided.

``cbook.normalize_kwargs`` only supports passing artists and artist classes as second argument
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Support for directly passing an alias mapping or None as second argument to
`.cbook.normalize_kwargs` has been deprecated.

``backend_svg.XMLWriter`` is deprecated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is an internal helper not intended for external use.

``image.thumbnail``
~~~~~~~~~~~~~~~~~~~

... is deprecated without replacement. Use :external:py:meth:`Pillow's
thumbnail method <PIL.Image.Image.thumbnail>` instead. See also the `Pillow
tutorial
<https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#create-jpeg-thumbnails>`_.

``testing.widgets.mock_event`` and ``testing.widgets.do_event``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

... are deprecated. Directly construct Event objects (typically `.MouseEvent` or
`.KeyEvent`) and pass them to ``canvas.callbacks.process()`` instead.
