
===========
API Changes
===========

This chapter is a log of changes to matplotlib that affect the
outward-facing API.  If updating matplotlib breaks your scripts, this
list may help describe what changes may be necessary in your code or
help figure out possible sources of the changes you are experiencing.

For new features that were added to matplotlib, please see
:ref:`whats-new`.

Changes in 1.4.x
================

Code changes
------------

* A major refactoring of the axes module was made. The axes module has been
  split into smaller modules:

    - the `_base` module, which contains a new private _AxesBase class. This
      class contains all methods except plotting and labelling methods.
    - the `axes` module, which contains the Axes class. This class inherits
      from _AxesBase, and contains all plotting and labelling methods.
    - the `_subplot` module, with all the classes concerning subplotting.

There are a couple of things that do not exists in the `axes` module's
namespace anymore. If you use them, you need to import them from their
original location:

  - math -> `import math`
  - ma -> `from numpy import ma`
  - cbook -> `from matplotlib import cbook`
  - docstring -> `from matplotlib import docstring`
  - is_sequence_of_strings -> `from matplotlib.cbook import is_sequence_of_strings`
  - is_string_like -> `from matplotlib.cbook import is_string_like`
  - iterable -> `from matplotlib.cbook import iterable`
  - itertools -> `import itertools`
  - martist -> `from matplotlib import artist as martist`
  - matplotlib -> `import matplotlib`
  - mcoll -> `from matplotlib import collections as mcoll`
  - mcolors -> `from matplotlib import colors as mcolors`
  - mcontour -> `from matplotlib import contour as mcontour`
  - mpatches -> `from matplotlib import patches as mpatches`
  - mpath -> `from matplotlib import path as mpath`
  - mquiver -> `from matplotlib import quiver as mquiver`
  - mstack -> `from matplotlib import stack as mstack`
  - mstream -> `from matplotlib import stream as mstream`
  - mtable -> `from matplotlib import table as mtable`

* As part of the refactoring to enable Qt5 support, the module
  `matplotlib.backends.qt4_compat` was renamed to
  `matplotlib.qt_compat`.  `qt4_compat` is deprecated in 1.4 and
  will be removed in 1.5.

* The :func:`~matplotlib.pyplot.errorbar` method has been changed such that
  the upper and lower limits (*lolims*, *uplims*, *xlolims*, *xuplims*) now
  point in the correct direction.

* The *fmt* kwarg for :func:`~matplotlib.pyplot.errorbar now supports
  the string 'none' to suppress drawing of a line and markers; use
  of the *None* object for this is deprecated. The default *fmt*
  value is changed to the empty string (''), so the line and markers
  are governed by the :func:`~matplotlib.pyplot.plot` defaults.

* A bug has been fixed in the path effects rendering of fonts, which now means
  that the font size is consistent with non-path effect fonts. See
  https://github.com/matplotlib/matplotlib/issues/2889 for more detail.

* The Sphinx extensions `ipython_directive` and
  `ipython_console_highlighting` have been moved to the IPython
  project itself.  While they remain in matplotlib for this release,
  they have been deprecated.  Update your extensions in `conf.py` to
  point to `IPython.sphinxext.ipython_directive` instead of
  `matplotlib.sphinxext.ipython_directive`.

* In `~matplotlib.finance`, almost all functions have been deprecated
  and replaced with a pair of functions name `*_ochl` and `*_ohlc`.
  The former is the 'open-close-high-low' order of quotes used
  previously in this module, and the latter is the
  'open-high-low-close' order that is standard in finance.

* For consistency the ``face_alpha`` keyword to
  :class:`matplotlib.patheffects.SimplePatchShadow` has been deprecated in
  favour of the ``alpha`` keyword. Similarly, the keyword ``offset_xy`` is now
  named ``offset`` across all :class:`~matplotlib.patheffects.AbstractPathEffect`s.
  ``matplotlib.patheffects._Base`` has
  been renamed to :class:`matplotlib.patheffects.AbstractPathEffect`.
  ``matplotlib.patheffect.ProxyRenderer`` has been renamed to
  :class:`matplotlib.patheffects.PathEffectRenderer` and is now a full
  RendererBase subclass.

* The artist used to draw the outline of a `colorbar` has been changed
  from a `matplotlib.lines.Line2D` to `matplotlib.patches.Polygon`,
  thus `colorbar.ColorbarBase.outline` is now a
  `matplotlib.patches.Polygon` object.

* The legend handler interface has changed from a callable, to any object
  which implements the ``legend_artists`` method (a deprecation phase will
  see this interface be maintained for v1.4). See
  :ref:`plotting-guide-legend` for further details. Further legend changes
  include:

   * :func:`matplotlib.axes.Axes._get_legend_handles` now returns a generator
     of handles, rather than a list.

   * The :func:`~matplotlib.pyplot.legend` function's "loc" positional
     argument has been deprecated. Use the "loc" keyword instead.

* The rcParams `savefig.transparent` has been added to control
  default transparency when saving figures.

* Slightly refactored the `Annotation` family.  The text location in
  `Annotation` is now handled entirely handled by the underlying `Text`
  object so `set_position` works as expected.  The attributes `xytext` and
  `textcoords` have been deprecated in favor of `xyann` and `anncoords` so
  that `Annotation` and `AnnotaionBbox` can share a common sensibly named
  api for getting/setting the location of the text or box.

    - `xyann` -> set the location of the annotation
    - `xy` -> set where the arrow points to
    - `anncoords` -> set the units of the annotation location
    - `xycoords` -> set the units of the point location
    - `set_position()` -> `Annotation` only set location of annotation

* `matplotlib.mlab.specgram`, `matplotlib.mlab.psd`,  `matplotlib.mlab.csd`,
  `matplotlib.mlab.cohere`, `matplotlib.mlab.cohere_pairs`,
  `matplotlib.pyplot.specgram`, `matplotlib.pyplot.psd`,
  `matplotlib.pyplot.csd`, and `matplotlib.pyplot.cohere` now raise
  ValueError where they previously raised AssertionError.

* For `matplotlib.mlab.psd`,  `matplotlib.mlab.csd`,
  `matplotlib.mlab.cohere`, `matplotlib.mlab.cohere_pairs`,
  `matplotlib.pyplot.specgram`, `matplotlib.pyplot.psd`,
  `matplotlib.pyplot.csd`, and `matplotlib.pyplot.cohere`, in cases
  where a shape (n, 1) array is returned, this is now converted to a (n, )
  array.  Previously, (n, m) arrays were averaged to an (n, ) array, but
  (n, 1) arrays were returend unchanged.  This change makes the dimensions
  consistent in both cases.

* Added the rcParam `axes.fromatter.useoffset` to control the default value
  of `useOffset` in `ticker.ScalarFormatter`

* Added `Formatter` sub-class `StrMethodFormatter` which
  does the exact same thing as `FormatStrFormatter`, but for new-style
  formatting strings.

* Deprecated `matplotlib.testing.image_util` and the only function within,
  `matplotlib.testing.image_util.autocontrast`. These will be removed
  completely in v1.5.0.

* The ``fmt`` argument of :meth:`~matplotlib.axes.Axes.plot_date` has been
  changed from ``bo`` to just ``o``, so color cycling can happen by default.

* Removed the class `FigureManagerQTAgg` and deprecated `NavigationToolbar2QTAgg`
  which will be removed in 1.5.

* Removed formerly public (non-prefixed) attributes `rect` and
  `drawRect` from `FigureCanvasQTAgg`; they were always an
  implementation detail of the (preserved) `drawRectangle()` function.

* The function signatures of `tight_bbox.adjust_bbox` and
  `tight_bbox.process_figure_for_rasterizing` have been changed. A new
  `fixed_dpi` parameter allows for overriding the `figure.dpi` setting
  instead of trying to deduce the intended behaviour from the file format.

* Added support for horizontal/vertical axes padding to
  `mpl_toolkits.axes_grid1.ImageGrid` --- argument ``axes_pad`` can now be
  tuple-like if separate axis padding is required.
  The original behavior is preserved.

* Added support for skewed transforms to `matplotlib.transforms.Affine2D`,
  which can be created using the `skew` and `skew_deg` methods.

* Added clockwise parameter to control sectors direction in `axes.pie`

* In `matplotlib.lines.Line2D` the `markevery` functionality has been extended.
  Previously an integer start-index and stride-length could be specified using
  either a two-element-list or a two-element-tuple.  Now this can only be done
  using a two-element-tuple.  If a two-element-list is used then it will be
  treated as numpy fancy indexing and only the two markers corresponding to the
  given indexes will be shown.

* removed prop kwarg from `mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`
  call.  It was passed through to the base-class `__init__` and is only used for
  setting padding.  Now `fontproperties` (which is what is really used to set
  the font properties of `AnchoredSizeBar`) is passed through in place of `prop`.
  If `fontpropreties` is not passed in, but `prop` is, then `prop` is used inplace
  of `fontpropreties`.  If both are passed in, `prop` is silently ignored.


* The use of the index 0 in `plt.subplot` and related commands is
  deprecated.  Due to a lack of validation calling `plt.subplots(2, 2,
  0)` does not raise an exception, but puts an axes in the _last_
  position.  This is due to the indexing in subplot being 1-based (to
  mirror MATLAB) so before indexing into the `GridSpec` object used to
  determine where the axes should go, 1 is subtracted off.  Passing in
  0 results in passing -1 to `GridSpec` which results in getting the
  last position back.  Even though this behavior is clearly wrong and
  not intended, we are going through a deprecation cycle in an
  abundance of caution that any users are exploiting this 'feature'.
  The use of 0 as an index will raise a warning in 1.4 and an
  exception in 1.5.

* Clipping is now off by default on offset boxes.

* matplotlib now uses a less-aggressive call to ``gc.collect(1)`` when
  closing figures to avoid major delays with large numbers of user objects
  in memory.

* The default clip value of *all* pie artists now defaults to ``False``.


Code removal
------------

* Removed ``mlab.levypdf``.  The code raised a numpy error (and has for
  a long time) and was not the standard form of the Levy distribution.
  ``scipy.stats.levy`` should be used instead


.. _changes_in_1_3:


Changes in 1.3.x
================

Changes in 1.3.1
----------------

It is rare that we make an API change in a bugfix release, however,
for 1.3.1 since 1.3.0 the following change was made:

- `text.Text.cached` (used to cache font objects) has been made into a
  private variable.  Among the obvious encapsulation benefit, this
  removes this confusing-looking member from the documentation.

- The method :meth:`~matplotlib.axes.Axes.hist` now always returns bin
  occupancies as an array of type `float`. Previously, it was sometimes
  an array of type `int`, depending on the call.

Code removal
------------

* The following items that were deprecated in version 1.2 or earlier
  have now been removed completely.

    - The Qt 3.x backends (`qt` and `qtagg`) have been removed in
      favor of the Qt 4.x backends (`qt4` and `qt4agg`).

    - The FltkAgg and Emf backends have been removed.

    - The `matplotlib.nxutils` module has been removed.  Use the
      functionality on `matplotlib.path.Path.contains_point` and
      friends instead.

    - Instead of `axes.Axes.get_frame`, use `axes.Axes.patch`.

    - The following `kwargs` to the `legend` function have been
      renamed:

      - `pad` -> `borderpad`
      - `labelsep` -> `labelspacing`
      - `handlelen` -> `handlelength`
      - `handletextsep` -> `handletextpad`
      - `axespad` -> `borderaxespad`

      Related to this, the following rcParams have been removed:

      - `legend.pad`, `legend.labelsep`, `legend.handlelen`,
        `legend.handletextsep` and `legend.axespad`

    - For the `hist` function, instead of `width`, use `rwidth`
      (relative width).

    - On `patches.Circle`, the `resolution` kwarg has been removed.
      For a circle made up of line segments, use
      `patches.CirclePolygon`.

    - The printing functions in the Wx backend have been removed due
      to the burden of keeping them up-to-date.

    - `mlab.liaupunov` has been removed.

    - `mlab.save`, `mlab.load`, `pylab.save` and `pylab.load` have
      been removed.  We recommend using `numpy.savetxt` and
      `numpy.loadtxt` instead.

    - `widgets.HorizontalSpanSelector` has been removed.  Use
      `widgets.SpanSelector` instead.

Code deprecation
----------------

* The CocoaAgg backend has been deprecated, with the possibility for
  deletion or resurrection in a future release.

* The top-level functions in `matplotlib.path` that are implemented in
  C++ were never meant to be public.  Instead, users should use the
  Pythonic wrappers for them in the `path.Path` and
  `collections.Collection` classes.  Use the following mapping to update
  your code:

    - `point_in_path` -> `path.Path.contains_point`
    - `get_path_extents` -> `path.Path.get_extents`
    - `point_in_path_collection` -> `collection.Collection.contains`
    - `path_in_path` -> `path.Path.contains_path`
    - `path_intersects_path` -> `path.Path.intersects_path`
    - `convert_path_to_polygons` -> `path.Path.to_polygons`
    - `cleanup_path` -> `path.Path.cleaned`
    - `points_in_path` -> `path.Path.contains_points`
    - `clip_path_to_rect` -> `path.Path.clip_to_bbox`

* `matplotlib.colors.normalize` and `matplotlib.colors.no_norm` have
  been deprecated in favour of `matplotlib.colors.Normalize` and
  `matplotlib.colors.NoNorm` respectively.

* The `ScalarMappable` class' `set_colorbar` is now
  deprecated. Instead, the
  :attr:`matplotlib.cm.ScalarMappable.colorbar` attribute should be
  used.  In previous matplotlib versions this attribute was an
  undocumented tuple of ``(colorbar_instance, colorbar_axes)`` but is
  now just ``colorbar_instance``. To get the colorbar axes it is
  possible to just use the
  :attr:`~matplotlib.colorbar.ColorbarBase.ax` attribute on a colorbar
  instance.

* The `~matplotlib.mpl` module is now deprecated. Those who relied on this
  module should transition to simply using ``import matplotlib as mpl``.

Code changes
------------

* :class:`~matplotlib.patches.Patch` now fully supports using RGBA values for
  its ``facecolor`` and ``edgecolor`` attributes, which enables faces and
  edges to have different alpha values. If the
  :class:`~matplotlib.patches.Patch` object's ``alpha`` attribute is set to
  anything other than ``None``, that value will override any alpha-channel
  value in both the face and edge colors. Previously, if
  :class:`~matplotlib.patches.Patch` had ``alpha=None``, the alpha component
  of ``edgecolor`` would be applied to both the edge and face.

* The optional ``isRGB`` argument to
  :meth:`~matplotlib.backend_bases.GraphicsContextBase.set_foreground` (and
  the other GraphicsContext classes that descend from it) has been renamed to
  ``isRGBA``, and should now only be set to ``True`` if the ``fg`` color
  argument is known to be an RGBA tuple.

* For :class:`~matplotlib.patches.Patch`, the ``capstyle`` used is now
  ``butt``, to be consistent with the default for most other objects, and to
  avoid problems with non-solid ``linestyle`` appearing solid when using a
  large ``linewidth``. Previously, :class:`~matplotlib.patches.Patch` used
  ``capstyle='projecting'``.

* `Path` objects can now be marked as `readonly` by passing
  `readonly=True` to its constructor.  The built-in path singletons,
  obtained through `Path.unit*` class methods return readonly paths.
  If you have code that modified these, you will need to make a
  deepcopy first, using either::

    import copy
    path = copy.deepcopy(Path.unit_circle())

    # or

    path = Path.unit_circle().deepcopy()

  Deep copying a `Path` always creates an editable (i.e. non-readonly)
  `Path`.

* The list at ``Path.NUM_VERTICES`` was replaced by a dictionary mapping
  Path codes to the number of expected vertices at
  :attr:`~matplotlib.path.Path.NUM_VERTICES_FOR_CODE`.

* To support XKCD style plots, the :func:`matplotlib.path.cleanup_path`
  method's signature was updated to require a sketch argument. Users of
  :func:`matplotlib.path.cleanup_path` are encouraged to use the new
  :meth:`~matplotlib.path.Path.cleaned` Path method.

* Data limits on a plot now start from a state of having "null"
  limits, rather than limits in the range (0, 1).  This has an effect
  on artists that only control limits in one direction, such as
  `axvline` and `axhline`, since their limits will not longer also
  include the range (0, 1).  This fixes some problems where the
  computed limits would be dependent on the order in which artists
  were added to the axes.

* Fixed a bug in setting the position for the right/top spine with data
  position type. Previously, it would draw the right or top spine at
  +1 data offset.

* In :class:`~matplotlib.patches.FancyArrow`, the default arrow head
  width, ``head_width``, has been made larger to produce a visible
  arrow head. The new value of this kwarg is ``head_width = 20 *
  width``.

* It is now possible to provide ``number of levels + 1`` colors in the case of
  `extend='both'` for contourf (or just ``number of levels`` colors for an
  extend value ``min`` or ``max``) such that the resulting colormap's
  ``set_under`` and ``set_over`` are defined appropriately. Any other number
  of colors will continue to behave as before (if more colors are provided
  than levels, the colors will be unused). A similar change has been applied
  to contour, where ``extend='both'`` would expect ``number of levels + 2``
  colors.

* A new keyword *extendrect* in :meth:`~matplotlib.pyplot.colorbar` and
  :class:`~matplotlib.colorbar.ColorbarBase` allows one to control the shape
  of colorbar extensions.

* The extension of :class:`~matplotlib.widgets.MultiCursor` to both vertical
  (default) and/or horizontal cursor implied that ``self.line`` is replaced
  by ``self.vline`` for vertical cursors lines and ``self.hline`` is added
  for the horizontal cursors lines.

* On POSIX platforms, the :func:`~matplotlib.cbook.report_memory` function
  raises :class:`NotImplementedError` instead of :class:`OSError` if the
  :command:`ps` command cannot be run.

* The :func:`matplotlib.cbook.check_output` function has been moved to
  :func:`matplotlib.compat.subprocess`.

Configuration and rcParams
--------------------------

* On Linux, the user-specific `matplotlibrc` configuration file is now
  located in `~/.config/matplotlib/matplotlibrc` to conform to the
  `XDG Base Directory Specification
  <http://standards.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_.

* The `font.*` rcParams now affect only text objects created after the
  rcParam has been set, and will not retroactively affect already
  existing text objects.  This brings their behavior in line with most
  other rcParams.

* Removed call of :meth:`~matplotlib.axes.Axes.grid` in
  :meth:`~matplotlib.pyplot.plotfile`. To draw the axes grid, set the
  ``axes.grid`` rcParam to *True*, or explicitly call
  :meth:`~matplotlib.axes.Axes.grid`.

Changes in 1.2.x
================

* The ``classic`` option of the rc parameter ``toolbar`` is deprecated
  and will be removed in the next release.

* The :meth:`~matplotlib.cbook.isvector` method has been removed since it
  is no longer functional.

* The `rasterization_zorder` property on `~matplotlib.axes.Axes` a
  zorder below which artists are rasterized.  This has defaulted to
  -30000.0, but it now defaults to `None`, meaning no artists will be
  rasterized.  In order to rasterize artists below a given zorder
  value, `set_rasterization_zorder` must be explicitly called.

* In :meth:`~matplotlib.axes.Axes.scatter`, and `~pyplot.scatter`,
  when specifying a marker using a tuple, the angle is now specified
  in degrees, not radians.

* Using :meth:`~matplotlib.axes.Axes.twinx` or
  :meth:`~matplotlib.axes.Axes.twiny` no longer overrides the current locaters
  and formatters on the axes.

* In :meth:`~matplotlib.axes.Axes.contourf`, the handling of the *extend*
  kwarg has changed.  Formerly, the extended ranges were mapped
  after to 0, 1 after being normed, so that they always corresponded
  to the extreme values of the colormap.  Now they are mapped
  outside this range so that they correspond to the special
  colormap values determined by the
  :meth:`~matplotlib.colors.Colormap.set_under` and
  :meth:`~matplotlib.colors.Colormap.set_over` methods, which
  default to the colormap end points.

* The new rc parameter ``savefig.format`` replaces ``cairo.format`` and
  ``savefig.extension``, and sets the default file format used by
  :meth:`matplotlib.figure.Figure.savefig`.

* In :meth:`~matplotlib.pyplot.pie` and :meth:`~matplotlib.Axes.pie`, one can
  now set the radius of the pie; setting the *radius* to 'None' (the default
  value), will result in a pie with a radius of 1 as before.

* Use of :func:`~matplotlib.projections.projection_factory` is now deprecated
  in favour of axes class identification using
  :func:`~matplotlib.projections.process_projection_requirements` followed by
  direct axes class invocation (at the time of writing, functions which do this
  are: :meth:`~matplotlib.figure.Figure.add_axes`,
  :meth:`~matplotlib.figure.Figure.add_subplot` and
  :meth:`~matplotlib.figure.Figure.gca`). Therefore::


      key = figure._make_key(*args, **kwargs)
      ispolar = kwargs.pop('polar', False)
      projection = kwargs.pop('projection', None)
      if ispolar:
          if projection is not None and projection != 'polar':
              raise ValueError('polar and projection args are inconsistent')
          projection = 'polar'
      ax = projection_factory(projection, self, rect, **kwargs)
      key = self._make_key(*args, **kwargs)

      # is now

      projection_class, kwargs, key = \
                         process_projection_requirements(self, *args, **kwargs)
      ax = projection_class(self, rect, **kwargs)

  This change means that third party objects can expose themselves as
  matplotlib axes by providing a ``_as_mpl_axes`` method. See
  :ref:`adding-new-scales` for more detail.

* A new keyword *extendfrac* in :meth:`~matplotlib.pyplot.colorbar` and
  :class:`~matplotlib.colorbar.ColorbarBase` allows one to control the size of
  the triangular minimum and maximum extensions on colorbars.

* A new keyword *capthick* in :meth:`~matplotlib.pyplot.errorbar` has been
  added as an intuitive alias to the *markeredgewidth* and *mew* keyword
  arguments, which indirectly controlled the thickness of the caps on
  the errorbars.  For backwards compatibility, specifying either of the
  original keyword arguments will override any value provided by
  *capthick*.

* Transform subclassing behaviour is now subtly changed. If your transform
  implements a non-affine transformation, then it should override the
  ``transform_non_affine`` method, rather than the generic ``transform`` method.
  Previously transforms would define ``transform`` and then copy the
  method into ``transform_non_affine``::

     class MyTransform(mtrans.Transform):
         def transform(self, xy):
             ...
         transform_non_affine = transform


  This approach will no longer function correctly and should be changed to::

     class MyTransform(mtrans.Transform):
         def transform_non_affine(self, xy):
             ...


* Artists no longer have ``x_isdata`` or ``y_isdata`` attributes; instead
  any artist's transform can be interrogated with
  ``artist_instance.get_transform().contains_branch(ax.transData)``

* Lines added to an axes now take into account their transform when updating the
  data and view limits. This means transforms can now be used as a pre-transform.
  For instance::

      >>> import matplotlib.pyplot as plt
      >>> import matplotlib.transforms as mtrans
      >>> ax = plt.axes()
      >>> ax.plot(range(10), transform=mtrans.Affine2D().scale(10) + ax.transData)
      >>> print(ax.viewLim)
      Bbox('array([[  0.,   0.],\n       [ 90.,  90.]])')

* One can now easily get a transform which goes from one transform's coordinate
  system to another, in an optimized way, using the new subtract method on a
  transform. For instance, to go from data coordinates to axes coordinates::

      >>> import matplotlib.pyplot as plt
      >>> ax = plt.axes()
      >>> data2ax = ax.transData - ax.transAxes
      >>> print(ax.transData.depth, ax.transAxes.depth)
      3, 1
      >>> print(data2ax.depth)
      2

  for versions before 1.2 this could only be achieved in a sub-optimal way,
  using ``ax.transData + ax.transAxes.inverted()`` (depth is a new concept,
  but had it existed it would return 4 for this example).

* ``twinx`` and ``twiny`` now returns an instance of SubplotBase if
  parent axes is an instance of SubplotBase.

* All Qt3-based backends are now deprecated due to the lack of py3k bindings.
  Qt and QtAgg backends will continue to work in v1.2.x for py2.6
  and py2.7. It is anticipated that the Qt3 support will be completely
  removed for the next release.

* :class:`~matplotlib.colors.ColorConverter`,
  :class:`~matplotlib.colors.Colormap` and
  :class:`~matplotlib.colors.Normalize` now subclasses ``object``

* ContourSet instances no longer have a ``transform`` attribute. Instead,
  access the transform with the ``get_transform`` method.

Changes in 1.1.x
================

* Added new :class:`matplotlib.sankey.Sankey` for generating Sankey diagrams.

* In :meth:`~matplotlib.pyplot.imshow`, setting *interpolation* to 'nearest'
  will now always mean that the nearest-neighbor interpolation is performed.
  If you want the no-op interpolation to be performed, choose 'none'.

* There were errors in how the tri-functions were handling input parameters
  that had to be fixed. If your tri-plots are not working correctly anymore,
  or you were working around apparent mistakes, please see issue #203 in the
  github tracker. When in doubt, use kwargs.

* The 'symlog' scale had some bad behavior in previous versions. This has now
  been fixed and users should now be able to use it without frustrations.
  The fixes did result in some minor changes in appearance for some users who
  may have been depending on the bad behavior.

* There is now a common set of markers for all plotting functions. Previously,
  some markers existed only for :meth:`~matplotlib.pyplot.scatter` or just for
  :meth:`~matplotlib.pyplot.plot`. This is now no longer the case. This merge
  did result in a conflict. The string 'd' now means "thin diamond" while
  'D' will mean "regular diamond".

Changes beyond 0.99.x
=====================

* The default behavior of :meth:`matplotlib.axes.Axes.set_xlim`,
  :meth:`matplotlib.axes.Axes.set_ylim`, and
  :meth:`matplotlib.axes.Axes.axis`, and their corresponding
  pyplot functions, has been changed: when view limits are
  set explicitly with one of these methods, autoscaling is turned
  off for the matching axis. A new *auto* kwarg is available to
  control this behavior. The limit kwargs have been renamed to
  *left* and *right* instead of *xmin* and *xmax*, and *bottom*
  and *top* instead of *ymin* and *ymax*.  The old names may still
  be used, however.

* There are five new Axes methods with corresponding pyplot
  functions to facilitate autoscaling, tick location, and tick
  label formatting, and the general appearance of ticks and
  tick labels:

  + :meth:`matplotlib.axes.Axes.autoscale` turns autoscaling
    on or off, and applies it.

  + :meth:`matplotlib.axes.Axes.margins` sets margins used to
    autoscale the :attr:`matplotlib.axes.Axes.viewLim` based on
    the :attr:`matplotlib.axes.Axes.dataLim`.

  + :meth:`matplotlib.axes.Axes.locator_params` allows one to
    adjust axes locator parameters such as *nbins*.

  + :meth:`matplotlib.axes.Axes.ticklabel_format` is a convenience
    method for controlling the :class:`matplotlib.ticker.ScalarFormatter`
    that is used by default with linear axes.

  + :meth:`matplotlib.axes.Axes.tick_params` controls direction, size,
    visibility, and color of ticks and their labels.

* The :meth:`matplotlib.axes.Axes.bar` method accepts a *error_kw*
  kwarg; it is a dictionary of kwargs to be passed to the
  errorbar function.

* The :meth:`matplotlib.axes.Axes.hist` *color* kwarg now accepts
  a sequence of color specs to match a sequence of datasets.

* The :class:`~matplotlib.collections.EllipseCollection` has been
  changed in two ways:

  + There is a new *units* option, 'xy', that scales the ellipse with
    the data units.  This matches the :class:'~matplotlib.patches.Ellipse`
    scaling.

  + The *height* and *width* kwargs have been changed to specify
    the height and width, again for consistency with
    :class:`~matplotlib.patches.Ellipse`, and to better match
    their names; previously they specified the half-height and
    half-width.

* There is a new rc parameter ``axes.color_cycle``, and the color
  cycle is now independent of the rc parameter ``lines.color``.
  :func:`matplotlib.Axes.set_default_color_cycle` is deprecated.

* You can now print several figures to one pdf file and modify the
  document information dictionary of a pdf file. See the docstrings
  of the class :class:`matplotlib.backends.backend_pdf.PdfPages` for
  more information.

* Removed configobj_ and `enthought.traits`_ packages, which are only
  required by the experimental traited config and are somewhat out of
  date. If needed, install them independently.

.. _configobj: http://www.voidspace.org.uk/python/configobj.html
.. _`enthought.traits`: http://code.enthought.com/projects/traits

* The new rc parameter ``savefig.extension`` sets the filename extension
  that is used by :meth:`matplotlib.figure.Figure.savefig` if its *fname*
  argument lacks an extension.

* In an effort to simplify the backend API, all clipping rectangles
  and paths are now passed in using GraphicsContext objects, even
  on collections and images.  Therefore::

    draw_path_collection(self, master_transform, cliprect, clippath,
                         clippath_trans, paths, all_transforms, offsets,
                         offsetTrans, facecolors, edgecolors, linewidths,
                         linestyles, antialiaseds, urls)

    # is now

    draw_path_collection(self, gc, master_transform, paths, all_transforms,
                         offsets, offsetTrans, facecolors, edgecolors,
                         linewidths, linestyles, antialiaseds, urls)


    draw_quad_mesh(self, master_transform, cliprect, clippath,
                   clippath_trans, meshWidth, meshHeight, coordinates,
                   offsets, offsetTrans, facecolors, antialiased,
                   showedges)

    # is now

    draw_quad_mesh(self, gc, master_transform, meshWidth, meshHeight,
                   coordinates, offsets, offsetTrans, facecolors,
                   antialiased, showedges)


    draw_image(self, x, y, im, bbox, clippath=None, clippath_trans=None)

    # is now

    draw_image(self, gc, x, y, im)

* There are four new Axes methods with corresponding pyplot
  functions that deal with unstructured triangular grids:

  + :meth:`matplotlib.axes.Axes.tricontour` draws contour lines
    on a triangular grid.

  + :meth:`matplotlib.axes.Axes.tricontourf` draws filled contours
    on a triangular grid.

  + :meth:`matplotlib.axes.Axes.tripcolor` draws a pseudocolor
    plot on a triangular grid.

  + :meth:`matplotlib.axes.Axes.triplot` draws a triangular grid
    as lines and/or markers.

Changes in 0.99
======================

* pylab no longer provides a load and save function.  These are
  available in matplotlib.mlab, or you can use numpy.loadtxt and
  numpy.savetxt for text files, or np.save and np.load for binary
  numpy arrays.

* User-generated colormaps can now be added to the set recognized
  by :func:`matplotlib.cm.get_cmap`.  Colormaps can be made the
  default and applied to the current image using
  :func:`matplotlib.pyplot.set_cmap`.

* changed use_mrecords default to False in mlab.csv2rec since this is
  partially broken

* Axes instances no longer have a "frame" attribute. Instead, use the
  new "spines" attribute. Spines is a dictionary where the keys are
  the names of the spines (e.g., 'left','right' and so on) and the
  values are the artists that draw the spines. For normal
  (rectilinear) axes, these artists are Line2D instances. For other
  axes (such as polar axes), these artists may be Patch instances.

* Polar plots no longer accept a resolution kwarg.  Instead, each Path
  must specify its own number of interpolation steps.  This is
  unlikely to be a user-visible change -- if interpolation of data is
  required, that should be done before passing it to matplotlib.

Changes for 0.98.x
==================
* psd(), csd(), and cohere() will now automatically wrap negative
  frequency components to the beginning of the returned arrays.
  This is much more sensible behavior and makes them consistent
  with specgram().  The previous behavior was more of an oversight
  than a design decision.

* Added new keyword parameters *nonposx*, *nonposy* to
  :class:`matplotlib.axes.Axes` methods that set log scale
  parameters.  The default is still to mask out non-positive
  values, but the kwargs accept 'clip', which causes non-positive
  values to be replaced with a very small positive value.

* Added new :func:`matplotlib.pyplot.fignum_exists` and
  :func:`matplotlib.pyplot.get_fignums`; they merely expose
  information that had been hidden in :mod:`matplotlib._pylab_helpers`.

* Deprecated numerix package.

* Added new :func:`matplotlib.image.imsave` and exposed it to the
  :mod:`matplotlib.pyplot` interface.

* Remove support for pyExcelerator in exceltools -- use xlwt
  instead

* Changed the defaults of acorr and xcorr to use usevlines=True,
  maxlags=10 and normed=True since these are the best defaults

* Following keyword parameters for :class:`matplotlib.label.Label` are now
  deprecated and new set of parameters are introduced. The new parameters
  are given as a fraction of the font-size. Also, *scatteryoffsets*,
  *fancybox* and *columnspacing* are added as keyword parameters.

        ================   ================
        Deprecated         New
        ================   ================
        pad                borderpad
        labelsep           labelspacing
        handlelen          handlelength
        handlestextsep     handletextpad
        axespad            borderaxespad
        ================   ================


* Removed the configobj and experimental traits rc support

* Modified :func:`matplotlib.mlab.psd`, :func:`matplotlib.mlab.csd`,
  :func:`matplotlib.mlab.cohere`, and :func:`matplotlib.mlab.specgram`
  to scale one-sided densities by a factor of 2.  Also, optionally
  scale the densities by the sampling frequency, which gives true values
  of densities that can be integrated by the returned frequency values.
  This also gives better MATLAB compatibility.  The corresponding
  :class:`matplotlib.axes.Axes` methods and :mod:`matplotlib.pyplot`
  functions were updated as well.

* Font lookup now uses a nearest-neighbor approach rather than an
  exact match.  Some fonts may be different in plots, but should be
  closer to what was requested.

* :meth:`matplotlib.axes.Axes.set_xlim`,
  :meth:`matplotlib.axes.Axes.set_ylim` now return a copy of the
  :attr:`viewlim` array to avoid modify-in-place surprises.

* :meth:`matplotlib.afm.AFM.get_fullname` and
  :meth:`matplotlib.afm.AFM.get_familyname` no longer raise an
  exception if the AFM file does not specify these optional
  attributes, but returns a guess based on the required FontName
  attribute.

* Changed precision kwarg in :func:`matplotlib.pyplot.spy`; default is
  0, and the string value 'present' is used for sparse arrays only to
  show filled locations.

* :class:`matplotlib.collections.EllipseCollection` added.

* Added ``angles`` kwarg to :func:`matplotlib.pyplot.quiver` for more
  flexible specification of the arrow angles.

* Deprecated (raise NotImplementedError) all the mlab2 functions from
  :mod:`matplotlib.mlab` out of concern that some of them were not
  clean room implementations.

* Methods :meth:`matplotlib.collections.Collection.get_offsets` and
  :meth:`matplotlib.collections.Collection.set_offsets` added to
  :class:`~matplotlib.collections.Collection` base class.

* :attr:`matplotlib.figure.Figure.figurePatch` renamed
  :attr:`matplotlib.figure.Figure.patch`;
  :attr:`matplotlib.axes.Axes.axesPatch` renamed
  :attr:`matplotlib.axes.Axes.patch`;
  :attr:`matplotlib.axes.Axes.axesFrame` renamed
  :attr:`matplotlib.axes.Axes.frame`.
  :meth:`matplotlib.axes.Axes.get_frame`, which returns
  :attr:`matplotlib.axes.Axes.patch`, is deprecated.

* Changes in the :class:`matplotlib.contour.ContourLabeler` attributes
  (:func:`matplotlib.pyplot.clabel` function) so that they all have a
  form like ``.labelAttribute``.  The three attributes that are most
  likely to be used by end users, ``.cl``, ``.cl_xy`` and
  ``.cl_cvalues`` have been maintained for the moment (in addition to
  their renamed versions), but they are deprecated and will eventually
  be removed.

* Moved several functions in :mod:`matplotlib.mlab` and
  :mod:`matplotlib.cbook` into a separate module
  :mod:`matplotlib.numerical_methods` because they were unrelated to
  the initial purpose of mlab or cbook and appeared more coherent
  elsewhere.

Changes for 0.98.1
==================

* Removed broken :mod:`matplotlib.axes3d` support and replaced it with
  a non-implemented error pointing to 0.91.x

Changes for 0.98.0
==================

* :func:`matplotlib.image.imread` now no longer always returns RGBA data---if
  the image is luminance or RGB, it will return a MxN or MxNx3 array
  if possible.  Also uint8 is no longer always forced to float.

* Rewrote the :class:`matplotlib.cm.ScalarMappable` callback
  infrastructure to use :class:`matplotlib.cbook.CallbackRegistry`
  rather than custom callback handling.  Any users of
  :meth:`matplotlib.cm.ScalarMappable.add_observer` of the
  :class:`~matplotlib.cm.ScalarMappable` should use the
  :attr:`matplotlib.cm.ScalarMappable.callbacks`
  :class:`~matplotlib.cbook.CallbackRegistry` instead.

* New axes function and Axes method provide control over the plot
  color cycle: :func:`matplotlib.axes.set_default_color_cycle` and
  :meth:`matplotlib.axes.Axes.set_color_cycle`.

* matplotlib now requires Python 2.4, so :mod:`matplotlib.cbook` will
  no longer provide :class:`set`, :func:`enumerate`, :func:`reversed`
  or :func:`izip` compatibility functions.

* In Numpy 1.0, bins are specified by the left edges only.  The axes
  method :meth:`matplotlib.axes.Axes.hist` now uses future Numpy 1.3
  semantics for histograms.  Providing ``binedges``, the last value gives
  the upper-right edge now, which was implicitly set to +infinity in
  Numpy 1.0.  This also means that the last bin doesn't contain upper
  outliers any more by default.

* New axes method and pyplot function,
  :func:`~matplotlib.pyplot.hexbin`, is an alternative to
  :func:`~matplotlib.pyplot.scatter` for large datasets.  It makes
  something like a :func:`~matplotlib.pyplot.pcolor` of a 2-D
  histogram, but uses hexagonal bins.

* New kwarg, ``symmetric``, in :class:`matplotlib.ticker.MaxNLocator`
  allows one require an axis to be centered around zero.

* Toolkits must now be imported from ``mpl_toolkits`` (not ``matplotlib.toolkits``)

Notes about the transforms refactoring
--------------------------------------

A major new feature of the 0.98 series is a more flexible and
extensible transformation infrastructure, written in Python/Numpy
rather than a custom C extension.

The primary goal of this refactoring was to make it easier to
extend matplotlib to support new kinds of projections.  This is
mostly an internal improvement, and the possible user-visible
changes it allows are yet to come.

See :mod:`matplotlib.transforms` for a description of the design of
the new transformation framework.

For efficiency, many of these functions return views into Numpy
arrays.  This means that if you hold on to a reference to them,
their contents may change.  If you want to store a snapshot of
their current values, use the Numpy array method copy().

The view intervals are now stored only in one place -- in the
:class:`matplotlib.axes.Axes` instance, not in the locator instances
as well.  This means locators must get their limits from their
:class:`matplotlib.axis.Axis`, which in turn looks up its limits from
the :class:`~matplotlib.axes.Axes`.  If a locator is used temporarily
and not assigned to an Axis or Axes, (e.g., in
:mod:`matplotlib.contour`), a dummy axis must be created to store its
bounds.  Call :meth:`matplotlib.ticker.Locator.create_dummy_axis` to
do so.

The functionality of :class:`Pbox` has been merged with
:class:`~matplotlib.transforms.Bbox`.  Its methods now all return
copies rather than modifying in place.

The following lists many of the simple changes necessary to update
code from the old transformation framework to the new one.  In
particular, methods that return a copy are named with a verb in the
past tense, whereas methods that alter an object in place are named
with a verb in the present tense.

:mod:`matplotlib.transforms`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================================================ ============================================================
Old method                                                   New method
============================================================ ============================================================
:meth:`Bbox.get_bounds`                                      :attr:`transforms.Bbox.bounds`
------------------------------------------------------------ ------------------------------------------------------------
:meth:`Bbox.width`                                           :attr:`transforms.Bbox.width`
------------------------------------------------------------ ------------------------------------------------------------
:meth:`Bbox.height`                                          :attr:`transforms.Bbox.height`
------------------------------------------------------------ ------------------------------------------------------------
`Bbox.intervalx().get_bounds()`                              :attr:`transforms.Bbox.intervalx`
`Bbox.intervalx().set_bounds()`                              [:attr:`Bbox.intervalx` is now a property.]
------------------------------------------------------------ ------------------------------------------------------------
`Bbox.intervaly().get_bounds()`                              :attr:`transforms.Bbox.intervaly`
`Bbox.intervaly().set_bounds()`                              [:attr:`Bbox.intervaly` is now a property.]
------------------------------------------------------------ ------------------------------------------------------------
:meth:`Bbox.xmin`                                            :attr:`transforms.Bbox.x0` or
                                                             :attr:`transforms.Bbox.xmin` [1]_
------------------------------------------------------------ ------------------------------------------------------------
:meth:`Bbox.ymin`                                            :attr:`transforms.Bbox.y0` or
                                                             :attr:`transforms.Bbox.ymin` [1]_
------------------------------------------------------------ ------------------------------------------------------------
:meth:`Bbox.xmax`                                            :attr:`transforms.Bbox.x1` or
                                                             :attr:`transforms.Bbox.xmax` [1]_
------------------------------------------------------------ ------------------------------------------------------------
:meth:`Bbox.ymax`                                            :attr:`transforms.Bbox.y1` or
                                                             :attr:`transforms.Bbox.ymax` [1]_
------------------------------------------------------------ ------------------------------------------------------------
`Bbox.overlaps(bboxes)`                                      `Bbox.count_overlaps(bboxes)`
------------------------------------------------------------ ------------------------------------------------------------
`bbox_all(bboxes)`                                           `Bbox.union(bboxes)`
                                                             [:meth:`transforms.Bbox.union` is a staticmethod.]
------------------------------------------------------------ ------------------------------------------------------------
`lbwh_to_bbox(l, b, w, h)`                                   `Bbox.from_bounds(x0, y0, w, h)`
                                                             [:meth:`transforms.Bbox.from_bounds` is a staticmethod.]
------------------------------------------------------------ ------------------------------------------------------------
`inverse_transform_bbox(trans, bbox)`                        `Bbox.inverse_transformed(trans)`
------------------------------------------------------------ ------------------------------------------------------------
`Interval.contains_open(v)`                                  `interval_contains_open(tuple, v)`
------------------------------------------------------------ ------------------------------------------------------------
`Interval.contains(v)`                                       `interval_contains(tuple, v)`
------------------------------------------------------------ ------------------------------------------------------------
`identity_transform()`                                       :class:`matplotlib.transforms.IdentityTransform`
------------------------------------------------------------ ------------------------------------------------------------
`blend_xy_sep_transform(xtrans, ytrans)`                     `blended_transform_factory(xtrans, ytrans)`
------------------------------------------------------------ ------------------------------------------------------------
`scale_transform(xs, ys)`                                    `Affine2D().scale(xs[, ys])`
------------------------------------------------------------ ------------------------------------------------------------
`get_bbox_transform(boxin, boxout)`                          `BboxTransform(boxin, boxout)` or
                                                             `BboxTransformFrom(boxin)` or
                                                             `BboxTransformTo(boxout)`
------------------------------------------------------------ ------------------------------------------------------------
`Transform.seq_xy_tup(points)`                               `Transform.transform(points)`
------------------------------------------------------------ ------------------------------------------------------------
`Transform.inverse_xy_tup(points)`                           `Transform.inverted().transform(points)`
============================================================ ============================================================

.. [1] The :class:`~matplotlib.transforms.Bbox` is bound by the points
   (x0, y0) to (x1, y1) and there is no defined order to these points,
   that is, x0 is not necessarily the left edge of the box.  To get
   the left edge of the :class:`Bbox`, use the read-only property
   :attr:`~matplotlib.transforms.Bbox.xmin`.

:mod:`matplotlib.axes`
~~~~~~~~~~~~~~~~~~~~~~

============================================================ ============================================================
Old method                                                   New method
============================================================ ============================================================
`Axes.get_position()`                                        :meth:`matplotlib.axes.Axes.get_position` [2]_
------------------------------------------------------------ ------------------------------------------------------------
`Axes.set_position()`                                        :meth:`matplotlib.axes.Axes.set_position` [3]_
------------------------------------------------------------ ------------------------------------------------------------
`Axes.toggle_log_lineary()`                                  :meth:`matplotlib.axes.Axes.set_yscale` [4]_
------------------------------------------------------------ ------------------------------------------------------------
`Subplot` class                                              removed.
============================================================ ============================================================

The :class:`Polar` class has moved to :mod:`matplotlib.projections.polar`.

.. [2] :meth:`matplotlib.axes.Axes.get_position` used to return a list
   of points, now it returns a :class:`matplotlib.transforms.Bbox`
   instance.

.. [3] :meth:`matplotlib.axes.Axes.set_position` now accepts either
   four scalars or a :class:`matplotlib.transforms.Bbox` instance.

.. [4] Since the recfactoring allows for more than two scale types
   ('log' or 'linear'), it no longer makes sense to have a toggle.
   `Axes.toggle_log_lineary()` has been removed.

:mod:`matplotlib.artist`
~~~~~~~~~~~~~~~~~~~~~~~~~~

============================================================ ============================================================
Old method                                                   New method
============================================================ ============================================================
`Artist.set_clip_path(path)`                                 `Artist.set_clip_path(path, transform)` [5]_
============================================================ ============================================================

.. [5] :meth:`matplotlib.artist.Artist.set_clip_path` now accepts a
   :class:`matplotlib.path.Path` instance and a
   :class:`matplotlib.transforms.Transform` that will be applied to
   the path immediately before clipping.

:mod:`matplotlib.collections`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================================================ ============================================================
Old method                                                   New method
============================================================ ============================================================
`linestyle`                                                  `linestyles` [6]_
============================================================ ============================================================

.. [6] Linestyles are now treated like all other collection
   attributes, i.e.  a single value or multiple values may be
   provided.

:mod:`matplotlib.colors`
~~~~~~~~~~~~~~~~~~~~~~~~

============================================================ ============================================================
Old method                                                   New method
============================================================ ============================================================
`ColorConvertor.to_rgba_list(c)`                             `ColorConvertor.to_rgba_array(c)`
                                                             [:meth:`matplotlib.colors.ColorConvertor.to_rgba_array`
                                                             returns an Nx4 Numpy array of RGBA color quadruples.]
============================================================ ============================================================

:mod:`matplotlib.contour`
~~~~~~~~~~~~~~~~~~~~~~~~~

============================================================ ============================================================
Old method                                                   New method
============================================================ ============================================================
`Contour._segments`                                          :meth:`matplotlib.contour.Contour.get_paths`` [Returns a
                                                             list of :class:`matplotlib.path.Path` instances.]
============================================================ ============================================================

:mod:`matplotlib.figure`
~~~~~~~~~~~~~~~~~~~~~~~~

============================================================ ============================================================
Old method                                                   New method
============================================================ ============================================================
`Figure.dpi.get()` / `Figure.dpi.set()`                      :attr:`matplotlib.figure.Figure.dpi` *(a property)*
============================================================ ============================================================

:mod:`matplotlib.patches`
~~~~~~~~~~~~~~~~~~~~~~~~~

============================================================ ============================================================
Old method                                                   New method
============================================================ ============================================================
`Patch.get_verts()`                                          :meth:`matplotlib.patches.Patch.get_path` [Returns a
                                                             :class:`matplotlib.path.Path` instance]
============================================================ ============================================================

:mod:`matplotlib.backend_bases`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================================================ ============================================================
Old method                                                   New method
============================================================ ============================================================
`GraphicsContext.set_clip_rectangle(tuple)`                  `GraphicsContext.set_clip_rectangle(bbox)`
------------------------------------------------------------ ------------------------------------------------------------
`GraphicsContext.get_clip_path()`                            `GraphicsContext.get_clip_path()` [7]_
------------------------------------------------------------ ------------------------------------------------------------
`GraphicsContext.set_clip_path()`                            `GraphicsContext.set_clip_path()` [8]_
============================================================ ============================================================

:class:`~matplotlib.backend_bases.RendererBase`
```````````````````````````````````````````````

New methods:

  * :meth:`draw_path(self, gc, path, transform, rgbFace)
    <matplotlib.backend_bases.RendererBase.draw_path>`

  * :meth:`draw_markers(self, gc, marker_path, marker_trans, path,
    trans, rgbFace)
    <matplotlib.backend_bases.RendererBase.draw_markers`

  * :meth:`draw_path_collection(self, master_transform, cliprect,
    clippath, clippath_trans, paths, all_transforms, offsets,
    offsetTrans, facecolors, edgecolors, linewidths, linestyles,
    antialiaseds)
    <matplotlib.backend_bases.RendererBase.draw_path_collection>`
    *[optional]*

Changed methods:

  * `draw_image(self, x, y, im, bbox)` is now
    :meth:`draw_image(self, x, y, im, bbox, clippath, clippath_trans)
    <matplotlib.backend_bases.RendererBase.draw_image>`

Removed methods:

  * `draw_arc`

  * `draw_line_collection`

  * `draw_line`

  * `draw_lines`

  * `draw_point`

  * `draw_quad_mesh`

  * `draw_poly_collection`

  * `draw_polygon`

  * `draw_rectangle`

  * `draw_regpoly_collection`

.. [7] :meth:`matplotlib.backend_bases.GraphicsContext.get_clip_path`
   returns a tuple of the form (*path*, *affine_transform*), where
   *path* is a :class:`matplotlib.path.Path` instance and
   *affine_transform* is a :class:`matplotlib.transforms.Affine2D`
   instance.

.. [8] :meth:`matplotlib.backend_bases.GraphicsContext.set_clip_path`
   now only accepts a :class:`matplotlib.transforms.TransformedPath`
   instance.

Changes for 0.91.2
==================

* For :func:`csv2rec`, checkrows=0 is the new default indicating all rows
  will be checked for type inference

* A warning is issued when an image is drawn on log-scaled axes, since
  it will not log-scale the image data.

* Moved :func:`rec2gtk` to :mod:`matplotlib.toolkits.gtktools`

* Moved :func:`rec2excel` to :mod:`matplotlib.toolkits.exceltools`

* Removed, dead/experimental ExampleInfo, Namespace and Importer
  code from :mod:`matplotlib.__init__`

Changes for 0.91.1
==================

Changes for 0.91.0
==================

* Changed :func:`cbook.is_file_like` to
  :func:`cbook.is_writable_file_like` and corrected behavior.

* Added ax kwarg to :func:`pyplot.colorbar` and
  :meth:`Figure.colorbar` so that one can specify the axes object from
  which space for the colorbar is to be taken, if one does not want to
  make the colorbar axes manually.

* Changed :func:`cbook.reversed` so it yields a tuple rather than a
  (index, tuple). This agrees with the python reversed builtin,
  and cbook only defines reversed if python doesn't provide the
  builtin.

* Made skiprows=1 the default on :func:`csv2rec`

* The gd and paint backends have been deleted.

* The errorbar method and function now accept additional kwargs
  so that upper and lower limits can be indicated by capping the
  bar with a caret instead of a straight line segment.

* The :mod:`matplotlib.dviread` file now has a parser for files like
  psfonts.map and pdftex.map, to map TeX font names to external files.

* The file :mod:`matplotlib.type1font` contains a new class for Type 1
  fonts.  Currently it simply reads pfa and pfb format files and
  stores the data in a way that is suitable for embedding in pdf
  files. In the future the class might actually parse the font to
  allow e.g.,  subsetting.

* :mod:`matplotlib.FT2Font` now supports :meth:`FT_Attach_File`. In
  practice this can be used to read an afm file in addition to a
  pfa/pfb file, to get metrics and kerning information for a Type 1
  font.

* The :class:`AFM` class now supports querying CapHeight and stem
  widths. The get_name_char method now has an isord kwarg like
  get_width_char.

* Changed :func:`pcolor` default to shading='flat'; but as noted now in the
  docstring, it is preferable to simply use the edgecolor kwarg.

* The mathtext font commands (``\cal``, ``\rm``, ``\it``, ``\tt``) now
  behave as TeX does: they are in effect until the next font change
  command or the end of the grouping.  Therefore uses of ``$\cal{R}$``
  should be changed to ``${\cal R}$``.  Alternatively, you may use the
  new LaTeX-style font commands (``\mathcal``, ``\mathrm``,
  ``\mathit``, ``\mathtt``) which do affect the following group,
  e.g., ``$\mathcal{R}$``.

* Text creation commands have a new default linespacing and a new
  ``linespacing`` kwarg, which is a multiple of the maximum vertical
  extent of a line of ordinary text.  The default is 1.2;
  ``linespacing=2`` would be like ordinary double spacing, for example.

* Changed default kwarg in
  :meth:`matplotlib.colors.Normalize.__init__`` to ``clip=False``;
  clipping silently defeats the purpose of the special over, under,
  and bad values in the colormap, thereby leading to unexpected
  behavior.  The new default should reduce such surprises.

* Made the emit property of :meth:`~matplotlib.axes.Axes.set_xlim` and
  :meth:`~matplotlib.axes.Axes.set_ylim` ``True`` by default; removed
  the Axes custom callback handling into a 'callbacks' attribute which
  is a :class:`~matplotlib.cbook.CallbackRegistry` instance.  This now
  supports the 'xlim_changed' and 'ylim_changed' Axes events.

Changes for 0.90.1
==================

::

    The file dviread.py has a (very limited and fragile) dvi reader
    for usetex support. The API might change in the future so don't
    depend on it yet.

    Removed deprecated support for a float value as a gray-scale;
    now it must be a string, like '0.5'.  Added alpha kwarg to
    ColorConverter.to_rgba_list.

    New method set_bounds(vmin, vmax) for formatters, locators sets
    the viewInterval and dataInterval from floats.

    Removed deprecated colorbar_classic.

    Line2D.get_xdata and get_ydata valid_only=False kwarg is replaced
    by orig=True.  When True, it returns the original data, otherwise
    the processed data (masked, converted)

    Some modifications to the units interface.
    units.ConversionInterface.tickers renamed to
    units.ConversionInterface.axisinfo and it now returns a
    units.AxisInfo object rather than a tuple.  This will make it
    easier to add axis info functionality (e.g., I added a default label
    on this iteration) w/o having to change the tuple length and hence
    the API of the client code every time new functionality is added.
    Also, units.ConversionInterface.convert_to_value is now simply
    named units.ConversionInterface.convert.

    Axes.errorbar uses Axes.vlines and Axes.hlines to draw its error
    limits int he vertical and horizontal direction.  As you'll see
    in the changes below, these functions now return a LineCollection
    rather than a list of lines.  The new return signature for
    errorbar is  ylins, caplines, errorcollections where
    errorcollections is a xerrcollection, yerrcollection

    Axes.vlines and Axes.hlines now create and returns a LineCollection, not a list
    of lines.  This is much faster.  The kwarg signature has changed,
    so consult the docs

    MaxNLocator accepts a new Boolean kwarg ('integer') to force
    ticks to integer locations.

    Commands that pass an argument to the Text constructor or to
    Text.set_text() now accept any object that can be converted
    with '%s'.  This affects xlabel(), title(), etc.

    Barh now takes a **kwargs dict instead of most of the old
    arguments. This helps ensure that bar and barh are kept in sync,
    but as a side effect you can no longer pass e.g., color as a
    positional argument.

    ft2font.get_charmap() now returns a dict that maps character codes
    to glyph indices (until now it was reversed)

    Moved data files into lib/matplotlib so that setuptools' develop
    mode works. Re-organized the mpl-data layout so that this source
    structure is maintained in the installation. (i.e., the 'fonts' and
    'images' sub-directories are maintained in site-packages.).
    Suggest removing site-packages/matplotlib/mpl-data and
    ~/.matplotlib/ttffont.cache before installing

Changes for 0.90.0
==================

::

    All artists now implement a "pick" method which users should not
    call.  Rather, set the "picker" property of any artist you want to
    pick on (the epsilon distance in points for a hit test) and
    register with the "pick_event" callback.  See
    examples/pick_event_demo.py for details

    Bar, barh, and hist have "log" binary kwarg: log=True
    sets the ordinate to a log scale.

    Boxplot can handle a list of vectors instead of just
    an array, so vectors can have different lengths.

    Plot can handle 2-D x and/or y; it plots the columns.

    Added linewidth kwarg to bar and barh.

    Made the default Artist._transform None (rather than invoking
    identity_transform for each artist only to have it overridden
    later).  Use artist.get_transform() rather than artist._transform,
    even in derived classes, so that the default transform will be
    created lazily as needed

    New LogNorm subclass of Normalize added to colors.py.
    All Normalize subclasses have new inverse() method, and
    the __call__() method has a new clip kwarg.

    Changed class names in colors.py to match convention:
    normalize -> Normalize, no_norm -> NoNorm.  Old names
    are still available for now.

    Removed obsolete pcolor_classic command and method.

    Removed lineprops and markerprops from the Annotation code and
    replaced them with an arrow configurable with kwarg arrowprops.
    See examples/annotation_demo.py - JDH

Changes for 0.87.7
==================

::

    Completely reworked the annotations API because I found the old
    API cumbersome.  The new design is much more legible and easy to
    read.  See matplotlib.text.Annotation and
    examples/annotation_demo.py

    markeredgecolor and markerfacecolor cannot be configured in
    matplotlibrc any more. Instead, markers are generally colored
    automatically based on the color of the line, unless marker colors
    are explicitly set as kwargs - NN

    Changed default comment character for load to '#' - JDH

    math_parse_s_ft2font_svg from mathtext.py & mathtext2.py now returns
    width, height, svg_elements. svg_elements is an instance of Bunch (
    cmbook.py) and has the attributes svg_glyphs and svg_lines, which are both
    lists.

    Renderer.draw_arc now takes an additional parameter, rotation.
    It specifies to draw the artist rotated in degrees anti-
    clockwise.  It was added for rotated ellipses.

    Renamed Figure.set_figsize_inches to Figure.set_size_inches to
    better match the get method, Figure.get_size_inches.

    Removed the copy_bbox_transform from transforms.py; added
    shallowcopy methods to all transforms.  All transforms already
    had deepcopy methods.

    FigureManager.resize(width, height): resize the window
    specified in pixels

    barh: x and y args have been renamed to width and bottom
    respectively, and their order has been swapped to maintain
    a (position, value) order.

    bar and barh: now accept kwarg 'edgecolor'.

    bar and barh: The left, height, width and bottom args can
    now all be scalars or sequences; see docstring.

    barh: now defaults to edge aligned instead of center
    aligned bars

    bar, barh and hist: Added a keyword arg 'align' that
    controls between edge or center bar alignment.

    Collections: PolyCollection and LineCollection now accept
    vertices or segments either in the original form [(x,y),
    (x,y), ...] or as a 2D numerix array, with X as the first column
    and Y as the second. Contour and quiver output the numerix
    form.  The transforms methods Bbox.update() and
    Transformation.seq_xy_tups() now accept either form.

    Collections: LineCollection is now a ScalarMappable like
    PolyCollection, etc.

    Specifying a grayscale color as a float is deprecated; use
    a string instead, e.g., 0.75 -> '0.75'.

    Collections: initializers now accept any mpl color arg, or
    sequence of such args; previously only a sequence of rgba
    tuples was accepted.

    Colorbar: completely new version and api; see docstring.  The
    original version is still accessible as colorbar_classic, but
    is deprecated.

    Contourf: "extend" kwarg replaces "clip_ends"; see docstring.
    Masked array support added to pcolormesh.

    Modified aspect-ratio handling:
        Removed aspect kwarg from imshow
        Axes methods:
            set_aspect(self, aspect, adjustable=None, anchor=None)
            set_adjustable(self, adjustable)
            set_anchor(self, anchor)
        Pylab interface:
            axis('image')

     Backend developers: ft2font's load_char now takes a flags
     argument, which you can OR together from the LOAD_XXX
     constants.

Changes for 0.86
================

::

     Matplotlib data is installed into the matplotlib module.
     This is similar to package_data.  This should get rid of
     having to check for many possibilities in _get_data_path().
     The MATPLOTLIBDATA env key is still checked first to allow
     for flexibility.

     1) Separated the color table data from cm.py out into
     a new file, _cm.py, to make it easier to find the actual
     code in cm.py and to add new colormaps. Everything
     from _cm.py is imported by cm.py, so the split should be
     transparent.
     2) Enabled automatic generation of a colormap from
     a list of colors in contour; see modified
     examples/contour_demo.py.
     3) Support for imshow of a masked array, with the
     ability to specify colors (or no color at all) for
     masked regions, and for regions that are above or
     below the normally mapped region.  See
     examples/image_masked.py.
     4) In support of the above, added two new classes,
     ListedColormap, and no_norm, to colors.py, and modified
     the Colormap class to include common functionality. Added
     a clip kwarg to the normalize class.

Changes for 0.85
================

::

    Made xtick and ytick separate props in rc

    made pos=None the default for tick formatters rather than 0 to
    indicate "not supplied"

    Removed "feature" of minor ticks which prevents them from
    overlapping major ticks.  Often you want major and minor ticks at
    the same place, and can offset the major ticks with the pad.  This
    could be made configurable

    Changed the internal structure of contour.py to a more OO style.
    Calls to contour or contourf in axes.py or pylab.py now return
    a ContourSet object which contains references to the
    LineCollections or PolyCollections created by the call,
    as well as the configuration variables that were used.
    The ContourSet object is a "mappable" if a colormap was used.

    Added a clip_ends kwarg to contourf. From the docstring:
             * clip_ends = True
               If False, the limits for color scaling are set to the
               minimum and maximum contour levels.
               True (default) clips the scaling limits.  Example:
               if the contour boundaries are V = [-100, 2, 1, 0, 1, 2, 100],
               then the scaling limits will be [-100, 100] if clip_ends
               is False, and [-3, 3] if clip_ends is True.
    Added kwargs linewidths, antialiased, and nchunk to contourf.  These
    are experimental; see the docstring.

    Changed Figure.colorbar():
        kw argument order changed;
        if mappable arg is a non-filled ContourSet, colorbar() shows
                lines instead hof polygons.
        if mappable arg is a filled ContourSet with clip_ends=True,
                the endpoints are not labelled, so as to give the
                correct impression of open-endedness.

    Changed LineCollection.get_linewidths to get_linewidth, for
    consistency.


Changes for 0.84
================

::

    Unified argument handling between hlines and vlines.  Both now
    take optionally a fmt argument (as in plot) and a keyword args
    that can be passed onto Line2D.

    Removed all references to "data clipping" in rc and lines.py since
    these were not used and not optimized.  I'm sure they'll be
    resurrected later with a better implementation when needed.

    'set' removed - no more deprecation warnings.  Use 'setp' instead.

    Backend developers: Added flipud method to image and removed it
    from to_str.  Removed origin kwarg from backend.draw_image.
    origin is handled entirely by the frontend now.

Changes for 0.83
================

::

  - Made HOME/.matplotlib the new config dir where the matplotlibrc
    file, the ttf.cache, and the tex.cache live.  The new default
    filenames in .matplotlib have no leading dot and are not hidden.
    e.g., the new names are matplotlibrc, tex.cache, and ttffont.cache.
    This is how ipython does it so it must be right.

    If old files are found, a warning is issued and they are moved to
    the new location.

  - backends/__init__.py no longer imports new_figure_manager,
    draw_if_interactive and show from the default backend, but puts
    these imports into a call to pylab_setup.  Also, the Toolbar is no
    longer imported from WX/WXAgg.  New usage:

      from backends import pylab_setup
      new_figure_manager, draw_if_interactive, show = pylab_setup()

  - Moved Figure.get_width_height() to FigureCanvasBase. It now
    returns int instead of float.

Changes for 0.82
================

::

  - toolbar import change in GTKAgg, GTKCairo and WXAgg

  - Added subplot config tool to GTK* backends -- note you must now
    import the NavigationToolbar2 from your backend of choice rather
    than from backend_gtk because it needs to know about the backend
    specific canvas -- see examples/embedding_in_gtk2.py.  Ditto for
    wx backend -- see examples/embedding_in_wxagg.py


  - hist bin change

      Sean Richards notes there was a problem in the way we created
      the binning for histogram, which made the last bin
      underrepresented.  From his post:

        I see that hist uses the linspace function to create the bins
        and then uses searchsorted to put the values in their correct
        bin. That's all good but I am confused over the use of linspace
        for the bin creation. I wouldn't have thought that it does
        what is needed, to quote the docstring it creates a "Linear
        spaced array from min to max". For it to work correctly
        shouldn't the values in the bins array be the same bound for
        each bin? (i.e. each value should be the lower bound of a
        bin). To provide the correct bins for hist would it not be
        something like

        def bins(xmin, xmax, N):
          if N==1: return xmax
          dx = (xmax-xmin)/N # instead of N-1
          return xmin + dx*arange(N)


       This suggestion is implemented in 0.81.  My test script with these
       changes does not reveal any bias in the binning

        from matplotlib.numerix.mlab import randn, rand, zeros, Float
        from matplotlib.mlab import hist, mean

        Nbins = 50
        Ntests = 200
        results = zeros((Ntests,Nbins), typecode=Float)
        for i in range(Ntests):
            print 'computing', i
            x = rand(10000)
            n, bins = hist(x, Nbins)
            results[i] = n
        print mean(results)


Changes for 0.81
================

::

  - pylab and artist "set" functions renamed to setp to avoid clash
    with python2.4 built-in set.  Current version will issue a
    deprecation warning which will be removed in future versions

  - imshow interpolation arguments changes for advanced interpolation
    schemes.  See help imshow, particularly the interpolation,
    filternorm and filterrad kwargs

  - Support for masked arrays has been added to the plot command and
    to the Line2D object.  Only the valid points are plotted.  A
    "valid_only" kwarg was added to the get_xdata() and get_ydata()
    methods of Line2D; by default it is False, so that the original
    data arrays are returned. Setting it to True returns the plottable
    points.

  - contour changes:

    Masked arrays: contour and contourf now accept masked arrays as
      the variable to be contoured.  Masking works correctly for
      contour, but a bug remains to be fixed before it will work for
      contourf.  The "badmask" kwarg has been removed from both
      functions.

     Level argument changes:

       Old version: a list of levels as one of the positional
       arguments specified the lower bound of each filled region; the
       upper bound of the last region was taken as a very large
       number.  Hence, it was not possible to specify that z values
       between 0 and 1, for example, be filled, and that values
       outside that range remain unfilled.

       New version: a list of N levels is taken as specifying the
       boundaries of N-1 z ranges.  Now the user has more control over
       what is colored and what is not.  Repeated calls to contourf
       (with different colormaps or color specifications, for example)
       can be used to color different ranges of z.  Values of z
       outside an expected range are left uncolored.

       Example:
         Old: contourf(z, [0, 1, 2]) would yield 3 regions: 0-1, 1-2, and >2.
         New: it would yield 2 regions: 0-1, 1-2.  If the same 3 regions were
         desired, the equivalent list of levels would be [0, 1, 2,
         1e38].

Changes for 0.80
================

::

  - xlim/ylim/axis always return the new limits regardless of
    arguments.  They now take kwargs which allow you to selectively
    change the upper or lower limits while leaving unnamed limits
    unchanged.  See help(xlim) for example

Changes for 0.73
================

::

  - Removed deprecated ColormapJet and friends

  - Removed all error handling from the verbose object

  - figure num of zero is now allowed

Changes for 0.72
================

::

  - Line2D, Text, and Patch copy_properties renamed update_from and
    moved into artist base class

  - LineCollecitons.color renamed to LineCollections.set_color for
    consistency with set/get introspection mechanism,

  - pylab figure now defaults to num=None, which creates a new figure
    with a guaranteed unique number

  - contour method syntax changed - now it is MATLAB compatible

      unchanged: contour(Z)
      old: contour(Z, x=Y, y=Y)
      new: contour(X, Y, Z)

    see http://matplotlib.sf.net/matplotlib.pylab.html#-contour


   - Increased the default resolution for save command.

   - Renamed the base attribute of the ticker classes to _base to avoid conflict
     with the base method.  Sitt for subs

   - subs=none now does autosubbing in the tick locator.

   - New subplots that overlap old will delete the old axes.  If you
     do not want this behavior, use fig.add_subplot or the axes
     command

Changes for 0.71
================

::

   Significant numerix namespace changes, introduced to resolve
   namespace clashes between python built-ins and mlab names.
   Refactored numerix to maintain separate modules, rather than
   folding all these names into a single namespace.  See the following
   mailing list threads for more information and background

     http://sourceforge.net/mailarchive/forum.php?thread_id=6398890&forum_id=36187
     http://sourceforge.net/mailarchive/forum.php?thread_id=6323208&forum_id=36187


   OLD usage

     from matplotlib.numerix import array, mean, fft

   NEW usage

     from matplotlib.numerix import array
     from matplotlib.numerix.mlab import mean
     from matplotlib.numerix.fft import fft

   numerix dir structure mirrors numarray (though it is an incomplete
   implementation)

     numerix
     numerix/mlab
     numerix/linear_algebra
     numerix/fft
     numerix/random_array

   but of course you can use 'numerix : Numeric' and still get the
   symbols.

   pylab still imports most of the symbols from Numerix, MLab, fft,
   etc, but is more cautious.  For names that clash with python names
   (min, max, sum), pylab keeps the builtins and provides the numeric
   versions with an a* prefix, e.g., (amin, amax, asum)

Changes for 0.70
================

::

   MplEvent factored into a base class Event and derived classes
   MouseEvent and KeyEvent

   Removed definct set_measurement in wx toolbar

Changes for 0.65.1
==================

::

  removed add_axes and add_subplot from backend_bases.  Use
  figure.add_axes and add_subplot instead.  The figure now manages the
  current axes with gca and sca for get and set current axes.  If you
  have code you are porting which called, e.g., figmanager.add_axes, you
  can now simply do figmanager.canvas.figure.add_axes.

Changes for 0.65
================

::


  mpl_connect and mpl_disconnect in the MATLAB interface renamed to
  connect and disconnect

  Did away with the text methods for angle since they were ambiguous.
  fontangle could mean fontstyle (obligue, etc) or the rotation of the
  text.  Use style and rotation instead.

Changes for 0.63
================

::

  Dates are now represented internally as float days since 0001-01-01,
  UTC.

  All date tickers and formatters are now in matplotlib.dates, rather
  than matplotlib.tickers

  converters have been abolished from all functions and classes.
  num2date and date2num are now the converter functions for all date
  plots

  Most of the date tick locators have a different meaning in their
  constructors.  In the prior implementation, the first argument was a
  base and multiples of the base were ticked.  e.g.,

    HourLocator(5)  # old: tick every 5 minutes

  In the new implementation, the explicit points you want to tick are
  provided as a number or sequence

     HourLocator(range(0,5,61))  # new: tick every 5 minutes

  This gives much greater flexibility.  I have tried to make the
  default constructors (no args) behave similarly, where possible.

  Note that YearLocator still works under the base/multiple scheme.
  The difference between the YearLocator and the other locators is
  that years are not recurrent.


  Financial functions:

    matplotlib.finance.quotes_historical_yahoo(ticker, date1, date2)

     date1, date2 are now datetime instances.  Return value is a list
     of quotes where the quote time is a float - days since gregorian
     start, as returned by date2num

     See examples/finance_demo.py for example usage of new API

Changes for 0.61
================

::

  canvas.connect is now deprecated for event handling.  use
  mpl_connect and mpl_disconnect instead.  The callback signature is
  func(event) rather than func(widget, event)

Changes for 0.60
================

::

  ColormapJet and Grayscale are deprecated.  For backwards
  compatibility, they can be obtained either by doing

    from matplotlib.cm import ColormapJet

  or

    from matplotlib.matlab import *

  They are replaced by cm.jet and cm.grey

Changes for 0.54.3
==================

::

  removed the set_default_font / get_default_font scheme from the
  font_manager to unify customization of font defaults with the rest of
  the rc scheme.  See examples/font_properties_demo.py and help(rc) in
  matplotlib.matlab.

Changes for 0.54
================

MATLAB interface
----------------

dpi
~~~

Several of the backends used a PIXELS_PER_INCH hack that I added to
try and make images render consistently across backends.  This just
complicated matters.  So you may find that some font sizes and line
widths appear different than before.  Apologies for the
inconvenience. You should set the dpi to an accurate value for your
screen to get true sizes.


pcolor and scatter
~~~~~~~~~~~~~~~~~~

There are two changes to the MATLAB interface API, both involving the
patch drawing commands.  For efficiency, pcolor and scatter have been
rewritten to use polygon collections, which are a new set of objects
from matplotlib.collections designed to enable efficient handling of
large collections of objects.  These new collections make it possible
to build large scatter plots or pcolor plots with no loops at the
python level, and are significantly faster than their predecessors.
The original pcolor and scatter functions are retained as
pcolor_classic and scatter_classic.

The return value from pcolor is a PolyCollection.  Most of the
propertes that are available on rectangles or other patches are also
available on PolyCollections, e.g., you can say::

  c = scatter(blah, blah)
  c.set_linewidth(1.0)
  c.set_facecolor('r')
  c.set_alpha(0.5)

or::

  c = scatter(blah, blah)
  set(c, 'linewidth', 1.0, 'facecolor', 'r', 'alpha', 0.5)


Because the collection is a single object, you no longer need to loop
over the return value of scatter or pcolor to set properties for the
entire list.

If you want the different elements of a collection to vary on a
property, e.g., to have different line widths, see matplotlib.collections
for a discussion on how to set the properties as a sequence.

For scatter, the size argument is now in points^2 (the area of the
symbol in points) as in MATLAB and is not in data coords as before.
Using sizes in data coords caused several problems.  So you will need
to adjust your size arguments accordingly or use scatter_classic.

mathtext spacing
~~~~~~~~~~~~~~~~

For reasons not clear to me (and which I'll eventually fix) spacing no
longer works in font groups.  However, I added three new spacing
commands which compensate for this '\ ' (regular space), '\/' (small
space) and '\hspace{frac}' where frac is a fraction of fontsize in
points.  You will need to quote spaces in font strings, is::

  title(r'$\rm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')



Object interface - Application programmers
------------------------------------------

Autoscaling
~~~~~~~~~~~

  The x and y axis instances no longer have autoscale view.  These are
  handled by axes.autoscale_view

Axes creation
~~~~~~~~~~~~~

    You should not instantiate your own Axes any more using the OO API.
    Rather, create a Figure as before and in place of::

      f = Figure(figsize=(5,4), dpi=100)
      a = Subplot(f, 111)
      f.add_axis(a)

    use::

      f = Figure(figsize=(5,4), dpi=100)
      a = f.add_subplot(111)

    That is, add_axis no longer exists and is replaced by::

      add_axes(rect, axisbg=defaultcolor, frameon=True)
      add_subplot(num, axisbg=defaultcolor, frameon=True)

Artist methods
~~~~~~~~~~~~~~

  If you define your own Artists, you need to rename the _draw method
  to draw

Bounding boxes
~~~~~~~~~~~~~~

   matplotlib.transforms.Bound2D is replaced by
   matplotlib.transforms.Bbox.  If you want to construct a bbox from
   left, bottom, width, height (the signature for Bound2D), use
   matplotlib.transforms.lbwh_to_bbox, as in

    bbox = clickBBox = lbwh_to_bbox(left, bottom, width, height)

   The Bbox has a different API than the Bound2D.  e.g., if you want to
   get the width and height of the bbox

     OLD::
        width  = fig.bbox.x.interval()
        height = fig.bbox.y.interval()

     New::
        width  = fig.bbox.width()
        height = fig.bbox.height()




Object constructors
~~~~~~~~~~~~~~~~~~~

  You no longer pass the bbox, dpi, or transforms to the various
  Artist constructors.  The old way or creating lines and rectangles
  was cumbersome because you had to pass so many attributes to the
  Line2D and Rectangle classes not related directly to the geometry
  and properties of the object.  Now default values are added to the
  object when you call axes.add_line or axes.add_patch, so they are
  hidden from the user.

  If you want to define a custom transformation on these objects, call
  o.set_transform(trans) where trans is a Transformation instance.

  In prior versions of you wanted to add a custom line in data coords,
  you would have to do

        l =  Line2D(dpi, bbox, x, y,
                    color = color,
                    transx = transx,
                    transy = transy,
                    )

  now all you need is

        l =  Line2D(x, y, color=color)

  and the axes will set the transformation for you (unless you have
  set your own already, in which case it will eave it unchanged)

Transformations
~~~~~~~~~~~~~~~

  The entire transformation architecture has been rewritten.
  Previously the x and y transformations where stored in the xaxis and
  yaxis instances.  The problem with this approach is it only allows
  for separable transforms (where the x and y transformations don't
  depend on one another).  But for cases like polar, they do.  Now
  transformations operate on x,y together.  There is a new base class
  matplotlib.transforms.Transformation and two concrete
  implementations, matplotlib.transforms.SeparableTransformation and
  matplotlib.transforms.Affine.  The SeparableTransformation is
  constructed with the bounding box of the input (this determines the
  rectangular coordinate system of the input, i.e., the x and y view
  limits), the bounding box of the display, and possibly nonlinear
  transformations of x and y.  The 2 most frequently used
  transformations, data coordinates -> display and axes coordinates ->
  display are available as ax.transData and ax.transAxes.  See
  alignment_demo.py which uses axes coords.

  Also, the transformations should be much faster now, for two reasons

   * they are written entirely in extension code

   * because they operate on x and y together, they can do the entire
     transformation in one loop.  Earlier I did something along the
     lines of::

       xt = sx*func(x) + tx
       yt = sy*func(y) + ty

     Although this was done in numerix, it still involves 6 length(x)
     for-loops (the multiply, add, and function evaluation each for x
     and y).  Now all of that is done in a single pass.


  If you are using transformations and bounding boxes to get the
  cursor position in data coordinates, the method calls are a little
  different now.  See the updated examples/coords_demo.py which shows
  you how to do this.

  Likewise, if you are using the artist bounding boxes to pick items
  on the canvas with the GUI, the bbox methods are somewhat
  different.  You will need to see the updated
  examples/object_picker.py.

  See unit/transforms_unit.py for many examples using the new
  transformations.


Changes for 0.50
================

::

  * refactored Figure class so it is no longer backend dependent.
    FigureCanvasBackend takes over the backend specific duties of the
    Figure.  matplotlib.backend_bases.FigureBase moved to
    matplotlib.figure.Figure.

  * backends must implement FigureCanvasBackend (the thing that
    controls the figure and handles the events if any) and
    FigureManagerBackend (wraps the canvas and the window for MATLAB
    interface).  FigureCanvasBase implements a backend switching
    mechanism

  * Figure is now an Artist (like everything else in the figure) and
    is totally backend independent

  * GDFONTPATH renamed to TTFPATH

  * backend faceColor argument changed to rgbFace

  * colormap stuff moved to colors.py

  * arg_to_rgb in backend_bases moved to class ColorConverter in
    colors.py

  * GD users must upgrade to gd-2.0.22 and gdmodule-0.52 since new gd
    features (clipping, antialiased lines) are now used.

  * Renderer must implement points_to_pixels

  Migrating code:

  MATLAB interface:

    The only API change for those using the MATLAB interface is in how
    you call figure redraws for dynamically updating figures.  In the
    old API, you did

      fig.draw()

    In the new API, you do

      manager = get_current_fig_manager()
      manager.canvas.draw()

    See the examples system_monitor.py, dynamic_demo.py, and anim.py

  API

    There is one important API change for application developers.
    Figure instances used subclass GUI widgets that enabled them to be
    placed directly into figures.  e.g., FigureGTK subclassed
    gtk.DrawingArea.  Now the Figure class is independent of the
    backend, and FigureCanvas takes over the functionality formerly
    handled by Figure.  In order to include figures into your apps,
    you now need to do, for example

      # gtk example
      fig = Figure(figsize=(5,4), dpi=100)
      canvas = FigureCanvasGTK(fig)  # a gtk.DrawingArea
      canvas.show()
      vbox.pack_start(canvas)

    If you use the NavigationToolbar, this in now intialized with a
    FigureCanvas, not a Figure.  The examples embedding_in_gtk.py,
    embedding_in_gtk2.py, and mpl_with_glade.py all reflect the new
    API so use these as a guide.

    All prior calls to

     figure.draw()  and
     figure.print_figure(args)

    should now be

     canvas.draw()  and
     canvas.print_figure(args)

    Apologies for the inconvenience.  This refactorization brings
    significant more freedom in developing matplotlib and should bring
    better plotting capabilities, so I hope the inconvenience is worth
    it.

Changes for 0.42
================

::

  * Refactoring AxisText to be backend independent.  Text drawing and
    get_window_extent functionality will be moved to the Renderer.

  * backend_bases.AxisTextBase is now text.Text module

  * All the erase and reset functionality removed from AxisText - not
    needed with double buffered drawing.  Ditto with state change.
    Text instances have a get_prop_tup method that returns a hashable
    tuple of text properties which you can use to see if text props
    have changed, e.g., by caching a font or layout instance in a dict
    with the prop tup as a key -- see RendererGTK.get_pango_layout in
    backend_gtk for an example.

  * Text._get_xy_display renamed Text.get_xy_display

  * Artist set_renderer and wash_brushes methods removed

  * Moved Legend class from matplotlib.axes into matplotlib.legend

  * Moved Tick, XTick, YTick, Axis, XAxis, YAxis from matplotlib.axes
    to matplotlib.axis

  * moved process_text_args to matplotlib.text

  * After getting Text handled in a backend independent fashion, the
    import process is much cleaner since there are no longer cyclic
    dependencies

  * matplotlib.matlab._get_current_fig_manager renamed to
    matplotlib.matlab.get_current_fig_manager to allow user access to
    the GUI window attribute, e.g., figManager.window for GTK and
    figManager.frame for wx

Changes for 0.40
================

::

  - Artist
      * __init__ takes a DPI instance and a Bound2D instance which is
        the bounding box of the artist in display coords
      * get_window_extent returns a Bound2D instance
      * set_size is removed; replaced by bbox and dpi
      * the clip_gc method is removed.  Artists now clip themselves with
        their box
      * added _clipOn boolean attribute.  If True, gc clip to bbox.

  - AxisTextBase
      * Initialized with a transx, transy which are Transform instances
      * set_drawing_area removed
      * get_left_right and get_top_bottom are replaced by get_window_extent

  - Line2D Patches now take transx, transy
      * Initialized with a transx, transy which are Transform instances

  - Patches
     * Initialized with a transx, transy which are Transform instances

  - FigureBase attributes dpi is a DPI intance rather than scalar and
    new attribute bbox is a Bound2D in display coords, and I got rid
    of the left, width, height, etc... attributes.  These are now
    accessible as, for example, bbox.x.min is left, bbox.x.interval()
    is width, bbox.y.max is top, etc...

  - GcfBase attribute pagesize renamed to figsize

  - Axes
      * removed figbg attribute
      * added fig instance to __init__
      * resizing is handled by figure call to resize.

  - Subplot
      * added fig instance to __init__

  - Renderer methods for patches now take gcEdge and gcFace instances.
    gcFace=None takes the place of filled=False

  - True and False symbols provided by cbook in a python2.3 compatible
    way

  - new module transforms supplies Bound1D, Bound2D and Transform
    instances and more

  - Changes to the MATLAB helpers API

    * _matlab_helpers.GcfBase is renamed by Gcf.  Backends no longer
      need to derive from this class.  Instead, they provide a factory
      function new_figure_manager(num, figsize, dpi).  The destroy
      method of the GcfDerived from the backends is moved to the derived
      FigureManager.

    * FigureManagerBase moved to backend_bases

    * Gcf.get_all_figwins renamed to Gcf.get_all_fig_managers

  Jeremy:

    Make sure to self._reset = False in AxisTextWX._set_font.  This was
    something missing in my backend code.
