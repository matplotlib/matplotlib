API Changes in 1.4.x
====================

Code changes
------------

* A major refactoring of the axes module was made. The axes module has been
  split into smaller modules:

    - the ``_base`` module, which contains a new private ``_AxesBase`` class.
      This class contains all methods except plotting and labelling methods.
    - the `~matplotlib.axes` module, which contains the `.axes.Axes` class.
      This class inherits from ``_AxesBase``, and contains all plotting and
      labelling methods.
    - the ``_subplot`` module, with all the classes concerning subplotting.

There are a couple of things that do not exists in the `~matplotlib.axes`
module's namespace anymore. If you use them, you need to import them from their
original location:

  - ``math`` -> ``import math``
  - ``ma`` -> ``from numpy import ma``
  - ``cbook`` -> ``from matplotlib import cbook``
  - ``docstring`` -> ``from matplotlib import docstring``
  - ``is_sequence_of_strings`` -> ``from matplotlib.cbook import is_sequence_of_strings``
  - ``is_string_like`` -> ``from matplotlib.cbook import is_string_like``
  - ``iterable`` -> ``from matplotlib.cbook import iterable``
  - ``itertools`` -> ``import itertools``
  - ``martist`` -> ``from matplotlib import artist as martist``
  - ``matplotlib`` -> ``import matplotlib``
  - ``mcoll`` -> ``from matplotlib import collections as mcoll``
  - ``mcolors`` -> ``from matplotlib import colors as mcolors``
  - ``mcontour`` -> ``from matplotlib import contour as mcontour``
  - ``mpatches`` -> ``from matplotlib import patches as mpatches``
  - ``mpath`` -> ``from matplotlib import path as mpath``
  - ``mquiver`` -> ``from matplotlib import quiver as mquiver``
  - ``mstack`` -> ``from matplotlib import stack as mstack``
  - ``mstream`` -> ``from matplotlib import stream as mstream``
  - ``mtable`` -> ``from matplotlib import table as mtable``

* As part of the refactoring to enable Qt5 support, the module
  ``matplotlib.backends.qt4_compat`` was renamed to
  ``matplotlib.backends.qt_compat``.  ``qt4_compat`` is deprecated in 1.4 and
  will be removed in 1.5.

* The :func:`~matplotlib.pyplot.errorbar` method has been changed such that
  the upper and lower limits (*lolims*, *uplims*, *xlolims*, *xuplims*) now
  point in the correct direction.

* The *fmt* kwarg for :func:`~matplotlib.pyplot.errorbar` now supports
  the string 'none' to suppress drawing of a line and markers; use
  of the *None* object for this is deprecated. The default *fmt*
  value is changed to the empty string (''), so the line and markers
  are governed by the :func:`~matplotlib.pyplot.plot` defaults.

* A bug has been fixed in the path effects rendering of fonts, which now means
  that the font size is consistent with non-path effect fonts. See
  https://github.com/matplotlib/matplotlib/issues/2889 for more detail.

* The Sphinx extensions ``ipython_directive`` and
  ``ipython_console_highlighting`` have been moved to the IPython
  project itself.  While they remain in Matplotlib for this release,
  they have been deprecated.  Update your extensions in :file:`conf.py` to
  point to ``IPython.sphinxext.ipython_directive`` instead of
  ``matplotlib.sphinxext.ipython_directive``.

* In ``matplotlib.finance``, almost all functions have been deprecated
  and replaced with a pair of functions name ``*_ochl`` and ``*_ohlc``.
  The former is the 'open-close-high-low' order of quotes used
  previously in this module, and the latter is the
  'open-high-low-close' order that is standard in finance.

* For consistency the ``face_alpha`` keyword to
  :class:`matplotlib.patheffects.SimplePatchShadow` has been deprecated in
  favour of the ``alpha`` keyword. Similarly, the keyword ``offset_xy`` is now
  named ``offset`` across all :class:`~matplotlib.patheffects.AbstractPathEffect`\ s.
  ``matplotlib.patheffects._Base`` has
  been renamed to :class:`matplotlib.patheffects.AbstractPathEffect`.
  ``matplotlib.patheffect.ProxyRenderer`` has been renamed to
  :class:`matplotlib.patheffects.PathEffectRenderer` and is now a full
  RendererBase subclass.

* The artist used to draw the outline of a `.Figure.colorbar` has been changed
  from a `matplotlib.lines.Line2D` to `matplotlib.patches.Polygon`, thus
  `.colorbar.ColorbarBase.outline` is now a `matplotlib.patches.Polygon`
  object.

* The legend handler interface has changed from a callable, to any object
  which implements the ``legend_artists`` method (a deprecation phase will
  see this interface be maintained for v1.4). See
  :doc:`/tutorials/intermediate/legend_guide` for further details. Further legend changes
  include:

   * ``matplotlib.axes.Axes._get_legend_handles`` now returns a generator of
     handles, rather than a list.

   * The :func:`~matplotlib.pyplot.legend` function's *loc* positional
     argument has been deprecated. Use the *loc* keyword argument instead.

* The :rc:`savefig.transparent` has been added to control
  default transparency when saving figures.

* Slightly refactored the `.Annotation` family.  The text location in
  `.Annotation` is now entirely handled by the underlying `.Text`
  object so ``.set_position`` works as expected.  The attributes *xytext* and
  *textcoords* have been deprecated in favor of *xyann* and *anncoords* so
  that `.Annotation` and `.AnnotationBbox` can share a common sensibly named
  api for getting/setting the location of the text or box.

    - *xyann* -> set the location of the annotation
    - *xy* -> set where the arrow points to
    - *anncoords* -> set the units of the annotation location
    - *xycoords* -> set the units of the point location
    - ``set_position()`` -> `.Annotation` only set location of annotation

* `matplotlib.mlab.specgram`, `matplotlib.mlab.psd`,  `matplotlib.mlab.csd`,
  `matplotlib.mlab.cohere`, ``matplotlib.mlab.cohere_pairs``,
  `matplotlib.pyplot.specgram`, `matplotlib.pyplot.psd`,
  `matplotlib.pyplot.csd`, and `matplotlib.pyplot.cohere` now raise
  ValueError where they previously raised AssertionError.

* For `matplotlib.mlab.psd`,  `matplotlib.mlab.csd`,
  `matplotlib.mlab.cohere`, ``matplotlib.mlab.cohere_pairs``,
  `matplotlib.pyplot.specgram`, `matplotlib.pyplot.psd`,
  `matplotlib.pyplot.csd`, and `matplotlib.pyplot.cohere`, in cases
  where a shape (n, 1) array is returned, this is now converted to a (n, )
  array.  Previously, (n, m) arrays were averaged to an (n, ) array, but
  (n, 1) arrays were returned unchanged.  This change makes the dimensions
  consistent in both cases.

* Added the :rc:`axes.formatter.useoffset` to control the default value
  of *useOffset* in `.ticker.ScalarFormatter`

* Added `.Formatter` sub-class `.StrMethodFormatter` which
  does the exact same thing as `.FormatStrFormatter`, but for new-style
  formatting strings.

* Deprecated ``matplotlib.testing.image_util`` and the only function within,
  ``matplotlib.testing.image_util.autocontrast``. These will be removed
  completely in v1.5.0.

* The ``fmt`` argument of :meth:`~matplotlib.axes.Axes.plot_date` has been
  changed from ``bo`` to just ``o``, so color cycling can happen by default.

* Removed the class ``FigureManagerQTAgg`` and deprecated
  ``NavigationToolbar2QTAgg`` which will be removed in 1.5.

* Removed formerly public (non-prefixed) attributes ``rect`` and
  ``drawRect`` from ``FigureCanvasQTAgg``; they were always an
  implementation detail of the (preserved) ``drawRectangle()`` function.

* The function signatures of `.tight_bbox.adjust_bbox` and
  `.tight_bbox.process_figure_for_rasterizing` have been changed. A new
  *fixed_dpi* parameter allows for overriding the ``figure.dpi`` setting
  instead of trying to deduce the intended behaviour from the file format.

* Added support for horizontal/vertical axes padding to
  `mpl_toolkits.axes_grid1.axes_grid.ImageGrid` --- argument *axes_pad* can now
  be tuple-like if separate axis padding is required.
  The original behavior is preserved.

* Added support for skewed transforms to `matplotlib.transforms.Affine2D`,
  which can be created using the `~.Affine2D.skew` and `~.Affine2D.skew_deg`
  methods.

* Added clockwise parameter to control sectors direction in `.axes.Axes.pie`

* In `matplotlib.lines.Line2D` the *markevery* functionality has been extended.
  Previously an integer start-index and stride-length could be specified using
  either a two-element-list or a two-element-tuple.  Now this can only be done
  using a two-element-tuple.  If a two-element-list is used then it will be
  treated as NumPy fancy indexing and only the two markers corresponding to the
  given indexes will be shown.

* Removed *prop* keyword argument from
  `mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar` call.  It was
  passed through to the base-class ``__init__`` and is only used for setting
  padding.  Now *fontproperties* (which is what is really used to set the font
  properties of `.AnchoredSizeBar`) is passed through in place of *prop*.  If
  *fontproperties* is not passed in, but *prop* is, then *prop* is used in
  place of *fontproperties*.  If both are passed in, *prop* is silently
  ignored.


* The use of the index 0 in `.pyplot.subplot` and related commands is
  deprecated.  Due to a lack of validation, calling ``plt.subplots(2, 2, 0)``
  does not raise an exception, but puts an axes in the _last_
  position.  This is due to the indexing in subplot being 1-based (to
  mirror MATLAB) so before indexing into the `.GridSpec` object used to
  determine where the axes should go, 1 is subtracted off.  Passing in
  0 results in passing -1 to `.GridSpec` which results in getting the
  last position back.  Even though this behavior is clearly wrong and
  not intended, we are going through a deprecation cycle in an
  abundance of caution that any users are exploiting this 'feature'.
  The use of 0 as an index will raise a warning in 1.4 and an
  exception in 1.5.

* Clipping is now off by default on offset boxes.

* Matplotlib now uses a less-aggressive call to ``gc.collect(1)`` when
  closing figures to avoid major delays with large numbers of user objects
  in memory.

* The default clip value of *all* pie artists now defaults to ``False``.


Code removal
------------

* Removed ``mlab.levypdf``.  The code raised a NumPy error (and has for
  a long time) and was not the standard form of the Levy distribution.
  ``scipy.stats.levy`` should be used instead
