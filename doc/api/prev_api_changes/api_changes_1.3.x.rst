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
  used.  In previous Matplotlib versions this attribute was an
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
  <https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html>`_.

* The `font.*` rcParams now affect only text objects created after the
  rcParam has been set, and will not retroactively affect already
  existing text objects.  This brings their behavior in line with most
  other rcParams.

* Removed call of :meth:`~matplotlib.axes.Axes.grid` in
  :meth:`~matplotlib.pyplot.plotfile`. To draw the axes grid, set the
  ``axes.grid`` rcParam to *True*, or explicitly call
  :meth:`~matplotlib.axes.Axes.grid`.
