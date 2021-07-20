

Changes for 0.98.0
==================

* :func:`matplotlib.image.imread` now no longer always returns RGBA data---if
  the image is luminance or RGB, it will return a MxN or MxNx3 array
  if possible.  Also uint8 is no longer always forced to float.

* Rewrote the :class:`matplotlib.cm.ScalarMappable` callback
  infrastructure to use :class:`matplotlib.cbook.CallbackRegistry`
  rather than custom callback handling.  Any users of
  ``matplotlib.cm.ScalarMappable.add_observer`` of the
  :class:`~matplotlib.cm.ScalarMappable` should use the
  :attr:`matplotlib.cm.ScalarMappable.callbacksSM`
  :class:`~matplotlib.cbook.CallbackRegistry` instead.

* New axes function and Axes method provide control over the plot
  color cycle: ``matplotlib.axes.set_default_color_cycle`` and
  ``matplotlib.axes.Axes.set_color_cycle``.

* Matplotlib now requires Python 2.4, so :mod:`matplotlib.cbook` will
  no longer provide :class:`set`, :func:`enumerate`, :func:`reversed`
  or ``izip`` compatibility functions.

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
bounds.  Call :meth:`matplotlib.ticker.TickHelper.create_dummy_axis` to
do so.

The functionality of ``Pbox`` has been merged with
:class:`~matplotlib.transforms.Bbox`.  Its methods now all return
copies rather than modifying in place.

The following lists many of the simple changes necessary to update
code from the old transformation framework to the new one.  In
particular, methods that return a copy are named with a verb in the
past tense, whereas methods that alter an object in place are named
with a verb in the present tense.

:mod:`matplotlib.transforms`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------------------------+------------------------------------------------------+
| Old method                                 | New method                                           |
+============================================+======================================================+
| ``Bbox.get_bounds``                        | :attr:`.transforms.Bbox.bounds`                      |
+--------------------------------------------+------------------------------------------------------+
| ``Bbox.width``                             | :attr:`transforms.Bbox.width                         |
|                                            | <.transforms.BboxBase.width>`                        |
+--------------------------------------------+------------------------------------------------------+
| ``Bbox.height``                            | :attr:`transforms.Bbox.height                        |
|                                            | <.transforms.BboxBase.height>`                       |
+--------------------------------------------+------------------------------------------------------+
| ``Bbox.intervalx().get_bounds()``          | :attr:`.transforms.Bbox.intervalx`                   |
| ``Bbox.intervalx().set_bounds()``          | [It is now a property.]                              |
+--------------------------------------------+------------------------------------------------------+
| ``Bbox.intervaly().get_bounds()``          | :attr:`.transforms.Bbox.intervaly`                   |
| ``Bbox.intervaly().set_bounds()``          | [It is now a property.]                              |
+--------------------------------------------+------------------------------------------------------+
| ``Bbox.xmin``                              | :attr:`.transforms.Bbox.x0` or                       |
|                                            | :attr:`transforms.Bbox.xmin                          |
|                                            | <.transforms.BboxBase.xmin>` [1]_                    |
+--------------------------------------------+------------------------------------------------------+
| ``Bbox.ymin``                              | :attr:`.transforms.Bbox.y0` or                       |
|                                            | :attr:`transforms.Bbox.ymin                          |
|                                            | <.transforms.BboxBase.ymin>` [1]_                    |
+--------------------------------------------+------------------------------------------------------+
| ``Bbox.xmax``                              | :attr:`.transforms.Bbox.x1` or                       |
|                                            | :attr:`transforms.Bbox.xmax                          |
|                                            | <.transforms.BboxBase.xmax>` [1]_                    |
+--------------------------------------------+------------------------------------------------------+
| ``Bbox.ymax``                              | :attr:`.transforms.Bbox.y1` or                       |
|                                            | :attr:`transforms.Bbox.ymax                          |
|                                            | <.transforms.BboxBase.ymax>` [1]_                    |
+--------------------------------------------+------------------------------------------------------+
| ``Bbox.overlaps(bboxes)``                  | `Bbox.count_overlaps(bboxes)                         |
|                                            | <.BboxBase.count_overlaps>`                          |
+--------------------------------------------+------------------------------------------------------+
| ``bbox_all(bboxes)``                       | `Bbox.union(bboxes) <.BboxBase.union>`               |
|                                            | [It is a staticmethod.]                              |
+--------------------------------------------+------------------------------------------------------+
| ``lbwh_to_bbox(l, b, w, h)``               | `Bbox.from_bounds(x0, y0, w, h) <.Bbox.from_bounds>` |
|                                            | [It is a staticmethod.]                              |
+--------------------------------------------+------------------------------------------------------+
| ``inverse_transform_bbox(trans, bbox)``    | ``bbox.inverse_transformed(trans)``                  |
|                                            |                                                      |
+--------------------------------------------+------------------------------------------------------+
| ``Interval.contains_open(v)``              | `interval_contains_open(tuple, v)                    |
|                                            | <.interval_contains_open>`                           |
+--------------------------------------------+------------------------------------------------------+
| ``Interval.contains(v)``                   | `interval_contains(tuple, v) <.interval_contains>`   |
+--------------------------------------------+------------------------------------------------------+
| ``identity_transform()``                   | :class:`.transforms.IdentityTransform`               |
+--------------------------------------------+------------------------------------------------------+
| ``blend_xy_sep_transform(xtrans, ytrans)`` | `blended_transform_factory(xtrans, ytrans)           |
|                                            | <.blended_transform_factory>`                        |
+--------------------------------------------+------------------------------------------------------+
| ``scale_transform(xs, ys)``                | `Affine2D().scale(xs[, ys]) <.Affine2D.scale>`       |
+--------------------------------------------+------------------------------------------------------+
| ``get_bbox_transform(boxin, boxout)``      | `BboxTransform(boxin, boxout) <.BboxTransform>` or   |
|                                            | `BboxTransformFrom(boxin) <.BboxTransformFrom>` or   |
|                                            | `BboxTransformTo(boxout) <.BboxTransformTo>`         |
+--------------------------------------------+------------------------------------------------------+
| ``Transform.seq_xy_tup(points)``           | `Transform.transform(points) <.Transform.transform>` |
+--------------------------------------------+------------------------------------------------------+
| ``Transform.inverse_xy_tup(points)``       | `Transform.inverted()                                |
|                                            | <.Transform.inverted>`.transform(points)             |
+--------------------------------------------+------------------------------------------------------+

.. [1] The :class:`~matplotlib.transforms.Bbox` is bound by the points
   (x0, y0) to (x1, y1) and there is no defined order to these points,
   that is, x0 is not necessarily the left edge of the box.  To get
   the left edge of the :class:`.Bbox`, use the read-only property
   :attr:`xmin <matplotlib.transforms.BboxBase.xmin>`.

:mod:`matplotlib.axes`
~~~~~~~~~~~~~~~~~~~~~~

============================= ==============================================
Old method                    New method
============================= ==============================================
``Axes.get_position()``       :meth:`matplotlib.axes.Axes.get_position` [2]_
----------------------------- ----------------------------------------------
``Axes.set_position()``       :meth:`matplotlib.axes.Axes.set_position` [3]_
----------------------------- ----------------------------------------------
``Axes.toggle_log_lineary()`` :meth:`matplotlib.axes.Axes.set_yscale` [4]_
----------------------------- ----------------------------------------------
``Subplot`` class             removed
============================= ==============================================

The ``Polar`` class has moved to :mod:`matplotlib.projections.polar`.

.. [2] :meth:`matplotlib.axes.Axes.get_position` used to return a list
   of points, now it returns a :class:`matplotlib.transforms.Bbox`
   instance.

.. [3] :meth:`matplotlib.axes.Axes.set_position` now accepts either
   four scalars or a :class:`matplotlib.transforms.Bbox` instance.

.. [4] Since the recfactoring allows for more than two scale types
   ('log' or 'linear'), it no longer makes sense to have a toggle.
   ``Axes.toggle_log_lineary()`` has been removed.

:mod:`matplotlib.artist`
~~~~~~~~~~~~~~~~~~~~~~~~

============================== ==============================================
Old method                     New method
============================== ==============================================
``Artist.set_clip_path(path)`` ``Artist.set_clip_path(path, transform)`` [5]_
============================== ==============================================

.. [5] :meth:`matplotlib.artist.Artist.set_clip_path` now accepts a
   :class:`matplotlib.path.Path` instance and a
   :class:`matplotlib.transforms.Transform` that will be applied to
   the path immediately before clipping.

:mod:`matplotlib.collections`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

=========== =================
Old method  New method
=========== =================
*linestyle* *linestyles* [6]_
=========== =================

.. [6] Linestyles are now treated like all other collection
   attributes, i.e.  a single value or multiple values may be
   provided.

:mod:`matplotlib.colors`
~~~~~~~~~~~~~~~~~~~~~~~~

================================== =====================================================
Old method                         New method
================================== =====================================================
``ColorConvertor.to_rgba_list(c)`` ``colors.to_rgba_array(c)``
                                   [:meth:`matplotlib.colors.to_rgba_array`
                                   returns an Nx4 NumPy array of RGBA color quadruples.]
================================== =====================================================

:mod:`matplotlib.contour`
~~~~~~~~~~~~~~~~~~~~~~~~~

===================== ===================================================
Old method            New method
===================== ===================================================
``Contour._segments`` ``matplotlib.contour.Contour.get_paths`` [Returns a
                      list of :class:`matplotlib.path.Path` instances.]
===================== ===================================================

:mod:`matplotlib.figure`
~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+--------------------------------------+
| Old method           | New method                           |
+======================+======================================+
| ``Figure.dpi.get()`` | :attr:`matplotlib.figure.Figure.dpi` |
| ``Figure.dpi.set()`` | *(a property)*                       |
+----------------------+--------------------------------------+

:mod:`matplotlib.patches`
~~~~~~~~~~~~~~~~~~~~~~~~~

===================== ====================================================
Old method            New method
===================== ====================================================
``Patch.get_verts()`` :meth:`matplotlib.patches.Patch.get_path` [Returns a
                      :class:`matplotlib.path.Path` instance]
===================== ====================================================

:mod:`matplotlib.backend_bases`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================================= ==========================================
Old method                                    New method
============================================= ==========================================
``GraphicsContext.set_clip_rectangle(tuple)`` `GraphicsContext.set_clip_rectangle(bbox)
                                              <.GraphicsContextBase.set_clip_rectangle>`
--------------------------------------------- ------------------------------------------
``GraphicsContext.get_clip_path()``           `GraphicsContext.get_clip_path()
                                              <.GraphicsContextBase.get_clip_path>` [7]_
--------------------------------------------- ------------------------------------------
``GraphicsContext.set_clip_path()``           `GraphicsContext.set_clip_path()
                                              <.GraphicsContextBase.set_clip_path>` [8]_
============================================= ==========================================

.. [7] :meth:`matplotlib.backend_bases.GraphicsContextBase.get_clip_path`
   returns a tuple of the form (*path*, *affine_transform*), where *path* is a
   :class:`matplotlib.path.Path` instance and *affine_transform* is a
   :class:`matplotlib.transforms.Affine2D` instance.

.. [8] :meth:`matplotlib.backend_bases.GraphicsContextBase.set_clip_path` now
   only accepts a :class:`matplotlib.transforms.TransformedPath` instance.

:class:`~matplotlib.backend_bases.RendererBase`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New methods:

  * :meth:`draw_path(self, gc, path, transform, rgbFace)
    <matplotlib.backend_bases.RendererBase.draw_path>`

  * :meth:`draw_markers(self, gc, marker_path, marker_trans, path,
    trans, rgbFace)
    <matplotlib.backend_bases.RendererBase.draw_markers>`

  * :meth:`draw_path_collection(self, master_transform, cliprect,
    clippath, clippath_trans, paths, all_transforms, offsets,
    offsetTrans, facecolors, edgecolors, linewidths, linestyles,
    antialiaseds)
    <matplotlib.backend_bases.RendererBase.draw_path_collection>`
    *[optional]*

Changed methods:

  * ``draw_image(self, x, y, im, bbox)`` is now
    :meth:`draw_image(self, x, y, im, bbox, clippath, clippath_trans)
    <matplotlib.backend_bases.RendererBase.draw_image>`

Removed methods:

  * ``draw_arc``

  * ``draw_line_collection``

  * ``draw_line``

  * ``draw_lines``

  * ``draw_point``

  * ``draw_quad_mesh``

  * ``draw_poly_collection``

  * ``draw_polygon``

  * ``draw_rectangle``

  * ``draw_regpoly_collection``
