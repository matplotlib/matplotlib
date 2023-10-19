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
    autoscale the ``matplotlib.axes.Axes.viewLim`` based on
    the ``matplotlib.axes.Axes.dataLim``.

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
  ``matplotlib.Axes.set_default_color_cycle`` is deprecated.

* You can now print several figures to one pdf file and modify the
  document information dictionary of a pdf file. See the docstrings
  of the class :class:`matplotlib.backends.backend_pdf.PdfPages` for
  more information.

* Removed configobj_ and `enthought.traits`_ packages, which are only
  required by the experimental traited config and are somewhat out of
  date. If needed, install them independently.

.. _configobj: http://www.voidspace.org.uk/python/configobj.html
.. _`enthought.traits`: http://code.enthought.com/pages/traits.html

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
