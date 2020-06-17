
What's new in Matplotlib 3.3.0
==============================

For a list of all of the issues and pull requests since the last
revision, see the :ref:`github-stats`.

.. contents:: Table of Contents
   :depth: 4

.. toctree::
   :maxdepth: 4


New `~.axes.Axes.axline` method
-------------------------------

A new `~.axes.Axes.axline` method has been added to draw infinitely long lines
that pass through two points.



.. plot::
   :include-source: True

   fig, ax = plt.subplots()

   ax.axline((.1, .1), slope=5, color='C0', label='by slope')
   ax.axline((.1, .2), (.8, .7), color='C3', label='by points')

   ax.legend()


Allow tick formatters to be set with str or function inputs
-----------------------------------------------------------

`~.Axis.set_major_formatter` and `~.Axis.set_minor_formatter`
now accept `str` or function inputs in addition to `~.ticker.Formatter`
instances. For a `str` a `~.ticker.StrMethodFormatter` is automatically
generated and used. For a function a `~.ticker.FuncFormatter` is automatically
generated and used.



Text color for legend labels
----------------------------

The text color of legend labels can now be set by passing a parameter
``labelcolor`` to `~.axes.Axes.legend`. The ``labelcolor`` keyword can be:

* A single color (either a string or RGBA tuple), which adjusts the text color
  of all the labels.
* A list or tuple, allowing the text color of each label to be set
  individually.
* ``linecolor``, which sets the text color of each label to match the
  corresponding line color.
* ``markerfacecolor``, which sets the text color of each label to match the
  corresponding marker face color.
* ``markeredgecolor``,  which sets the text color of each label to match the
  corresponding marker edge color.



Provisional API for composing semantic axes layouts from text or nested lists
-----------------------------------------------------------------------------

The `.Figure` class has a provisional method to generate complex grids
of named `.axes.Axes` based on nested list input or ASCII art:

.. plot::
   :include-source: True

   axd = plt.figure(constrained_layout=True).subplot_mosaic(
     [["Top", "Top", "Edge"],
      ["Left", ".",  "Edge"]]
   )
   for k, ax in axd.items():
       ax.text(0.5, 0.5, k,
               ha='center', va='center', fontsize=36,
               color='darkgrey')

or as a string (with single-character Axes labels):

.. plot::
   :include-source: True

   axd = plt.figure(constrained_layout=True).subplot_mosaic(
   """
   TTE
   L.E
   """)
   for k, ax in axd.items():
       ax.text(0.5, 0.5, k,
               ha='center', va='center', fontsize=36,
               color='darkgrey')



See :ref:`sphx_glr_tutorials_provisional_mosaic.py` for more
details and examples.


Setting axes box aspect
-----------------------

It is now possible to set the aspect of an axes box directly via
`~.Axes.set_box_aspect`. The box aspect is the ratio between axes height
and axes width in physical units, independent of the data limits.
This is useful to e.g. produce a square plot, independent of the data it
contains, or to have a usual plot with the same axes dimensions next to
an image plot with fixed (data-)aspect.

For use cases check out the :doc:`Axes box aspect
</gallery/subplots_axes_and_figures/axes_box_aspect>` example.


`.Axes.sharex`, `.Axes.sharey`
------------------------------

These new methods allow sharing axes *immediately* after creating them.  For
example, they can be used to selectively link some axes created all together
using `~.Figure.subplots`.

Note that they may *not* be used to share axes after any operation (e.g.,
drawing) has occurred on them.


Align labels to Axes edges
--------------------------

`~.axes.Axes.set_xlabel`, `~.axes.Axes.set_ylabel` and `.ColorbarBase.set_label`
support a parameter ``loc`` for simplified positioning. Supported values are
'left', 'center', or 'right'. The default is controlled via
:rc:`xaxis.labelposition` and :rc:`yaxis.labelposition`; the Colorbar label
takes the rcParam based on its orientation.


New "extend" keyword to colors.BoundaryNorm
-------------------------------------------

`~.colors.BoundaryNorm` now has an *extend* keyword argument, analogous to
*extend* in `~.axes.Axes.contourf`. When set to 'both', 'min', or 'max', it
maps the corresponding out-of-range values to `~.colors.Colormap` lookup-table
indices near the appropriate ends of their range so that the colors for out-of
range values are adjacent to, but distinct from, their in-range neighbors. The
colorbar inherits the *extend* argument from the norm, so with
``extend='both'``, for example, the colorbar will have triangular extensions
for out-of-range values with colors that differ from adjacent in-range colors.

  .. plot::

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import numpy as np

    # Make the data
    dx, dy = 0.05, 0.05
    y, x = np.mgrid[slice(1, 5 + dy, dy),
                    slice(1, 5 + dx, dx)]
    z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
    z = z[:-1, :-1]

    # Z roughly varies between -1 and +1.
    # Color boundary levels range from -0.8 to 0.8, so there are out-of-bounds
    # areas.
    levels = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
    cmap = plt.get_cmap('PiYG')

    fig, axs = plt.subplots(nrows=2, constrained_layout=True, sharex=True)

    # Before this change:
    norm = BoundaryNorm(levels, ncolors=cmap.N)
    im = axs[0].pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=axs[0], extend='both')
    axs[0].axis([x.min(), x.max(), y.min(), y.max()])
    axs[0].set_title("Colorbar with extend='both'")

    # With the new keyword:
    norm = BoundaryNorm(levels, ncolors=cmap.N, extend='both')
    im = axs[1].pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=axs[1])  # note that the colorbar is updated accordingly
    axs[1].axis([x.min(), x.max(), y.min(), y.max()])
    axs[1].set_title("BoundaryNorm with extend='both'")

    plt.show()


``GridSpec.subplots()``
-----------------------

The `.GridSpec` class gained a `~.GridSpecBase.subplots` method, so that one
can write ::

    fig.add_gridspec(2, 2, height_ratios=[3, 1]).subplots()

as an alternative to ::

    fig.subplots(2, 2, gridspec_kw={"height_ratios": [3, 1]})


`matplotlib.rc_context` is now a `contextlib.contextmanager`
------------------------------------------------------------

`matplotlib.rc_context` can now be used as a decorator (technically, it is now
implemented as a `contextlib.contextmanager`), e.g. ::

    @rc_context({"lines.linewidth": 2})
    def some_function(...):
        ...

rcParams for controlling default "raise window" behavior
--------------------------------------------------------
The new config option :rc:`figure.raise_window` allows to disable
raising the plot window when calling `~.pyplot.show` or `~.pyplot.pause`.
``MacOSX`` backend is currently not supported.



``imshow`` now coerces 3D arrays with depth 1 to 2D
---------------------------------------------------

Starting from this version arrays of size MxNx1 will be coerced into MxN
for displaying. This means commands like ``plt.imshow(np.random.rand(3, 3, 1))``
will no longer return an error message that the image shape is invalid.


``Axes3D`` no longer distorts the 3D plot to match the 2D aspect ratio
----------------------------------------------------------------------

Plots made with :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D` were previously
stretched to fit a square bounding box. As this stretching was done after
the projection from 3D to 2D, it resulted in distorted images if non-square
bounding boxes were used.  As of 3.3, this no longer occurs.

Currently, modes of setting the aspect (via
`~mpl_toolkits.mplot3d.axes3d.Axes3D.set_aspect`) in data space are
not supported for Axes3D but may be in the future.  If you want to
simulate having equal aspect in data space, set the ratio of your data
limits to match the value of `~.get_box_aspect`.  To control these
ratios use the `~mpl_toolkits.mplot3d.axes3d.Axes3D.set_box_aspect`
method which accepts the ratios as a 3-tuple of X:Y:Z.  The default
aspect ratio is 4:4:3.


3D axes now support minor ticks
-------------------------------

Home/Forward/Backward buttons now work with 3D axes
---------------------------------------------------



``savefig()`` gained a *backend* keyword argument
-------------------------------------------------

The *backend* keyword argument to ``savefig`` can now be used to pick the
rendering backend without having to globally set the backend; e.g. one can save
PDFs using the pgf backend with ``savefig("file.pdf", backend="pgf")``.


Offset text is now set to the top when using ``axis.tick_top()``
----------------------------------------------------------------

Solves the issue that the power indicator (e.g. 1e4) stayed on the bottom, even if the ticks were on the top.


Pcolor and Pcolormesh now accept ``shading='nearest'`` and ``'auto'``
---------------------------------------------------------------------

Previously `.axes.Axes.pcolor` and  `.axes.Axes.pcolormesh` handled
the situation where *x* and *y* have the same (respective) size as *C* by
dropping the last row and column of *C*, and *x* and *y* are regarded as the
edges of the remaining rows and columns in *C*.  However, many users want
*x* and *y* centered on the rows and columns of *C*.

To accommodate this, ``shading='nearest'`` and ``shading='auto'`` are new
allowed strings for the *shading* keyword argument. ``'nearest'`` will center
the color on *x* and *y* if *x* and *y* have the same dimensions as *C*
(otherwise an error will be thrown). ``shading='auto'`` will choose 'flat' or
'nearest' based on the size of *X*, *Y*, *C*.

If ``shading='flat'`` then *X*, and *Y* should have dimensions one larger
than *C*.  If *X* and *Y* have the same dimensions as *C*, then the previous
behavior is used and the last row and column of *C* are dropped, and a
DeprecationWarning is emitted.

Users can also specify this by the new :rc:`pcolor.shading` in their
``.matplotlibrc`` or via `.rcParams`.

See :doc:`pcolormesh </gallery/images_contours_and_fields/pcolormesh_grids>`
for examples.



Set zorder of contour labels
----------------------------

`~.axes.Axes.clabel` now accepts a *zorder* keyword argument making it easier
to set the *zorder* of contour labels. If not specified, the default *zorder*
of clabels used to always be 3 (i.e. the default *zorder* of `~.text.Text`)
irrespective of the *zorder* passed to
`~.axes.Axes.contour`/`~.axes.Axes.contourf`.  The new default *zorder* for
clabels has been changed to (``2 + zorder`` passed to `~.axes.Axes.contour` /
`~.axes.Axes.contourf`).


Simple syntax to select fonts by absolute path
----------------------------------------------

Fonts can now be selected by passing an absolute `pathlib.Path` to the *font*
keyword argument of `.Text`.


Add generalized "mathtext.fallback" rcParam
-------------------------------------------

New  :rc:`mathtext.fallback` rcParam. Takes "cm", "stix", "stixsans"
or "none" to turn fallback off. The rcParam *mathtext.fallback_to_cm* is
deprecated, but if used, will override new fallback.


Lines now accept ``MarkerStyle`` instances as input
---------------------------------------------------

Similar to `~.Axes.scatter`, `~.Axes.plot` and `~.lines.Line2D` now accept
`~.markers.MarkerStyle` instances as input for the *marker* parameter::

    plt.plot(..., marker=matplotlib.markers.MarkerStyle("D"))


Cursor text now uses a number of significant digits matching pointing precision
-------------------------------------------------------------------------------

Previously, the x/y position displayed by the cursor text would usually include
far more significant digits than the mouse pointing precision (typically one
pixel).  This is now fixed for linear scales.


Qt zoom rectangle now black and white
-------------------------------------

This makes it visible even over a dark background.


Functions to compute a Path's size
----------------------------------

Various functions were added to `~.bezier.BezierSegment` and `~.path.Path` to
allow computation of the shape/size of a `~.path.Path` and its composite Bezier
curves.

In addition to the fixes below, `~.bezier.BezierSegment` has gained more
documentation and usability improvements, including properties that contain its
dimension, degree, control_points, and more.

Better interface for Path segment iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`~.path.Path.iter_bezier` iterates through the `~.bezier.BezierSegment`'s that
make up the Path. This is much more useful typically than the existing
`~.path.Path.iter_segments` function, which returns the absolute minimum amount
of information possible to reconstruct the Path.

Fixed bug that computed a Path's Bbox incorrectly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Historically, `~.path.Path.get_extents` has always simply returned the Bbox of
a curve's control points, instead of the Bbox of the curve itself. While this is
a correct upper bound for the path's extents, it can differ dramatically from
the Path's actual extents for non-linear Bezier curves.


Dates now use a modern epoch
----------------------------

Matplotlib converts dates to days since an epoch using `.dates.date2num` (via
`matplotlib.units`).  Previously, an epoch of ``0000-12-31T00:00:00`` was used
so that ``0001-01-01`` was converted to 1.0.  An epoch so distant in the
past meant that a modern date was not able to preserve microseconds because
2000 years times the 2^(-52) resolution of a 64-bit float gives 14
microseconds.

Here we change the default epoch to the more reasonable UNIX default of
``1970-01-01T00:00:00`` which for a modern date has 0.35 microsecond
resolution.  (Finer resolution is not possible because we rely on
`datetime.datetime` for the date locators). Access to the epoch is provided
by `~.dates.get_epoch`, and there is a new :rc:`date.epoch` rcParam.  The user
may also call `~.dates.set_epoch`, but it must be set *before* any date
conversion or plotting is used.

If you have data stored as ordinal floats in the old epoch, a simple
conversion (using the new epoch) is::

    new_ordinal = old_ordinal + mdates.date2num(np.datetime64('0000-12-31'))




`~.axes.Axes.set_title` gains a *y* keyword argument to control auto positioning
--------------------------------------------------------------------------------

`~.axes.Axes.set_title` tries to auto-position the title to avoid any
decorators on the top x-axis.  This is not always desirable so now
*y* is an explicit keyword argument of `~.axes.Axes.set_title`.  It
defaults to *None* which means to use auto-positioning.  If a value is
supplied (i.e. the pre-3.0 default was ``y=1.0``) then auto-positioning is
turned off.  This can also be set with the new rcParameter :rc:`axes.titley`.


tight_layout now supports suptitle
----------------------------------


Add :rc:`contour.linewidth` to rcParams
---------------------------------------

The new config option :rc:`contour.linewidth` allows to control the default
line width of contours as a float. When set to ``None``, the line widths fall
back to :rc:`lines.linewidth`. The config value is overridden as usual by the
*linewidths* argument passed to `~.axes.Axes.contour` when it is not set to
``None``.


The SVG backend can now render hatches with transparency
--------------------------------------------------------

The SVG backend now respects the hatch stroke alpha. Useful applications are,
among others, semi-transparent hatches as a subtle way to differentiate columns
in bar plots.


Saving SVG now supports adding metadata
---------------------------------------

When saving SVG files, metadata can now be passed which will be saved in the
file using `Dublin Core`_ and `RDF`_. A list of valid metadata can be found in
the documentation for `.FigureCanvasSVG.print_svg`.

.. _Dublin Core: https://www.dublincore.org/specifications/dublin-core/
.. _RDF: https://www.w3.org/1999/.status/PR-rdf-syntax-19990105/status

Saving PDF metadata via PGF now consistent with PDF backend
-----------------------------------------------------------

When saving PDF files using the PGF backend, passed metadata will be
interpreted in the same way as with the PDF backend.  Previously, this metadata
was only accepted by the PGF backend when saving a multi-page PDF with
`.backend_pgf.PdfPages`, but is now allowed when saving a single figure, as
well.

`.backend_bases.key_press_handler` and `.backend_bases.button_press_handler` simplifications
--------------------------------------------------------------------------------------------

These event handlers can now be directly connected to a canvas with
``canvas.mpl_connect("key_press_event", key_press_handler)`` and
``canvas.mpl_connect("button_press_event", button_press_handler)``, rather than
having to write wrapper functions that fill in the (now optional) *canvas* and
*toolbar* parameters.
