r"""
.. _axes_grid1_users-guide-index:

==========================================
Overview of :mod:`mpl_toolkits.axes_grid1`
==========================================

:mod:`.axes_grid1` provides the following features:

- Helper classes (ImageGrid_, RGBAxes_, AxesDivider_) to ease the layout of
  axes displaying images with a fixed aspect ratio while satisfying additional
  constraints (matching the heights of a colorbar and an image, or fixing the
  padding between images);
- ParasiteAxes_ (twinx/twiny-like features so that you can plot different data
  (e.g., different y-scale) in a same Axes);
- AnchoredArtists_ (custom artists which are placed at an anchored position,
  similarly to legends).

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_demo_axes_grid_001.png
   :target: ../../gallery/axes_grid1/demo_axes_grid.html
   :align: center

axes_grid1
==========

ImageGrid
---------

In Matplotlib, axes location and size are usually specified in normalized
figure coordinates (0 = bottom left, 1 = top right), which makes
it difficult to achieve a fixed (absolute) padding between images.
`~.axes_grid1.axes_grid.ImageGrid` can be used to achieve such a padding; see
its docs for detailed API information.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axesgrid_001.png
   :target: ../../gallery/axes_grid1/simple_axesgrid.html
   :align: center

* The position of each axes is determined at the drawing time (see
  AxesDivider_), so that the size of the entire grid fits in the
  given rectangle (like the aspect of axes). Note that in this example,
  the paddings between axes are fixed even if you changes the figure
  size.

* Axes in the same column share their x-axis, and axes in the same row share
  their y-axis (in the sense of `~.Axes.sharex`, `~.Axes.sharey`).
  Additionally, Axes in the same column all have the same width, and axes in
  the same row all have the same height.  These widths and heights are scaled
  in proportion to the axes' view limits (xlim or ylim).

  .. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axesgrid2_001.png
     :target: ../../gallery/axes_grid1/simple_axesgrid2.html
     :align: center

The examples below show what you can do with ImageGrid.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_demo_axes_grid_001.png
   :target: ../../gallery/axes_grid1/demo_axes_grid.html
   :align: center

AxesDivider Class
-----------------

Behind the scenes, ImageGrid (and RGBAxes, described below) rely on
`~.axes_grid1.axes_divider.AxesDivider`, whose role is to calculate the
location of the axes at drawing time.

Users typically do not need to directly instantiate dividers
by calling `~.axes_grid1.axes_divider.AxesDivider`; instead,
`~.axes_grid1.axes_divider.make_axes_locatable` can be used to create a divider
for an axes::

  ax = subplot(1, 1, 1)
  divider = make_axes_locatable(ax)

`.AxesDivider.append_axes` can then be used to create a new axes on a given
side ("left", "right", "top", "bottom") of the original axes.

colorbar whose height (or width) in sync with the master axes
-------------------------------------------------------------

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_colorbar_001.png
   :target: ../../gallery/axes_grid1/simple_colorbar.html
   :align: center

scatter_hist.py with AxesDivider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :doc:`/gallery/lines_bars_and_markers/scatter_hist` example can be
rewritten using `~.axes_grid1.axes_divider.make_axes_locatable`::

    axScatter = plt.subplot()
    axScatter.scatter(x, y)
    axScatter.set_aspect(1.)

    # create new axes on the right and on the top of the current axes.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", size=1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", size=1.2, pad=0.1, sharey=axScatter)

    # the scatter plot:
    # histograms
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

See the full source code below.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_scatter_hist_locatable_axes_001.png
   :target: ../../gallery/axes_grid1/scatter_hist_locatable_axes.html
   :align: center

The :doc:`/gallery/axes_grid1/scatter_hist_locatable_axes` using the
AxesDivider has some advantages over the
original :doc:`/gallery/lines_bars_and_markers/scatter_hist` in Matplotlib.
For example, you can set the aspect ratio of the scatter plot, even with the
x-axis or y-axis is shared accordingly.

ParasiteAxes
------------

The ParasiteAxes is an axes whose location is identical to its host
axes. The location is adjusted in the drawing time, thus it works even
if the host change its location (e.g., images).

In most cases, you first create a host axes, which provides a few
method that can be used to create parasite axes. They are *twinx*,
*twiny* (which are similar to twinx and twiny in the matplotlib) and
*twin*. *twin* takes an arbitrary transformation that maps between the
data coordinates of the host axes and the parasite axes.  *draw*
method of the parasite axes are never called. Instead, host axes
collects artists in parasite axes and draw them as if they belong to
the host axes, i.e., artists in parasite axes are merged to those of
the host axes and then drawn according to their zorder.  The host and
parasite axes modifies some of the axes behavior. For example, color
cycle for plot lines are shared between host and parasites. Also, the
legend command in host, creates a legend that includes lines in the
parasite axes.  To create a host axes, you may use *host_subplot* or
*host_axes* command.

Example 1. twinx
~~~~~~~~~~~~~~~~

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_parasite_simple_001.png
   :target: ../../gallery/axes_grid1/parasite_simple.html
   :align: center

Example 2. twin
~~~~~~~~~~~~~~~

*twin* without a transform argument assumes that the parasite axes has the
same data transform as the host. This can be useful when you want the
top(or right)-axis to have different tick-locations, tick-labels, or
tick-formatter for bottom(or left)-axis. ::

  ax2 = ax.twin() # now, ax2 is responsible for "top" axis and "right" axis
  ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi],
                 labels=["0", r"$\frac{1}{2}\pi$",
                         r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axisline4_001.png
   :target: ../../gallery/axes_grid1/simple_axisline4.html
   :align: center

A more sophisticated example using twin. Note that if you change the
x-limit in the host axes, the x-limit of the parasite axes will change
accordingly.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_parasite_simple2_001.png
   :target: ../../gallery/axes_grid1/parasite_simple2.html
   :align: center

AnchoredArtists
---------------

:mod:`.axes_grid1.anchored_artists` is a collection of artists whose location
is anchored to the (axes) bbox, similarly to legends.  These artists derive
from `.offsetbox.OffsetBox`, and the artist need to be drawn in canvas
coordinates.  There is limited support for arbitrary transforms.  For example,
the ellipse in the example below will have width and height in data coordinate.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_anchored_artists_001.png
   :target: ../../gallery/axes_grid1/simple_anchored_artists.html
   :align: center

InsetLocator
------------

.. seealso::
   `.Axes.inset_axes` and `.Axes.indicate_inset_zoom` in the main library.

:mod:`.axes_grid1.inset_locator` provides helper classes and functions to
place inset axes at an anchored position of the parent axes, similarly to
AnchoredArtist.

`.inset_locator.inset_axes` creates an inset axes whose size is either fixed,
or a fixed proportion of the parent axes::

    inset_axes = inset_axes(parent_axes,
                            width="30%",  # width = 30% of parent_bbox
                            height=1.,  # height = 1 inch
                            loc='lower left')

creates an inset axes whose width is 30% of the parent axes and whose
height is fixed at 1 inch.

`.inset_locator.zoomed_inset_axes` creates an inset axes whose data scale is
that of the parent axes multiplied by some factor, e.g. ::

    inset_axes = zoomed_inset_axes(ax,
                                   0.5,  # zoom = 0.5
                                   loc='upper right')

creates an inset axes whose data scale is half of the parent axes.  This can be
useful to mark the zoomed area on the parent axes:

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_inset_locator_demo_001.png
   :target: ../../gallery/axes_grid1/inset_locator_demo.html
   :align: center

`.inset_locator.mark_inset` allows marking the location of the area represented
by the inset axes:

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_inset_locator_demo2_001.png
   :target: ../../gallery/axes_grid1/inset_locator_demo2.html
   :align: center

RGBAxes
-------

RGBAxes is a helper class to conveniently show RGB composite
images. Like ImageGrid, the location of axes are adjusted so that the
area occupied by them fits in a given rectangle.  Also, the xaxis and
yaxis of each axes are shared. ::

    from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

    fig = plt.figure()
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8], pad=0.0)
    r, g, b = get_rgb()  # r, g, b are 2D images.
    ax.imshow_rgb(r, g, b)

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_demo_axes_rgb_001.png
   :target: ../../gallery/axes_grid1/demo_axes_rgb.html
   :align: center

AxesDivider
===========

The :mod:`mpl_toolkits.axes_grid1.axes_divider` module provides helper classes
to adjust the axes positions of a set of images at drawing time.

* :mod:`~mpl_toolkits.axes_grid1.axes_size` provides a class of
  units that are used to determine the size of each axes. For example,
  you can specify a fixed size.

* `~mpl_toolkits.axes_grid1.axes_divider.Divider` is the class that
  calculates the axes position. It divides the given rectangular area into
  several areas. The divider is initialized by setting the lists of horizontal
  and vertical sizes on which the division will be based. Then use
  :meth:`~mpl_toolkits.axes_grid1.axes_divider.Divider.new_locator`, which
  returns a callable object that can be used to set the axes_locator of the
  axes.

Here, we demonstrate how to achieve the following layout: we want to position
axes in a 3x4 grid (note that `.Divider` makes row indices start from the
*bottom*\(!) of the grid):

.. code-block:: none

   +--------+--------+--------+--------+
   | (2, 0) | (2, 1) | (2, 2) | (2, 3) |
   +--------+--------+--------+--------+
   | (1, 0) | (1, 1) | (1, 2) | (1, 3) |
   +--------+--------+--------+--------+
   | (0, 0) | (0, 1) | (0, 2) | (0, 3) |
   +--------+--------+--------+--------+

such that the bottom row has a fixed height of 2 (inches) and the top two rows
have a height ratio of 2 (middle) to 3 (top).  (For example, if the grid has
a size of 7 inches, the bottom row will be 2 inches, the middle row also 2
inches, and the top row 3 inches.)

These constraints are specified using classes from the
:mod:`~mpl_toolkits.axes_grid1.axes_size` module, namely::

    from mpl_toolkits.axes_grid1.axes_size import Fixed, Scaled
    vert = [Fixed(2), Scaled(2), Scaled(3)]

(More generally, :mod:`~mpl_toolkits.axes_grid1.axes_size` classes define a
``get_size(renderer)`` method that returns a pair of floats -- a relative size,
and an absolute size.  ``Fixed(2).get_size(renderer)`` returns ``(0, 2)``;
``Scaled(2).get_size(renderer)`` returns ``(2, 0)``.)

We use these constraints to initialize a `.Divider` object::

    rect = [0.2, 0.2, 0.6, 0.6]  # Position of the grid in the figure.
    vert = [Fixed(2), Scaled(2), Scaled(3)]  # As above.
    horiz = [...]  # Some other horizontal constraints.
    divider = Divider(fig, rect, horiz, vert)

then use `.Divider.new_locator` to create an `.AxesLocator` instance for a
given grid entry::

    locator = divider.new_locator(nx=0, ny=1)  # Grid entry (1, 0).

and make it responsible for locating the axes::

    ax.set_axes_locator(locator)

The `.AxesLocator` is a callable object that returns the location and size of
the cell at the first column and the second row.

Locators that spans over multiple cells can be created with, e.g.::

    # Columns #0 and #1 ("0-2 range"), row #1.
    locator = divider.new_locator(nx=0, nx1=2, ny=1)

See the example,

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axes_divider1_001.png
   :target: ../../gallery/axes_grid1/simple_axes_divider1.html
   :align: center

You can also adjust the size of each axes according to its x or y
data limits (AxesX and AxesY).

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axes_divider3_001.png
   :target: ../../gallery/axes_grid1/simple_axes_divider3.html
   :align: center
"""
