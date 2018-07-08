r"""
==============================
Overview of axes_grid1 toolkit
==============================

Controlling the layout of plots with the axes_grid toolkit.

.. _axes_grid1_users-guide-index:

What is axes_grid1 toolkit?
===========================

*axes_grid1* is a collection of helper classes to ease displaying
(multiple) images with matplotlib.  In matplotlib, the axes location
(and size) is specified in the normalized figure coordinates, which
may not be ideal for displaying images that needs to have a given
aspect ratio.  For example, it helps if you have a colorbar whose
height always matches that of the image.  `ImageGrid`_, `RGB Axes`_ and
`AxesDivider`_ are helper classes that deals with adjusting the
location of (multiple) Axes.  They provides a framework to adjust the
position of multiple axes at the drawing time. `ParasiteAxes`_
provides twinx(or twiny)-like features so that you can plot different
data (e.g., different y-scale) in a same Axes. `AnchoredArtists`_
includes custom artists which are placed at some anchored position,
like the legend.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_demo_axes_grid_001.png
   :target: ../../gallery/axes_grid1/demo_axes_grid.html
   :align: center
   :scale: 50

   Demo Axes Grid


axes_grid1
==========

ImageGrid
---------


A class that creates a grid of Axes. In matplotlib, the axes location
(and size) is specified in the normalized figure coordinates. This may
not be ideal for images that needs to be displayed with a given aspect
ratio.  For example, displaying images of a same size with some fixed
padding between them cannot be easily done in matplotlib. ImageGrid is
used in such case.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axesgrid_001.png
   :target: ../../gallery/axes_grid1/simple_axesgrid.html
   :align: center
   :scale: 50

   Simple Axesgrid

* The position of each axes is determined at the drawing time (see
  `AxesDivider`_), so that the size of the entire grid fits in the
  given rectangle (like the aspect of axes). Note that in this example,
  the paddings between axes are fixed even if you changes the figure
  size.

* axes in the same column has a same axes width (in figure
  coordinate), and similarly, axes in the same row has a same
  height. The widths (height) of the axes in the same row (column) are
  scaled according to their view limits (xlim or ylim).

  .. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axesgrid2_001.png
     :target: ../../gallery/axes_grid1/simple_axesgrid2.html
     :align: center
     :scale: 50

     Simple Axes Grid

* xaxis are shared among axes in a same column. Similarly, yaxis are
  shared among axes in a same row. Therefore, changing axis properties
  (view limits, tick location, etc. either by plot commands or using
  your mouse in interactive backends) of one axes will affect all
  other shared axes.



When initialized, ImageGrid creates given number (*ngrids* or *ncols* *
*nrows* if *ngrids* is None) of Axes instances. A sequence-like
interface is provided to access the individual Axes instances (e.g.,
grid[0] is the first Axes in the grid. See below for the order of
axes).



ImageGrid takes following arguments,


 ============= ========   ================================================
 Name          Default    Description
 ============= ========   ================================================
 fig
 rect
 nrows_ncols              number of rows and cols. e.g., (2,2)
 ngrids        None       number of grids. nrows x ncols if None
 direction     "row"      increasing direction of axes number. [row|column]
 axes_pad      0.02       pad between axes in inches
 add_all       True       Add axes to figures if True
 share_all     False      xaxis & yaxis of all axes are shared if True
 aspect        True       aspect of axes
 label_mode    "L"        location of tick labels thaw will be displayed.
                          "1" (only the lower left axes),
                          "L" (left most and bottom most axes),
                          or "all".
 cbar_mode     None       [None|single|each]
 cbar_location "right"    [right|top]
 cbar_pad      None       pad between image axes and colorbar axes
 cbar_size     "5%"       size of the colorbar
 axes_class    None
 ============= ========   ================================================

 *rect*
  specifies the location of the grid. You can either specify
  coordinates of the rectangle to be used (e.g., (0.1, 0.1, 0.8, 0.8)
  as in the Axes), or the subplot-like position (e.g., "121").

 *direction*
  means the increasing direction of the axes number.

 *aspect*
  By default (False), widths and heights of axes in the grid are
  scaled independently. If True, they are scaled according to their
  data limits (similar to aspect parameter in mpl).

 *share_all*
  if True, xaxis and yaxis of all axes are shared.

 *direction*
  direction of increasing axes number.  For "row",

   +---------+---------+
   | grid[0] | grid[1] |
   +---------+---------+
   | grid[2] | grid[3] |
   +---------+---------+

  For "column",

   +---------+---------+
   | grid[0] | grid[2] |
   +---------+---------+
   | grid[1] | grid[3] |
   +---------+---------+

You can also create a colorbar (or colorbars). You can have colorbar
for each axes (cbar_mode="each"), or you can have a single colorbar
for the grid (cbar_mode="single"). The colorbar can be placed on your
right, or top. The axes for each colorbar is stored as a *cbar_axes*
attribute.



The examples below show what you can do with ImageGrid.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_demo_axes_grid_001.png
   :target: ../../gallery/axes_grid1/demo_axes_grid.html
   :align: center
   :scale: 50

   Demo Axes Grid


AxesDivider Class
-----------------

Behind the scene, the ImageGrid class and the RGBAxes class utilize the
AxesDivider class, whose role is to calculate the location of the axes
at drawing time. While a more about the AxesDivider is (will be)
explained in (yet to be written) AxesDividerGuide, direct use of the
AxesDivider class will not be necessary for most users.  The
axes_divider module provides a helper function make_axes_locatable,
which can be useful. It takes a existing axes instance and create a
divider for it. ::

  ax = subplot(1,1,1)
  divider = make_axes_locatable(ax)




*make_axes_locatable* returns an instance of the AxesLocator class,
derived from the Locator. It provides *append_axes* method that
creates a new axes on the given side of ("top", "right", "bottom" and
"left") of the original axes.



colorbar whose height (or width) in sync with the master axes
-------------------------------------------------------------

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_colorbar_001.png
   :target: ../../gallery/axes_grid1/simple_colorbar.html
   :align: center
   :scale: 50

   Simple Colorbar




scatter_hist.py with AxesDivider
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The "scatter_hist.py" example in mpl can be rewritten using
*make_axes_locatable*. ::

    axScatter = subplot(111)
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
   :scale: 50

   Scatter Hist


The scatter_hist using the AxesDivider has some advantage over the
original scatter_hist.py in mpl. For example, you can set the aspect
ratio of the scatter plot, even with the x-axis or y-axis is shared
accordingly.


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
parasite axes.  To create a host axes, you may use *host_suplot* or
*host_axes* command.


Example 1. twinx
~~~~~~~~~~~~~~~~

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_parasite_simple_001.png
   :target: ../../gallery/axes_grid1/parasite_simple.html
   :align: center
   :scale: 50

   Parasite Simple

Example 2. twin
~~~~~~~~~~~~~~~

*twin* without a transform argument assumes that the parasite axes has the
same data transform as the host. This can be useful when you want the
top(or right)-axis to have different tick-locations, tick-labels, or
tick-formatter for bottom(or left)-axis. ::

  ax2 = ax.twin() # now, ax2 is responsible for "top" axis and "right" axis
  ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
  ax2.set_xticklabels(["0", r"$\frac{1}{2}\pi$",
                       r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])


.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axisline4_001.png
   :target: ../../gallery/axes_grid1/simple_axisline4.html
   :align: center
   :scale: 50

   Simple Axisline4



A more sophisticated example using twin. Note that if you change the
x-limit in the host axes, the x-limit of the parasite axes will change
accordingly.


.. figure:: ../../gallery/axes_grid1/images/sphx_glr_parasite_simple2_001.png
   :target: ../../gallery/axes_grid1/parasite_simple2.html
   :align: center
   :scale: 50

   Parasite Simple2


AnchoredArtists
---------------

It's a collection of artists whose location is anchored to the (axes)
bbox, like the legend. It is derived from *OffsetBox* in mpl, and
artist need to be drawn in the canvas coordinate. But, there is a
limited support for an arbitrary transform. For example, the ellipse
in the example below will have width and height in the data
coordinate.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_anchored_artists_001.png
   :target: ../../gallery/axes_grid1/simple_anchored_artists.html
   :align: center
   :scale: 50

   Simple Anchored Artists


InsetLocator
------------

:mod:`mpl_toolkits.axes_grid1.inset_locator` provides helper classes
and functions to place your (inset) axes at the anchored position of
the parent axes, similarly to AnchoredArtist.

Using :func:`mpl_toolkits.axes_grid1.inset_locator.inset_axes`, you
can have inset axes whose size is either fixed, or a fixed proportion
of the parent axes. For example,::

    inset_axes = inset_axes(parent_axes,
                            width="30%", # width = 30% of parent_bbox
                            height=1., # height : 1 inch
                            loc='lower left')

creates an inset axes whose width is 30% of the parent axes and whose
height is fixed at 1 inch.

You may creates your inset whose size is determined so that the data
scale of the inset axes to be that of the parent axes multiplied by
some factor. For example, ::

    inset_axes = zoomed_inset_axes(ax,
                                   0.5, # zoom = 0.5
                                   loc='upper right')

creates an inset axes whose data scale is half of the parent axes.
Here is complete examples.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_inset_locator_demo_001.png
   :target: ../../gallery/axes_grid1/inset_locator_demo.html
   :align: center
   :scale: 50

   Inset Locator Demo

For example, :func:`zoomed_inset_axes` can be used when you want the
inset represents the zoom-up of the small portion in the parent axes.
And :mod:`~mpl_toolkits/axes_grid/inset_locator` provides a helper
function :func:`mark_inset` to mark the location of the area
represented by the inset axes.

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_inset_locator_demo2_001.png
   :target: ../../gallery/axes_grid1/inset_locator_demo2.html
   :align: center
   :scale: 50

   Inset Locator Demo2


RGB Axes
~~~~~~~~

RGBAxes is a helper class to conveniently show RGB composite
images. Like ImageGrid, the location of axes are adjusted so that the
area occupied by them fits in a given rectangle.  Also, the xaxis and
yaxis of each axes are shared. ::

    from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

    fig = plt.figure(1)
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])

    r, g, b = get_rgb() # r,g,b are 2-d images
    ax.imshow_rgb(r, g, b,
                  origin="lower", interpolation="nearest")


.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_rgb_001.png
   :target: ../../gallery/axes_grid1/simple_rgb.html
   :align: center
   :scale: 50

   Simple Rgb


AxesDivider
===========

The axes_divider module provides helper classes to adjust the axes
positions of a set of images at drawing time.

* :mod:`~mpl_toolkits.axes_grid1.axes_size` provides a class of
  units that are used to determine the size of each axes. For example,
  you can specify a fixed size.

* :class:`~mpl_toolkits.axes_grid1.axes_size.Divider` is the class
  that calculates the axes position. It divides the given
  rectangular area into several areas. The divider is initialized by
  setting the lists of horizontal and vertical sizes on which the division
  will be based. Then use
  :meth:`~mpl_toolkits.axes_grid1.axes_size.Divider.new_locator`,
  which returns a callable object that can be used to set the
  axes_locator of the axes.


First, initialize the divider by specifying its grids, i.e.,
horizontal and vertical.

for example,::

    rect = [0.2, 0.2, 0.6, 0.6]
    horiz=[h0, h1, h2, h3]
    vert=[v0, v1, v2]
    divider = Divider(fig, rect, horiz, vert)

where, rect is a bounds of the box that will be divided and h0,..h3,
v0,..v2 need to be an instance of classes in the
:mod:`~mpl_toolkits.axes_grid1.axes_size`.  They have *get_size* method
that returns a tuple of two floats. The first float is the relative
size, and the second float is the absolute size. Consider a following
grid.

+-----+-----+-----+-----+
| v0  |     |     |     |
+-----+-----+-----+-----+
| v1  |     |     |     |
+-----+-----+-----+-----+
|h0,v2| h1  | h2  | h3  |
+-----+-----+-----+-----+


* v0 => 0, 2
* v1 => 2, 0
* v2 => 3, 0

The height of the bottom row is always 2 (axes_divider internally
assumes that the unit is inches). The first and the second rows have a
height ratio of 2:3. For example, if the total height of the grid is 6,
then the first and second row will each occupy 2/(2+3) and 3/(2+3) of
(6-1) inches. The widths of the horizontal columns will be similarly
determined. When the aspect ratio is set, the total height (or width) will
be adjusted accordingly.


The :mod:`mpl_toolkits.axes_grid1.axes_size` contains several classes
that can be used to set the horizontal and vertical configurations. For
example, for vertical configuration one could use::

  from mpl_toolkits.axes_grid1.axes_size import Fixed, Scaled
  vert = [Fixed(2), Scaled(2), Scaled(3)]

After you set up the divider object, then you create a locator
instance that will be given to the axes object.::

     locator = divider.new_locator(nx=0, ny=1)
     ax.set_axes_locator(locator)

The return value of the new_locator method is an instance of the
AxesLocator class. It is a callable object that returns the
location and size of the cell at the first column and the second row.
You may create a locator that spans over multiple cells.::

     locator = divider.new_locator(nx=0, nx=2, ny=1)

The above locator, when called, will return the position and size of
the cells spanning the first and second column and the first row. In
this example, it will return [0:2, 1].

See the example,

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axes_divider2_001.png
   :target: ../../gallery/axes_grid1/simple_axes_divider2.html
   :align: center
   :scale: 50

   Simple Axes Divider2

You can adjust the size of each axes according to its x or y
data limits (AxesX and AxesY).

.. figure:: ../../gallery/axes_grid1/images/sphx_glr_simple_axes_divider3_001.png
   :target: ../../gallery/axes_grid1/simple_axes_divider3.html
   :align: center
   :scale: 50

   Simple Axes Divider3
"""
