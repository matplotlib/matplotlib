========
Overview
========

The matplotlib AxesGrid toolkit is a collection of helper classes,
mainly to ease displaying (multiple) images in matplotlib.

.. contents::
   :depth: 1
   :local:

`AxesGrid`_, `RGB Axes`_ and `AxesDivider`_ are helper classes that
deals with adjusting the location of (multiple) Axes, mainly for
displaying images.  It provides a framework to adjust the position of
multiple axes at the drawing time.  `ParasiteAxes`_ provides twinx(or
twiny)-like features so that you can plot different data (e.g.,
different y-scale) in a same Axes. `AxisLine`_ is a custom Axes
class. Unlike default Axes in matpotlib, each axis (left, right, top
and bottom) is associated with a separate artist (which is resposible
to draw axis-line, ticks, ticklabels, label). `AnchoredArtists`_
includes custom artists which are placed at some anchored position,
like the legend.




AxesGrid
========


A class that creates a grid of Axes. In matplotlib, the axes location
(and size) is specified in the normalized figure coordinates. This may
not be ideal for images that needs to be displayed with a given aspect
ratio.  For example, displaying images of a same size with some fixed
padding between them cannot be easily done in matplotlib. AxesGrid is
used in such case.

.. plot:: mpl_toolkits/axes_grid/figures/simple_axesgrid.py
   :include-source:

* The postion of each axes is determined at the drawing time (see
  `AxesDivider`_), so that the size of the entire grid fits in the
  given rectangle (like the aspec of axes). Note that in this example,
  the paddings between axes are fixed even if you changes the figure
  size.

* axes in the same column has a same axes width (in figure
  coordinate), and similarly, axes in the same row has a same
  height. The widths (height) of the axes in the same row (column) are
  scaled according to their view limits (xlim or ylim).

  .. plot:: mpl_toolkits/axes_grid/figures/simple_axesgrid2.py
     :include-source:

* xaxis are shared among axes in a same column. Similarly, yaxis are
  shared among axes in a same row. Therefore, changing axis properties
  (view limits, tick location, etc. either by plot commands or using
  your mouse in interactive backends) of one axes will affect all
  other shared axes.



When initialized, AxesGrid creates given number (*ngrids* or *ncols* *
*nrows* if *ngrids* is None) of Axes instances. A sequence-like
interface is provided to access the individual Axes instances (e.g.,
grid[0] is the first Axes in the grid. See below for the order of
axes).



AxesGrid takes following arguments,


 ============= ========   ================================================
 Name          Default    Description
 ============= ========   ================================================
 fig
 rect
 nrows_ncols              number of rows and cols. e.g. (2,2)
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
  By default (False), widths and heigths of axes in the grid are
  scaled independently. If True, they are scaled according to their
  data limits (similar to aspect parameter in mpl).

 *share_all*
  if True, xaxis  and yaxis of all axes are shared.

 *direction*
  direction of increasing axes number.   For "row",

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

You can also create a colorbar (or colobars). You can have colorbar
for each axes (cbar_mode="each"), or you can have a single colorbar
for the grid (cbar_mode="single"). The colorbar can be placed on your
right, or top. The axes for each colorbar is stored as a *cbar_axes*
attribute.



The examples below show what you can do with AxesGrid.

.. plot:: mpl_toolkits/axes_grid/figures/demo_axes_grid.py


RGB Axes
========

RGBAxes is a helper clase to conveniently show RGB composite
images. Like AxesGrid, the location of axes are adjusted so that the
area occupied by them fits in a given rectangle.  Also, the xaxis and
yaxis of each axes are shared. ::

    from mpl_toolkits.axes_grid.axes_rgb import RGBAxes

    fig = plt.figure(1)
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])

    r, g, b = get_rgb() # r,g,b are 2-d images
    ax.imshow_rgb(r, g, b,
                  origin="lower", interpolation="nearest")


.. plot:: mpl_toolkits/axes_grid/figures/simple_rgb.py



AxesDivider
===========

Behind the scene, the AxesGrid class and the RGBAxes class utilize the
AxesDivider class, whose role is to calculate the location of the axes
at drawing time. While a more about the AxesDivider is (will be)
explained in (yet to be written) AxesDividerGuide, direct use of the
AxesDivider class will not be necessary for most users.  The
axes_divider module provides a helper function make_axes_locatable,
which can be useful. It takes a exisitng axes instance and create a
divider for it. ::

	ax = subplot(1,1,1)
	divider = make_axes_locatable(ax)




*make_axes_locatable* returns an isntance of the AxesLocator class,
derived from the Locator. It has *new_vertical*, and *new_horizontal*
methods. The *new_vertical* (*new_horizontal*) creates a new axes on
the upper (right) side of the original axes.


scatter_hist.py with AxesDivider
--------------------------------

The "scatter_hist.py" example in mpl can be rewritten using
*make_axes_locatable*. ::

    from mpl_toolkits.axes_grid import make_axes_locatable

    axScatter = subplot(111)
    divider = make_axes_locatable(axScatter)

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    axHistx = divider.new_vertical(1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.new_horizontal(1.2, pad=0.1, sharey=axScatter)

    fig.add_axes(axHistx)
    fig.add_axes(axHisty)


    # the scatter plot:
    axScatter.scatter(x, y)
    axScatter.set_aspect(1.)

    # histograms
    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

See the full source code below.


.. plot:: mpl_toolkits/axes_grid/figures/scatter_hist.py


The scatter_hist using the AxesDivider has some advantage over the
original scatter_hist.py in mpl. For example, you can set the aspect
ratio of the scatter plot, even with the x-axis or y-axis is shared
accordingly.


ParasiteAxes
============

The ParasiteAxes is a axes whose location is identical to its host
axes. The location is adjusted in the drawing time, thus it works even
if the host change its location (e.g., images). It provides *twinx*,
*twiny* (similar to twinx and twiny in the matplotlib). Also it
provides *twin*, which takes an arbitraty tranfromation that maps
between the data coordinates of the host and the parasite axes.
Artists in each axes are mergred and drawn acrroding to their zorder.
It also modifies some behavior of the axes. For example, color cycle
for plot lines are shared between host and parasites. Also, the legend
command in host, creates a legend that includes lines in the parasite
axes.

Example 1. twinx
----------------

.. plot:: mpl_toolkits/axes_grid/figures/parasite_simple.py
   :include-source:

Example 2. twin
---------------

A more sophiscated example using twin. Note that if you change the
x-limit in the host axes, the x-limit of the parasite axes will change
accordingly.


.. plot:: mpl_toolkits/axes_grid/figures/parasite_simple2.py



AxisLine
========

AxisLine is a custom (and very experimenta) Axes class, where each
axis (left, right, top and bottom) have a separate artist associated
(which is resposible to draw axis-line, ticks, ticklabels, label).
Also, you can create your own axis, which can pass through a fixed
position in the axes coordinate, or a fixed position in the data
coordinate (i.e., the axis floats around when viewlimit changes).

Most of the class in this toolkit is based on this class. And it has
not been tested extensibly. You may go back to the original mpl
behanvior, by ::

  ax.toggle_axisline(False)

The axes class, by default, provides 4 artists which are responsible
to draw axis in "left","right","bottom" and "top". They are accessed
as ax.axis["left"], ax.axis["right"], and so on, i.e., ax.axis is a
dictionary that contains artists (note that ax.axis is still a
callable methods and it behaves as an original Axes.axis method in
mpl).

For example, you can hide right, and top axis by ::

  ax.axis["right"].set_visible(False)
  ax.axis["top"].set_visible(False)


.. plot:: mpl_toolkits/axes_grid/figures/simple_axisline3.py


SubplotZero gives you two more additional (floating?) axis of x=0 and
y=0 (in data coordinate)

.. plot:: mpl_toolkits/axes_grid/figures/simple_axisline2.py
   :include-source:


Axisline with ParasiteAxes
--------------------------

Most of axes class in the axes_grid toolkit, including ParasiteAxes,
is based on the Axisline axes. The combination of the two can be
useful in some case. For example, you can have different tick-location,
tick-label, or tick-formatter for bottom and top (or left and right)
axis. ::

  ax2 = ax.twin() # now, ax2 is responsible for "top" axis and "right" axis
  ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
  ax2.set_xticklabels(["0", r"$\frac{1}{2}\pi$",
                       r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])


.. plot:: mpl_toolkits/axes_grid/figures/simple_axisline4.py


AxisLine Axes lets you create a custom axis, ::

    # make new (right-side) yaxis, but wth some offset
    offset = (20, 0)
    new_axisline = ax.get_grid_helper().new_fixed_axis
    ax.axis["right2"] = new_axisline(loc="right",
                                     offset=offset)


And, you can use it with parasiteAxes.


.. plot:: mpl_toolkits/axes_grid/figures/demo_parasite_axes2.py


AnchoredArtists
===============

It's a collection of artists whose location is anchored to the (axes)
bbox, like the legend. It is derived from *OffsetBox* in mpl, and
artist need to be drawn in the canvas coordinate. But, there is a
limited support for an arbitrary transform. For example, the ellipse
in the example below will have width and height in the data
coordinate.

.. plot:: mpl_toolkits/axes_grid/figures/simple_anchored_artists.py
   :include-source:


InsetLocator
============

:mod:`mpl_toolkits.axes_grid.inset_locator` provides helper classes
and functions to place your (inset) axes at the anchored position of
the parent axes, similarly to AnchoredArtis.

Using :func:`mpl_toolkits.axes_grid.inset_locator.inset_axes`, you
can have inset axes whose size is either fixed, or a fixed proportion
of the parent axes. For example,::

    inset_axes = inset_axes(parent_axes,
                            width="30%", # width = 30% of parent_bbox
                            height=1., # height : 1 inch
                            loc=3)

creates an inset axes whose width is 30% of the parent axes and whose
height is fixed at 1 inch.

You may creates your inset whose size is determined so that the data
scale of the inset axes to be that of the parent axes multiplied by
some factor. For example, ::

    inset_axes = zoomed_inset_axes(ax,
                                   0.5, # zoom = 0.5
                                   loc=1)

creates an inset axes whose data scale is half of the parent axes.
Here is complete examples.

.. plot:: mpl_toolkits/axes_grid/figures/inset_locator_demo.py

For example, :func:`zoomed_inset_axes` can be used when you want the
inset represents the zoom-up of the small portion in the parent axes.
And :mod:`~mpl_toolkits/axes_grid/inset_locator` provides a helper
function :func:`mark_inset` to mark the location of the area
represented by the inset axes.

.. plot:: mpl_toolkits/axes_grid/figures/inset_locator_demo2.py
   :include-source:


Curvelinear Grid
================

You can draw a cuvelinear grid and ticks. Also a floating axis can be
created. See :ref:`axislines-manual` for more details.

.. plot:: mpl_toolkits/axes_grid/figures/demo_floating_axis.py


