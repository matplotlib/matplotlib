==============================
Overview of axes_grid1 toolkit
==============================

What is axes_grid1 toolkit?
===========================

The matplotlib axes_grid1 toolkit is a collection of helper classes,
mainly to ease displaying (multiple) images in matplotlib.

.. contents::
   :depth: 1
   :local:

.. note:: 
   AxesGrid toolkit has been a part of matplotlib since v
   0.99. Originally, the toolkit had a single namespace of 
   *axes_grid*. In more recent version, the toolkit 
   has divided into two separate namespace (*axes_grid1* and *axisartist*).
   While *axes_grid* namespace is maintained for the backward compatibility,
   use of *axes_grid1* and *axisartist* is recommended.


*axes_grid1* is a collection of helper classes to ease displaying
(multiple) images with matplotlib.  In matplotlib, the axes location
(and size) is specified in the normalized figure coordinates, which
may not be ideal for displaying images that needs to have a given
aspect ratio.  For example, it helps you to have a colorbar whose
height always matches that of the image.  `ImageGrid`_, `RGB Axes`_ and
`AxesDivider`_ are helper classes that deals with adjusting the
location of (multiple) Axes.  They provides a framework to adjust the
position of multiple axes at the drawing time. `ParasiteAxes`_
provides twinx(or twiny)-like features so that you can plot different
data (e.g., different y-scale) in a same Axes. `AnchoredArtists`_
includes custom artists which are placed at some anchored position,
like the legend.

.. plot:: mpl_toolkits/axes_grid1/examples/demo_axes_grid.py


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

.. plot:: mpl_toolkits/axes_grid1/examples/simple_axesgrid.py
   :include-source:

* The position of each axes is determined at the drawing time (see
  `AxesDivider`_), so that the size of the entire grid fits in the
  given rectangle (like the aspect of axes). Note that in this example,
  the paddings between axes are fixed even if you changes the figure
  size.

* axes in the same column has a same axes width (in figure
  coordinate), and similarly, axes in the same row has a same
  height. The widths (height) of the axes in the same row (column) are
  scaled according to their view limits (xlim or ylim).

  .. plot:: mpl_toolkits/axes_grid1/examples/simple_axesgrid2.py
     :include-source:

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

You can also create a colorbar (or colorbars). You can have colorbar
for each axes (cbar_mode="each"), or you can have a single colorbar
for the grid (cbar_mode="single"). The colorbar can be placed on your
right, or top. The axes for each colorbar is stored as a *cbar_axes*
attribute.



The examples below show what you can do with ImageGrid.

.. plot:: mpl_toolkits/axes_grid1/examples/demo_axes_grid.py


AxesDivider
-----------

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

.. plot:: mpl_toolkits/axes_grid1/figures/simple_colorbar.py
   :include-source:




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

.. plot:: mpl_toolkits/axes_grid1/examples/scatter_hist.py


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

.. plot:: mpl_toolkits/axes_grid1/figures/parasite_simple.py
   :include-source:

Example 2. twin
~~~~~~~~~~~~~~~

*twin* without a transform argument treat the parasite axes to have a
same data transform as the host. This can be useful when you want the
top(or right)-axis to have different tick-locations, tick-labels, or
tick-formatter for bottom(or left)-axis. ::

  ax2 = ax.twin() # now, ax2 is responsible for "top" axis and "right" axis
  ax2.set_xticks([0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi])
  ax2.set_xticklabels(["0", r"$\frac{1}{2}\pi$",
                       r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"])


.. plot:: mpl_toolkits/axes_grid1/examples/simple_axisline4.py



A more sophisticated example using twin. Note that if you change the
x-limit in the host axes, the x-limit of the parasite axes will change
accordingly.


.. plot:: mpl_toolkits/axes_grid1/examples/parasite_simple2.py


AnchoredArtists
---------------

It's a collection of artists whose location is anchored to the (axes)
bbox, like the legend. It is derived from *OffsetBox* in mpl, and
artist need to be drawn in the canvas coordinate. But, there is a
limited support for an arbitrary transform. For example, the ellipse
in the example below will have width and height in the data
coordinate.

.. plot:: mpl_toolkits/axes_grid1/examples/simple_anchored_artists.py
   :include-source:


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

.. plot:: mpl_toolkits/axes_grid1/examples/inset_locator_demo.py

For example, :func:`zoomed_inset_axes` can be used when you want the
inset represents the zoom-up of the small portion in the parent axes.
And :mod:`~mpl_toolkits/axes_grid/inset_locator` provides a helper
function :func:`mark_inset` to mark the location of the area
represented by the inset axes.

.. plot:: mpl_toolkits/axes_grid1/examples/inset_locator_demo2.py
   :include-source:


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


.. plot:: mpl_toolkits/axes_grid1/figures/simple_rgb.py
