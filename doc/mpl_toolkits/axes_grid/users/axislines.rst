.. _axislines-manual:

=========
Axislines
=========

Axislines includes a derived Axes implementation. The
biggest difference is that the artists responsible to draw axis line,
ticks, ticklabel and axis labels are separated out from the mpl's Axis
class, which are much more than artists in the original
mpl. This change was strongly motivated to support curvlinear
grid. Here are a few things that axes_grid.axislines.Axes is different
from original Axes from mpl.

* Axis elements (axis line(spine), ticks, ticklabel and axis labels)
  are drawn by a AxisArtist instance. Unlike Axis, left, right, top
  and bottom axis are drawn by separate artists. And each of them may
  have different tick location and different tick labels.

* gridlines are drawn by a Gridlines instance. The change was
  motivated that in curvelinear coordinate, a gridline may not cross
  axislines (i.e., no associated ticks). In the original Axes class,
  gridlines are tied to ticks.

* ticklines can be rotated if necessary (i.e, along the gridlines)

In summary, all these changes was to support

* a curvelinear grid.
* a floating axis

.. plot:: mpl_toolkits/axes_grid/figures/demo_floating_axis.py


*axes_grid.axislines.Axes* defines a *axis* attribute, which is a
dictionary of AxisArtist instances. By default, the dictionary has 4
AxisArtist instances, responsible for drawing of left, right, bottom
and top axis.

xaxis and yaxis attributes are still available, however they are set
to not visible. As separate artists are used for rendering axis, some
axis-related method in mpl may have no effect.
In addition to AxisArtist instances, the axes_grid.axislines.Axes will
have *gridlines* attribute (Gridlines), which obviously draws grid
lines. 

In both AxisArtist and Gridlines, the calculation of tick and grid
location is delegated to an instance of GridHelper class.
axes_grid.axislines.Axes class uses GridHelperRectlinear as a grid
helper. The GridHelperRectlinear class is a wrapper around the *xaxis*
and *yaxis* of mpl's original Axes, and it was meant to work as the
way how mpl's original axes works. For example, tick location changes
using set_ticks method and etc. should work as expected. But change in
artist properties (e.g., color) will not work in general, although
some effort has been made so that some often-change attributes (color,
etc.) are respected.


AxisArtist
==========

AxisArtist can be considered as a container artist with following
attributes which will draw ticks, labels, etc.

 * line
 * major_ticks, major_ticklabels
 * minor_ticks, minor_ticklabels
 * offsetText
 * label


line
----

Derived from Line2d class. Responsible for drawing a spinal(?) line.

major_ticks, minor_ticks
------------------------

Derived from Line2d class. Note that ticks are markers.


major_ticklabels, minor_ticklabels
----------------------------------

Derived from Text. Note that it is not a list of Text artist, but a
single artist (similar to a collection).

axislabel
---------

Derived from Text.


Default AxisArtists
-------------------

By default, following for axis artists are defined.::

  ax.axis["left"], ax.axis["bottom"], ax.axis["right"], ax.axis["top"]

The ticklabels and axislabel of the top and the right axis are set to
not visible.


HowTo
=====

1. Changing tick locations and label.

  Same as the original mpl's axes.::

   ax.set_xticks([1,2,3])

2. Changing axis properties like color, etc.

  Change the properties of appropriate artists. For example, to change
  the color of the ticklabels::

    ax.axis["left"].major_ticklabels.set_color("r")


GridHelper
==========

To actually define a curvelinear coordinate, you have to use your own
grid helper. A generalised version of grid helper class is supplied
and this class should be suffice in most of cases. A user may provide
two functions which defines a transformation (and its inverse pair)
from the curved coordinate to (rectlinear) image coordinate. Note that
while ticks and grids are drawn for curved coordinate, the data
transform of the axes itself (ax.transData) is still rectlinear
(image) coordinate. ::


    from  mpl_toolkits.axes_grid.grid_helper_curvelinear \
         import GridHelperCurveLinear
    from mpl_toolkits.axes_grid.axislines import Subplot

    # from curved coordinate to rectlinear coordinate.
    def tr(x, y): 
        x, y = np.asarray(x), np.asarray(y)
        return x, y-x

    # from rectlinear coordinate to curved coordinate.
    def inv_tr(x,y):
        x, y = np.asarray(x), np.asarray(y)
        return x, y+x


    grid_helper = GridHelperCurveLinear((tr, inv_tr))

    ax1 = Subplot(fig, 1, 1, 1, grid_helper=grid_helper)

    fig.add_subplot(ax1)


You may use matplotlib's Transform instance instead (but a
inverse transformation must be defined). Often, coordinate range in a
curved coordinate system may have a limited range, or may have
cycles. In those cases, a more customized version of grid helper is
required. ::


    import  mpl_toolkits.axes_grid.angle_helper as angle_helper

    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degree
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()


    # extreme finder :  find a range of coordinate.
    # 20, 20 : number of sampling points along x, y direction
    # The first coordinate (longitude, but theta in polar) 
    #   has a cycle of 360 degree.
    # The second coordinate (latitude, but radius in polar)  has a minimum of 0
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle = 360,
                                                     lat_cycle = None,
                                                     lon_minmax = None,
                                                     lat_minmax = (0, np.inf),
                                                     )

    # Find a grid values appropriate for the coordinate (degree,
    # minute, second). The argument is a approximate number of grids.
    grid_locator1 = angle_helper.LocatorDMS(12)

    # And also uses an appropriate formatter.  Note that,the
    # acceptable Locator and Formatter class is a bit different than
    # that of mpl's, and you cannot directly use mpl's Locator and
    # Formatter here (but may be possible in the future).
    tick_formatter1 = angle_helper.FormatterDMS()

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )


Again, the *transData* of the axes is still a rectlinear coordinate
(image coordinate). You may manually do conversion between two
coordinates, or you may use Parasite Axes for convenience.::

    ax1 = SubplotHost(fig, 1, 2, 2, grid_helper=grid_helper)

    # A parasite axes with given transform
    ax2 = ParasiteAxesAuxTrans(ax1, tr, "equal")
    # note that ax2.transData == tr + ax1.transData
    # Anthing you draw in ax2 will match the ticks and grids of ax1.
    ax1.parasites.append(ax2)


.. plot:: mpl_toolkits/axes_grid/figures/demo_curvelinear_grid.py



FloatingAxis
============

A floating axis is an axis one of whose data coordinate is fixed, i.e,
its location is not fixed in Axes coordinate but changes as axes data
limits changes. A floating axis can be created using
*new_floating_axis* method. However, it is your responsibility that
the resulting AxisArtist is properly added to the axes. A recommended
way is to add it as an item of Axes's axis attribute.::

    # floating axis whose first (index starts from 0) coordinate 
    # (theta) is fixed at 60

    ax1.axis["lat"] = axis = ax1.new_floating_axis(0, 60)
    axis.label.set_text(r"$\theta = 60^{\circ}$")
    axis.label.set_visible(True)


See the first example of this page.

Current Limitations and TODO's
==============================

The code need more refinement. Here is a incomplete list of issues and TODO's

* No easy way to support a user customized tick location (for
  curvelinear grid). A new Locator class needs to be created.

* FloatingAxis may have coordinate limits, e.g., a floating axis of x
  = 0, but y only spans from 0 to 1.

* The location of axislabel of FloatingAxis needs to be optionally
  given as a coordinate value. ex, a floating axis of x=0 with label at y=1
