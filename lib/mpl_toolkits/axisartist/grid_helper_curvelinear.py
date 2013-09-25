"""
An experimental support for curvilinear grid.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import zip

from itertools import chain
from .grid_finder import GridFinder

from  .axislines import AxisArtistHelper, GridHelperBase
from  .axis_artist import AxisArtist
from matplotlib.transforms import Affine2D, IdentityTransform
import numpy as np

from matplotlib.path import Path

class FixedAxisArtistHelper(AxisArtistHelper.Fixed):
    """
    Helper class for a fixed axis.
    """

    def __init__(self, grid_helper, side, nth_coord_ticks=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """

        super(FixedAxisArtistHelper, self).__init__( \
            loc=side,
            )

        self.grid_helper = grid_helper
        if nth_coord_ticks is None:
            nth_coord_ticks = self.nth_coord
        self.nth_coord_ticks = nth_coord_ticks

        self.side = side
        self._limits_inverted = False

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)

        if self.nth_coord == 0:
            xy1, xy2 = axes.get_ylim()
        else:
            xy1, xy2 = axes.get_xlim()

        if xy1 > xy2:
            self._limits_inverted = True
        else:
            self._limits_inverted = False


    def change_tick_coord(self, coord_number=None):
        if coord_number is None:
            self.nth_coord_ticks = 1 - self.nth_coord_ticks
        elif coord_number in [0, 1]:
            self.nth_coord_ticks = coord_number
        else:
            raise Exception("wrong coord number")


    def get_tick_transform(self, axes):
        return axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""

        g = self.grid_helper

        if self._limits_inverted:
            side = {"left":"right","right":"left",
                    "top":"bottom", "bottom":"top"}[self.side]
        else:
            side = self.side

        ti1 = g.get_tick_iterator(self.nth_coord_ticks, side)
        ti2 = g.get_tick_iterator(1-self.nth_coord_ticks, side, minor=True)

        #ti2 = g.get_tick_iterator(1-self.nth_coord_ticks, self.side, minor=True)

        return chain(ti1, ti2), iter([])



class FloatingAxisArtistHelper(AxisArtistHelper.Floating):

    def __init__(self, grid_helper, nth_coord, value, axis_direction=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """

        super(FloatingAxisArtistHelper, self).__init__(nth_coord,
                                                       value,
                                                       )
        self.value = value
        self.grid_helper = grid_helper
        self._extremes = None, None

        self._get_line_path = None # a method that returns a Path.
        self._line_num_points = 100 # number of points to create a line

    def set_extremes(self, e1, e2):
        self._extremes = e1, e2

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)

        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()
        grid_finder = self.grid_helper.grid_finder
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                              x1, y1, x2, y2)

        extremes = list(extremes)
        e1, e2 = self._extremes # ranges of other coordinates
        if self.nth_coord == 0:
            if e1 is not None:
                extremes[2] = max(e1, extremes[2])
            if e2 is not None:
                extremes[3] = min(e2, extremes[3])
        elif self.nth_coord == 1:
            if e1 is not None:
                extremes[0] = max(e1, extremes[0])
            if e2 is not None:
                extremes[1] = min(e2, extremes[1])

        grid_info = dict()
        lon_min, lon_max, lat_min, lat_max = extremes
        lon_levs, lon_n, lon_factor = \
                  grid_finder.grid_locator1(lon_min, lon_max)
        lat_levs, lat_n, lat_factor = \
                  grid_finder.grid_locator2(lat_min, lat_max)
        grid_info["extremes"] = extremes

        grid_info["lon_info"] = lon_levs, lon_n, lon_factor
        grid_info["lat_info"] = lat_levs, lat_n, lat_factor

        grid_info["lon_labels"] = grid_finder.tick_formatter1("bottom",
                                                              lon_factor,
                                                              lon_levs)

        grid_info["lat_labels"] = grid_finder.tick_formatter2("bottom",
                                                              lat_factor,
                                                              lat_levs)

        grid_finder = self.grid_helper.grid_finder

        #e1, e2 = self._extremes # ranges of other coordinates
        if self.nth_coord == 0:
            xx0 = np.linspace(self.value, self.value, self._line_num_points)
            yy0 = np.linspace(extremes[2], extremes[3], self._line_num_points)
            xx, yy = grid_finder.transform_xy(xx0, yy0)
        elif self.nth_coord == 1:
            xx0 = np.linspace(extremes[0], extremes[1], self._line_num_points)
            yy0 = np.linspace(self.value, self.value, self._line_num_points)
            xx, yy = grid_finder.transform_xy(xx0, yy0)

        grid_info["line_xy"] = xx, yy
        self.grid_info = grid_info

    def get_axislabel_transform(self, axes):
        return Affine2D() #axes.transData

    def get_axislabel_pos_angle(self, axes):

        extremes = self.grid_info["extremes"]

        if self.nth_coord == 0:
            xx0 = self.value
            yy0 = (extremes[2]+extremes[3])/2.
            dxx, dyy = 0., abs(extremes[2]-extremes[3])/1000.
        elif self.nth_coord == 1:
            xx0 = (extremes[0]+extremes[1])/2.
            yy0 = self.value
            dxx, dyy = abs(extremes[0]-extremes[1])/1000., 0.

        grid_finder = self.grid_helper.grid_finder
        xx1, yy1 = grid_finder.transform_xy([xx0], [yy0])

        trans_passingthrough_point = axes.transData + axes.transAxes.inverted()
        p = trans_passingthrough_point.transform_point([xx1[0], yy1[0]])


        if (0. <= p[0] <= 1.) and (0. <= p[1] <= 1.):
            xx1c, yy1c = axes.transData.transform_point([xx1[0], yy1[0]])
            xx2, yy2 = grid_finder.transform_xy([xx0+dxx], [yy0+dyy])
            xx2c, yy2c = axes.transData.transform_point([xx2[0], yy2[0]])

            return (xx1c, yy1c), np.arctan2(yy2c-yy1c, xx2c-xx1c)/np.pi*180.
        else:
            return None, None




    def get_tick_transform(self, axes):
        return IdentityTransform() #axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label, (optionally) tick_label"""

        grid_finder = self.grid_helper.grid_finder

        lat_levs, lat_n, lat_factor = self.grid_info["lat_info"]
        lat_levs = np.asarray(lat_levs)
        if lat_factor is not None:
            yy0 = lat_levs / lat_factor
            dy = 0.01 / lat_factor
        else:
            yy0 = lat_levs
            dy = 0.01

        lon_levs, lon_n, lon_factor = self.grid_info["lon_info"]
        lon_levs = np.asarray(lon_levs)
        if lon_factor is not None:
            xx0 = lon_levs / lon_factor
            dx = 0.01 / lon_factor
        else:
            xx0 = lon_levs
            dx = 0.01

        if None in self._extremes:
            e0, e1 = self._extremes
        else:
            e0, e1 = sorted(self._extremes)
        if e0 is None:
            e0 = -np.inf
        if e1 is None:
            e1 = np.inf

        if self.nth_coord == 0:
            mask = (e0 <= yy0) & (yy0 <= e1)
            #xx0, yy0 = xx0[mask], yy0[mask]
            yy0 = yy0[mask]
        elif self.nth_coord == 1:
            mask = (e0 <= xx0) & (xx0 <= e1)
            #xx0, yy0 = xx0[mask], yy0[mask]
            xx0 = xx0[mask]

        def transform_xy(x, y):
            x1, y1 = grid_finder.transform_xy(x, y)
            x2y2 = axes.transData.transform(np.array([x1, y1]).transpose())
            x2, y2 = x2y2.transpose()
            return x2, y2

        # find angles
        if self.nth_coord == 0:
            xx0 = np.empty_like(yy0)
            xx0.fill(self.value)

            xx1, yy1 = transform_xy(xx0, yy0)

            xx00 = xx0.copy()
            xx00[xx0+dx>e1] -= dx
            xx1a, yy1a = transform_xy(xx00, yy0)
            xx1b, yy1b = transform_xy(xx00+dx, yy0)

            xx2a, yy2a = transform_xy(xx0, yy0)
            xx2b, yy2b = transform_xy(xx0, yy0+dy)

            labels = self.grid_info["lat_labels"]
            labels = [l for l, m in zip(labels, mask) if m]

        elif self.nth_coord == 1:
            yy0 = np.empty_like(xx0)
            yy0.fill(self.value)

            xx1, yy1 = transform_xy(xx0, yy0)

            xx1a, yy1a = transform_xy(xx0, yy0)
            xx1b, yy1b = transform_xy(xx0, yy0+dy)

            xx00 = xx0.copy()
            xx00[xx0+dx>e1] -= dx
            xx2a, yy2a = transform_xy(xx00, yy0)
            xx2b, yy2b = transform_xy(xx00+dx, yy0)

            labels = self.grid_info["lon_labels"]
            labels = [l for l, m in zip(labels, mask) if m]


        def f1():
            dd = np.arctan2(yy1b-yy1a, xx1b-xx1a) # angle normal
            dd2 = np.arctan2(yy2b-yy2a, xx2b-xx2a) # angle tangent
            mm = ((yy1b-yy1a)==0.) & ((xx1b-xx1a)==0.) # mask where dd1 is not defined
            dd[mm] = dd2[mm]+3.14159/2.
            #dd = np.arctan2(yy2-yy1, xx2-xx1) # angle normal
            #dd2 = np.arctan2(yy3-yy1, xx3-xx1) # angle tangent
            #mm = ((yy2-yy1)==0.) & ((xx2-xx1)==0.) # mask where dd1 is not defined
            #dd[mm] = dd2[mm]+3.14159/2.

            #dd += 3.14159

            #dd = np.arctan2(xx2-xx1, angle_tangent-yy1)
            trans_tick = self.get_tick_transform(axes)
            tr2ax = trans_tick + axes.transAxes.inverted()
            for x, y, d, d2, lab in zip(xx1, yy1, dd, dd2, labels):
                c2 = tr2ax.transform_point((x, y))
                delta=0.00001
                if (0. -delta<= c2[0] <= 1.+delta) and \
                       (0. -delta<= c2[1] <= 1.+delta):
                    d1 = d/3.14159*180.
                    d2 = d2/3.14159*180.
                    yield [x, y], d1, d2, lab

        return f1(), iter([])

    def get_line_transform(self, axes):
        return axes.transData

    def get_line(self, axes):
        self.update_lim(axes)
        x, y = self.grid_info["line_xy"]

        if self._get_line_path is None:
            return Path(list(zip(x, y)))
        else:
            return self._get_line_path(axes, x, y)




class GridHelperCurveLinear(GridHelperBase):

    def __init__(self, aux_trans,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        """
        aux_trans : a transform from the source (curved) coordinate to
        target (rectilinear) coordinate. An instance of MPL's Transform
        (inverse transform should be defined) or a tuple of two callable
        objects which defines the transform and its inverse. The callables
        need take two arguments of array of source coordinates and
        should return two target coordinates:
          e.g., x2, y2 = trans(x1, y1)
        """
        super(GridHelperCurveLinear, self).__init__()

        self.grid_info = None
        self._old_values = None
        #self._grid_params = dict()
        self._aux_trans = aux_trans

        self.grid_finder = GridFinder(aux_trans,
                                      extreme_finder,
                                      grid_locator1,
                                      grid_locator2,
                                      tick_formatter1,
                                      tick_formatter2)


    def update_grid_finder(self, aux_trans=None, **kw):

        if aux_trans is not None:
            self.grid_finder.update_transform(aux_trans)

        self.grid_finder.update(**kw)
        self.invalidate()


    def _update(self, x1, x2, y1, y2):
        "bbox in 0-based image coordinates"
        # update wcsgrid

        if self.valid() and self._old_values == (x1, x2, y1, y2):
            return

        self._update_grid(x1, y1, x2, y2)

        self._old_values = (x1, x2, y1, y2)

        self._force_update = False


    def new_fixed_axis(self, loc,
                       nth_coord=None,
                       axis_direction=None,
                       offset=None,
                       axes=None):


        if axes is None:
            axes = self.axes

        if axis_direction is None:
            axis_direction = loc
        _helper = FixedAxisArtistHelper(self, loc,
                                        #nth_coord,
                                        nth_coord_ticks=nth_coord,
                                        )

        axisline = AxisArtist(axes, _helper, axis_direction=axis_direction)

        return axisline


    def new_floating_axis(self, nth_coord,
                          value,
                          axes=None,
                          axis_direction="bottom"
                          ):

        if axes is None:
            axes = self.axes

        _helper = FloatingAxisArtistHelper( \
            self, nth_coord, value, axis_direction)

        axisline = AxisArtist(axes, _helper)

        #_helper = FloatingAxisArtistHelper(self, nth_coord,
        #                                   value,
        #                                   label_direction=label_direction,
        #                                   )

        #axisline = AxisArtistFloating(axes, _helper,
        #                              axis_direction=axis_direction)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        #axisline.major_ticklabels.set_visible(True)
        #axisline.minor_ticklabels.set_visible(False)

        #axisline.major_ticklabels.set_rotate_along_line(True)
        #axisline.set_rotate_label_along_line(True)

        return axisline


    def _update_grid(self, x1, y1, x2, y2):

        self.grid_info = self.grid_finder.get_grid_info(x1, y1, x2, y2)


    def get_gridlines(self, which="major", axis="both"):
        grid_lines = []

        if axis in ["both", "x"]:
            for gl in self.grid_info["lon"]["lines"]:
                grid_lines.extend(gl)
        if axis in ["both", "y"]:
            for gl in self.grid_info["lat"]["lines"]:
                grid_lines.extend(gl)

        return grid_lines


    def get_tick_iterator(self, nth_coord, axis_side, minor=False):

        #axisnr = dict(left=0, bottom=1, right=2, top=3)[axis_side]
        angle_tangent = dict(left=90, right=90, bottom=0, top=0)[axis_side]
        #angle = [0, 90, 180, 270][axisnr]
        lon_or_lat = ["lon", "lat"][nth_coord]
        if not minor: # major ticks
            def f():
                for (xy, a), l in zip(self.grid_info[lon_or_lat]["tick_locs"][axis_side],
                                    self.grid_info[lon_or_lat]["tick_labels"][axis_side]):
                    angle_normal = a
                    yield xy, angle_normal, angle_tangent, l
        else:
            def f():
                for (xy, a), l in zip(self.grid_info[lon_or_lat]["tick_locs"][axis_side],
                                    self.grid_info[lon_or_lat]["tick_labels"][axis_side]):
                    angle_normal = a
                    yield xy, angle_normal, angle_tangent, ""
                #for xy, a, l in self.grid_info[lon_or_lat]["ticks"][axis_side]:
                #    yield xy, a, ""

        return f()



def test3():

    import numpy as np
    from matplotlib.transforms import Transform
    from matplotlib.path import Path

    class MyTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution):
            """
            Create a new Aitoff transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Aitoff space.
            """
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, ll):
            x = ll[:, 0:1]
            y  = ll[:, 1:2]

            return np.concatenate((x, y-x), 1)

        transform.__doc__ = Transform.transform.__doc__

        transform_non_affine = transform
        transform_non_affine.__doc__ = Transform.transform_non_affine.__doc__

        def transform_path(self, path):
            vertices = path.vertices
            ipath = path.interpolated(self._resolution)
            return Path(self.transform(ipath.vertices), ipath.codes)
        transform_path.__doc__ = Transform.transform_path.__doc__

        transform_path_non_affine = transform_path
        transform_path_non_affine.__doc__ = Transform.transform_path_non_affine.__doc__

        def inverted(self):
            return MyTransformInv(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    class MyTransformInv(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, ll):
            x = ll[:, 0:1]
            y  = ll[:, 1:2]

            return np.concatenate((x, y+x), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return MyTransform(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__



    import matplotlib.pyplot as plt
    fig = plt.figure(1)
    fig.clf()
    tr = MyTransform(1)
    grid_helper = GridHelperCurveLinear(tr)


    from mpl_toolkits.axes_grid1.parasite_axes import host_subplot_class_factory
    from .axislines import Axes

    SubplotHost = host_subplot_class_factory(Axes)

    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    fig.add_subplot(ax1)

    ax2 = ParasiteAxesAuxTrans(ax1, tr, "equal")
    ax1.parasites.append(ax2)
    ax2.plot([3, 6], [5.0, 10.])

    ax1.set_aspect(1.)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)

    ax1.grid(True)
    plt.draw()



def curvelinear_test2(fig):
    """
    polar projection, but in a rectangular box.
    """
    global ax1
    import numpy as np
    from . import angle_helper
    from matplotlib.projections import PolarAxes
    from matplotlib.transforms import Affine2D

    from mpl_toolkits.axes_grid.parasite_axes import SubplotHost, \
         ParasiteAxesAuxTrans
    import matplotlib.cbook as cbook

    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degree
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

    # polar projection, which involves cycle, and also has limits in
    # its coordinates, needs a special method to find the extremes
    # (min, max of the coordinate within the view).

    # 20, 20 : number of sampling points along x, y direction
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle = 360,
                                                     lat_cycle = None,
                                                     lon_minmax = None,
                                                     lat_minmax = (0, np.inf),
                                                     )

    grid_locator1 = angle_helper.LocatorDMS(5)
    # Find a grid values appropriate for the coordinate (degree,
    # minute, second).

    tick_formatter1 = angle_helper.FormatterDMS()
    # And also uses an appropriate formatter.  Note that,the
    # acceptable Locator and Formatter class is a bit different than
    # that of mpl's, and you cannot directly use mpl's Locator and
    # Formatter here (but may be possible in the future).

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )


    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    # make ticklabels of right and top axis visible.
    ax1.axis["right"].major_ticklabels.set_visible(True)
    ax1.axis["top"].major_ticklabels.set_visible(True)

    # let right axis shows ticklabels for 1st coordinate (angle)
    ax1.axis["right"].get_helper().nth_coord_ticks=0
    # let bottom axis shows ticklabels for 2nd coordinate (radius)
    ax1.axis["bottom"].get_helper().nth_coord_ticks=1

    fig.add_subplot(ax1)

    grid_helper = ax1.get_grid_helper()
    ax1.axis["lat"] = axis = grid_helper.new_floating_axis(0, 60, axes=ax1)
    axis.label.set_text("Test")
    axis.label.set_visible(True)
    #axis._extremes = 2, 10
    #axis.label.set_text("Test")
    #axis.major_ticklabels.set_visible(False)
    #axis.major_ticks.set_visible(False)
    axis.get_helper()._extremes=2, 10

    ax1.axis["lon"] = axis = grid_helper.new_floating_axis(1, 6, axes=ax1)
    #axis.major_ticklabels.set_visible(False)
    #axis.major_ticks.set_visible(False)
    axis.label.set_text("Test 2")
    axis.get_helper()._extremes=-180, 90

    # A parasite axes with given transform
    ax2 = ParasiteAxesAuxTrans(ax1, tr, "equal")
    # note that ax2.transData == tr + ax1.transData
    # Anthing you draw in ax2 will match the ticks and grids of ax1.
    ax1.parasites.append(ax2)
    intp = cbook.simple_linear_interpolation
    ax2.plot(intp(np.array([0, 30]), 50),
             intp(np.array([10., 10.]), 50))

    ax1.set_aspect(1.)
    ax1.set_xlim(-5, 12)
    ax1.set_ylim(-5, 10)

    ax1.grid(True)


def curvelinear_test3(fig):
    """
    polar projection, but in a rectangular box.
    """
    global ax1, axis
    import numpy as np
    from . import angle_helper
    from matplotlib.projections import PolarAxes
    from matplotlib.transforms import Affine2D

    from mpl_toolkits.axes_grid.parasite_axes import SubplotHost

    # PolarAxes.PolarTransform takes radian. However, we want our coordinate
    # system in degree
    tr = Affine2D().scale(np.pi/180., 1.) + PolarAxes.PolarTransform()

    # polar projection, which involves cycle, and also has limits in
    # its coordinates, needs a special method to find the extremes
    # (min, max of the coordinate within the view).

    # 20, 20 : number of sampling points along x, y direction
    extreme_finder = angle_helper.ExtremeFinderCycle(20, 20,
                                                     lon_cycle = 360,
                                                     lat_cycle = None,
                                                     lon_minmax = None,
                                                     lat_minmax = (0, np.inf),
                                                     )

    grid_locator1 = angle_helper.LocatorDMS(12)
    # Find a grid values appropriate for the coordinate (degree,
    # minute, second).

    tick_formatter1 = angle_helper.FormatterDMS()
    # And also uses an appropriate formatter.  Note that,the
    # acceptable Locator and Formatter class is a bit different than
    # that of mpl's, and you cannot directly use mpl's Locator and
    # Formatter here (but may be possible in the future).

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )


    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    for axis in list(six.itervalues(ax1.axis)):
        axis.set_visible(False)

    fig.add_subplot(ax1)

    grid_helper = ax1.get_grid_helper()
    ax1.axis["lat1"] = axis = grid_helper.new_floating_axis(0, 130,
                                                            axes=ax1,
                                                            axis_direction="left"
                                                            )
    axis.label.set_text("Test")
    axis.label.set_visible(True)
    axis.get_helper()._extremes=0.001, 10



    grid_helper = ax1.get_grid_helper()
    ax1.axis["lat2"] = axis = grid_helper.new_floating_axis(0, 50, axes=ax1,
                                                            axis_direction="right")
    axis.label.set_text("Test")
    axis.label.set_visible(True)
    axis.get_helper()._extremes=0.001, 10

    ax1.axis["lon"] = axis = grid_helper.new_floating_axis(1, 10,
                                                           axes=ax1,
                                                           axis_direction="bottom")
    axis.label.set_text("Test 2")
    axis.get_helper()._extremes= 50, 130
    axis.major_ticklabels.set_axis_direction("top")
    axis.label.set_axis_direction("top")

    grid_helper.grid_finder.grid_locator1.den = 5
    grid_helper.grid_finder.grid_locator2._nbins = 5


#     # A parasite axes with given transform
#     ax2 = ParasiteAxesAuxTrans(ax1, tr, "equal")
#     # note that ax2.transData == tr + ax1.transData
#     # Anthing you draw in ax2 will match the ticks and grids of ax1.
#     ax1.parasites.append(ax2)
#     intp = cbook.simple_linear_interpolation
#     ax2.plot(intp(np.array([0, 30]), 50),
#              intp(np.array([10., 10.]), 50))

    ax1.set_aspect(1.)
    ax1.set_xlim(-5, 12)
    ax1.set_ylim(-5, 10)

    ax1.grid(True)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plt.figure(1, figsize=(5, 5))
    fig.clf()

    #test3()
    #curvelinear_test2(fig)
    curvelinear_test3(fig)

    #plt.draw()
    plt.show()
