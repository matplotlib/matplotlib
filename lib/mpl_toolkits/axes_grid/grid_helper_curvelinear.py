"""
An experimental support for curvelinear grid.
"""

from itertools import chain
from mpl_toolkits.axes_grid.grid_finder import GridFinder

from  mpl_toolkits.axes_grid.axislines import \
     AxisArtistHelper, GridHelperBase, AxisArtist
from matplotlib.transforms import Affine2D
import numpy as np

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
            label_direction=None)

        self.grid_helper = grid_helper
        if nth_coord_ticks is None:
            nth_coord_ticks = self.nth_coord
        self.nth_coord_ticks = nth_coord_ticks

        self.side = side

    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)

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

        ti1 = g.get_tick_iterator(self.nth_coord_ticks, self.side)
        ti2 = g.get_tick_iterator(1-self.nth_coord_ticks, self.side, minor=True)

        #ti2 = g.get_tick_iterator(1-self.nth_coord_ticks, self.side, minor=True)

        return chain(ti1, ti2), iter([])


class FloatingAxisArtistHelper(AxisArtistHelper.Floating):

    def __init__(self, grid_helper, nth_coord, value, label_direction=None):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """

        super(FloatingAxisArtistHelper, self).__init__(nth_coord,
                                                       value,
                                                       label_direction,
                                                       )
        self.value = value
        self.grid_helper = grid_helper


    def update_lim(self, axes):
        self.grid_helper.update_lim(axes)

        x1, x2 = axes.get_xlim()
        y1, y2 = axes.get_ylim()
        grid_finder = self.grid_helper.grid_finder
        extremes = grid_finder.extreme_finder(grid_finder.inv_transform_xy,
                                              x1, y1, x2, y2)

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
        if self.nth_coord == 0:
            xx0 = np.linspace(self.value, self.value, 100)
            yy0 = np.linspace(extremes[2], extremes[3], 100)
            xx, yy = grid_finder.transform_xy(xx0, yy0)
        elif self.nth_coord == 1:
            xx0 = np.linspace(extremes[0], extremes[1], 100)
            yy0 = np.linspace(self.value, self.value, 100)
            xx, yy = grid_finder.transform_xy(xx0, yy0)

        grid_info["line_xy"] = xx, yy
        self.grid_info = grid_info


    def get_label_pos(self, axes, with_angle=False):

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
            if with_angle:
                xx1c, yy1c = axes.transData.transform_point([xx1[0], yy1[0]])
                xx2, yy2 = grid_finder.transform_xy([xx0+dxx], [yy0+dyy])
                xx2c, yy2c = axes.transData.transform_point([xx2[0], yy2[0]])

                return (xx1c, yy1c), Affine2D(), \
                       np.arctan2(yy2c-yy1c, xx2c-xx1c)
            else:
                return p, axes.transAxes
        else:
            if with_angle:
                return None, None, None
            else:
                return None, None




    def get_ticklabel_offset_transform(self, axes,
                                       pad_points, fontprops,
                                       renderer,
                                       ):

        tr, va, ha = self._get_label_offset_transform(pad_points, fontprops,
                                                      renderer,
                                                      None,
                                                      )

        a = self._ticklabel_angles[self.label_direction]
        return Affine2D(), "baseline", "center", 0

    def get_tick_transform(self, axes):
        return axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""

        grid_finder = self.grid_helper.grid_finder

        # find angles
        if self.nth_coord == 0:
            lat_levs, lat_n, lat_factor = self.grid_info["lat_info"]
            if lat_factor is not None:
                yy0 = lat_levs / lat_factor
                dy = 0.01 / lat_factor
            else:
                yy0 = lat_levs
                dy = 0.01
            xx0 = np.empty_like(yy0)
            xx0.fill(self.value)
            xx1, yy1 = grid_finder.transform_xy(xx0, yy0)
            xx2, yy2 = grid_finder.transform_xy(xx0, yy0+dy)
            labels = self.grid_info["lat_labels"]
        elif self.nth_coord == 1:
            lon_levs, lon_n, lon_factor = self.grid_info["lon_info"]
            if lon_factor is not None:
                xx0 = lon_levs / lon_factor
                dx = 0.01 / lon_factor
            else:
                xx0 = lon_levs
                dx = 0.01
            yy0 = np.empty_like(xx0)
            yy0.fill(self.value)
            xx1, yy1 = grid_finder.transform_xy(xx0, yy0)
            xx2, yy2 = grid_finder.transform_xy(xx0+dx, yy0)
            labels = self.grid_info["lon_labels"]

        if self.label_direction == "top":
            da = 180.
        else:
            da = 0.

        def f1():
            dd = np.arctan2(yy2-yy1, xx2-xx1)
            trans_tick = self.get_tick_transform(axes)
            tr2ax = trans_tick + axes.transAxes.inverted()
            for x, y, d, lab in zip(xx1, yy1, dd, labels):
                c2 = tr2ax.transform_point((x, y))
                delta=0.00001
                if (0. -delta<= c2[0] <= 1.+delta) and \
                       (0. -delta<= c2[1] <= 1.+delta):
                    yield [x, y], d/3.14159*180.+da, lab

        return f1(), iter([])


    def get_line_transform(self, axes):
        return axes.transData

    def get_line(self, axes):
        self.update_lim(axes)
        from matplotlib.path import Path
        xx, yy = self.grid_info["line_xy"]

        return Path(zip(xx, yy))




class GridHelperCurveLinear(GridHelperBase):

    def __init__(self, aux_trans,
                 extreme_finder=None,
                 grid_locator1=None,
                 grid_locator2=None,
                 tick_formatter1=None,
                 tick_formatter2=None):
        """
        aux_trans : a transform from the source (curved) coordinate to
        target (rectlinear) coordinate. An instance of MPL's Transform
        (inverse transform should be defined) or a tuple of two callable
        objects which defines the transform and its inverse. The callables
        need take two arguments of array of source coordinates and
        should return two target coordinates:
          e.g. x2, y2 = trans(x1, y1)
        """
        super(GridHelperCurveLinear, self).__init__()

        self.grid_info = None
        self._old_values = None
        #self._grid_params = dict()

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
                       tick_direction="in",
                       label_direction=None,
                       offset=None,
                       axes=None):


        if axes is None:
            axes = self.axes

        _helper = FixedAxisArtistHelper(self, loc,
                                        #nth_coord,
                                        nth_coord_ticks=nth_coord)

        axisline = AxisArtist(axes, _helper)

        return axisline


    def new_floating_axis(self, nth_coord,
                          value,
                          tick_direction="in",
                          label_direction=None,
                          axes=None,
                          ):

        if label_direction is None:
            label_direction = "top"

        _helper = FloatingAxisArtistHelper(self, nth_coord,
                                           value,
                                           label_direction=label_direction,
                                           )

        axisline = AxisArtist(axes, _helper)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        #axisline.major_ticklabels.set_visible(True)
        #axisline.minor_ticklabels.set_visible(False)

        axisline.major_ticklabels.set_rotate_along_line(True)
        axisline.set_rotate_label_along_line(True)

        return axisline


    def _update_grid(self, x1, y1, x2, y2):

        self.grid_info = self.grid_finder.get_grid_info(x1, y1, x2, y2)


    def get_gridlines(self):
        grid_lines = []
        for gl in self.grid_info["lat"]["lines"]:
            grid_lines.extend(gl)
        for gl in self.grid_info["lon"]["lines"]:
            grid_lines.extend(gl)

        return grid_lines


    def get_tick_iterator(self, nth_coord, axis_side, minor=False):

        axisnr = dict(left=0, bottom=1, right=2, top=3)[axis_side]
        angle = [0, 90, 180, 270][axisnr]
        lon_or_lat = ["lon", "lat"][nth_coord]
        if not minor: # major ticks
            def f():
                for (xy, a), l in zip(self.grid_info[lon_or_lat]["tick_locs"][axis_side],
                                    self.grid_info[lon_or_lat]["tick_labels"][axis_side]):
                    yield xy, a, l
        else:
            def f():
                for (xy, a), l in zip(self.grid_info[lon_or_lat]["tick_locs"][axis_side],
                                    self.grid_info[lon_or_lat]["tick_labels"][axis_side]):
                    yield xy, a, ""
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
    from mpl_toolkits.axes_grid.parasite_axes import SubplotHost, \
         ParasiteAxesAuxTrans
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
    import  mpl_toolkits.axes_grid.angle_helper as angle_helper
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
    #axis.label.set_text("Test")
    #axis.major_ticklabels.set_visible(False)
    #axis.major_ticks.set_visible(False)

    ax1.axis["lon"] = axis = grid_helper.new_floating_axis(1, 6, axes=ax1)
    #axis.major_ticklabels.set_visible(False)
    #axis.major_ticks.set_visible(False)
    axis.label.set_text("Test 2")

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig = plt.figure(1, figsize=(5, 5))
    fig.clf()

    #curvelinear_test1(fig)
    curvelinear_test2(fig)

    plt.draw()
    plt.show()


