"""
An experimental support for curvelinear grid.
"""

from itertools import chain
from grid_finder import GridFinderMplTransform, GridFinder

from  mpl_toolkits.axes_grid.axislines import \
     AxisArtistHelper, GridHelperBase, AxisArtist
from matplotlib.transforms import Transform


class FixedAxisArtistHelper(AxisArtistHelper.Fixed):

    def __init__(self, grid_helper, side, nth_coord, nth_coord_ticks):
        """
        nth_coord = along which coordinate value varies.
         nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis
        """

        super(FixedAxisArtistHelper, self).__init__( \
            loc=side,
            nth_coord=nth_coord,
            passingthrough_point=None,
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
        self._grid_params = dict()

        if isinstance(aux_trans, Transform):
            self.grid_finder = GridFinderMplTransform(aux_trans,
                                                      extreme_finder,
                                                      grid_locator1,
                                                      grid_locator2,
                                                      tick_formatter1,
                                                      tick_formatter2)
        else:
            trans, inv_trans = aux_trans
            self.grid_finder = GridFinder(trans, inv_trans,
                                          extreme_finder,
                                          grid_locator1,
                                          grid_locator2,
                                          tick_formatter1,
                                          tick_formatter2)


    def _update(self, x1, x2, y1, y2):
        "bbox in 0-based image coordinates"
        # update wcsgrid

        if self._force_update is False and \
               self._old_values == (x1, x2, y1, y2,
                                    self.get_grid_params()):
            return

        self._update_grid(x1, y1, x2, y2)

        self._old_values = (x1, x2, y1, y2, self.get_grid_params().copy())

        self._force_update = False


    def new_fixed_axis(self, loc,
                       nth_coord=None, passthrough_point=None,
                       tick_direction="in",
                       label_direction=None,
                       offset=None,
                       axes=None):


        if axes is None:
            axes = self.axes

        _helper = FixedAxisArtistHelper(self, loc,
                                        nth_coord, nth_coord_ticks=None)

        axisline = AxisArtist(axes, _helper)

        return axisline



    def update_grid_params(self, **ka):
        pass

    def get_grid_params(self):
        return self._grid_params

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




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test3()
    plt.show()

