import math

import numpy as npy
from matplotlib.numerix import npyma as ma

import matplotlib
rcParams = matplotlib.rcParams
from matplotlib.artist import kwdocd
from matplotlib.axes import Axes
from matplotlib import cbook
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.ticker import Formatter, Locator, NullLocator, FixedLocator, NullFormatter
from matplotlib.transforms import Affine2D, Affine2DBase, Bbox, \
    BboxTransformTo, IdentityTransform, Transform, TransformWrapper

class GeoAxes(Axes):
    """
    An abstract base class for geographic projections
    """
    class ThetaFormatter(Formatter):
        """
        Used to format the theta tick labels.  Converts the native
        unit of radians into degrees and adds a degree symbol.
        """
        def __init__(self, round_to=1.0):
            self._round_to = round_to

        def __call__(self, x, pos=None):
            degrees = (x / npy.pi) * 180.0
            degrees = round(degrees / self._round_to) * self._round_to
            # \u00b0 : degree symbol
            return u"%d\u00b0" % degrees

    RESOLUTION = 75

    def cla(self):
        Axes.cla(self)

        self.set_longitude_grid(30)
        self.set_latitude_grid(15)
        self.set_longitude_grid_ends(75)
        self.xaxis.set_minor_locator(NullLocator())
        self.yaxis.set_minor_locator(NullLocator())
        self.xaxis.set_ticks_position('none')
        self.yaxis.set_ticks_position('none')

        self.grid(rcParams['axes.grid'])

        Axes.set_xlim(self, -npy.pi, npy.pi)
        Axes.set_ylim(self, -npy.pi / 2.0, npy.pi / 2.0)

    def _set_lim_and_transforms(self):
        # A (possibly non-linear) projection on the (already scaled) data
        self.transProjection = self._get_core_transform(self.RESOLUTION)

        self.transAffine = self._get_affine_transform()

        self.transAxes = BboxTransformTo(self.bbox)

        # The complete data transformation stack -- from data all the
        # way to display coordinates
        self.transData = \
            self.transProjection + \
            self.transAffine + \
            self.transAxes

        # This is the transform for longitude ticks.
        self._xaxis_pretransform = \
            Affine2D() \
            .scale(1.0, self._longitude_cap * 2.0) \
            .translate(0.0, -self._longitude_cap)
        self._xaxis_transform = \
            self._xaxis_pretransform + \
            self.transData
        self._xaxis_text1_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, 4.0)
        self._xaxis_text2_transform = \
            Affine2D().scale(1.0, 0.0) + \
            self.transData + \
            Affine2D().translate(0.0, -4.0)

        # This is the transform for latitude ticks.
        yaxis_stretch = Affine2D().scale(npy.pi * 2.0, 1.0).translate(-npy.pi, 0.0)
        yaxis_space = Affine2D().scale(1.0, 1.1)
        self._yaxis_transform = \
            yaxis_stretch + \
            self.transData
        yaxis_text_base = \
            yaxis_stretch + \
            self.transProjection + \
            (yaxis_space + \
             self.transAffine + \
             self.transAxes)
        self._yaxis_text1_transform = \
            yaxis_text_base + \
            Affine2D().translate(-8.0, 0.0)
        self._yaxis_text2_transform = \
            yaxis_text_base + \
            Affine2D().translate(8.0, 0.0)

    def _get_affine_transform(self):
        transform = self._get_core_transform(1)
        xscale, _ = transform.transform_point((npy.pi, 0))
        _, yscale = transform.transform_point((0, npy.pi / 2.0))
        return Affine2D() \
            .scale(0.5 / xscale, 0.5 / yscale) \
            .translate(0.5, 0.5)

    def update_layout(self, renderer):
        t_text, b_text = self.xaxis.get_text_heights(renderer)
        l_text, r_text = self.yaxis.get_text_widths(renderer)
        originalPosition = self.get_position(True)
        title_offset = (b_text - originalPosition.transformed(
                self.figure.transFigure).height) / 2.0
        self.titleOffsetTrans.clear().translate(0, title_offset)

    def get_xaxis_transform(self):
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad):
        return self._xaxis_text1_transform, 'bottom', 'center'

    def get_xaxis_text2_transform(self, pad):
        return self._xaxis_text2_transform, 'top', 'center'

    def get_yaxis_transform(self):
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad):
        return self._yaxis_text1_transform, 'center', 'right'

    def get_yaxis_text2_transform(self, pad):
        return self._yaxis_text2_transform, 'center', 'left'

    def get_axes_patch(self):
        return Circle((0.5, 0.5), 0.5)

    def set_yscale(self, *args, **kwargs):
        if args[0] != 'linear':
            raise NotImplementedError

    set_xscale = set_yscale

    def set_xlim(self, *args, **kwargs):
        Axes.set_xlim(self, -npy.pi, npy.pi)
        Axes.set_ylim(self, -npy.pi / 2.0, npy.pi / 2.0)

    set_ylim = set_xlim

    def format_coord(self, long, lat):
        'return a format string formatting the coordinate'
        long = long * (180.0 / npy.pi)
        lat = lat * (180.0 / npy.pi)
        if lat >= 0.0:
            ns = 'N'
        else:
            ns = 'S'
        if long >= 0.0:
            ew = 'E'
        else:
            ew = 'W'
        return u'%f\u00b0%s, %f\u00b0%s' % (abs(lat), ns, abs(long), ew)

    def set_longitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.
        """
        number = (360.0 / degrees) + 1
        self.xaxis.set_major_locator(
            FixedLocator(
                npy.linspace(-npy.pi, npy.pi, number, True)[1:-1]))
        self._logitude_degrees = degrees
        self.xaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_latitude_grid(self, degrees):
        """
        Set the number of degrees between each longitude grid.
        """
        number = (180.0 / degrees) + 1
        self.yaxis.set_major_locator(
            FixedLocator(
                npy.linspace(-npy.pi / 2.0, npy.pi / 2.0, number, True)[1:-1]))
        self._latitude_degrees = degrees
        self.yaxis.set_major_formatter(self.ThetaFormatter(degrees))

    def set_longitude_grid_ends(self, degrees):
        """
        Set the latitude(s) at which to stop drawing the longitude grids.
        """
        self._longitude_cap = degrees * (npy.pi / 180.0)
        self._xaxis_pretransform \
            .clear() \
            .scale(1.0, self._longitude_cap * 2.0) \
            .translate(0.0, -self._longitude_cap)

    def get_data_ratio(self):
        '''
        Return the aspect ratio of the data itself.
        '''
        return 1.0

    ### Interactive panning

    def can_zoom(self):
        """
        Return True if this axes support the zoom box
        """
        return False

    def start_pan(self, x, y, button):
        pass

    def end_pan(self):
        pass

    def drag_pan(self, button, key, x, y):
        pass


class AitoffAxes(GeoAxes):
    name = 'aitoff'

    class AitoffTransform(Transform):
        """
        The base Aitoff transform.
        """
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
            longitude = ll[:, 0:1]
            latitude  = ll[:, 1:2]

            # Pre-compute some values
            half_long = longitude / 2.0
            cos_latitude = npy.cos(latitude)

            alpha = npy.arccos(cos_latitude * npy.cos(half_long))
            # Mask this array, or we'll get divide-by-zero errors
            alpha = ma.masked_where(alpha == 0.0, alpha)
            # We want unnormalized sinc.  numpy.sinc gives us normalized
            sinc_alpha = ma.sin(alpha) / alpha

            x = (cos_latitude * npy.sin(half_long)) / sinc_alpha
            y = (npy.sin(latitude) / sinc_alpha)
            x.set_fill_value(0.0)
            y.set_fill_value(0.0)
            return npy.concatenate((x.filled(), y.filled()), 1)
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
            return AitoffAxes.InvertedAitoffTransform(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedAitoffTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, xy):
            # MGDTODO: Math is hard ;(
            return xy
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return AitoffAxes.AitoffTransform(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    def __init__(self, *args, **kwargs):
        self._longitude_cap = npy.pi / 2.0
        GeoAxes.__init__(self, *args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.AitoffTransform(resolution)


class HammerAxes(GeoAxes):
    name = 'hammer'

    class HammerTransform(Transform):
        """
        The base Hammer transform.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution):
            """
            Create a new Hammer transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Hammer space.
            """
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, ll):
            longitude = ll[:, 0:1]
            latitude  = ll[:, 1:2]

            # Pre-compute some values
            half_long = longitude / 2.0
            cos_latitude = npy.cos(latitude)
            sqrt2 = npy.sqrt(2.0)

            alpha = 1.0 + cos_latitude * npy.cos(half_long)
            x = (2.0 * sqrt2) * (cos_latitude * npy.sin(half_long)) / alpha
            y = (sqrt2 * npy.sin(latitude)) / alpha
            return npy.concatenate((x, y), 1)
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
            return HammerAxes.InvertedHammerTransform(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedHammerTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]

            quarter_x = 0.25 * x
            half_y = 0.5 * y
            z = npy.sqrt(1.0 - quarter_x*quarter_x - half_y*half_y)
            longitude = 2 * npy.arctan((z*x) / (2.0 * (2.0*z*z - 1.0)))
            latitude = npy.arcsin(y*z)
            return npy.concatenate((longitude, latitude), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return HammerAxes.HammerTransform(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    def __init__(self, *args, **kwargs):
        self._longitude_cap = npy.pi / 2.0
        GeoAxes.__init__(self, *args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.HammerTransform(resolution)


class MollweideAxes(GeoAxes):
    name = 'mollweide'

    class MollweideTransform(Transform):
        """
        The base Mollweide transform.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution):
            """
            Create a new Mollweide transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Mollweide space.
            """
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, ll):
            longitude = ll[:, 0:1]
            latitude  = ll[:, 1:2]

            aux = 2.0 * npy.arcsin((2.0 * latitude) / npy.pi)
            x = (2.0 * npy.sqrt(2.0) * longitude * npy.cos(aux)) / npy.pi
            y = (npy.sqrt(2.0) * npy.sin(aux))

            return npy.concatenate((x, y), 1)
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
            return MollweideAxes.InvertedMollweideTransform(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedMollweideTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, resolution):
            Transform.__init__(self)
            self._resolution = resolution

        def transform(self, xy):
            # MGDTODO: Math is hard ;(
            return xy
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return MollweideAxes.MollweideTransform(self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    def __init__(self, *args, **kwargs):
        self._longitude_cap = npy.pi / 2.0
        GeoAxes.__init__(self, *args, **kwargs)
        self.set_aspect(0.5, adjustable='box', anchor='C')
        self.cla()

    def _get_core_transform(self, resolution):
        return self.MollweideTransform(resolution)


class LambertAxes(GeoAxes):
    name = 'lambert'

    class LambertTransform(Transform):
        """
        The base Lambert transform.
        """
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, center_longitude, center_latitude, resolution):
            """
            Create a new Lambert transform.  Resolution is the number of steps
            to interpolate between each input line segment to approximate its
            path in curved Lambert space.
            """
            Transform.__init__(self)
            self._resolution = resolution
            self._center_longitude = center_longitude
            self._center_latitude = center_latitude

        def transform(self, ll):
            longitude = ll[:, 0:1]
            latitude  = ll[:, 1:2]
            clong = self._center_longitude
            clat = self._center_latitude
            cos_lat = npy.cos(latitude)
            sin_lat = npy.sin(latitude)
            diff_long = longitude - clong
            cos_diff_long = npy.cos(diff_long)

            inner_k = (1.0 +
                       npy.sin(clat)*sin_lat +
                       npy.cos(clat)*cos_lat*cos_diff_long)
            # Prevent divide-by-zero problems
            inner_k = npy.where(inner_k == 0.0, 1e-15, inner_k)
            k = npy.sqrt(2.0 / inner_k)
            x = k*cos_lat*npy.sin(diff_long)
            y = k*(npy.cos(clat)*sin_lat -
                   npy.sin(clat)*cos_lat*cos_diff_long)

            return npy.concatenate((x, y), 1)
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
            return LambertAxes.InvertedLambertTransform(
                self._center_longitude,
                self._center_latitude,
                self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    class InvertedLambertTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def __init__(self, center_longitude, center_latitude, resolution):
            Transform.__init__(self)
            self._resolution = resolution
            self._center_longitude = center_longitude
            self._center_latitude = center_latitude

        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            clong = self._center_longitude
            clat = self._center_latitude
            p = npy.sqrt(x*x + y*y)
            p = npy.where(p == 0.0, 1e-9, p)
            c = 2.0 * npy.arcsin(0.5 * p)
            sin_c = npy.sin(c)
            cos_c = npy.cos(c)

            lat = npy.arcsin(cos_c*npy.sin(clat) +
                             ((y*sin_c*npy.cos(clat)) / p))
            long = clong + npy.arctan(
                (x*sin_c) / (p*npy.cos(clat)*cos_c - y*npy.sin(clat)*sin_c))

            return npy.concatenate((long, lat), 1)
        transform.__doc__ = Transform.transform.__doc__

        def inverted(self):
            return LambertAxes.LambertTransform(
                self._center_longitude,
                self._center_latitude,
                self._resolution)
        inverted.__doc__ = Transform.inverted.__doc__

    def __init__(self, *args, **kwargs):
        self._longitude_cap = npy.pi / 2.0
        self._center_longitude = kwargs.pop("center_longitude", 0.0)
        self._center_latitude = kwargs.pop("center_latitude", 0.0)
        GeoAxes.__init__(self, *args, **kwargs)
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.cla()

    def cla(self):
        GeoAxes.cla(self)
        self.yaxis.set_major_formatter(NullFormatter())

    def _get_core_transform(self, resolution):
        return self.LambertTransform(
            self._center_longitude,
            self._center_latitude,
            resolution)

    def _get_affine_transform(self):
        return Affine2D() \
            .scale(0.25) \
            .translate(0.5, 0.5)
