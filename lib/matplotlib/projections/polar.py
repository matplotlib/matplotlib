import math

import numpy as npy

from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.ticker import Locator
from matplotlib.transforms import Affine2D, Affine2DBase, Bbox, BboxTransform, \
    IdentityTransform, Transform, TransformWrapper

class PolarAxes(Axes):
    name = 'polar'
    
    class PolarTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, tr):
            xy = npy.zeros(tr.shape, npy.float_)
            t = tr[:, 0:1]
            r = tr[:, 1:2]
            x = xy[:, 0:1]
            y = xy[:, 1:2]
            x += r * npy.cos(t)
            y += r * npy.sin(t)
            return xy
        transform_non_affine = transform

        def inverted(self):
            return PolarAxes.InvertedPolarTransform()

    class PolarAffine(Affine2DBase):
        def __init__(self, limits):
            Affine2DBase.__init__(self)
            self._limits = limits
            self.set_children(limits)
            self._mtx = None

        def get_matrix(self):
            if self._invalid:
                ymax = self._limits.ymax
                affine = Affine2D() \
                    .scale(0.5 / ymax) \
                    .translate(0.5, 0.5)
                self._mtx = affine.get_matrix()
                self._inverted = None
                self._invalid = 0
            return self._mtx
    
    class InvertedPolarTransform(Transform):
        input_dims = 2
        output_dims = 2
        is_separable = False

        def transform(self, xy):
            x = xy[:, 0:1]
            y = xy[:, 1:]
            r = npy.sqrt(x*x + y*y)
            theta = npy.arccos(x / r)
            theta = npy.where(y < 0, 2 * npy.pi - theta, theta)
            return npy.concatenate((theta, r), 1)

        def inverted(self):
            return PolarAxes.PolarTransform()

    class ThetaLocator(Locator):
        pass
        
    def __init__(self, *args, **kwargs):
        Axes.__init__(self, *args, **kwargs)
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.cla()
    __init__.__doc__ = Axes.__init__.__doc__

    def cla(self):
        Axes.cla(self)

        self.xaxis.set_major_locator(PolarAxes.ThetaLocator())
    
    def _set_transData(self):
        # A (possibly non-linear) projection on the (already scaled) data
        self.transProjection = self.PolarTransform()

        # An affine transformation on the data, generally to limit the
        # range of the axes
        self.transProjectionAffine = self.PolarAffine(self.viewLim)
        
        self.transData = self.transScale + self.transProjection + \
            self.transProjectionAffine + self.transAxes

        self._xaxis_transform = (
            self.PolarTransform() +
            self.PolarAffine(Bbox.unit()) +
            self.transAxes)
        self._yaxis_transform = (
            Affine2D().scale(npy.pi * 0.5, 1.0) +
            self.transData)

    def get_xaxis_text1_transform(self, pixelPad):
        return (Affine2D().translate(0.0, 1.05) +
                self._xaxis_transform)
        
    def get_axes_patch(self):
        return Circle((0.5, 0.5), 0.5)
        
    def drag_pan(self, button, key, startx, starty, dx, dy, start_lim, start_trans):
        def format_deltas(key, dt, dr):
            if key=='t':
                dr = 0
            elif key=='r':
                dt = 0
            return (dt,dr)
        
        if button == 1:
            inverse = start_trans.inverted()
            startt, startr = inverse.transform_point((startx, starty))
            t, r = inverse.transform_point((startx + dx, starty + dy))
            
            dt, dr = t - startt, r - startr
            dt, dr = format_deltas(key, dt, dr)
            t, r = startt + dt, startr + dr

            # Deal with r
            scale = r / startr
            self.set_ylim(start_lim.ymin, start_lim.ymax / scale)

            # Deal with theta
            dt0 = t - startt
            dt1 = startt - t
            if abs(dt1) < abs(dt0):
                dt = abs(dt1) * sign(dt0) * -1.0
            else:
                dt = dt0 * -1.0
            self.set_xlim(start_lim.xmin - dt, start_lim.xmin - dt + npy.pi*2.0)
        
    def set_rmax(self, rmax):
        self.viewLim.ymax = rmax

    def get_rmax(self):
        return self.viewLim.ymax

    def set_rscale(self, *args, **kwargs):
        return self.set_yscale(*args, **kwargs)

    def set_xscale(self, *args, **kwargs):
        raise NotImplementedError("You can not set the xscale on a polar plot.")
    
    def format_coord(self, theta, r):
        'return a format string formatting the coordinate'
        thetas = self.format_xdata(theta)
        rs = self.format_ydata(r)
        theta /= math.pi
        return u'theta=%spi, r=%s' % (thetas, rs)

    def get_data_ratio(self):
        '''
        Return the aspect ratio of the data itself.  For a polar plot,
        this should always be 1.0
        '''
        return 1.0
    
# These are a couple of failed attempts to project a polar plot using
# cubic bezier curves.
        
#         def transform_path(self, path):
#             twopi = 2.0 * npy.pi
#             halfpi = 0.5 * npy.pi
            
#             vertices = path.vertices
#             t0 = vertices[0:-1, 0]
#             t1 = vertices[1:  , 0]
#             td = npy.where(t1 > t0, t1 - t0, twopi - (t0 - t1))
#             maxtd = td.max()
#             interpolate = npy.ceil(maxtd / halfpi)
#             if interpolate > 1.0:
#                 vertices = self.interpolate(vertices, interpolate)

#             vertices = self.transform(vertices)

#             result = npy.zeros((len(vertices) * 3 - 2, 2), npy.float_)
#             codes = mpath.Path.CURVE4 * npy.ones((len(vertices) * 3 - 2, ), mpath.Path.code_type)
#             result[0] = vertices[0]
#             codes[0] = mpath.Path.MOVETO

#             kappa = 4.0 * ((npy.sqrt(2.0) - 1.0) / 3.0)
#             kappa = 0.5
            
#             p0   = vertices[0:-1]
#             p1   = vertices[1:  ]

#             x0   = p0[:, 0:1]
#             y0   = p0[:, 1: ]
#             b0   = ((y0 - x0) - y0) / ((x0 + y0) - x0)
#             a0   = y0 - b0*x0

#             x1   = p1[:, 0:1]
#             y1   = p1[:, 1: ]
#             b1   = ((y1 - x1) - y1) / ((x1 + y1) - x1)
#             a1   = y1 - b1*x1

#             x = -(a0-a1) / (b0-b1)
#             y = a0 + b0*x

#             xk = (x - x0) * kappa + x0
#             yk = (y - y0) * kappa + y0

#             result[1::3, 0:1] = xk
#             result[1::3, 1: ] = yk

#             xk = (x - x1) * kappa + x1
#             yk = (y - y1) * kappa + y1

#             result[2::3, 0:1] = xk
#             result[2::3, 1: ] = yk
            
#             result[3::3] = p1

#             print vertices[-2:]
#             print result[-2:]
            
#             return mpath.Path(result, codes)
            
#             twopi = 2.0 * npy.pi
#             halfpi = 0.5 * npy.pi
            
#             vertices = path.vertices
#             t0 = vertices[0:-1, 0]
#             t1 = vertices[1:  , 0]
#             td = npy.where(t1 > t0, t1 - t0, twopi - (t0 - t1))
#             maxtd = td.max()
#             interpolate = npy.ceil(maxtd / halfpi)

#             print "interpolate", interpolate
#             if interpolate > 1.0:
#                 vertices = self.interpolate(vertices, interpolate)
            
#             result = npy.zeros((len(vertices) * 3 - 2, 2), npy.float_)
#             codes = mpath.Path.CURVE4 * npy.ones((len(vertices) * 3 - 2, ), mpath.Path.code_type)
#             result[0] = vertices[0]
#             codes[0] = mpath.Path.MOVETO

#             kappa = 4.0 * ((npy.sqrt(2.0) - 1.0) / 3.0)
#             tkappa = npy.arctan(kappa)
#             hyp_kappa = npy.sqrt(kappa*kappa + 1.0)

#             t0 = vertices[0:-1, 0]
#             t1 = vertices[1:  , 0]
#             r0 = vertices[0:-1, 1]
#             r1 = vertices[1:  , 1]

#             td = npy.where(t1 > t0, t1 - t0, twopi - (t0 - t1))
#             td_scaled = td / (npy.pi * 0.5)
#             rd = r1 - r0
#             r0kappa = r0 * kappa * td_scaled
#             r1kappa = r1 * kappa * td_scaled
#             ravg_kappa = ((r1 + r0) / 2.0) * kappa * td_scaled

#             result[1::3, 0] = t0 + (tkappa * td_scaled)
#             result[1::3, 1] = r0*hyp_kappa
#             # result[1::3, 1] = r0 / npy.cos(tkappa * td_scaled) # npy.sqrt(r0*r0 + ravg_kappa*ravg_kappa)

#             result[2::3, 0] = t1 - (tkappa * td_scaled)
#             result[2::3, 1] = r1*hyp_kappa
#             # result[2::3, 1] = r1 / npy.cos(tkappa * td_scaled) # npy.sqrt(r1*r1 + ravg_kappa*ravg_kappa)
            
#             result[3::3, 0] = t1
#             result[3::3, 1] = r1

#             print vertices[:6], result[:6], t0[:6], t1[:6], td[:6], td_scaled[:6], tkappa
#             result = self.transform(result)
#             return mpath.Path(result, codes)
#         transform_path_non_affine = transform_path

        
