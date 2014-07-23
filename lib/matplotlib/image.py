"""
The image module supports basic image loading, rescaling and display
operations.

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import warnings
import math

import numpy as np
from numpy import ma

from matplotlib import rcParams
import matplotlib.artist as martist
from matplotlib.artist import allow_rasterization
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.cbook as cbook

# For clarity, names from _image are given explicitly in this module:
import matplotlib._image as _image
import matplotlib._png as _png

# For user convenience, the names from _image are also imported into
# the image namespace:
from matplotlib._image import *

from matplotlib.transforms import BboxBase, Bbox, IdentityTransform
import matplotlib.transforms as mtransforms


class _AxesImageBase(martist.Artist, cm.ScalarMappable):
    zorder = 0
    # map interpolation strings to module constants
    _interpd = {
        'none': _image.NEAREST,  # fall back to nearest when not supported
        'nearest': _image.NEAREST,
        'bilinear': _image.BILINEAR,
        'bicubic': _image.BICUBIC,
        'spline16': _image.SPLINE16,
        'spline36': _image.SPLINE36,
        'hanning': _image.HANNING,
        'hamming': _image.HAMMING,
        'hermite': _image.HERMITE,
        'kaiser': _image.KAISER,
        'quadric': _image.QUADRIC,
        'catrom': _image.CATROM,
        'gaussian': _image.GAUSSIAN,
        'bessel': _image.BESSEL,
        'mitchell': _image.MITCHELL,
        'sinc': _image.SINC,
        'lanczos': _image.LANCZOS,
        'blackman': _image.BLACKMAN,
    }

    # reverse interp dict
    _interpdr = dict([(v, k) for k, v in six.iteritems(_interpd)])

    interpnames = list(six.iterkeys(_interpd))

    def __str__(self):
        return "AxesImage(%g,%g;%gx%g)" % tuple(self.axes.bbox.bounds)

    def __init__(self, ax,
                 cmap=None,
                 norm=None,
                 interpolation=None,
                 origin=None,
                 filternorm=1,
                 filterrad=4.0,
                 resample=False,
                 **kwargs
                 ):
        """
        interpolation and cmap default to their rc settings

        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        extent is data axes (left, right, bottom, top) for making image plots
        registered with data plots.  Default is to label the pixel
        centers with the zero-based row and column indices.

        Additional kwargs are matplotlib.artist properties

        """
        martist.Artist.__init__(self)
        cm.ScalarMappable.__init__(self, norm, cmap)

        if origin is None:
            origin = rcParams['image.origin']
        self.origin = origin
        self.set_filternorm(filternorm)
        self.set_filterrad(filterrad)
        self._filterrad = filterrad

        self.set_interpolation(interpolation)
        self.set_resample(resample)
        self.axes = ax

        self._imcache = None

        # this is an experimental attribute, if True, unsampled image
        # will be drawn using the affine transform that are
        # appropriately skewed so that the given position
        # corresponds to the actual position in the coordinate. -JJL
        self._image_skew_coordinate = None

        self.update(kwargs)

    def get_size(self):
        """Get the numrows, numcols of the input image"""
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        return self._A.shape[:2]

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on
        all backends

        ACCEPTS: float
        """
        martist.Artist.set_alpha(self, alpha)
        self._imcache = None

    def changed(self):
        """
        Call this whenever the mappable is changed so observers can
        update state
        """
        self._imcache = None
        self._rgbacache = None
        cm.ScalarMappable.changed(self)

    def make_image(self, magnification=1.0):
        raise RuntimeError('The make_image method must be overridden.')

    def _get_unsampled_image(self, A, image_extents, viewlim):
        """
        convert numpy array A with given extents ([x1, x2, y1, y2] in
        data coordinate) into the Image, given the viewlim (should be a
        bbox instance).  Image will be clipped if the extents is
        significantly larger than the viewlim.
        """
        xmin, xmax, ymin, ymax = image_extents
        dxintv = xmax-xmin
        dyintv = ymax-ymin

        # the viewport scale factor
        if viewlim.width == 0.0 and dxintv == 0.0:
            sx = 1.0
        else:
            sx = dxintv/viewlim.width
        if viewlim.height == 0.0 and dyintv == 0.0:
            sy = 1.0
        else:
            sy = dyintv/viewlim.height
        numrows, numcols = A.shape[:2]
        if sx > 2:
            x0 = (viewlim.x0-xmin)/dxintv * numcols
            ix0 = max(0, int(x0 - self._filterrad))
            x1 = (viewlim.x1-xmin)/dxintv * numcols
            ix1 = min(numcols, int(x1 + self._filterrad))
            xslice = slice(ix0, ix1)
            xmin_old = xmin
            xmin = xmin_old + ix0*dxintv/numcols
            xmax = xmin_old + ix1*dxintv/numcols
            dxintv = xmax - xmin
            sx = dxintv/viewlim.width
        else:
            xslice = slice(0, numcols)

        if sy > 2:
            y0 = (viewlim.y0-ymin)/dyintv * numrows
            iy0 = max(0, int(y0 - self._filterrad))
            y1 = (viewlim.y1-ymin)/dyintv * numrows
            iy1 = min(numrows, int(y1 + self._filterrad))
            if self.origin == 'upper':
                yslice = slice(numrows-iy1, numrows-iy0)
            else:
                yslice = slice(iy0, iy1)
            ymin_old = ymin
            ymin = ymin_old + iy0*dyintv/numrows
            ymax = ymin_old + iy1*dyintv/numrows
            dyintv = ymax - ymin
            sy = dyintv/viewlim.height
        else:
            yslice = slice(0, numrows)

        if xslice != self._oldxslice or yslice != self._oldyslice:
            self._imcache = None
            self._oldxslice = xslice
            self._oldyslice = yslice

        if self._imcache is None:
            if self._A.dtype == np.uint8 and self._A.ndim == 3:
                im = _image.frombyte(self._A[yslice, xslice, :], 0)
                im.is_grayscale = False
            else:
                if self._rgbacache is None:
                    x = self.to_rgba(self._A, bytes=False)
                    # Avoid side effects: to_rgba can return its argument
                    # unchanged.
                    if np.may_share_memory(x, self._A):
                        x = x.copy()
                    # premultiply the colors
                    x[..., 0:3] *= x[..., 3:4]
                    x = (x * 255).astype(np.uint8)
                    self._rgbacache = x
                else:
                    x = self._rgbacache
                im = _image.frombyte(x[yslice, xslice, :], 0)
                if self._A.ndim == 2:
                    im.is_grayscale = self.cmap.is_gray()
                else:
                    im.is_grayscale = False
            self._imcache = im

            if self.origin == 'upper':
                im.flipud_in()
        else:
            im = self._imcache

        return im, xmin, ymin, dxintv, dyintv, sx, sy

    @staticmethod
    def _get_rotate_and_skew_transform(x1, y1, x2, y2, x3, y3):
        """
        Retuen a transform that does
         (x1, y1) -> (x1, y1)
         (x2, y2) -> (x2, y2)
         (x2, y1) -> (x3, y3)

        It was intended to derive a skew transform that preserve the
        lower-left corner (x1, y1) and top-right corner(x2,y2), but
        change the the lower-right-corner(x2, y1) to a new position
        (x3, y3).
        """
        tr1 = mtransforms.Affine2D()
        tr1.translate(-x1, -y1)
        x2a, y2a = tr1.transform_point((x2, y2))
        x3a, y3a = tr1.transform_point((x3, y3))

        inv_mat = 1. / (x2a*y3a-y2a*x3a) * np.mat([[y3a, -y2a], [-x3a, x2a]])

        a, b = (inv_mat * np.mat([[x2a], [x2a]])).flat
        c, d = (inv_mat * np.mat([[y2a], [0]])).flat

        tr2 = mtransforms.Affine2D.from_values(a, c, b, d, 0, 0)

        tr = (tr1 + tr2 +
              mtransforms.Affine2D().translate(x1, y1)).inverted().get_affine()

        return tr

    def _draw_unsampled_image(self, renderer, gc):
        """
        draw unsampled image. The renderer should support a draw_image method
        with scale parameter.
        """
        trans = self.get_transform()  # axes.transData

        # convert the coordinates to the intermediate coordinate (ic).
        # The transformation from the ic to the canvas is a pure
        # affine transform.

        # A straight-forward way is to use the non-affine part of the
        # original transform for conversion to the ic.

        # firs, convert the image extent to the ic
        x_llc, x_trc, y_llc, y_trc = self.get_extent()

        xy = trans.transform(np.array([(x_llc, y_llc),
                                       (x_trc, y_trc)]))

        _xx1, _yy1 = xy[0]
        _xx2, _yy2 = xy[1]

        extent_in_ic = _xx1, _xx2, _yy1, _yy2

        # define trans_ic_to_canvas : unless _image_skew_coordinate is
        # set, it is simply a affine part of the original transform.
        if self._image_skew_coordinate:
            # skew the image when required.
            x_lrc, y_lrc = self._image_skew_coordinate
            xy2 = trans.transform(np.array([(x_lrc, y_lrc)]))
            _xx3, _yy3 = xy2[0]

            tr_rotate_skew = self._get_rotate_and_skew_transform(_xx1, _yy1,
                                                                 _xx2, _yy2,
                                                                 _xx3, _yy3)
            trans_ic_to_canvas = tr_rotate_skew
        else:
            trans_ic_to_canvas = IdentityTransform()

        # Now, viewLim in the ic.  It can be rotated and can be
        # skewed. Make it big enough.
        x1, y1, x2, y2 = self.axes.bbox.extents
        trans_canvas_to_ic = trans_ic_to_canvas.inverted()
        xy_ = trans_canvas_to_ic.transform(np.array([(x1, y1),
                                                     (x2, y1),
                                                     (x2, y2),
                                                     (x1, y2)]))
        x1_, x2_ = min(xy_[:, 0]), max(xy_[:, 0])
        y1_, y2_ = min(xy_[:, 1]), max(xy_[:, 1])
        viewLim_in_ic = Bbox.from_extents(x1_, y1_, x2_, y2_)

        # get the image, sliced if necessary. This is done in the ic.
        im, xmin, ymin, dxintv, dyintv, sx, sy = \
            self._get_unsampled_image(self._A, extent_in_ic, viewLim_in_ic)

        if im is None:
            return  # I'm not if this check is required. -JJL

        fc = self.axes.patch.get_facecolor()
        bg = mcolors.colorConverter.to_rgba(fc, 0)
        im.set_bg(*bg)

        # image input dimensions
        im.reset_matrix()
        numrows, numcols = im.get_size()

        if numrows <= 0 or numcols <= 0:
            return
        im.resize(numcols, numrows)  # just to create im.bufOut that
                                     # is required by backends. There
                                     # may be better solution -JJL

        im._url = self.get_url()
        im._gid = self.get_gid()

        renderer.draw_image(gc, xmin, ymin, im, dxintv, dyintv,
                            trans_ic_to_canvas)

    def _check_unsampled_image(self, renderer):
        """
        return True if the image is better to be drawn unsampled.
        The derived class needs to override it.
        """
        return False

    @allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible():
            return
        if (self.axes.get_xscale() != 'linear' or
            self.axes.get_yscale() != 'linear'):
            warnings.warn("Images are not supported on non-linear axes.")

        l, b, widthDisplay, heightDisplay = self.axes.bbox.bounds
        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_alpha(self.get_alpha())

        if self._check_unsampled_image(renderer):
            self._draw_unsampled_image(renderer, gc)
        else:
            if self._image_skew_coordinate is not None:
                warnings.warn("Image will not be shown"
                              " correctly with this backend.")

            im = self.make_image(renderer.get_image_magnification())
            if im is None:
                return
            im._url = self.get_url()
            im._gid = self.get_gid()
            renderer.draw_image(gc, l, b, im)
        gc.restore()

    def contains(self, mouseevent):
        """
        Test whether the mouse event occured within the image.
        """
        if six.callable(self._contains):
            return self._contains(self, mouseevent)
        # TODO: make sure this is consistent with patch and patch
        # collection on nonlinear transformed coordinates.
        # TODO: consider returning image coordinates (shouldn't
        # be too difficult given that the image is rectilinear
        x, y = mouseevent.xdata, mouseevent.ydata
        xmin, xmax, ymin, ymax = self.get_extent()
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        #print x, y, xmin, xmax, ymin, ymax
        if x is not None and y is not None:
            inside = ((x >= xmin) and (x <= xmax) and
                      (y >= ymin) and (y <= ymax))
        else:
            inside = False

        return inside, {}

    def write_png(self, fname, noscale=False):
        """Write the image to png file with fname"""
        im = self.make_image()
        if im is None:
            return
        if noscale:
            numrows, numcols = im.get_size()
            im.reset_matrix()
            im.set_interpolation(0)
            im.resize(numcols, numrows)
        im.flipud_out()
        rows, cols, buffer = im.as_rgba_str()
        _png.write_png(buffer, cols, rows, fname)

    def set_data(self, A):
        """
        Set the image array

        ACCEPTS: numpy/PIL Image A
        """
        # check if data is PIL Image without importing Image
        if hasattr(A, 'getpixel'):
            self._A = pil_to_array(A)
        else:
            self._A = cbook.safe_masked_invalid(A)

        if (self._A.dtype != np.uint8 and
            not np.can_cast(self._A.dtype, np.float)):
            raise TypeError("Image data can not convert to float")

        if (self._A.ndim not in (2, 3) or
            (self._A.ndim == 3 and self._A.shape[-1] not in (3, 4))):
            raise TypeError("Invalid dimensions for image data")

        self._imcache = None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None

    def set_array(self, A):
        """
        Retained for backwards compatibility - use set_data instead

        ACCEPTS: numpy array A or PIL Image"""
        # This also needs to be here to override the inherited
        # cm.ScalarMappable.set_array method so it is not invoked
        # by mistake.

        self.set_data(A)

    def get_interpolation(self):
        """
        Return the interpolation method the image uses when resizing.

        One of 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
        'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom',
        'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', or 'none'.

        """
        return self._interpolation

    def set_interpolation(self, s):
        """
        Set the interpolation method the image uses when resizing.

        if None, use a value from rc setting. If 'none', the image is
        shown as is without interpolating. 'none' is only supported in
        agg, ps and pdf backends and will fall back to 'nearest' mode
        for other backends.

        ACCEPTS: ['nearest' | 'bilinear' | 'bicubic' | 'spline16' |
          'spline36' | 'hanning' | 'hamming' | 'hermite' | 'kaiser' |
          'quadric' | 'catrom' | 'gaussian' | 'bessel' | 'mitchell' |
          'sinc' | 'lanczos' | 'none' |]

        """
        if s is None:
            s = rcParams['image.interpolation']
        s = s.lower()
        if s not in self._interpd:
            raise ValueError('Illegal interpolation string')
        self._interpolation = s

    def set_resample(self, v):
        """
        Set whether or not image resampling is used

        ACCEPTS: True|False
        """
        if v is None:
            v = rcParams['image.resample']
        self._resample = v

    def get_resample(self):
        """Return the image resample boolean"""
        return self._resample

    def set_filternorm(self, filternorm):
        """
        Set whether the resize filter norms the weights -- see
        help for imshow

        ACCEPTS: 0 or 1
        """
        if filternorm:
            self._filternorm = 1
        else:
            self._filternorm = 0

    def get_filternorm(self):
        """Return the filternorm setting"""
        return self._filternorm

    def set_filterrad(self, filterrad):
        """
        Set the resize filter radius only applicable to some
        interpolation schemes -- see help for imshow

        ACCEPTS: positive float
        """
        r = float(filterrad)
        assert(r > 0)
        self._filterrad = r

    def get_filterrad(self):
        """return the filterrad setting"""
        return self._filterrad


class AxesImage(_AxesImageBase):
    def __str__(self):
        return "AxesImage(%g,%g;%gx%g)" % tuple(self.axes.bbox.bounds)

    def __init__(self, ax,
                 cmap=None,
                 norm=None,
                 interpolation=None,
                 origin=None,
                 extent=None,
                 filternorm=1,
                 filterrad=4.0,
                 resample=False,
                 **kwargs
                 ):

        """
        interpolation and cmap default to their rc settings

        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        extent is data axes (left, right, bottom, top) for making image plots
        registered with data plots.  Default is to label the pixel
        centers with the zero-based row and column indices.

        Additional kwargs are matplotlib.artist properties

        """

        self._extent = extent

        _AxesImageBase.__init__(self, ax,
                                cmap=cmap,
                                norm=norm,
                                interpolation=interpolation,
                                origin=origin,
                                filternorm=filternorm,
                                filterrad=filterrad,
                                resample=resample,
                                **kwargs
                                )

    def make_image(self, magnification=1.0):
        if self._A is None:
            raise RuntimeError('You must first set the image'
                               ' array or the image attribute')

        # image is created in the canvas coordinate.
        x1, x2, y1, y2 = self.get_extent()
        trans = self.get_transform()
        xy = trans.transform(np.array([(x1, y1),
                                       (x2, y2),
                                       ]))
        _x1, _y1 = xy[0]
        _x2, _y2 = xy[1]

        transformed_viewLim = mtransforms.TransformedBbox(self.axes.viewLim,
                                                          trans)

        im, xmin, ymin, dxintv, dyintv, sx, sy = \
            self._get_unsampled_image(self._A, [_x1, _x2, _y1, _y2],
                                      transformed_viewLim)

        fc = self.axes.patch.get_facecolor()
        bg = mcolors.colorConverter.to_rgba(fc, 0)
        im.set_bg(*bg)

        # image input dimensions
        im.reset_matrix()
        numrows, numcols = im.get_size()
        if numrows < 1 or numcols < 1:   # out of range
            return None
        im.set_interpolation(self._interpd[self._interpolation])

        im.set_resample(self._resample)

        # the viewport translation
        if dxintv == 0.0:
            tx = 0.0
        else:
            tx = (xmin-transformed_viewLim.x0)/dxintv * numcols
        if dyintv == 0.0:
            ty = 0.0
        else:
            ty = (ymin-transformed_viewLim.y0)/dyintv * numrows

        im.apply_translation(tx, ty)

        l, b, r, t = self.axes.bbox.extents
        widthDisplay = ((round(r*magnification) + 0.5) -
                        (round(l*magnification) - 0.5))
        heightDisplay = ((round(t*magnification) + 0.5) -
                         (round(b*magnification) - 0.5))

        # resize viewport to display
        rx = widthDisplay / numcols
        ry = heightDisplay / numrows
        im.apply_scaling(rx*sx, ry*sy)
        im.resize(int(widthDisplay+0.5), int(heightDisplay+0.5),
                  norm=self._filternorm, radius=self._filterrad)
        return im

    def _check_unsampled_image(self, renderer):
        """
        return True if the image is better to be drawn unsampled.
        """
        if self.get_interpolation() == "none":
            if renderer.option_scale_image():
                return True
            else:
                warnings.warn("The backend (%s) does not support "
                              "interpolation='none'. The image will be "
                              "interpolated with 'nearest` "
                              "mode." % renderer.__class__)

        return False

    def set_extent(self, extent):
        """
        extent is data axes (left, right, bottom, top) for making image plots

        This updates ax.dataLim, and, if autoscaling, sets viewLim
        to tightly fit the image, regardless of dataLim.  Autoscaling
        state is not changed, so following this with ax.autoscale_view
        will redo the autoscaling in accord with dataLim.

        """
        self._extent = extent

        xmin, xmax, ymin, ymax = extent
        corners = (xmin, ymin), (xmax, ymax)
        self.axes.update_datalim(corners)
        if self.axes._autoscaleXon:
            self.axes.set_xlim((xmin, xmax), auto=None)
        if self.axes._autoscaleYon:
            self.axes.set_ylim((ymin, ymax), auto=None)

    def get_extent(self):
        """Get the image extent: left, right, bottom, top"""
        if self._extent is not None:
            return self._extent
        else:
            sz = self.get_size()
            #print 'sz', sz
            numrows, numcols = sz
            if self.origin == 'upper':
                return (-0.5, numcols-0.5, numrows-0.5, -0.5)
            else:
                return (-0.5, numcols-0.5, -0.5, numrows-0.5)


class NonUniformImage(AxesImage):
    def __init__(self, ax, **kwargs):
        """
        kwargs are identical to those for AxesImage, except
        that 'interpolation' defaults to 'nearest', and 'bilinear'
        is the only alternative.
        """
        interp = kwargs.pop('interpolation', 'nearest')
        AxesImage.__init__(self, ax,
                           **kwargs)
        self.set_interpolation(interp)

    def _check_unsampled_image(self, renderer):
        """
        return False. Do not use unsampled image.
        """
        return False

    def make_image(self, magnification=1.0):
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        A = self._A
        if len(A.shape) == 2:
            if A.dtype != np.uint8:
                A = self.to_rgba(A, bytes=True)
                self.is_grayscale = self.cmap.is_gray()
            else:
                A = np.repeat(A[:, :, np.newaxis], 4, 2)
                A[:, :, 3] = 255
                self.is_grayscale = True
        else:
            if A.dtype != np.uint8:
                A = (255*A).astype(np.uint8)
            if A.shape[2] == 3:
                B = np.zeros(tuple(list(A.shape[0:2]) + [4]), np.uint8)
                B[:, :, 0:3] = A
                B[:, :, 3] = 255
                A = B
            self.is_grayscale = False

        x0, y0, v_width, v_height = self.axes.viewLim.bounds
        l, b, r, t = self.axes.bbox.extents
        width = (round(r) + 0.5) - (round(l) - 0.5)
        height = (round(t) + 0.5) - (round(b) - 0.5)
        width *= magnification
        height *= magnification
        im = _image.pcolor(self._Ax, self._Ay, A,
                           height, width,
                           (x0, x0+v_width, y0, y0+v_height),
                           self._interpd[self._interpolation])

        fc = self.axes.patch.get_facecolor()
        bg = mcolors.colorConverter.to_rgba(fc, 0)
        im.set_bg(*bg)
        im.is_grayscale = self.is_grayscale
        return im

    def set_data(self, x, y, A):
        """
        Set the grid for the pixel centers, and the pixel values.

          *x* and *y* are 1-D ndarrays of lengths N and M, respectively,
             specifying pixel centers

          *A* is an (M,N) ndarray or masked array of values to be
            colormapped, or a (M,N,3) RGB array, or a (M,N,4) RGBA
            array.
        """
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        A = cbook.safe_masked_invalid(A)
        if len(x.shape) != 1 or len(y.shape) != 1\
           or A.shape[0:2] != (y.shape[0], x.shape[0]):
            raise TypeError("Axes don't match array shape")
        if len(A.shape) not in [2, 3]:
            raise TypeError("Can only plot 2D or 3D data")
        if len(A.shape) == 3 and A.shape[2] not in [1, 3, 4]:
            raise TypeError("3D arrays must have three (RGB) "
                            "or four (RGBA) color components")
        if len(A.shape) == 3 and A.shape[2] == 1:
            A.shape = A.shape[0:2]
        self._A = A
        self._Ax = x
        self._Ay = y
        self._imcache = None

        # I am adding this in accor with _AxesImageBase.set_data --
        # examples/pylab_examples/image_nonuniform.py was breaking on
        # the call to _get_unsampled_image when the oldxslice attr was
        # accessed - JDH 3/3/2010
        self._oldxslice = None
        self._oldyslice = None

    def set_array(self, *args):
        raise NotImplementedError('Method not supported')

    def set_interpolation(self, s):
        if s is not None and s not in ('nearest', 'bilinear'):
            raise NotImplementedError('Only nearest neighbor and '
                                      'bilinear interpolations are supported')
        AxesImage.set_interpolation(self, s)

    def get_extent(self):
        if self._A is None:
            raise RuntimeError('Must set data first')
        return self._Ax[0], self._Ax[-1], self._Ay[0], self._Ay[-1]

    def set_filternorm(self, s):
        pass

    def set_filterrad(self, s):
        pass

    def set_norm(self, norm):
        if self._A is not None:
            raise RuntimeError('Cannot change colors after loading data')
        cm.ScalarMappable.set_norm(self, norm)

    def set_cmap(self, cmap):
        if self._A is not None:
            raise RuntimeError('Cannot change colors after loading data')
        cm.ScalarMappable.set_cmap(self, cmap)


class PcolorImage(martist.Artist, cm.ScalarMappable):
    """
    Make a pcolor-style plot with an irregular rectangular grid.

    This uses a variation of the original irregular image code,
    and it is used by pcolorfast for the corresponding grid type.
    """
    def __init__(self, ax,
                 x=None,
                 y=None,
                 A=None,
                 cmap=None,
                 norm=None,
                 **kwargs
                 ):
        """
        cmap defaults to its rc setting

        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        Additional kwargs are matplotlib.artist properties

        """
        martist.Artist.__init__(self)
        cm.ScalarMappable.__init__(self, norm, cmap)
        self.axes = ax
        self._rgbacache = None
        # There is little point in caching the image itself because
        # it needs to be remade if the bbox or viewlim change,
        # so caching does help with zoom/pan/resize.
        self.update(kwargs)
        self.set_data(x, y, A)

    def make_image(self, magnification=1.0):
        if self._A is None:
            raise RuntimeError('You must first set the image array')
        fc = self.axes.patch.get_facecolor()
        bg = mcolors.colorConverter.to_rgba(fc, 0)
        bg = (np.array(bg)*255).astype(np.uint8)
        l, b, r, t = self.axes.bbox.extents
        width = (round(r) + 0.5) - (round(l) - 0.5)
        height = (round(t) + 0.5) - (round(b) - 0.5)
        width = width * magnification
        height = height * magnification
        if self._rgbacache is None:
            A = self.to_rgba(self._A, bytes=True)
            self._rgbacache = A
            if self._A.ndim == 2:
                self.is_grayscale = self.cmap.is_gray()
        else:
            A = self._rgbacache
        vl = self.axes.viewLim
        im = _image.pcolor2(self._Ax, self._Ay, A,
                            height,
                            width,
                            (vl.x0, vl.x1, vl.y0, vl.y1),
                            bg)
        im.is_grayscale = self.is_grayscale
        return im

    def changed(self):
        self._rgbacache = None
        cm.ScalarMappable.changed(self)

    @allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible():
            return
        im = self.make_image(renderer.get_image_magnification())
        gc = renderer.new_gc()
        gc.set_clip_rectangle(self.axes.bbox.frozen())
        gc.set_clip_path(self.get_clip_path())
        gc.set_alpha(self.get_alpha())
        renderer.draw_image(gc,
                            round(self.axes.bbox.xmin),
                            round(self.axes.bbox.ymin),
                            im)
        gc.restore()

    def set_data(self, x, y, A):
        A = cbook.safe_masked_invalid(A)
        if x is None:
            x = np.arange(0, A.shape[1]+1, dtype=np.float64)
        else:
            x = np.asarray(x, np.float64).ravel()
        if y is None:
            y = np.arange(0, A.shape[0]+1, dtype=np.float64)
        else:
            y = np.asarray(y, np.float64).ravel()

        if A.shape[:2] != (y.size-1, x.size-1):
            print(A.shape)
            print(y.size)
            print(x.size)
            raise ValueError("Axes don't match array shape")
        if A.ndim not in [2, 3]:
            raise ValueError("A must be 2D or 3D")
        if A.ndim == 3 and A.shape[2] == 1:
            A.shape = A.shape[:2]
        self.is_grayscale = False
        if A.ndim == 3:
            if A.shape[2] in [3, 4]:
                if ((A[:, :, 0] == A[:, :, 1]).all() and
                    (A[:, :, 0] == A[:, :, 2]).all()):
                    self.is_grayscale = True
            else:
                raise ValueError("3D arrays must have RGB or RGBA as last dim")
        self._A = A
        self._Ax = x
        self._Ay = y
        self._rgbacache = None

    def set_array(self, *args):
        raise NotImplementedError('Method not supported')

    def set_alpha(self, alpha):
        """
        Set the alpha value used for blending - not supported on
        all backends

        ACCEPTS: float
        """
        martist.Artist.set_alpha(self, alpha)
        self.update_dict['array'] = True


class FigureImage(martist.Artist, cm.ScalarMappable):
    zorder = 0

    def __init__(self, fig,
                 cmap=None,
                 norm=None,
                 offsetx=0,
                 offsety=0,
                 origin=None,
                 **kwargs
                 ):

        """
        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        kwargs are an optional list of Artist keyword args
        """
        martist.Artist.__init__(self)
        cm.ScalarMappable.__init__(self, norm, cmap)
        if origin is None:
            origin = rcParams['image.origin']
        self.origin = origin
        self.figure = fig
        self.ox = offsetx
        self.oy = offsety
        self.update(kwargs)
        self.magnification = 1.0

    def contains(self, mouseevent):
        """Test whether the mouse event occured within the image."""
        if six.callable(self._contains):
            return self._contains(self, mouseevent)
        xmin, xmax, ymin, ymax = self.get_extent()
        xdata, ydata = mouseevent.x, mouseevent.y
        #print xdata, ydata, xmin, xmax, ymin, ymax
        if xdata is not None and ydata is not None:
            inside = ((xdata >= xmin) and (xdata <= xmax) and
                      (ydata >= ymin) and (ydata <= ymax))
        else:
            inside = False

        return inside, {}

    def get_size(self):
        """Get the numrows, numcols of the input image"""
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        return self._A.shape[:2]

    def get_extent(self):
        """Get the image extent: left, right, bottom, top"""
        numrows, numcols = self.get_size()
        return (-0.5+self.ox, numcols-0.5+self.ox,
                -0.5+self.oy, numrows-0.5+self.oy)

    def set_data(self, A):
        """Set the image array."""
        cm.ScalarMappable.set_array(self, cbook.safe_masked_invalid(A))

    def set_array(self, A):
        """Deprecated; use set_data for consistency with other image types."""
        self.set_data(A)

    def make_image(self, magnification=1.0):
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        x = self.to_rgba(self._A, bytes=True)
        self.magnification = magnification
        # if magnification is not one, we need to resize
        ismag = magnification != 1
        #if ismag: raise RuntimeError
        if ismag:
            isoutput = 0
        else:
            isoutput = 1
        im = _image.frombyte(x, isoutput)
        fc = self.figure.get_facecolor()
        im.set_bg(*mcolors.colorConverter.to_rgba(fc, 0))
        im.is_grayscale = (self.cmap.name == "gray" and
                           len(self._A.shape) == 2)

        if ismag:
            numrows, numcols = self.get_size()
            numrows *= magnification
            numcols *= magnification
            im.set_interpolation(_image.NEAREST)
            im.resize(numcols, numrows)
        if self.origin == 'upper':
            im.flipud_out()

        return im

    @allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible():
            return
        # todo: we should be able to do some cacheing here
        im = self.make_image(renderer.get_image_magnification())
        gc = renderer.new_gc()
        gc.set_clip_rectangle(self.figure.bbox)
        gc.set_clip_path(self.get_clip_path())
        gc.set_alpha(self.get_alpha())
        renderer.draw_image(gc, round(self.ox), round(self.oy), im)
        gc.restore()

    def write_png(self, fname):
        """Write the image to png file with fname"""
        im = self.make_image()
        rows, cols, buffer = im.as_rgba_str()
        _png.write_png(buffer, cols, rows, fname)


class BboxImage(_AxesImageBase):
    """The Image class whose size is determined by the given bbox."""
    def __init__(self, bbox,
                 cmap=None,
                 norm=None,
                 interpolation=None,
                 origin=None,
                 filternorm=1,
                 filterrad=4.0,
                 resample=False,
                 interp_at_native=True,
                 **kwargs
                 ):

        """
        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        interp_at_native is a flag that determines whether or not
        interpolation should still be applied when the image is
        displayed at its native resolution.  A common use case for this
        is when displaying an image for annotational purposes; it is
        treated similarly to Photoshop (interpolation is only used when
        displaying the image at non-native resolutions).


        kwargs are an optional list of Artist keyword args

        """
        _AxesImageBase.__init__(self, ax=None,
                                cmap=cmap,
                                norm=norm,
                                interpolation=interpolation,
                                origin=origin,
                                filternorm=filternorm,
                                filterrad=filterrad,
                                resample=resample,
                                **kwargs
                                )

        self.bbox = bbox
        self.interp_at_native = interp_at_native

    def get_window_extent(self, renderer=None):
        if renderer is None:
            renderer = self.get_figure()._cachedRenderer

        if isinstance(self.bbox, BboxBase):
            return self.bbox
        elif six.callable(self.bbox):
            return self.bbox(renderer)
        else:
            raise ValueError("unknown type of bbox")

    def contains(self, mouseevent):
        """Test whether the mouse event occured within the image."""
        if six.callable(self._contains):
            return self._contains(self, mouseevent)

        if not self.get_visible():  # or self.get_figure()._renderer is None:
            return False, {}

        x, y = mouseevent.x, mouseevent.y
        inside = self.get_window_extent().contains(x, y)

        return inside, {}

    def get_size(self):
        """Get the numrows, numcols of the input image"""
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        return self._A.shape[:2]

    def make_image(self, renderer, magnification=1.0):
        if self._A is None:
            raise RuntimeError('You must first set the image '
                               'array or the image attribute')

        if self._imcache is None:
            if self._A.dtype == np.uint8 and len(self._A.shape) == 3:
                im = _image.frombyte(self._A, 0)
                im.is_grayscale = False
            else:
                if self._rgbacache is None:
                    x = self.to_rgba(self._A, bytes=True)
                    self._rgbacache = x
                else:
                    x = self._rgbacache
                im = _image.frombyte(x, 0)
                if len(self._A.shape) == 2:
                    im.is_grayscale = self.cmap.is_gray()
                else:
                    im.is_grayscale = False
            self._imcache = im

            if self.origin == 'upper':
                im.flipud_in()
        else:
            im = self._imcache

        # image input dimensions
        im.reset_matrix()

        im.set_interpolation(self._interpd[self._interpolation])

        im.set_resample(self._resample)

        l, b, r, t = self.get_window_extent(renderer).extents  # bbox.extents
        widthDisplay = abs(round(r) - round(l))
        heightDisplay = abs(round(t) - round(b))
        widthDisplay *= magnification
        heightDisplay *= magnification

        numrows, numcols = self._A.shape[:2]

        if (not self.interp_at_native and
            widthDisplay == numcols and heightDisplay == numrows):
            im.set_interpolation(0)

        # resize viewport to display
        rx = widthDisplay / numcols
        ry = heightDisplay / numrows
        #im.apply_scaling(rx*sx, ry*sy)
        im.apply_scaling(rx, ry)
        #im.resize(int(widthDisplay+0.5), int(heightDisplay+0.5),
        #          norm=self._filternorm, radius=self._filterrad)
        im.resize(int(widthDisplay), int(heightDisplay),
                  norm=self._filternorm, radius=self._filterrad)
        return im

    @allow_rasterization
    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible():
            return
        # todo: we should be able to do some cacheing here
        image_mag = renderer.get_image_magnification()
        im = self.make_image(renderer, image_mag)
        x0, y0, x1, y1 = self.get_window_extent(renderer).extents
        gc = renderer.new_gc()
        self._set_gc_clip(gc)
        gc.set_alpha(self.get_alpha())
        #gc.set_clip_path(self.get_clip_path())

        l = np.min([x0, x1])
        b = np.min([y0, y1])
        renderer.draw_image(gc, round(l), round(b), im)
        gc.restore()


def imread(fname, format=None):
    """
    Read an image from a file into an array.

    *fname* may be a string path or a Python file-like object.  If
    using a file object, it must be opened in binary mode.

    If *format* is provided, will try to read file of that type,
    otherwise the format is deduced from the filename.  If nothing can
    be deduced, PNG is tried.

    Return value is a :class:`numpy.array`.  For grayscale images, the
    return array is MxN.  For RGB images, the return value is MxNx3.
    For RGBA images the return value is MxNx4.

    matplotlib can only read PNGs natively, but if `PIL
    <http://www.pythonware.com/products/pil/>`_ is installed, it will
    use it to load the image and return an array (if possible) which
    can be used with :func:`~matplotlib.pyplot.imshow`.
    """

    def pilread(fname):
        """try to load the image with PIL or return None"""
        try:
            from PIL import Image
        except ImportError:
            return None
        if cbook.is_string_like(fname):
            # force close the file after reading the image
            with open(fname, "rb") as fh:
                image = Image.open(fh)
                return pil_to_array(image)
        else:
            image = Image.open(fname)
            return pil_to_array(image)

    handlers = {'png': _png.read_png, }
    if format is None:
        if cbook.is_string_like(fname):
            basename, ext = os.path.splitext(fname)
            ext = ext.lower()[1:]
        elif hasattr(fname, 'name'):
            basename, ext = os.path.splitext(fname.name)
            ext = ext.lower()[1:]
        else:
            ext = 'png'
    else:
        ext = format

    if ext not in handlers:
        im = pilread(fname)
        if im is None:
            raise ValueError('Only know how to handle extensions: %s; '
                             'with PIL installed matplotlib can handle '
                             'more images' % list(six.iterkeys(handlers.keys)))
        return im

    handler = handlers[ext]

    # To handle Unicode filenames, we pass a file object to the PNG
    # reader extension, since Python handles them quite well, but it's
    # tricky in C.
    if cbook.is_string_like(fname):
        with open(fname, 'rb') as fd:
            return handler(fd)
    else:
        return handler(fname)


def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None,
           origin=None, dpi=100):
    """
    Save an array as in image file.

    The output formats available depend on the backend being used.

    Arguments:
      *fname*:
        A string containing a path to a filename, or a Python file-like object.
        If *format* is *None* and *fname* is a string, the output
        format is deduced from the extension of the filename.
      *arr*:
        An MxN (luminance), MxNx3 (RGB) or MxNx4 (RGBA) array.
    Keyword arguments:
      *vmin*/*vmax*: [ None | scalar ]
        *vmin* and *vmax* set the color scaling for the image by fixing the
        values that map to the colormap color limits. If either *vmin*
        or *vmax* is None, that limit is determined from the *arr*
        min/max value.
      *cmap*:
        cmap is a colors.Colormap instance, eg cm.jet.
        If None, default to the rc image.cmap value.
      *format*:
        One of the file extensions supported by the active
        backend.  Most backends support png, pdf, ps, eps and svg.
      *origin*
        [ 'upper' | 'lower' ] Indicates where the [0,0] index of
        the array is in the upper left or lower left corner of
        the axes. Defaults to the rc image.origin value.
      *dpi*
        The DPI to store in the metadata of the file.  This does not affect the
        resolution of the output image.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    figsize = [x / float(dpi) for x in (arr.shape[1], arr.shape[0])]
    fig = Figure(figsize=figsize, dpi=dpi, frameon=False)
    canvas = FigureCanvas(fig)
    im = fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=dpi, format=format, transparent=True)


def pil_to_array(pilImage):
    """
    Load a PIL image and return it as a numpy array.  For grayscale
    images, the return array is MxN.  For RGB images, the return value
    is MxNx3.  For RGBA images the return value is MxNx4
    """
    def toarray(im, dtype=np.uint8):
        """Return a 1D array of dtype."""
        # Pillow wants us to use "tobytes"
        if hasattr(im, 'tobytes'):
            x_str = im.tobytes('raw', im.mode)
        else:
            x_str = im.tostring('raw', im.mode)
        x = np.fromstring(x_str, dtype)
        return x

    if pilImage.mode in ('RGBA', 'RGBX'):
        im = pilImage  # no need to convert images
    elif pilImage.mode == 'L':
        im = pilImage  # no need to luminance images
        # return MxN luminance array
        x = toarray(im)
        x.shape = im.size[1], im.size[0]
        return x
    elif pilImage.mode == 'RGB':
        #return MxNx3 RGB array
        im = pilImage  # no need to RGB images
        x = toarray(im)
        x.shape = im.size[1], im.size[0], 3
        return x
    elif pilImage.mode.startswith('I;16'):
        # return MxN luminance array of uint16
        im = pilImage
        if im.mode.endswith('B'):
            x = toarray(im, '>u2')
        else:
            x = toarray(im, '<u2')
        x.shape = im.size[1], im.size[0]
        return x.astype('=u2')
    else:  # try to convert to an rgba image
        try:
            im = pilImage.convert('RGBA')
        except ValueError:
            raise RuntimeError('Unknown image mode')

    # return MxNx4 RGBA array
    x = toarray(im)
    x.shape = im.size[1], im.size[0], 4
    return x


def thumbnail(infile, thumbfile, scale=0.1, interpolation='bilinear',
              preview=False):
    """
    make a thumbnail of image in *infile* with output filename
    *thumbfile*.

      *infile* the image file -- must be PNG or PIL readable if you
         have `PIL <http://www.pythonware.com/products/pil/>`_ installed

      *thumbfile*
        the thumbnail filename

      *scale*
        the scale factor for the thumbnail

      *interpolation*
        the interpolation scheme used in the resampling


      *preview*
        if True, the default backend (presumably a user interface
        backend) will be used which will cause a figure to be raised
        if :func:`~matplotlib.pyplot.show` is called.  If it is False,
        a pure image backend will be used depending on the extension,
        'png'->FigureCanvasAgg, 'pdf'->FigureCanvasPdf,
        'svg'->FigureCanvasSVG


    See examples/misc/image_thumbnail.py.

    .. htmlonly::

        :ref:`misc-image_thumbnail`

    Return value is the figure instance containing the thumbnail

    """
    basedir, basename = os.path.split(infile)
    baseout, extout = os.path.splitext(thumbfile)

    im = imread(infile)
    rows, cols, depth = im.shape

    # this doesn't really matter, it will cancel in the end, but we
    # need it for the mpl API
    dpi = 100

    height = float(rows)/dpi*scale
    width = float(cols)/dpi*scale

    extension = extout.lower()

    if preview:
        # let the UI backend do everything
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(width, height), dpi=dpi)
    else:
        if extension == '.png':
            from matplotlib.backends.backend_agg \
                import FigureCanvasAgg as FigureCanvas
        elif extension == '.pdf':
            from matplotlib.backends.backend_pdf \
                import FigureCanvasPdf as FigureCanvas
        elif extension == '.svg':
            from matplotlib.backends.backend_svg \
                import FigureCanvasSVG as FigureCanvas
        else:
            raise ValueError("Can only handle "
                             "extensions 'png', 'svg' or 'pdf'")

        from matplotlib.figure import Figure
        fig = Figure(figsize=(width, height), dpi=dpi)
        canvas = FigureCanvas(fig)

    ax = fig.add_axes([0, 0, 1, 1], aspect='auto',
                      frameon=False, xticks=[], yticks=[])

    basename, ext = os.path.splitext(basename)
    ax.imshow(im, aspect='auto', resample=True, interpolation=interpolation)
    fig.savefig(thumbfile, dpi=dpi)
    return fig
