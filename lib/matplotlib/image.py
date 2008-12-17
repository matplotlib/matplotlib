"""
The image module supports basic image loading, rescaling and display
operations.

"""
from __future__ import division
import os, warnings

import numpy as np
from numpy import ma

from matplotlib import rcParams
from matplotlib import artist as martist
from matplotlib import colors as mcolors
from matplotlib import cm

# For clarity, names from _image are given explicitly in this module:
from matplotlib import _image
from matplotlib import _png

# For user convenience, the names from _image are also imported into
# the image namespace:
from matplotlib._image import *

class AxesImage(martist.Artist, cm.ScalarMappable):
    zorder = 1
    # map interpolation strings to module constants
    _interpd = {
        'nearest'  : _image.NEAREST,
        'bilinear' : _image.BILINEAR,
        'bicubic'  : _image.BICUBIC,
        'spline16' : _image.SPLINE16,
        'spline36' : _image.SPLINE36,
        'hanning'  : _image.HANNING,
        'hamming'  : _image.HAMMING,
        'hermite'  : _image.HERMITE,
        'kaiser'   : _image.KAISER,
        'quadric'  : _image.QUADRIC,
        'catrom'   : _image.CATROM,
        'gaussian' : _image.GAUSSIAN,
        'bessel'   : _image.BESSEL,
        'mitchell' : _image.MITCHELL,
        'sinc'     : _image.SINC,
        'lanczos'  : _image.LANCZOS,
        'blackman' : _image.BLACKMAN,
    }

    # reverse interp dict
    _interpdr = dict([ (v,k) for k,v in _interpd.items()])

    interpnames = _interpd.keys()

    def __str__(self):
        return "AxesImage(%g,%g;%gx%g)" % tuple(self.axes.bbox.bounds)

    def __init__(self, ax,
                 cmap = None,
                 norm = None,
                 interpolation=None,
                 origin=None,
                 extent=None,
                 filternorm=1,
                 filterrad=4.0,
                 resample = False,
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

        if origin is None: origin = rcParams['image.origin']
        self.origin = origin
        self._extent = extent
        self.set_filternorm(filternorm)
        self.set_filterrad(filterrad)
        self._filterrad = filterrad

        self.set_interpolation(interpolation)
        self.set_resample(resample)
        self.axes = ax

        self._imcache = None

        self.update(kwargs)

    def get_size(self):
        'Get the numrows, numcols of the input image'
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
        if self._A is None:
            raise RuntimeError('You must first set the image array or the image attribute')

        xmin, xmax, ymin, ymax = self.get_extent()
        dxintv = xmax-xmin
        dyintv = ymax-ymin

        # the viewport scale factor
        sx = dxintv/self.axes.viewLim.width
        sy = dyintv/self.axes.viewLim.height
        numrows, numcols = self._A.shape[:2]
        if sx > 2:
            x0 = (self.axes.viewLim.x0-xmin)/dxintv * numcols
            ix0 = max(0, int(x0 - self._filterrad))
            x1 = (self.axes.viewLim.x1-xmin)/dxintv * numcols
            ix1 = min(numcols, int(x1 + self._filterrad))
            xslice = slice(ix0, ix1)
            xmin_old = xmin
            xmin = xmin_old + ix0*dxintv/numcols
            xmax = xmin_old + ix1*dxintv/numcols
            dxintv = xmax - xmin
            sx = dxintv/self.axes.viewLim.width
        else:
            xslice = slice(0, numcols)

        if sy > 2:
            y0 = (self.axes.viewLim.y0-ymin)/dyintv * numrows
            iy0 = max(0, int(y0 - self._filterrad))
            y1 = (self.axes.viewLim.y1-ymin)/dyintv * numrows
            iy1 = min(numrows, int(y1 + self._filterrad))
            if self.origin == 'upper':
                yslice = slice(numrows-iy1, numrows-iy0)
            else:
                yslice = slice(iy0, iy1)
            ymin_old = ymin
            ymin = ymin_old + iy0*dyintv/numrows
            ymax = ymin_old + iy1*dyintv/numrows
            dyintv = ymax - ymin
            sy = dyintv/self.axes.viewLim.height
        else:
            yslice = slice(0, numrows)

        if xslice != self._oldxslice or yslice != self._oldyslice:
            self._imcache = None
            self._oldxslice = xslice
            self._oldyslice = yslice

        if self._imcache is None:
            if self._A.dtype == np.uint8 and len(self._A.shape) == 3:
                im = _image.frombyte(self._A[yslice,xslice,:], 0)
                im.is_grayscale = False
            else:
                if self._rgbacache is None:
                    x = self.to_rgba(self._A, self._alpha)
                    self._rgbacache = x
                else:
                    x = self._rgbacache
                im = _image.fromarray(x[yslice,xslice], 0)
                if len(self._A.shape) == 2:
                    im.is_grayscale = self.cmap.is_gray()
                else:
                    im.is_grayscale = False
            self._imcache = im

            if self.origin=='upper':
                im.flipud_in()
        else:
            im = self._imcache

        fc = self.axes.patch.get_facecolor()
        bg = mcolors.colorConverter.to_rgba(fc, 0)
        im.set_bg( *bg)

        # image input dimensions
        im.reset_matrix()
        numrows, numcols = im.get_size()

        im.set_interpolation(self._interpd[self._interpolation])

        im.set_resample(self._resample)

        # the viewport translation
        tx = (xmin-self.axes.viewLim.x0)/dxintv * numcols
        ty = (ymin-self.axes.viewLim.y0)/dyintv * numrows

        l, b, r, t = self.axes.bbox.extents
        widthDisplay = (round(r) + 0.5) - (round(l) - 0.5)
        heightDisplay = (round(t) + 0.5) - (round(b) - 0.5)
        widthDisplay *= magnification
        heightDisplay *= magnification
        im.apply_translation(tx, ty)

        # resize viewport to display
        rx = widthDisplay / numcols
        ry = heightDisplay  / numrows
        im.apply_scaling(rx*sx, ry*sy)
        im.resize(int(widthDisplay+0.5), int(heightDisplay+0.5),
                  norm=self._filternorm, radius=self._filterrad)
        return im


    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible(): return
        if (self.axes.get_xscale() != 'linear' or
            self.axes.get_yscale() != 'linear'):
            warnings.warn("Images are not supported on non-linear axes.")
        im = self.make_image(renderer.get_image_magnification())
        im._url = self.get_url()
        l, b, widthDisplay, heightDisplay = self.axes.bbox.bounds
        clippath, affine = self.get_transformed_clip_path_and_affine()
        renderer.draw_image(round(l), round(b), im, self.axes.bbox.frozen(),
                            clippath, affine)

    def contains(self, mouseevent):
        """Test whether the mouse event occured within the image.
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        # TODO: make sure this is consistent with patch and patch
        # collection on nonlinear transformed coordinates.
        # TODO: consider returning image coordinates (shouldn't
        # be too difficult given that the image is rectilinear
        x, y = mouseevent.xdata, mouseevent.ydata
        xmin, xmax, ymin, ymax = self.get_extent()
        if xmin > xmax:
            xmin,xmax = xmax,xmin
        if ymin > ymax:
            ymin,ymax = ymax,ymin
        #print x, y, xmin, xmax, ymin, ymax
        if x is not None and y is not None:
            inside = x>=xmin and x<=xmax and y>=ymin and y<=ymax
        else:
            inside = False

        return inside,{}

    def write_png(self, fname, noscale=False):
        """Write the image to png file with fname"""
        im = self.make_image()
        if noscale:
            numrows, numcols = im.get_size()
            im.reset_matrix()
            im.set_interpolation(0)
            im.resize(numcols, numrows)
        im.flipud_out()
        rows, cols, buffer = im.as_rgba_str()
        _png.write_png(buffer, cols, rows, fname)

    def set_data(self, A, shape=None):
        """
        Set the image array

        ACCEPTS: numpy/PIL Image A"""
        # check if data is PIL Image without importing Image
        if hasattr(A,'getpixel'):
            self._A = pil_to_array(A)
        elif ma.isMA(A):
            self._A = A
        else:
            self._A = np.asarray(A) # assume array

        if self._A.dtype != np.uint8 and not np.can_cast(self._A.dtype, np.float):
            raise TypeError("Image data can not convert to float")

        if (self._A.ndim not in (2, 3) or
            (self._A.ndim == 3 and self._A.shape[-1] not in (3, 4))):
            raise TypeError("Invalid dimensions for image data")

        self._imcache =None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None

    def set_array(self, A):
        """
        retained for backwards compatibility - use set_data instead

        ACCEPTS: numpy array A or PIL Image"""
        # This also needs to be here to override the inherited
        # cm.ScalarMappable.set_array method so it is not invoked
        # by mistake.

        self.set_data(A)



    def set_extent(self, extent):
        """extent is data axes (left, right, bottom, top) for making image plots
        """
        self._extent = extent

        xmin, xmax, ymin, ymax = extent
        corners = (xmin, ymin), (xmax, ymax)
        self.axes.update_datalim(corners)
        if self.axes._autoscaleon:
            self.axes.set_xlim((xmin, xmax))
            self.axes.set_ylim((ymin, ymax))

    def get_interpolation(self):
        """
        Return the interpolation method the image uses when resizing.

        One of 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning',
        'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
        'bessel', 'mitchell', 'sinc', 'lanczos',
        """
        return self._interpolation

    def set_interpolation(self, s):
        """
        Set the interpolation method the image uses when resizing.

        ACCEPTS: ['nearest' | 'bilinear' | 'bicubic' | 'spline16' |
          'spline36' | 'hanning' | 'hamming' | 'hermite' | 'kaiser' |
          'quadric' | 'catrom' | 'gaussian' | 'bessel' | 'mitchell' |
          'sinc' | 'lanczos' | ]

        """
        if s is None: s = rcParams['image.interpolation']
        s = s.lower()
        if s not in self._interpd:
            raise ValueError('Illegal interpolation string')
        self._interpolation = s

    def set_resample(self, v):
        if v is None: v = rcParams['image.resample']
        self._resample = v

    def get_interpolation(self):
        return self._resample

    def get_extent(self):
        'get the image extent: left, right, bottom, top'
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

    def set_filternorm(self, filternorm):
        """Set whether the resize filter norms the weights -- see
        help for imshow

        ACCEPTS: 0 or 1
        """
        if filternorm:
            self._filternorm = 1
        else:
            self._filternorm = 0

    def get_filternorm(self):
        'return the filternorm setting'
        return self._filternorm

    def set_filterrad(self, filterrad):
        """Set the resize filter radius only applicable to some
        interpolation schemes -- see help for imshow

        ACCEPTS: positive float
        """
        r = float(filterrad)
        assert(r>0)
        self._filterrad = r

    def get_filterrad(self):
        'return the filterrad setting'
        return self._filterrad


class NonUniformImage(AxesImage):
    def __init__(self, ax,
                 **kwargs
                ):
        interp = kwargs.pop('interpolation', 'nearest')
        AxesImage.__init__(self, ax,
                           **kwargs)
        AxesImage.set_interpolation(self, interp)

    def make_image(self, magnification=1.0):
        if self._A is None:
            raise RuntimeError('You must first set the image array')
        x0, y0, v_width, v_height = self.axes.viewLim.bounds
        l, b, r, t = self.axes.bbox.extents
        width = (round(r) + 0.5) - (round(l) - 0.5)
        height = (round(t) + 0.5) - (round(b) - 0.5)
        width *= magnification
        height *= magnification
        im = _image.pcolor(self._Ax, self._Ay, self._A,
                           height, width,
                           (x0, x0+v_width, y0, y0+v_height),
                           self._interpd[self._interpolation])

        fc = self.axes.patch.get_facecolor()
        bg = mcolors.colorConverter.to_rgba(fc, 0)
        im.set_bg(*bg)
        im.is_grayscale = self.is_grayscale
        return im

    def set_data(self, x, y, A):
        x = np.asarray(x,np.float32)
        y = np.asarray(y,np.float32)
        if not ma.isMA(A):
            A = np.asarray(A)
        if len(x.shape) != 1 or len(y.shape) != 1\
           or A.shape[0:2] != (y.shape[0], x.shape[0]):
            raise TypeError("Axes don't match array shape")
        if len(A.shape) not in [2, 3]:
            raise TypeError("Can only plot 2D or 3D data")
        if len(A.shape) == 3 and A.shape[2] not in [1, 3, 4]:
            raise TypeError("3D arrays must have three (RGB) or four (RGBA) color components")
        if len(A.shape) == 3 and A.shape[2] == 1:
            A.shape = A.shape[0:2]
        if len(A.shape) == 2:
            if A.dtype != np.uint8:
                A = (self.cmap(self.norm(A))*255).astype(np.uint8)
                self.is_grayscale = self.cmap.is_gray()
            else:
                A = np.repeat(A[:,:,np.newaxis], 4, 2)
                A[:,:,3] = 255
                self.is_grayscale = True
        else:
            if A.dtype != np.uint8:
                A = (255*A).astype(np.uint8)
            if A.shape[2] == 3:
                B = zeros(tuple(list(A.shape[0:2]) + [4]), np.uint8)
                B[:,:,0:3] = A
                B[:,:,3] = 255
                A = B
            self.is_grayscale = False
        self._A = A
        self._Ax = x
        self._Ay = y
        self._imcache = None

    def set_array(self, *args):
        raise NotImplementedError('Method not supported')

    def set_interpolation(self, s):
        if s != None and not s in ('nearest','bilinear'):
            raise NotImplementedError('Only nearest neighbor and bilinear interpolations are supported')
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
        cm.ScalarMappable.set_cmap(self, norm)

class PcolorImage(martist.Artist, cm.ScalarMappable):
    '''
    Make a pcolor-style plot with an irregular rectangular grid.

    This uses a variation of the original irregular image code,
    and it is used by pcolorfast for the corresponding grid type.
    '''
    def __init__(self, ax,
                 x=None,
                 y=None,
                 A=None,
                 cmap = None,
                 norm = None,
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
        if self.check_update('array'):
            A = self.to_rgba(self._A, alpha=self._alpha, bytes=True)
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

    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible(): return
        im = self.make_image(renderer.get_image_magnification())
        renderer.draw_image(round(self.axes.bbox.xmin),
                            round(self.axes.bbox.ymin),
                            im,
                            self.axes.bbox.frozen(),
                            *self.get_transformed_clip_path_and_affine())


    def set_data(self, x, y, A):
        if not ma.isMA(A):
            A = np.asarray(A)
        if x is None:
            x = np.arange(0, A.shape[1]+1, dtype=np.float64)
        else:
            x = np.asarray(x, np.float64).ravel()
        if y is None:
            y = np.arange(0, A.shape[0]+1, dtype=np.float64)
        else:
            y = np.asarray(y, np.float64).ravel()

        if A.shape[:2] != (y.size-1, x.size-1):
            print A.shape
            print y.size
            print x.size
            raise ValueError("Axes don't match array shape")
        if A.ndim not in [2, 3]:
            raise ValueError("A must be 2D or 3D")
        if A.ndim == 3 and A.shape[2] == 1:
            A.shape = A.shape[:2]
        self.is_grayscale = False
        if A.ndim == 3:
            if A.shape[2] in [3, 4]:
                if (A[:,:,0] == A[:,:,1]).all() and (A[:,:,0] == A[:,:,2]).all():
                    self.is_grayscale = True
            else:
                raise ValueError("3D arrays must have RGB or RGBA as last dim")
        self._A = A
        self._Ax = x
        self._Ay = y
        self.update_dict['array'] = True

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
    zorder = 1
    def __init__(self, fig,
                 cmap = None,
                 norm = None,
                 offsetx = 0,
                 offsety = 0,
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
        if origin is None: origin = rcParams['image.origin']
        self.origin = origin
        self.figure = fig
        self.ox = offsetx
        self.oy = offsety
        self.update(kwargs)
        self.magnification = 1.0

    def contains(self, mouseevent):
        """Test whether the mouse event occured within the image.
        """
        if callable(self._contains): return self._contains(self,mouseevent)
        xmin, xmax, ymin, ymax = self.get_extent()
        xdata, ydata = mouseevent.x, mouseevent.y
        #print xdata, ydata, xmin, xmax, ymin, ymax
        if xdata is not None and ydata is not None:
            inside = xdata>=xmin and xdata<=xmax and ydata>=ymin and ydata<=ymax
        else:
            inside = False

        return inside,{}

    def get_size(self):
        'Get the numrows, numcols of the input image'
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        return self._A.shape[:2]

    def get_extent(self):
        'get the image extent: left, right, bottom, top'
        numrows, numcols = self.get_size()
        return (-0.5+self.ox, numcols-0.5+self.ox,
                -0.5+self.oy, numrows-0.5+self.oy)

    def make_image(self, magnification=1.0):
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        x = self.to_rgba(self._A, self._alpha)
        self.magnification = magnification
        # if magnification is not one, we need to resize
        ismag = magnification!=1
        #if ismag: raise RuntimeError
        if ismag:
            isoutput = 0
        else:
            isoutput = 1
        im = _image.fromarray(x, isoutput)
        fc = self.figure.get_facecolor()
        im.set_bg( *mcolors.colorConverter.to_rgba(fc, 0) )
        im.is_grayscale = (self.cmap.name == "gray" and
                           len(self._A.shape) == 2)

        if ismag:
            numrows, numcols = self.get_size()
            numrows *= magnification
            numcols *= magnification
            im.set_interpolation(_image.NEAREST)
            im.resize(numcols, numrows)
        if self.origin=='upper':
            im.flipud_out()

        return im

    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible(): return
        # todo: we should be able to do some cacheing here
        im = self.make_image(renderer.get_image_magnification())
        renderer.draw_image(round(self.ox), round(self.oy), im, self.figure.bbox,
                            *self.get_transformed_clip_path_and_affine())

    def write_png(self, fname):
        """Write the image to png file with fname"""
        im = self.make_image()
        rows, cols, buffer = im.as_rgba_str()
        _png.write_png(buffer, cols, rows, fname)

def imread(fname):
    """
    Return image file in *fname* as :class:`numpy.array`.

    Return value is a :class:`numpy.array`.  For grayscale images, the
    return array is MxN.  For RGB images, the return value is MxNx3.
    For RGBA images the return value is MxNx4.

    matplotlib can only read PNGs natively, but if `PIL
    <http://www.pythonware.com/products/pil/>`_ is installed, it will
    use it to load the image and return an array (if possible) which
    can be used with :func:`~matplotlib.pyplot.imshow`.

    TODO: support RGB and grayscale return values in _image.readpng
    """

    def pilread():
        'try to load the image with PIL or return None'
        try: import Image
        except ImportError: return None
        image = Image.open( fname )
        return pil_to_array(image)


    handlers = {'png' :_png.read_png,
                }
    basename, ext = os.path.splitext(fname)
    ext = ext.lower()[1:]

    if ext not in handlers.keys():
        im = pilread()
        if im is None:
            raise ValueError('Only know how to handle extensions: %s; with PIL installed matplotlib can handle more images' % handlers.keys())
        return im

    handler = handlers[ext]
    return handler(fname)



def pil_to_array( pilImage ):
    """
    load a PIL image and return it as a numpy array of uint8.  For
    grayscale images, the return array is MxN.  For RGB images, the
    return value is MxNx3.  For RGBA images the return value is MxNx4
    """
    def toarray(im):
        'return a 1D array of floats'
        x_str = im.tostring('raw',im.mode,0,-1)
        x = np.fromstring(x_str,np.uint8)
        return x

    if pilImage.mode in ('RGBA', 'RGBX'):
        im = pilImage # no need to convert images
    elif pilImage.mode=='L':
        im = pilImage # no need to luminance images
        # return MxN luminance array
        x = toarray(im)
        x.shape = im.size[1], im.size[0]
        return x
    elif pilImage.mode=='RGB':
        #return MxNx3 RGB array
        im = pilImage # no need to RGB images
        x = toarray(im)
        x.shape = im.size[1], im.size[0], 3
        return x

    else: # try to convert to an rgba image
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
        'png'->FigureCanvasAgg, 'pdf'->FigureCanvasPDF,
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
        if extension=='.png':
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        elif extension=='.pdf':
            from matplotlib.backends.backend_pdf import FigureCanvasPDF as FigureCanvas
        elif extension=='.svg':
            from matplotlib.backends.backend_svg import FigureCanvasSVG as FigureCanvas
        else:
            raise ValueError("Can only handle extensions 'png', 'svg' or 'pdf'")

        from matplotlib.figure import Figure
        fig = Figure(figsize=(width, height), dpi=dpi)
        canvas = FigureCanvas(fig)

    ax = fig.add_axes([0,0,1,1], aspect='auto', frameon=False, xticks=[], yticks=[])

    basename, ext = os.path.splitext(basename)
    ax.imshow(im, aspect='auto', resample=True, interpolation='bilinear')
    fig.savefig(thumbfile, dpi=dpi)
    return fig
