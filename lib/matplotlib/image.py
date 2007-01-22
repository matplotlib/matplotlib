"""
The image module supports basic image loading, rescaling and display
operations.

"""
from __future__ import division
import sys, os
from matplotlib import rcParams
from artist import Artist
from colors import colorConverter
import cm
import numerix
import numerix.ma as ma
from numerix import arange, asarray, UInt8, Float32, repeat, NewAxis, typecode
import _image



class AxesImage(Artist, cm.ScalarMappable):

    def __init__(self, ax,
                 cmap = None,
                 norm = None,
                 interpolation=None,
                 origin=None,
                 extent=None,
                 filternorm=1,
                 filterrad=4.0,
                 **kwargs
                 ):

        """
        interpolation and cmap default to their rc settings

        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        extent is a data xmin, xmax, ymin, ymax for making image plots
        registered with data plots.  Default is the image dimensions
        in pixels

        Additional kwargs are matplotlib.artist properties

        """
        Artist.__init__(self)
        cm.ScalarMappable.__init__(self, norm, cmap)

        if origin is None: origin = rcParams['image.origin']
        self.origin = origin
        self._extent = extent
        self.set_filternorm(filternorm)
        self.set_filterrad(filterrad)


        # map interpolation strings to module constants
        self._interpd = {
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
        self._interpdr = dict([ (v,k) for k,v in self._interpd.items()])

        if interpolation is None: interpolation = rcParams['image.interpolation']

        self.set_interpolation(interpolation)
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
        Artist.set_alpha(self, alpha)
        self._imcache = None

    def changed(self):
        """
        Call this whenever the mappable is changed so observers can
        update state
        """
        self._imcache = None
        cm.ScalarMappable.changed(self)

        
    def make_image(self, magnification=1.0):
        if self._A is None:
            raise RuntimeError('You must first set the image array or the image attribute')

        if self._imcache is None:
            if typecode(self._A) == UInt8:
                im = _image.frombyte(self._A, 0)
                im.is_grayscale = False
            else:
                x = self.to_rgba(self._A, self._alpha)
                im = _image.fromarray(x, 0)
                if len(self._A.shape) == 2:
                    im.is_grayscale = self.cmap.is_gray()
                else:
                    im.is_grayscale = False
            self._imcache = im

            if self.origin=='upper':
                im.flipud_in()
        else:
            im = self._imcache

        bg = colorConverter.to_rgba(self.axes.get_frame().get_facecolor(), 0)
        im.set_bg( *bg)

        # image input dimensions
        im.reset_matrix()
        numrows, numcols = im.get_size()

        im.set_interpolation(self._interpd[self._interpolation])

        xmin, xmax, ymin, ymax = self.get_extent()
        dxintv = xmax-xmin
        dyintv = ymax-ymin

        # the viewport scale factor
        sx = dxintv/self.axes.viewLim.width()
        sy = dyintv/self.axes.viewLim.height()

        if im.get_interpolation()!=_image.NEAREST:
            im.apply_translation(-1, -1)

        # the viewport translation
        tx = (xmin-self.axes.viewLim.xmin())/dxintv * numcols
        ty = (ymin-self.axes.viewLim.ymin())/dyintv * numrows

        l, b, widthDisplay, heightDisplay = self.axes.bbox.get_bounds()
        widthDisplay *= magnification
        heightDisplay *= magnification

        im.apply_translation(tx, ty)
        im.apply_scaling(sx, sy)

        # resize viewport to display
        rx = widthDisplay / numcols
        ry = heightDisplay  / numrows
        im.apply_scaling(rx, ry)

        im.resize(int(widthDisplay+0.5), int(heightDisplay+0.5),
                  norm=self._filternorm, radius=self._filterrad)

        return im


    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible(): return
        im = self.make_image(renderer.get_image_magnification())
        l, b, widthDisplay, heightDisplay = self.axes.bbox.get_bounds()
        renderer.draw_image(l, b, im, self.axes.bbox)

    def pick(self, mouseevent):
        'return true if the data coords of mouse click are within the extent of the image'
        if not self.pickable(): return 
        picker = self.get_picker()
        if callable(picker):
            hit, props = picker(self, mouseevent)
            if hit:
                self.figure.canvas.pick_event(mouseevent, self, **props)
        elif picker:
            xmin, xmax, ymin, ymax = self.get_extent()
            xdata, ydata = mouseevent.xdata, mouseevent.ydata
            #print xdata, ydata, xmin, xmax, ymin, ymax
            if (xdata is not None and ydata is not None and
                xdata>=xmin and xdata<=xmax and ydata>=ymin and ydata<=ymax):
                self.figure.canvas.pick_event(mouseevent, self)
            
        
    def write_png(self, fname, noscale=False):
        """Write the image to png file with fname"""
        im = self.make_image()
        if noscale:
            numrows,numcols = im.get_size()
            im.reset_matrix()
            im.set_interpolation(0)
            im.resize(numcols, numrows)
        im.flipud_out()
        im.write_png(fname)

    def set_data(self, A, shape=None):
        """
        Set the image array

        ACCEPTS: numeric/numarray/PIL Image A"""
        # check if data is PIL Image without importing Image
        if hasattr(A,'getpixel'): X = pil_to_array(A)
        else: X = ma.asarray(A) # assume array

        if (typecode(X) != UInt8
            or len(X.shape) != 3
            or X.shape[2] > 4
            or X.shape[2] < 3):
            cm.ScalarMappable.set_array(self, X)
        else:
            self._A = X

        self._imcache =None

    def set_array(self, A):
        """
        retained for backwards compatibility - use set_data instead

        ACCEPTS: numeric/numarray/PIL Image A"""


        self.set_data(A)

    def get_interpolation(self):
        """
        Return the interpolation method the image uses when resizing.

        One of

        'bicubic', 'bilinear', 'blackman100', 'blackman256', 'blackman64',
        'nearest', 'sinc144', 'sinc256', 'sinc64', 'spline16', 'spline36'
        """
        return self._interpolation

    def set_interpolation(self, s):
        """
        Set the interpolation method the image uses when resizing.

        ACCEPTS: ['bicubic' | 'bilinear' | 'blackman100' | 'blackman256' | 'blackman64', 'nearest' | 'sinc144' | 'sinc256' | 'sinc64' | 'spline16' | 'spline36']
        """

        s = s.lower()
        if not self._interpd.has_key(s):
            raise ValueError('Illegal interpolation string')
        self._interpolation = s

    def get_extent(self):
        'get the image extent: xmin, xmax, ymin, ymax'
        if self._extent is not None:
            return self._extent
        else:
            numrows, numcols = self.get_size()
            iwidth, iheight = numcols, numrows
            #return 0, width, 0, height
            tmp, tmp, dwidth, dheight = self.axes.bbox.get_bounds()
            sx = dwidth  / iwidth
            sy = dheight / iheight

            return 0, 1.0/sx*dwidth, 0, 1.0/sy*dheight

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


class NonUniformImage(AxesImage):
    def __init__(self, ax,
                 cmap = None,
                 norm = None,
                 extent=None,
                ):
        AxesImage.__init__(self, ax,
                           cmap = cmap,
                           norm = norm,
                           extent=extent,
                           interpolation = 'nearest',
                           origin = 'lower',
                          )

    def make_image(self, magnification=1.0):
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        x0, y0, v_width, v_height = self.axes.viewLim.get_bounds()
        l, b, width, height = self.axes.bbox.get_bounds()
	width *= magnification
	height *= magnification
        im = _image.pcolor(self._Ax, self._Ay, self._A,
                           height, width,
                           (x0, x0+v_width, y0, y0+v_height),
                          )

        bg = colorConverter.to_rgba(self.axes.get_frame().get_facecolor(), 0)
        im.set_bg(*bg)
        return im

    def set_data(self, x, y, A):
        x = asarray(x).astype(Float32)
        y = asarray(y).astype(Float32)
        A = asarray(A)
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
            if typecode(A) != UInt8:
                A = (self.cmap(self.norm(A))*255).astype(UInt8)
            else:
                A = repeat(A[:,:,NewAxis], 4, 2)
                A[:,:,3] = 255
        else:
            if typecode(A) != UInt8:
                A = (255*A).astype(UInt8)
            if A.shape[2] == 3:
                B = zeros(tuple(list(A.shape[0:2]) + [4]), UInt8)
                B[:,:,0:3] = A
                B[:,:,3] = 255
                A = B
        self._A = A
        self._Ax = x
        self._Ay = y
        self._imcache = None

    def set_array(self, *args):
        raise NotImplementedError('Method not supported')

    def set_interpolation(self, s):
        if s != 'nearest':
            raise NotImplementedError('Only nearest neighbor supported')

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




class FigureImage(Artist, cm.ScalarMappable):
    def __init__(self, fig,
                 cmap = None,
                 norm = None,
                 offsetx = 0,
                 offsety = 0,
                 origin=None,
                 ):

        """
        cmap is a colors.Colormap instance
        norm is a colors.Normalize instance to map luminance to 0-1

        """
        Artist.__init__(self)
        cm.ScalarMappable.__init__(self, norm, cmap)
        if origin is None: origin = rcParams['image.origin']
        self.origin = origin
        self.figure = fig
        self.ox = offsetx
        self.oy = offsety


    def get_size(self):
        'Get the numrows, numcols of the input image'
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        return self._A.shape[:2]

    def make_image(self, magnification=1.0):
        # had to introduce argument magnification to satisfy the unit test
        # figimage_demo.py. I have no idea, how magnification should be used
        # within the function. It should be !=1.0 only for non-default DPI
        # settings in the PS backend, as introduced by patch #1562394
        # Probably Nicholas Young should look over this code and see, how
        # magnification should be handled correctly.
        if self._A is None:
            raise RuntimeError('You must first set the image array')

        x = self.to_rgba(self._A, self._alpha)

        im = _image.fromarray(x, 1)
        im.set_bg( *colorConverter.to_rgba(self.figure.get_facecolor(), 0) )
        im.is_grayscale = (self.cmap.name == "gray" and
                           len(self._A.shape) == 2)
        if self.origin=='upper':
            im.flipud_out()

        return im

    def draw(self, renderer, *args, **kwargs):
        if not self.get_visible(): return
        im = self.make_image()
        renderer.draw_image(self.ox, self.oy, im, self.figure.bbox)

    def write_png(self, fname):
        """Write the image to png file with fname"""
        im = self.make_image()
        im.write_png(fname)

def imread(fname):
    """
    return image file in fname as numerix array

    Return value is a MxNx4 array of 0-1 normalized floats

    """
    handlers = {'png' :_image.readpng,
                }
    basename, ext = os.path.splitext(fname)
    ext = ext.lower()[1:]
    if ext not in handlers.keys():
        raise ValueError('Only know how to handled extensions: %s' % handlers.keys())

    handler = handlers[ext]
    return handler(fname)



def pil_to_array( pilImage ):
    if pilImage.mode in ('RGBA', 'RGBX'):
        im = pilImage # no need to convert images in rgba format
    else: # try to convert to an rgba image
        try:
            im = pilImage.convert('RGBA')
        except ValueError:
            raise RuntimeError('Unknown image mode')

    x_str = im.tostring('raw',im.mode,0,-1)
    x = numerix.fromstring(x_str,numerix.UInt8)
    x.shape = im.size[1], im.size[0], 4
    return x
