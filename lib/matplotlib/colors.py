"""
A class for converting color arguments to RGB

This class instantiates a single instance colorConverter that is used
to convert matlab color strings to RGB.  RGB is a tuple of float RGB
values in the range 0-1.

Commands which take color arguments can use several formats to specify
the colors.  For the basic builtin colors, you can use a single letter

      b  : blue
      g  : green
      r  : red
      c  : cyan
      m  : magenta
      y  : yellow
      k  : black 
      w  : white


For a greater range of colors, you have two options.  You can specify
the color using an html hex string, as in

      color = '#eeefff'

or you can pass an R,G,B tuple, where each of R,G,B are in the range
[0,1].
"""
from numerix import MLab, array, arange, take, put, Float, Int, where, \
     zeros, asarray, sort, searchsorted, sometrue, ravel, divide
from types import IntType, FloatType
from cbook import True, False, enumerate, is_string_like, iterable

def looks_like_color(c):
    if is_string_like(c):
        if len(c)==1: return True
        elif len(s)==7 and c.startswith('#') and len(s)==7: return True
        else: return False
    elif iterable(c) and len(c)==3:
        try:
            rgb = [float(val) for val in c]
            return True
        except:
            return False
    else:
        return False

def rgb2hex(rgb):
    'Given a len 3 rgb tuple of 0-1 floats, return the hex string'
    def fmt(val):
        h=hex(int(val*255))[2:]
        if len(h) < 2: return '0%s'%h
        else: return h
    return '#%s' % ''.join([fmt(val) for val in rgb])
    
def hex2color(s):
    "Convert hex string (like html uses, eg, #efefef) to a r,g,b tuple"
    if s.find('#')!=0 or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef#')
    r,g,b = map(lambda x: int('0x' + x, 16)/256.0, (s[1:3], s[3:5], s[5:7]))
    return r,g,b

class ColorConverter:
    colors = {
        'b' : (0.0, 0.0, 1.0),
        'g' : (0.0, 0.5, 0.0),
        'r' : (1.0, 0.0, 0.0),
        'c' : (0.0, 0.75, 0.75),
        'm' : (0.75, 0, 0.75),
        'y' : (0.75, 0.75, 0),
        'k' : (0.0, 0.0, 0.0),
        'w' : (1.0, 1.0, 1.0),
        }

    cache = {}
    def to_rgb(self, arg):
        """
        returns a tuple of three floats from 0-1.  arg can be a matlab
        format string, a html hex color string, an rgb tuple, or a
        float between 0 and 1.  In the latter case, grayscale is used
        """
        try: return self.cache[arg]
        except KeyError: pass
        except TypeError: # could be unhashable rgb seq
            arg = tuple(arg)
            try: self.cache[arg]
            except KeyError: pass

        
        try: float(arg)
        except: 
            if is_string_like(arg) and len(arg)==7 and arg[0]=='#':
                color =   hex2color(arg)
            else:
                # see if it looks like rgb.  If so, just return arg
                try: float(arg[2])
                except: color = self.colors.get(arg, (0.0, 0.0, 0.0))
                else: color =  tuple(arg)
        else:
            if arg>=0 and arg<=1:
                color =  (arg,arg,arg)
            else:
                msg = 'Floating point color arg must be between 0 and 1\n' +\
                      'Found %1.2f' % arg
                raise RuntimeError(msg)

        self.cache[arg] = color
        return color

    def to_rgba(self, arg, alpha=1.0):
        """
        returns a tuple of four floats from 0-1.  arg can be a matlab
        format string, a html hex color string, an rgb tuple, or a
        float between 0 and 1.  In the latter case, grayscale is used
        """
        r,g,b = self.to_rgb(arg)
        return r,g,b,alpha

colorConverter = ColorConverter()

def makeMappingArray(N, data):
    """Create an N-element 1-d lookup table
    
    data represented by a list of x,y0,y1 mapping correspondences.
    Each element in this list represents how a value between 0 and 1
    (inclusive) represented by x is mapped to a corresponding value
    between 0 and 1 (inclusive). The two values of y are to allow 
    for discontinuous mapping functions (say as might be found in a
    sawtooth) where y0 represents the value of y for values of x
    <= to that given, and y1 is the value to be used for x > than
    that given). The list must start with x=0, end with x=1, and 
    all values of x must be in increasing order. Values between
    the given mapping points are determined by simple linear interpolation.
    
    The function returns an array "result" where result[x*(N-1)]
    gives the closest value for values of x between 0 and 1.
    """
    try:
        adata = array(data)
    except:
        raise TypeError("data must be convertable to an array")
    shape = adata.shape
    if len(shape) != 2 and shape[1] != 3:
        raise ValueError("data must be nx3 format")

    x  = adata[:,0]
    y0 = adata[:,1]
    y1 = adata[:,2]
    if x[0] != 0. or x[-1] != 1.0:
        raise ValueError(
           "data mapping points must start with x=0. and end with x=1")
    if sometrue(sort(x)-x):
        raise ValueError(
           "data mapping points must have x in increasing order")
    # begin generation of lookup table
    x = x * (N-1)
    lut = zeros((N,), Float)
    xind = arange(float(N))
    ind = searchsorted(x, xind)[1:-1]
    
    lut[1:-1] = ( divide(xind[1:-1] - take(x,ind-1),
                         take(x,ind)-take(x,ind-1) )
                *(take(y0,ind)-take(y1,ind-1)) + take(y1,ind-1))
    lut[0] = y1[0]
    lut[-1] = y0[-1]
    # ensure that the lut is confined to values between 0 and 1 by clipping it
    lut = where(lut > 1., 1., lut)
    lut = where(lut < 0., 0., lut)
    return lut
    
    
class Colormap:
    """Basis abstract class for all scalar to rgb mappings"""
    def __init__(self, name, N=256):
        """Public class attributes:
            self.N:       number of rgb quantization levels
            self.name:    name of colormap
        """
        raise NotImplementedError("Abstract class only")
    def __call__(self, X, alpha=1.0):
        raise NotImplementedError("Abstract class only")           
    
class LinearSegmentedColormap(Colormap):
    """Colormap objects based on lookup tables using linear segments.
    
    The lookup transfer function is a simple linear function between 
    defined intensities. There is no limit to the number of segments
    that may be defined. Though as the segment intervals start containing
    fewer and fewer array locations, there will be inevitable quantization
    errors
    """
    def __init__(self, name, segmentdata, N=256):
        """Create color map from linear mapping segments
        
        segmentdata argument is a dictionary with a red, green and blue
        entries. Each entry should be a list of x, y0, y1 tuples.
        See makeMappingArray for details
        """
        self.N = N
        self.name=name
        self._red_lut   = makeMappingArray(N, segmentdata['red'])
        self._green_lut = makeMappingArray(N, segmentdata['green'])
        self._blue_lut  = makeMappingArray(N, segmentdata['blue'])
        
    def __call__(self, X, alpha=1.0):
        """
        X is either a scalar or an array (of any dimension).
        If scalar, a tuple of rgba values is returned, otherwise
        an array with the new shape = oldshape+(4,).  Any values
        that are outside the 0,1 interval are clipped to that
        interval before generating rgb values.  
        Alpha must be a scalar
        """
        alpha = min(alpha, 1.0) # alpha must be between 0 and 1
        alpha = max(alpha, 0.0)
        if type(X) in [IntType, FloatType]:
            vtype = 'scalar'
            xa = array([X])
        else:
            vtype = 'array'
            xa = array(X)

        xa = where(xa>1.,1.,xa)
        xa = where(xa<0.,0.,xa)
        xa = (xa *(self.N-1)).astype(Int)
        rgba = zeros(xa.shape+(4,), Float)
        rgba[...,0] = take(self._red_lut, xa)
        rgba[...,1] = take(self._green_lut, xa)
        rgba[...,2] = take(self._blue_lut, xa)
        rgba[...,3] = alpha
        if vtype == 'scalar':
            rgba = tuple(rgba[0,:])
        return rgba

    
class normalize:
    def __init__(self, vmin=None, vmax=None):
        """
        Normalize a given value to the 0-1 range

        If vmin or vmax is not given, they are taken from the input's
        minimum and maximum value respectively.  If the given value
        falls outside the range, the returned value will be 0 or 1,
        whichever is closest. Returns 0 if vmin==vmax. Works with
        scalars or arrays.
        """
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, value):

        vmin = self.vmin
        vmax = self.vmax
        if type(value) in [IntType, FloatType]:
            vtype = 'scalar'
            val = array([value])
        else:
            vtype = 'array'
            val = array(value)
        if vmin is None or vmax is None:
            rval = ravel(val)
            if vmin is None:
                vmin = min(rval)
            if vmax is None:
                vmax = max(rval)
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin==vmax:
            return 0.*value
        else:
            val = where(val<vmin, vmin, val)
            val = where(val>vmax, vmax, val)
            result = divide(val-vmin, vmax-vmin)
        if vtype == 'scalar':
            result = result[0]
        return result

    def autoscale(self, A):
        if not self.scaled():
            rval = ravel(A)
            if self.vmin is None:
                self.vmin = min(rval)
            if self.vmax is None:
                self.vmax = max(rval)

    def scaled(self):
        'return true if vmin and vmax set'
        return not (self.vmin is None or self.vmax is None)

    def is_mappable(self):
        return self._A is not None and self._A.shape<=2



