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

Finally, legal html names for colors, like 'red', 'burlywood' and
'chartreuse' are supported.
"""

from numerix import MLab, array, arange, take, put, Float, Int, where, \
     zeros, asarray, sort, searchsorted, sometrue, ravel, divide
from numerix import min as nxmin
from numerix import max as nxmax
from types import IntType, FloatType
from cbook import enumerate, is_string_like, iterable

cnames = {
    'aliceblue'            : '#F0F8FF',  	 
    'antiquewhite'         : '#FAEBD7', 	
    'aqua'                 : '#00FFFF', 	
    'aquamarine'           : '#7FFFD4', 	
    'azure'                : '#F0FFFF', 	
    'beige'                : '#F5F5DC', 	
    'bisque'               : '#FFE4C4', 	
    'black'                : '#000000', 	
    'blanchedalmond'       : '#FFEBCD', 	
    'blue'                 : '#0000FF', 	
    'blueviolet'           : '#8A2BE2', 	
    'brown'                : '#A52A2A', 	
    'burlywood'            : '#DEB887', 	
    'cadetblue'            : '#5F9EA0', 	
    'chartreuse'           : '#7FFF00', 	
    'chocolate'            : '#D2691E', 	
    'coral'                : '#FF7F50', 	
    'cornflowerblue'       : '#6495ED', 	
    'cornsilk'             : '#FFF8DC', 	
    'crimson'              : '#DC143C', 	
    'cyan'                 : '#00FFFF', 	
    'darkblue'             : '#00008B', 	
    'darkcyan'             : '#008B8B', 	
    'darkgoldenrod'        : '#B8860B', 	
    'darkgray'             : '#A9A9A9', 	
    'darkgreen'            : '#006400', 	
    'darkkhaki'            : '#BDB76B', 	
    'darkmagenta'          : '#8B008B', 	
    'darkolivegreen'       : '#556B2F', 	
    'darkorange'           : '#FF8C00', 	
    'darkorchid'           : '#9932CC', 	
    'darkred'              : '#8B0000', 	
    'darksalmon'           : '#E9967A', 	
    'darkseagreen'         : '#8FBC8F', 	
    'darkslateblue'        : '#483D8B', 	
    'darkslategray'        : '#2F4F4F', 	
    'darkturquoise'        : '#00CED1', 	
    'darkviolet'           : '#9400D3', 	
    'deeppink'             : '#FF1493', 	
    'deepskyblue'          : '#00BFFF', 	
    'dimgray'              : '#696969', 	
    'dodgerblue'           : '#1E90FF', 	
    'firebrick'            : '#B22222', 	
    'floralwhite'          : '#FFFAF0', 	
    'forestgreen'          : '#228B22', 	
    'fuchsia'              : '#FF00FF', 	
    'gainsboro'            : '#DCDCDC', 	
    'ghostwhite'           : '#F8F8FF', 	
    'gold'                 : '#FFD700', 	
    'goldenrod'            : '#DAA520', 	
    'gray'                 : '#808080', 	
    'green'                : '#008000', 	
    'greenyellow'          : '#ADFF2F', 	
    'honeydew'             : '#F0FFF0', 	
    'hotpink'              : '#FF69B4', 	
    'indianred'            : '#CD5C5C', 	
    'indigo'               : '#4B0082', 	
    'ivory'                : '#FFFFF0', 	
    'khaki'                : '#F0E68C', 	
    'lavender'             : '#E6E6FA', 	
    'lavenderblush'        : '#FFF0F5', 	
    'lawngreen'            : '#7CFC00', 	
    'lemonchiffon'         : '#FFFACD', 	
    'lightblue'            : '#ADD8E6', 	
    'lightcoral'           : '#F08080', 	
    'lightcyan'            : '#E0FFFF', 	
    'lightgoldenrodyellow' : '#FAFAD2', 	
    'lightgreen'           : '#90EE90', 	
    'lightgrey'            : '#D3D3D3',  	
    'lightpink'            : '#FFB6C1', 	
    'lightsalmon'          : '#FFA07A', 	
    'lightseagreen'        : '#20B2AA', 	
    'lightskyblue'         : '#87CEFA', 	
    'lightslategray'       : '#778899',  	
    'lightsteelblue'       : '#B0C4DE', 	
    'lightyellow'          : '#FFFFE0', 	
    'lime'                 : '#00FF00', 	
    'limegreen'            : '#32CD32', 	
    'linen'                : '#FAF0E6', 	
    'magenta'              : '#FF00FF', 	
    'maroon'               : '#800000', 	
    'mediumaquamarine'     : '#66CDAA', 	
    'mediumblue'           : '#0000CD', 	
    'mediumorchid'         : '#BA55D3', 	
    'mediumpurple'         : '#9370DB', 	
    'mediumseagreen'       : '#3CB371', 	
    'mediumslateblue'      : '#7B68EE', 	
    'mediumspringgreen'    : '#00FA9A', 	
    'mediumturquoise'      : '#48D1CC', 	
    'mediumvioletred'      : '#C71585', 	
    'midnightblue'         : '#191970', 	
    'mintcream'            : '#F5FFFA', 	
    'mistyrose'            : '#FFE4E1', 	
    'moccasin'             : '#FFE4B5', 	
    'navajowhite'          : '#FFDEAD', 	
    'navy'                 : '#000080', 	
    'oldlace'              : '#FDF5E6', 	
    'olive'                : '#808000', 	
    'olivedrab'            : '#6B8E23', 	
    'orange'               : '#FFA500', 	
    'orangered'            : '#FF4500', 	
    'orchid'               : '#DA70D6',  	
    'palegoldenrod'        : '#EEE8AA', 	
    'palegreen'            : '#98FB98', 	
    'palevioletred'        : '#AFEEEE', 	
    'papayawhip'           : '#FFEFD5', 	
    'peachpuff'            : '#FFDAB9', 	
    'peru'                 : '#CD853F', 	
    'pink'                 : '#FFC0CB', 	
    'plum'                 : '#DDA0DD', 	
    'powderblue'           : '#B0E0E6', 	
    'purple'               : '#800080', 	
    'red'                  : '#FF0000',  	
    'rosybrown'            : '#BC8F8F', 	
    'royalblue'            : '#4169E1', 	
    'saddlebrown'          : '#8B4513', 	
    'salmon'               : '#FA8072', 	
    'sandybrown'           : '#FAA460', 	
    'seagreen'             : '#2E8B57',  	
    'seashell'             : '#FFF5EE', 	
    'sienna'               : '#A0522D',  	
    'silver'               : '#C0C0C0', 	
    'skyblue'              : '#87CEEB', 	
    'slateblue'            : '#6A5ACD',  	
    'slategray'            : '#708090', 	
    'snow'                 : '#FFFAFA', 	
    'springgreen'          : '#00FF7F', 	
    'steelblue'            : '#4682B4', 	
    'tan'                  : '#D2B48C', 	
    'teal'                 : '#008080', 	
    'thistle'              : '#D8BFD8', 	
    'tomato'               : '#FF6347', 	
    'turquoise'            : '#40E0D0', 	
    'violet'               : '#EE82EE', 	
    'wheat'                : '#F5DEB3', 	
    'white'                : '#FFFFFF', 	
    'whitesmoke'           : '#F5F5F5', 	
    'yellow'               : '#FFFF00', 	
    'yellowgreen'          : '#9ACD32',
    'black'                : '#000000',	
    'navy'                 : '#000080',	
    'darkblue'             : '#00008B',	
    'mediumblue'           : '#0000CD',	
    'blue'                 : '#0000FF',	
    'darkgreen'            : '#006400',	
    'green'                : '#008000',	
    'teal'                 : '#008080',	
    'darkcyan'             : '#008B8B',	
    'deepskyblue'          : '#00BFFF',	
    'darkturquoise'        : '#00CED1',	
    'mediumspringgreen'    : '#00FA9A',	
    'lime'                 : '#00FF00',	
    'springgreen'          : '#00FF7F',	
    'aqua'                 : '#00FFFF',	
    'cyan'                 : '#00FFFF',	
    'midnightblue'         : '#191970',	
    'dodgerblue'           : '#1E90FF',	
    'lightseagreen'        : '#20B2AA',	
    'forestgreen'          : '#228B22',	
    'seagreen'             : '#2E8B57', 	
    'darkslategray'        : '#2F4F4F',	
    'limegreen'            : '#32CD32',	
    'mediumseagreen'       : '#3CB371',	
    'turquoise'            : '#40E0D0',	
    'royalblue'            : '#4169E1',	
    'steelblue'            : '#4682B4',	
    'darkslateblue'        : '#483D8B',	
    'mediumturquoise'      : '#48D1CC',	
    'indigo'               : '#4B0082',	
    'darkolivegreen'       : '#556B2F',	
    'cadetblue'            : '#5F9EA0',	
    'cornflowerblue'       : '#6495ED',	
    'mediumaquamarine'     : '#66CDAA',	
    'dimgray'              : '#696969',	
    'slateblue'            : '#6A5ACD', 	
    'olivedrab'            : '#6B8E23',	
    'slategray'            : '#708090',	
    'lightslategray'       : '#778899', 	
    'mediumslateblue'      : '#7B68EE',	
    'lawngreen'            : '#7CFC00',	
    'chartreuse'           : '#7FFF00',	
    'aquamarine'           : '#7FFFD4',	
    'maroon'               : '#800000',	
    'purple'               : '#800080',	
    'olive'                : '#808000',	
    'gray'                 : '#808080',	
    'skyblue'              : '#87CEEB',	
    'lightskyblue'         : '#87CEFA',	
    'blueviolet'           : '#8A2BE2',	
    'darkred'              : '#8B0000',	
    'darkmagenta'          : '#8B008B',	
    'saddlebrown'          : '#8B4513',	
    'darkseagreen'         : '#8FBC8F',	
    'lightgreen'           : '#90EE90',	
    'mediumpurple'         : '#9370DB',	
    'darkviolet'           : '#9400D3',	
    'palegreen'            : '#98FB98',	
    'darkorchid'           : '#9932CC',	
    'yellowgreen'          : '#9ACD32',	
    'sienna'               : '#A0522D', 	
    'brown'                : '#A52A2A',	
    'darkgray'             : '#A9A9A9',	
    'lightblue'            : '#ADD8E6',	
    'greenyellow'          : '#ADFF2F',	
    'palevioletred'        : '#AFEEEE',	
    'lightsteelblue'       : '#B0C4DE',	
    'powderblue'           : '#B0E0E6',	
    'firebrick'            : '#B22222',	
    'darkgoldenrod'        : '#B8860B',	
    'mediumorchid'         : '#BA55D3',	
    'rosybrown'            : '#BC8F8F',	
    'darkkhaki'            : '#BDB76B',	
    'silver'               : '#C0C0C0',	
    'mediumvioletred'      : '#C71585',	
    'indianred'            : '#CD5C5C',	
    'peru'                 : '#CD853F',	
    'chocolate'            : '#D2691E',	
    'tan'                  : '#D2B48C',	
    'lightgrey'            : '#D3D3D3', 	
    'thistle'              : '#D8BFD8',	
    'orchid'               : '#DA70D6', 	
    'goldenrod'            : '#DAA520',	
    'crimson'              : '#DC143C',	
    'gainsboro'            : '#DCDCDC',	
    'plum'                 : '#DDA0DD',	
    'burlywood'            : '#DEB887',	
    'lightcyan'            : '#E0FFFF',	
    'lavender'             : '#E6E6FA',	
    'darksalmon'           : '#E9967A',	
    'violet'               : '#EE82EE',	
    'palegoldenrod'        : '#EEE8AA',	
    'lightcoral'           : '#F08080',	
    'khaki'                : '#F0E68C',	
    'aliceblue'            : '#F0F8FF',	
    'honeydew'             : '#F0FFF0',	
    'azure'                : '#F0FFFF',	
    'wheat'                : '#F5DEB3',	
    'beige'                : '#F5F5DC',	
    'whitesmoke'           : '#F5F5F5',	
    'mintcream'            : '#F5FFFA',	
    'ghostwhite'           : '#F8F8FF',	
    'salmon'               : '#FA8072',	
    'sandybrown'           : '#FAA460',	
    'antiquewhite'         : '#FAEBD7',	
    'linen'                : '#FAF0E6',	
    'lightgoldenrodyellow' : '#FAFAD2',	
    'oldlace'              : '#FDF5E6',	
    'red'                  : '#FF0000', 	
    'fuchsia'              : '#FF00FF',	
    'magenta'              : '#FF00FF',	
    'deeppink'             : '#FF1493',	
    'orangered'            : '#FF4500',	
    'tomato'               : '#FF6347',	
    'hotpink'              : '#FF69B4',	
    'coral'                : '#FF7F50',	
    'darkorange'           : '#FF8C00',	
    'lightsalmon'          : '#FFA07A',	
    'orange'               : '#FFA500',	
    'lightpink'            : '#FFB6C1',	
    'pink'                 : '#FFC0CB',	
    'gold'                 : '#FFD700',	
    'peachpuff'            : '#FFDAB9',	
    'navajowhite'          : '#FFDEAD',	
    'moccasin'             : '#FFE4B5',	
    'bisque'               : '#FFE4C4',	
    'mistyrose'            : '#FFE4E1',	
    'blanchedalmond'       : '#FFEBCD',	
    'papayawhip'           : '#FFEFD5',	
    'lavenderblush'        : '#FFF0F5',	
    'seashell'             : '#FFF5EE',	
    'cornsilk'             : '#FFF8DC',	
    'lemonchiffon'         : '#FFFACD',	
    'floralwhite'          : '#FFFAF0',	
    'snow'                 : '#FFFAFA',	
    'yellow'               : '#FFFF00',	
    'lightyellow'          : '#FFFFE0',	
    'ivory'                : '#FFFFF0',	
    'white'                : '#FFFFFF',	
    }

def looks_like_color(c):
    if is_string_like(c):
        if cnames.has_key(c): return True
        elif len(c)==1: return True
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
    return '#%02x%02x%02x' % tuple([round(val*255) for val in rgb])
    
def hex2color(s):
    "Convert hex string (like html uses, eg, #efefef) to a r,g,b tuple"
    if s[0]!='#' or len(s)!=7:
        raise ValueError('s must be a hex string like "#efefef"')
    return tuple([int(n, 16)/255.0 for n in (s[1:3], s[3:5], s[5:7])])

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

        color = None
        try: float(arg)
        except: 
            if is_string_like(arg):
                hex = cnames.get(arg)
                if hex is not None: arg = hex
                if len(arg)==7 and arg[0]=='#':
                    color =   hex2color(arg)
            if color is None:
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
        self._isinit = False
        self._segmentdata = segmentdata
    def _init(self):   
        self._red_lut   = makeMappingArray(self.N, self._segmentdata['red'])
        self._green_lut = makeMappingArray(self.N, self._segmentdata['green'])
        self._blue_lut  = makeMappingArray(self.N, self._segmentdata['blue'])
        self._isinit = True
        
    def __call__(self, X, alpha=1.0):
        """
        X is either a scalar or an array (of any dimension).
        If scalar, a tuple of rgba values is returned, otherwise
        an array with the new shape = oldshape+(4,).  Any values
        that are outside the 0,1 interval are clipped to that
        interval before generating rgb values.  
        Alpha must be a scalar
        """
        if not self._isinit: self._init()
        alpha = min(alpha, 1.0) # alpha must be between 0 and 1
        alpha = max(alpha, 0.0)
        if type(X) in [IntType, FloatType]:
            vtype = 'scalar'
            xa = array([X])
        else:
            vtype = 'array'
            xa = asarray(X)

        # assume the data is properly normalized
        #xa = where(xa>1.,1.,xa)
        #xa = where(xa<0.,0.,xa)


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
            val = asarray(value)
        if vmin is None or vmax is None:
            rval = ravel(val)
            if vmin is None: vmin = nxmin(rval)
            if vmax is None: vmax = nxmax(rval)
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin==vmax:
            return 0.*value
        else:
            
            val = where(val<vmin, vmin, val)
            val = where(val>vmax, vmax, val)
            result = (1.0/(vmax-vmin))*(val-vmin)
        if vtype == 'scalar':
            result = result[0]
        return result

    def autoscale(self, A):
        if not self.scaled():
            rval = ravel(A)
            if self.vmin is None: self.vmin = nxmin(rval)
            if self.vmax is None: self.vmax = nxmax(rval)

    def scaled(self):
        'return true if vmin and vmax set'
        return (self.vmin is not None and self.vmax is not None)

    def is_mappable(self):
        return hasattr(self, '_A') and self._A is not None and self._A.shape<=2


    
    
