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
import re

from numerix import array, arange, take, put, Float, Int, putmask, \
     zeros, asarray, sort, searchsorted, sometrue, ravel, divide,\
     ones, typecode, typecodes, alltrue, clip
from numerix.mlab import amin, amax
import numerix.ma as ma
import numerix as nx
from cbook import enumerate, is_string_like, iterable
from matplotlib import rcParams
import warnings

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
    warnings.warn('Use is_color_like instead!', DeprecationWarning)
    if is_string_like(c):
        if cnames.has_key(c): return True
        elif len(c)==1: return True
        elif len(c)==7 and c.startswith('#') and len(c)==7: return True
        else: return False
    elif iterable(c) and len(c)==3:
        try:
            rgb = [float(val) for val in c]
            return True
        except:
            return False
    else:
        return False

def is_color_like(c):
    try:
        colorConverter.to_rgb(c)
        return True
    except ValueError:
        return False


def rgb2hex(rgb):
    'Given a len 3 rgb tuple of 0-1 floats, return the hex string'
    return '#%02x%02x%02x' % tuple([round(val*255) for val in rgb])

hexColorPattern = re.compile("\A#[a-fA-F0-9]{6}\Z")

def hex2color(s):
    """
    Take a hex string 's' and return the corresponding rgb 3-tuple
    Example: #efefef -> (0.93725, 0.93725, 0.93725)
    """
    if not isinstance(s, basestring):
        raise TypeError('hex2color requires a string argument')
    if hexColorPattern.match(s) is None:
        raise ValueError('invalid hex color string "%s"' % s)
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
    def to_rgb(self, arg, warn=True):
        """
        Returns an RGB tuple of three floats from 0-1.

        arg can be an RGB sequence or a string in any of several forms:
            1) a letter from the set 'rgbcmykw'
            2) a hex color string, like '#00FFFF'
            3) a standard name, like 'aqua'
            4) a float, like '0.4', indicating gray on a 0-1 scale
        """
        # warn kwarg will go away when float-as-grayscale does
        try: return self.cache[arg]
        except KeyError: pass
        except TypeError: # could be unhashable rgb seq
            arg = tuple(arg)
            try: self.cache[arg]
            except KeyError: pass
            except TypeError:
                raise ValueError('to_rgb: unhashable even inside a tuple')

        try:
            if is_string_like(arg):
                str1 = cnames.get(arg, arg)
                if str1.startswith('#'):
                    color = hex2color(str1)
                else:
                    try:
                        color = self.colors[arg]
                    except KeyError:
                        color = tuple([float(arg)]*3)
            elif iterable(arg):   # streamline this after removing float case
                color = tuple(arg[:3])
                if [x for x in color if (x < 0) or  (x > 1)]:
                    raise ValueError('to_rgb: Invalid rgb arg "%s"' % (str(arg)))
            elif isinstance(arg, (float,int)):
                #raise Exception('number is %s' % str(arg))
                if warn: warnings.warn(
                    "For gray use a string, '%s', not a float, %s" %
                                                (str(arg), str(arg)),
                                                DeprecationWarning)
                else: self._gray = True
                if 0 <= arg <= 1:
                    color = (arg,arg,arg)
                else:
                    raise ValueError('Floating point color arg must be between 0 and 1')
            else:
                raise ValueError('to_rgb: Invalid rgb arg "%s"' % (str(arg)))

            self.cache[arg] = color

        except (KeyError, ValueError, TypeError), exc:
            raise ValueError('to_rgb: Invalid rgb arg "%s"\n%s' % (str(arg), exc))

        return color

    def to_rgba(self, arg, alpha=None, warn=True):
        """
        Returns an RGBA tuple of four floats from 0-1.

        For acceptable values of arg, see to_rgb.  In
        addition, arg may already be an rgba sequence, in which
        case it is returned unchanged if the alpha kwarg is None,
        or takes on the specified alpha.
        """
        if not is_string_like(arg) and iterable(arg):
            if len(arg) == 4 and alpha is None:
                return tuple(arg)
            r,g,b = arg[:3]
        else:
            r,g,b = self.to_rgb(arg, warn)
        if alpha is None:
            alpha = 1.0
        return r,g,b,alpha

    def to_rgba_list(self, c):
        """
        Returns a list of rgba tuples.

        Accepts a single mpl color spec or a sequence of specs.
        If the sequence is a list, the list items are changed in place.
        """
        # This can be improved after removing float-as-grayscale.
        if not is_string_like(c):
            try:
                N = len(c) # raises TypeError if it is not a sequence
                # Temporary hack: keep single rgb or rgba from being
                # treated as grayscale.
                if N==3 or N==4:
                    L = [x for x in c if x>=0 and x<=1]
                    if len(L) == N:
                        raise ValueError
                # If c is a list, we need to return the same list but
                # with modified items so that items can be appended to
                # it. This is needed for examples/dynamic_collections.py.
                if not isinstance(c, list): # specific; don't need duck-typing
                    c = list(c)
                self._gray = False
                for i, cc in enumerate(c):
                    c[i] = self.to_rgba(cc, warn=False)  # change in place
                if self._gray:
                    msg = "In argument %s: use string, not float, for grayscale" % str(c)
                    warnings.warn(msg, DeprecationWarning)
                return c
            except (ValueError, TypeError):
                pass
        try:
            return [self.to_rgba(c)]
        except (ValueError, TypeError):
            raise TypeError('c must be a matplotlib color arg or a sequence of them')



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
    clip(lut, 0.0, 1.0)
    #lut = where(lut > 1., 1., lut)
    #lut = where(lut < 0., 0., lut)
    return lut


class Colormap:
    """Base class for all scalar to rgb mappings

        Important methods:
            set_bad()
            set_under()
            set_over()
    """
    def __init__(self, name, N=256):
        """Public class attributes:
            self.N:       number of rgb quantization levels
            self.name:    name of colormap

        """
        self.name = name
        self.N = N
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0) # If bad, don't paint anything.
        self._rgba_under = None
        self._rgba_over = None
        self._i_under = N
        self._i_over = N+1
        self._i_bad = N+2
        self._isinit = False


    def __call__(self, X, alpha=1.0):
        """
        X is either a scalar or an array (of any dimension).
        If scalar, a tuple of rgba values is returned, otherwise
        an array with the new shape = oldshape+(4,). If the X-values
        are integers, then they are used as indices into the array.
        If they are floating point, then they must be in the
        interval (0.0, 1.0).
        Alpha must be a scalar.
        """

        if not self._isinit: self._init()
        alpha = min(alpha, 1.0) # alpha must be between 0 and 1
        alpha = max(alpha, 0.0)
        self._lut[:-3, -1] = alpha
        mask_bad = None
        if not iterable(X):
            vtype = 'scalar'
            xa = array([X])
        else:
            vtype = 'array'
            xma = ma.asarray(X)
            xa = xma.filled(0)
            mask_bad = ma.getmask(xma)
        if typecode(xa) in typecodes['Float']:
            putmask(xa, xa==1.0, 0.9999999) #Treat 1.0 as slightly less than 1.
            xa = (xa * self.N).astype(Int)
        # Set the over-range indices before the under-range;
        # otherwise the under-range values get converted to over-range.
        putmask(xa, xa>self.N-1, self._i_over)
        putmask(xa, xa<0, self._i_under)
        if mask_bad is not None and mask_bad.shape == xa.shape:
            putmask(xa, mask_bad, self._i_bad)
        rgba = take(self._lut, xa)
        if vtype == 'scalar':
            rgba = tuple(rgba[0,:])
        return rgba

    def set_bad(self, color = 'k', alpha = 1.0):
        '''Set color to be used for masked values.
        '''
        self._rgba_bad = colorConverter.to_rgba(color, alpha)
        if self._isinit: self._set_extremes()

    def set_under(self, color = 'k', alpha = 1.0):
        '''Set color to be used for low out-of-range values.
           Requires norm.clip = False
        '''
        self._rgba_under = colorConverter.to_rgba(color, alpha)
        if self._isinit: self._set_extremes()

    def set_over(self, color = 'k', alpha = 1.0):
        '''Set color to be used for high out-of-range values.
           Requires norm.clip = False
        '''
        self._rgba_over = colorConverter.to_rgba(color, alpha)
        if self._isinit: self._set_extremes()

    def _set_extremes(self):
        if self._rgba_under:
            self._lut[self._i_under] = self._rgba_under
        else:
            self._lut[self._i_under] = self._lut[0]
        if self._rgba_over:
            self._lut[self._i_over] = self._rgba_over
        else:
            self._lut[self._i_over] = self._lut[self.N-1]
        self._lut[self._i_bad] = self._rgba_bad

    def _init():
        '''Generate the lookup table, self._lut'''
        raise NotImplementedError("Abstract class only")

    def is_gray(self):
        return (alltrue(self._lut[:,0] == self._lut[:,1])
                    and alltrue(self._lut[:,0] == self._lut[:,2]))


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
        self.monochrome = False  # True only if all colors in map are identical;
                                 # needed for contouring.
        Colormap.__init__(self, name, N)
        self._segmentdata = segmentdata

    def _init(self):
        self._lut = ones((self.N + 3, 4), Float)
        self._lut[:-3, 0] = makeMappingArray(self.N, self._segmentdata['red'])
        self._lut[:-3, 1] = makeMappingArray(self.N, self._segmentdata['green'])
        self._lut[:-3, 2] = makeMappingArray(self.N, self._segmentdata['blue'])
        self._isinit = True
        self._set_extremes()


class ListedColormap(LinearSegmentedColormap):
    """Colormap object generated from a list of colors.

    Color boundaries are evenly spaced.  This is intended for simulating
    indexed color selection, but may be useful for generating
    special colormaps also.
    """
    def __init__(self, colors, name = 'from_list', N = None):
        """
        """
        self.colors = colors
        self.monochrome = False  # True only if all colors in map are identical;
                                 # needed for contouring.
        if N is None:
            N = len(self.colors)
        else:
            if is_string_like(self.colors):
                self.colors = [self.colors] * N
                self.monochrome = True
            elif iterable(self.colors):
                if len(self.colors) == 1:
                    self.monochrome = True
                if len(self.colors) < N:
                    self.colors = list(self.colors) * N
                    del(self.colors[N:])
            else:
                try: gray = float(self.colors)
                except TypeError: pass
                else:  self.colors = [gray] * N
                self.monochrome = True
        Colormap.__init__(self, name, N)


    def _init(self):
        rgb = array([colorConverter.to_rgb(c)
                    for c in self.colors], Float)
        self._lut = zeros((self.N + 3, 4), Float)
        self._lut[:-3, :-1] = rgb
        self._lut[:-3, -1] = 1
        self._isinit = True
        self._set_extremes()


class Normalize:
    """
    Normalize a given value to the 0-1 range
    """
    def __init__(self, vmin=None, vmax=None, clip = True):
        """
        If vmin or vmax is not given, they are taken from the input's
        minimum and maximum value respectively.  If clip is True and
        the given value falls outside the range, the returned value
        will be 0 or 1, whichever is closer. Returns 0 if vmin==vmax.
        Works with scalars or arrays, including masked arrays.  If
        clip is True, masked values are set to 1; otherwise they
        remain masked.
        """
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        if isinstance(value, (int, float)):
            vtype = 'scalar'
            val = ma.array([value])
        else:
            vtype = 'array'
            val = ma.asarray(value)

        self.autoscale(val)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin==vmax:
            return 0.*value
        else:
            if clip:
                mask = ma.getmask(val)
                val = ma.array(nx.clip(val.filled(vmax), vmin, vmax),
                                mask=mask)
            result = (val-vmin) * (1.0/(vmax-vmin))
        if vtype == 'scalar':
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax

        if isinstance(value, (int, float)):
            return vmin + value * (vmax - vmin)
        else:
            val = ma.asarray(value)
            return vmin + val * (vmax - vmin)


    def autoscale(self, A):
        if not self.scaled():
            if self.vmin is None: self.vmin = ma.minimum(A)
            if self.vmax is None: self.vmax = ma.maximum(A)

    def scaled(self):
        'return true if vmin and vmax set'
        return (self.vmin is not None and self.vmax is not None)

class LogNorm(Normalize):
    """
    Normalize a given value to the 0-1 range on a log scale
    """
    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        if isinstance(value, (int, float)):
            vtype = 'scalar'
            val = ma.array([value])
        else:
            vtype = 'array'
            val = ma.asarray(value)
        self.autoscale(val)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin<=0:
            raise ValueError("values must all be positive")
        elif vmin==vmax:
            return 0.*value
        else:
            if clip:
                mask = ma.getmask(val)
                val = ma.array(nx.clip(val.filled(vmax), vmin, vmax),
                                mask=mask)
            result = (ma.log(val)-nx.log(vmin))/(nx.log(vmax)-nx.log(vmin))
        if vtype == 'scalar':
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax

        if isinstance(value, (int, float)):
            return vmin * pow((vmax/vmin), value)
        else:
            val = ma.asarray(value)
            return vmin * ma.power((vmax/vmin), val)



class NoNorm(Normalize):
    '''
    Dummy replacement for Normalize, for the case where we
    want to use indices directly in a ScalarMappable.
    '''
    def __call__(self, value, clip=None):
        return value

    def inverse(self, value):
        return value

# compatibility with earlier class names that violated convention:
normalize = Normalize
no_norm = NoNorm

