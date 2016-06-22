"""
A module for converting numbers or color arguments to *RGB* or *RGBA*

*RGB* and *RGBA* are sequences of, respectively, 3 or 4 floats in the
range 0-1.

This module includes functions and classes for color specification
conversions, and for mapping numbers to colors in a 1-D array of colors called
a colormap. Colormapping typically involves two steps: a data array is first
mapped onto the range 0-1 using an instance of :class:`Normalize` or of a
subclass; then this number in the 0-1 range is mapped to a color using an
instance of a subclass of :class:`Colormap`.  Two are provided here:
:class:`LinearSegmentedColormap`, which is used to generate all the built-in
colormap instances, but is also useful for making custom colormaps, and
:class:`ListedColormap`, which is used for generating a custom colormap from a
list of color specifications.

The module also provides a single instance, *colorConverter*, of the
:class:`ColorConverter` class providing methods for converting single color
specifications or sequences of them to *RGB* or *RGBA*.

Commands which take color arguments can use several formats to specify
the colors.  For the basic built-in colors, you can use a single letter

    - b: blue
    - g: green
    - r: red
    - c: cyan
    - m: magenta
    - y: yellow
    - k: black
    - w: white

Gray shades can be given as a string encoding a float in the 0-1 range, e.g.::

    color = '0.75'

For a greater range of colors, you have two options.  You can specify the
color using an html hex string, as in::

      color = '#eeefff'

or you can pass an *R* , *G* , *B* tuple, where each of *R* , *G* , *B* are in
the range [0,1].

Finally, legal html names for colors, like 'red', 'burlywood' and 'chartreuse'
are supported.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six.moves import zip

import warnings
import re
import numpy as np
from numpy import ma
import matplotlib.cbook as cbook

cnames = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#A52A2A',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksage':             '#598556',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgreen':           '#90EE90',
    'lightgray':            '#D3D3D3',
    'lightpink':            '#FFB6C1',
    'lightsage':            '#BCECAC',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sage':                 '#87AE73',
    'sandybrown':           '#FAA460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32'}


# add british equivs
for k, v in list(six.iteritems(cnames)):
    if k.find('gray') >= 0:
        k = k.replace('gray', 'grey')
        cnames[k] = v


def is_color_like(c):
    'Return *True* if *c* can be converted to *RGB*'
    try:
        colorConverter.to_rgb(c)
        return True
    except ValueError:
        return False


def rgb2hex(rgb):
    'Given an rgb or rgba sequence of 0-1 floats, return the hex string'
    a = '#%02x%02x%02x' % tuple([int(np.round(val * 255)) for val in rgb[:3]])
    return a

hexColorPattern = re.compile("\A#[a-fA-F0-9]{6}\Z")


def hex2color(s):
    """
    Take a hex string *s* and return the corresponding rgb 3-tuple
    Example: #efefef -> (0.93725, 0.93725, 0.93725)
    """
    if not isinstance(s, six.string_types):
        raise TypeError('hex2color requires a string argument')
    if hexColorPattern.match(s) is None:
        raise ValueError('invalid hex color string "%s"' % s)
    return tuple([int(n, 16) / 255.0 for n in (s[1:3], s[3:5], s[5:7])])


class ColorConverter(object):
    """
    Provides methods for converting color specifications to *RGB* or *RGBA*

    Caching is used for more efficient conversion upon repeated calls
    with the same argument.

    Ordinarily only the single instance instantiated in this module,
    *colorConverter*, is needed.
    """
    colors = {
        'b': (0.0, 0.0, 1.0),
        'g': (0.0, 0.5, 0.0),
        'r': (1.0, 0.0, 0.0),
        'c': (0.0, 0.75, 0.75),
        'm': (0.75, 0, 0.75),
        'y': (0.75, 0.75, 0),
        'k': (0.0, 0.0, 0.0),
        'w': (1.0, 1.0, 1.0), }

    cache = {}

    def to_rgb(self, arg):
        """
        Returns an *RGB* tuple of three floats from 0-1.

        *arg* can be an *RGB* or *RGBA* sequence or a string in any of
        several forms:

            1) a letter from the set 'rgbcmykw'
            2) a hex color string, like '#00FFFF'
            3) a standard name, like 'aqua'
            4) a string representation of a float, like '0.4',
               indicating gray on a 0-1 scale

        if *arg* is *RGBA*, the *A* will simply be discarded.
        """
        # Gray must be a string to distinguish 3-4 grays from RGB or RGBA.

        try:
            return self.cache[arg]
        except KeyError:
            pass
        except TypeError:  # could be unhashable rgb seq
            arg = tuple(arg)
            try:
                return self.cache[arg]
            except KeyError:
                pass
            except TypeError:
                raise ValueError(
                    'to_rgb: arg "%s" is unhashable even inside a tuple'
                    % (str(arg),))

        try:
            if cbook.is_string_like(arg):
                argl = arg.lower()
                color = self.colors.get(argl, None)
                if color is None:
                    str1 = cnames.get(argl, argl)
                    if str1.startswith('#'):
                        color = hex2color(str1)
                    else:
                        fl = float(argl)
                        if fl < 0 or fl > 1:
                            raise ValueError(
                                'gray (string) must be in range 0-1')
                        color = (fl,)*3
            elif cbook.iterable(arg):
                if len(arg) > 4 or len(arg) < 3:
                    raise ValueError(
                        'sequence length is %d; must be 3 or 4' % len(arg))
                color = tuple(arg[:3])
                if [x for x in color if (float(x) < 0) or (x > 1)]:
                    # This will raise TypeError if x is not a number.
                    raise ValueError(
                        'number in rbg sequence outside 0-1 range')
            else:
                raise ValueError(
                    'cannot convert argument to rgb sequence')

            self.cache[arg] = color

        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                'to_rgb: Invalid rgb arg "%s"\n%s' % (str(arg), exc))
            # Error messages could be improved by handling TypeError
            # separately; but this should be rare and not too hard
            # for the user to figure out as-is.
        return color

    def to_rgba(self, arg, alpha=None):
        """
        Returns an *RGBA* tuple of four floats from 0-1.

        For acceptable values of *arg*, see :meth:`to_rgb`.
        In addition, if *arg* is "none" (case-insensitive),
        then (0,0,0,0) will be returned.
        If *arg* is an *RGBA* sequence and *alpha* is not *None*,
        *alpha* will replace the original *A*.
        """
        try:
            if arg.lower() == 'none':
                return (0.0, 0.0, 0.0, 0.0)
        except AttributeError:
            pass

        try:
            if not cbook.is_string_like(arg) and cbook.iterable(arg):
                if len(arg) == 4:
                    if any(float(x) < 0 or x > 1 for x in arg):
                        raise ValueError(
                            'number in rbga sequence outside 0-1 range')
                    if alpha is None:
                        return tuple(arg)
                    if alpha < 0.0 or alpha > 1.0:
                        raise ValueError("alpha must be in range 0-1")
                    return arg[0], arg[1], arg[2], alpha
                if len(arg) == 3:
                    r, g, b = arg
                    if any(float(x) < 0 or x > 1 for x in arg):
                        raise ValueError(
                            'number in rbg sequence outside 0-1 range')
                else:
                    raise ValueError(
                            'length of rgba sequence should be either 3 or 4')
            else:
                r, g, b = self.to_rgb(arg)
            if alpha is None:
                alpha = 1.0
            return r, g, b, alpha
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'to_rgba: Invalid rgba arg "%s"\n%s' % (str(arg), exc))

    def to_rgba_array(self, c, alpha=None):
        """
        Returns a numpy array of *RGBA* tuples.

        Accepts a single mpl color spec or a sequence of specs.

        Special case to handle "no color": if *c* is "none" (case-insensitive),
        then an empty array will be returned.  Same for an empty list.
        """
        try:
            nc = len(c)
        except TypeError:
            raise ValueError(
                "Cannot convert argument type %s to rgba array" % type(c))
        try:
            if nc == 0 or c.lower() == 'none':
                return np.zeros((0, 4), dtype=np.float)
        except AttributeError:
            pass
        try:
            # Single value? Put it in an array with a single row.
            return np.array([self.to_rgba(c, alpha)], dtype=np.float)
        except ValueError:
            if isinstance(c, np.ndarray):
                if c.ndim != 2 and c.dtype.kind not in 'SU':
                    raise ValueError("Color array must be two-dimensional")
                if (c.ndim == 2 and c.shape[1] == 4 and c.dtype.kind == 'f'):
                    if (c.ravel() > 1).any() or (c.ravel() < 0).any():
                        raise ValueError(
                            "number in rgba sequence is outside 0-1 range")
                    result = np.asarray(c, np.float)
                    if alpha is not None:
                        if alpha > 1 or alpha < 0:
                            raise ValueError("alpha must be in 0-1 range")
                        result[:, 3] = alpha
                    return result
                    # This alpha operation above is new, and depends
                    # on higher levels to refrain from setting alpha
                    # to values other than None unless there is
                    # intent to override any existing alpha values.

            # It must be some other sequence of color specs.
            result = np.zeros((nc, 4), dtype=np.float)
            for i, cc in enumerate(c):
                result[i] = self.to_rgba(cc, alpha)
            return result


colorConverter = ColorConverter()


def makeMappingArray(N, data, gamma=1.0):
    """Create an *N* -element 1-d lookup table

    *data* represented by a list of x,y0,y1 mapping correspondences.
    Each element in this list represents how a value between 0 and 1
    (inclusive) represented by x is mapped to a corresponding value
    between 0 and 1 (inclusive). The two values of y are to allow
    for discontinuous mapping functions (say as might be found in a
    sawtooth) where y0 represents the value of y for values of x
    <= to that given, and y1 is the value to be used for x > than
    that given). The list must start with x=0, end with x=1, and
    all values of x must be in increasing order. Values between
    the given mapping points are determined by simple linear interpolation.

    Alternatively, data can be a function mapping values between 0 - 1
    to 0 - 1.

    The function returns an array "result" where ``result[x*(N-1)]``
    gives the closest value for values of x between 0 and 1.
    """

    if six.callable(data):
        xind = np.linspace(0, 1, N) ** gamma
        lut = np.clip(np.array(data(xind), dtype=np.float), 0, 1)
        return lut

    try:
        adata = np.array(data)
    except:
        raise TypeError("data must be convertable to an array")
    shape = adata.shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError("data must be nx3 format")

    x = adata[:, 0]
    y0 = adata[:, 1]
    y1 = adata[:, 2]

    if x[0] != 0. or x[-1] != 1.0:
        raise ValueError(
            "data mapping points must start with x=0. and end with x=1")
    if np.sometrue(np.sort(x) - x):
        raise ValueError(
            "data mapping points must have x in increasing order")
    # begin generation of lookup table
    x = x * (N - 1)
    lut = np.zeros((N,), np.float)
    xind = (N - 1) * np.linspace(0, 1, N) ** gamma
    ind = np.searchsorted(x, xind)[1:-1]

    distance = (xind[1:-1] - x[ind - 1]) / (x[ind] - x[ind - 1])
    lut[1:-1] = distance * (y0[ind] - y1[ind - 1]) + y1[ind - 1]
    lut[0] = y1[0]
    lut[-1] = y0[-1]
    # ensure that the lut is confined to values between 0 and 1 by clipping it
    return np.clip(lut, 0.0, 1.0)


class Colormap(object):
    """
    Baseclass for all scalar to RGBA mappings.

    Typically Colormap instances are used to convert data values (floats) from
    the interval ``[0, 1]`` to the RGBA color that the respective Colormap
    represents. For scaling of data into the ``[0, 1]`` interval see
    :class:`matplotlib.colors.Normalize`. It is worth noting that
    :class:`matplotlib.cm.ScalarMappable` subclasses make heavy use of this
    ``data->normalize->map-to-color`` processing chain.

    """
    def __init__(self, name, N=256):
        r"""
        Parameters
        ----------
        name : str
            The name of the colormap.
        N : int
            The number of rgb quantization levels.

        """
        self.name = name
        self.N = int(N)  # ensure that N is always int
        self._rgba_bad = (0.0, 0.0, 0.0, 0.0)  # If bad, don't paint anything.
        self._rgba_under = None
        self._rgba_over = None
        self._i_under = self.N
        self._i_over = self.N + 1
        self._i_bad = self.N + 2
        self._isinit = False

        #: When this colormap exists on a scalar mappable and colorbar_extend
        #: is not False, colorbar creation will pick up ``colorbar_extend`` as
        #: the default value for the ``extend`` keyword in the
        #: :class:`matplotlib.colorbar.Colorbar` constructor.
        self.colorbar_extend = False

    def __call__(self, X, alpha=None, bytes=False):
        """
        Parameters
        ----------
        X : scalar, ndarray
            The data value(s) to convert to RGBA.
            For floats, X should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, X should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float, None
            Alpha must be a scalar between 0 and 1, or None.
        bytes : bool
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be uint8s in the interval
            ``[0, 255]``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, othewise an array of
        RGBA values with a shape of ``X.shape + (4, )``.

        """
        # See class docstring for arg/kwarg documentation.
        if not self._isinit:
            self._init()
        mask_bad = None
        if not cbook.iterable(X):
            vtype = 'scalar'
            xa = np.array([X])
        else:
            vtype = 'array'
            xma = ma.array(X, copy=True)  # Copy here to avoid side effects.
            mask_bad = xma.mask           # Mask will be used below.
            xa = xma.filled()             # Fill to avoid infs, etc.
            del xma

        # Calculations with native byteorder are faster, and avoid a
        # bug that otherwise can occur with putmask when the last
        # argument is a numpy scalar.
        if not xa.dtype.isnative:
            xa = xa.byteswap().newbyteorder()

        if xa.dtype.kind == "f":
            # Treat 1.0 as slightly less than 1.
            vals = np.array([1, 0], dtype=xa.dtype)
            almost_one = np.nextafter(*vals)
            cbook._putmask(xa, xa == 1.0, almost_one)
            # The following clip is fast, and prevents possible
            # conversion of large positive values to negative integers.

            xa *= self.N
            np.clip(xa, -1, self.N, out=xa)

            # ensure that all 'under' values will still have negative
            # value after casting to int
            cbook._putmask(xa, xa < 0.0, -1)
            xa = xa.astype(int)
        # Set the over-range indices before the under-range;
        # otherwise the under-range values get converted to over-range.
        cbook._putmask(xa, xa > self.N - 1, self._i_over)
        cbook._putmask(xa, xa < 0, self._i_under)
        if mask_bad is not None:
            if mask_bad.shape == xa.shape:
                cbook._putmask(xa, mask_bad, self._i_bad)
            elif mask_bad:
                xa.fill(self._i_bad)
        if bytes:
            lut = (self._lut * 255).astype(np.uint8)
        else:
            lut = self._lut.copy()  # Don't let alpha modify original _lut.

        if alpha is not None:
            alpha = min(alpha, 1.0)  # alpha must be between 0 and 1
            alpha = max(alpha, 0.0)
            if bytes:
                alpha = int(alpha * 255)
            if (lut[-1] == 0).all():
                lut[:-1, -1] = alpha
                # All zeros is taken as a flag for the default bad
                # color, which is no color--fully transparent.  We
                # don't want to override this.
            else:
                lut[:, -1] = alpha
                # If the bad value is set to have a color, then we
                # override its alpha just as for any other value.

        rgba = np.empty(shape=xa.shape + (4,), dtype=lut.dtype)
        lut.take(xa, axis=0, mode='clip', out=rgba)
        if vtype == 'scalar':
            rgba = tuple(rgba[0, :])
        return rgba

    def set_bad(self, color='k', alpha=None):
        """Set color to be used for masked values.
        """
        self._rgba_bad = colorConverter.to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def set_under(self, color='k', alpha=None):
        """Set color to be used for low out-of-range values.
           Requires norm.clip = False
        """
        self._rgba_under = colorConverter.to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def set_over(self, color='k', alpha=None):
        """Set color to be used for high out-of-range values.
           Requires norm.clip = False
        """
        self._rgba_over = colorConverter.to_rgba(color, alpha)
        if self._isinit:
            self._set_extremes()

    def _set_extremes(self):
        if self._rgba_under:
            self._lut[self._i_under] = self._rgba_under
        else:
            self._lut[self._i_under] = self._lut[0]
        if self._rgba_over:
            self._lut[self._i_over] = self._rgba_over
        else:
            self._lut[self._i_over] = self._lut[self.N - 1]
        self._lut[self._i_bad] = self._rgba_bad

    def _init(self):
        """Generate the lookup table, self._lut"""
        raise NotImplementedError("Abstract class only")

    def is_gray(self):
        if not self._isinit:
            self._init()
        return (np.alltrue(self._lut[:, 0] == self._lut[:, 1]) and
                np.alltrue(self._lut[:, 0] == self._lut[:, 2]))

    def _resample(self, lutsize):
        """
        Return a new color map with *lutsize* entries.
        """
        raise NotImplementedError()


class LinearSegmentedColormap(Colormap):
    """Colormap objects based on lookup tables using linear segments.

    The lookup table is generated using linear interpolation for each
    primary color, with the 0-1 domain divided into any number of
    segments.
    """
    def __init__(self, name, segmentdata, N=256, gamma=1.0):
        """Create color map from linear mapping segments

        segmentdata argument is a dictionary with a red, green and blue
        entries. Each entry should be a list of *x*, *y0*, *y1* tuples,
        forming rows in a table. Entries for alpha are optional.

        Example: suppose you want red to increase from 0 to 1 over
        the bottom half, green to do the same over the middle half,
        and blue over the top half.  Then you would use::

            cdict = {'red':   [(0.0,  0.0, 0.0),
                               (0.5,  1.0, 1.0),
                               (1.0,  1.0, 1.0)],

                     'green': [(0.0,  0.0, 0.0),
                               (0.25, 0.0, 0.0),
                               (0.75, 1.0, 1.0),
                               (1.0,  1.0, 1.0)],

                     'blue':  [(0.0,  0.0, 0.0),
                               (0.5,  0.0, 0.0),
                               (1.0,  1.0, 1.0)]}

        Each row in the table for a given color is a sequence of
        *x*, *y0*, *y1* tuples.  In each sequence, *x* must increase
        monotonically from 0 to 1.  For any input value *z* falling
        between *x[i]* and *x[i+1]*, the output value of a given color
        will be linearly interpolated between *y1[i]* and *y0[i+1]*::

            row i:   x  y0  y1
                           /
                          /
            row i+1: x  y0  y1

        Hence y0 in the first row and y1 in the last row are never used.


        .. seealso::

               :meth:`LinearSegmentedColormap.from_list`
               Static method; factory function for generating a
               smoothly-varying LinearSegmentedColormap.

               :func:`makeMappingArray`
               For information about making a mapping array.
        """
        # True only if all colors in map are identical; needed for contouring.
        self.monochrome = False
        Colormap.__init__(self, name, N)
        self._segmentdata = segmentdata
        self._gamma = gamma

    def _init(self):
        self._lut = np.ones((self.N + 3, 4), np.float)
        self._lut[:-3, 0] = makeMappingArray(
            self.N, self._segmentdata['red'], self._gamma)
        self._lut[:-3, 1] = makeMappingArray(
            self.N, self._segmentdata['green'], self._gamma)
        self._lut[:-3, 2] = makeMappingArray(
            self.N, self._segmentdata['blue'], self._gamma)
        if 'alpha' in self._segmentdata:
            self._lut[:-3, 3] = makeMappingArray(
                self.N, self._segmentdata['alpha'], 1)
        self._isinit = True
        self._set_extremes()

    def set_gamma(self, gamma):
        """
        Set a new gamma value and regenerate color map.
        """
        self._gamma = gamma
        self._init()

    @staticmethod
    def from_list(name, colors, N=256, gamma=1.0):
        """
        Make a linear segmented colormap with *name* from a sequence
        of *colors* which evenly transitions from colors[0] at val=0
        to colors[-1] at val=1.  *N* is the number of rgb quantization
        levels.
        Alternatively, a list of (value, color) tuples can be given
        to divide the range unevenly.
        """

        if not cbook.iterable(colors):
            raise ValueError('colors must be iterable')

        if cbook.iterable(colors[0]) and len(colors[0]) == 2 and \
                not cbook.is_string_like(colors[0]):
            # List of value, color pairs
            vals, colors = list(zip(*colors))
        else:
            vals = np.linspace(0., 1., len(colors))

        cdict = dict(red=[], green=[], blue=[], alpha=[])
        for val, color in zip(vals, colors):
            r, g, b, a = colorConverter.to_rgba(color)
            cdict['red'].append((val, r, r))
            cdict['green'].append((val, g, g))
            cdict['blue'].append((val, b, b))
            cdict['alpha'].append((val, a, a))

        return LinearSegmentedColormap(name, cdict, N, gamma)

    def _resample(self, lutsize):
        """
        Return a new color map with *lutsize* entries.
        """
        return LinearSegmentedColormap(self.name, self._segmentdata, lutsize)


class ListedColormap(Colormap):
    """Colormap object generated from a list of colors.

    This may be most useful when indexing directly into a colormap,
    but it can also be used to generate special colormaps for ordinary
    mapping.
    """
    def __init__(self, colors, name='from_list', N=None):
        """
        Make a colormap from a list of colors.

        *colors*
            a list of matplotlib color specifications,
            or an equivalent Nx3 or Nx4 floating point array
            (*N* rgb or rgba values)
        *name*
            a string to identify the colormap
        *N*
            the number of entries in the map.  The default is *None*,
            in which case there is one colormap entry for each
            element in the list of colors.  If::

                N < len(colors)

            the list will be truncated at *N*.  If::

                N > len(colors)

            the list will be extended by repetition.
        """
        self.colors = colors
        self.monochrome = False  # True only if all colors in map are
                                 # identical; needed for contouring.
        if N is None:
            N = len(self.colors)
        else:
            if (cbook.is_string_like(self.colors) and
                    cbook.is_hashable(self.colors)):
                self.colors = [self.colors] * N
                self.monochrome = True
            elif cbook.iterable(self.colors):
                self.colors = list(self.colors)  # in case it was a tuple
                if len(self.colors) == 1:
                    self.monochrome = True
                if len(self.colors) < N:
                    self.colors = list(self.colors) * N
                del(self.colors[N:])
            else:
                try:
                    gray = float(self.colors)
                except TypeError:
                    pass
                else:
                    self.colors = [gray] * N
                self.monochrome = True
        Colormap.__init__(self, name, N)

    def _init(self):
        rgba = colorConverter.to_rgba_array(self.colors)
        self._lut = np.zeros((self.N + 3, 4), np.float)
        self._lut[:-3] = rgba
        self._isinit = True
        self._set_extremes()

    def _resample(self, lutsize):
        """
        Return a new color map with *lutsize* entries.
        """
        colors = self(np.linspace(0, 1, lutsize))
        return ListedColormap(colors, name=self.name)


class Normalize(object):
    """
    A class which, when called, can normalize data into
    the ``[0.0, 1.0]`` interval.

    """
    def __init__(self, vmin=None, vmax=None, clip=False):
        """
        If *vmin* or *vmax* is not given, they are initialized from the
        minimum and maximum value respectively of the first input
        processed.  That is, *__call__(A)* calls *autoscale_None(A)*.
        If *clip* is *True* and the given value falls outside the range,
        the returned value will be 0 or 1, whichever is closer.
        Returns 0 if::

            vmin==vmax

        Works with scalars or arrays, including masked arrays.  If
        *clip* is *True*, masked values are set to 1; otherwise they
        remain masked.  Clipping silently defeats the purpose of setting
        the over, under, and masked colors in the colormap, so it is
        likely to lead to surprises; therefore the default is
        *clip* = *False*.
        """
        self.vmin = vmin
        self.vmax = vmax
        self.clip = clip

    @staticmethod
    def process_value(value):
        """
        Homogenize the input *value* for easy and efficient normalization.

        *value* can be a scalar or sequence.

        Returns *result*, *is_scalar*, where *result* is a
        masked array matching *value*.  Float dtypes are preserved;
        integer types with two bytes or smaller are converted to
        np.float32, and larger types are converted to np.float.
        Preserving float32 when possible, and using in-place operations,
        can greatly improve speed for large arrays.

        Experimental; we may want to add an option to force the
        use of float32.
        """
        if cbook.iterable(value):
            is_scalar = False
            result = ma.asarray(value)
            if result.dtype.kind == 'f':
                # this is overkill for lists of floats, but required
                # to support pd.Series as input until we can reliable
                # determine if result and value share memory in all cases
                # (list, tuple, deque, ndarray, Series, ...)
                result = result.copy()
            elif result.dtype.itemsize > 2:
                result = result.astype(np.float)
            else:
                result = result.astype(np.float32)
        else:
            is_scalar = True
            result = ma.array([value]).astype(np.float)
        return result, is_scalar

    def __call__(self, value, clip=None):
        """
        Normalize *value* data in the ``[vmin, vmax]`` interval into
        the ``[0.0, 1.0]`` interval and return it.  *clip* defaults
        to *self.clip* (which defaults to *False*).  If not already
        initialized, *vmin* and *vmax* are initialized using
        *autoscale_None(value)*.
        """
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if vmin == vmax:
            result.fill(0)   # Or should it be all masked?  Or 0.5?
        elif vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)
            # ma division is very slow; we can take a shortcut
            # use np.asarray so data passed in as an ndarray subclass are
            # interpreted as an ndarray. See issue #6622.
            resdat = np.asarray(result.data)
            resdat -= vmin
            resdat /= (vmax - vmin)
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin = float(self.vmin)
        vmax = float(self.vmax)

        if cbook.iterable(value):
            val = ma.asarray(value)
            return vmin + val * (vmax - vmin)
        else:
            return vmin + value * (vmax - vmin)

    def autoscale(self, A):
        """
        Set *vmin*, *vmax* to min, max of *A*.
        """
        self.vmin = ma.min(A)
        self.vmax = ma.max(A)

    def autoscale_None(self, A):
        ' autoscale only None-valued vmin or vmax'
        if self.vmin is None and np.size(A) > 0:
            self.vmin = ma.min(A)
        if self.vmax is None and np.size(A) > 0:
            self.vmax = ma.max(A)

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

        result, is_scalar = self.process_value(value)

        result = ma.masked_less_equal(result, 0, copy=False)

        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin <= 0:
            raise ValueError("values must all be positive")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)
            # in-place equivalent of above can be much faster
            resdat = result.data
            mask = result.mask
            if mask is np.ma.nomask:
                mask = (resdat <= 0)
            else:
                mask |= resdat <= 0
            cbook._putmask(resdat, mask, 1)
            np.log(resdat, resdat)
            resdat -= np.log(vmin)
            resdat /= (np.log(vmax) - np.log(vmin))
            result = np.ma.array(resdat, mask=mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = ma.asarray(value)
            return vmin * ma.power((vmax / vmin), val)
        else:
            return vmin * pow((vmax / vmin), value)

    def autoscale(self, A):
        """
        Set *vmin*, *vmax* to min, max of *A*.
        """
        A = ma.masked_less_equal(A, 0, copy=False)
        self.vmin = ma.min(A)
        self.vmax = ma.max(A)

    def autoscale_None(self, A):
        ' autoscale only None-valued vmin or vmax'
        if self.vmin is not None and self.vmax is not None:
            return
        A = ma.masked_less_equal(A, 0, copy=False)
        if self.vmin is None:
            self.vmin = ma.min(A)
        if self.vmax is None:
            self.vmax = ma.max(A)


class SymLogNorm(Normalize):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is a
    need to have a range around zero that is linear.  The parameter
    *linthresh* allows the user to specify the size of this range
    (-*linthresh*, *linthresh*).
    """
    def __init__(self,  linthresh, linscale=1.0,
                 vmin=None, vmax=None, clip=False):
        """
        *linthresh*:
        The range within which the plot is linear (to
        avoid having the plot go to infinity around zero).

        *linscale*:
        This allows the linear range (-*linthresh* to *linthresh*)
        to be stretched relative to the logarithmic range.  Its
        value is the number of decades to use for each half of the
        linear range.  For example, when *linscale* == 1.0 (the
        default), the space used for the positive and negative
        halves of the linear range will be equal to one decade in
        the logarithmic range. Defaults to 1.
        """
        Normalize.__init__(self, vmin, vmax, clip)
        self.linthresh = float(linthresh)
        self._linscale_adj = (linscale / (1.0 - np.e ** -1))

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        vmin, vmax = self.vmin, self.vmax

        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)
            # in-place equivalent of above can be much faster
            resdat = self._transform(result.data)
            resdat -= self._lower
            resdat /= (self._upper - self._lower)

        if is_scalar:
            result = result[0]
        return result

    def _transform(self, a):
        """
        Inplace transformation.
        """
        masked = np.abs(a) > self.linthresh
        sign = np.sign(a[masked])
        log = (self._linscale_adj + np.log(np.abs(a[masked]) / self.linthresh))
        log *= sign * self.linthresh
        a[masked] = log
        a[~masked] *= self._linscale_adj
        return a

    def _inv_transform(self, a):
        """
        Inverse inplace Transformation.
        """
        masked = np.abs(a) > (self.linthresh * self._linscale_adj)
        sign = np.sign(a[masked])
        exp = np.exp(sign * a[masked] / self.linthresh - self._linscale_adj)
        exp *= sign * self.linthresh
        a[masked] = exp
        a[~masked] /= self._linscale_adj
        return a

    def _transform_vmin_vmax(self):
        """
        Calculates vmin and vmax in the transformed system.
        """
        vmin, vmax = self.vmin, self.vmax
        arr = np.array([vmax, vmin]).astype(np.float)
        self._upper, self._lower = self._transform(arr)

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        val = ma.asarray(value)
        val = val * (self._upper - self._lower) + self._lower
        return self._inv_transform(val)

    def autoscale(self, A):
        """
        Set *vmin*, *vmax* to min, max of *A*.
        """
        self.vmin = ma.min(A)
        self.vmax = ma.max(A)
        self._transform_vmin_vmax()

    def autoscale_None(self, A):
        """ autoscale only None-valued vmin or vmax """
        if self.vmin is not None and self.vmax is not None:
            pass
        if self.vmin is None:
            self.vmin = ma.min(A)
        if self.vmax is None:
            self.vmax = ma.max(A)
        self._transform_vmin_vmax()


class PowerNorm(Normalize):
    """
    Normalize a given value to the ``[0, 1]`` interval with a power-law
    scaling. This will clip any negative data points to 0.
    """
    def __init__(self, gamma, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            result.fill(0)
        else:
            res_mask = result.data < 0
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)
            resdat = result.data
            resdat -= vmin
            np.power(resdat, gamma, resdat)
            resdat /= (vmax - vmin) ** gamma

            result = np.ma.array(resdat, mask=result.mask, copy=False)
            result[res_mask] = 0
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        gamma = self.gamma
        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = ma.asarray(value)
            return ma.power(val, 1. / gamma) * (vmax - vmin) + vmin
        else:
            return pow(value, 1. / gamma) * (vmax - vmin) + vmin

    def autoscale(self, A):
        """
        Set *vmin*, *vmax* to min, max of *A*.
        """
        self.vmin = ma.min(A)
        if self.vmin < 0:
            self.vmin = 0
            warnings.warn("Power-law scaling on negative values is "
                          "ill-defined, clamping to 0.")

        self.vmax = ma.max(A)

    def autoscale_None(self, A):
        ' autoscale only None-valued vmin or vmax'
        if self.vmin is None and np.size(A) > 0:
            self.vmin = ma.min(A)
            if self.vmin < 0:
                self.vmin = 0
                warnings.warn("Power-law scaling on negative values is "
                              "ill-defined, clamping to 0.")

        if self.vmax is None and np.size(A) > 0:
            self.vmax = ma.max(A)


class BoundaryNorm(Normalize):
    """
    Generate a colormap index based on discrete intervals.

    Unlike :class:`Normalize` or :class:`LogNorm`,
    :class:`BoundaryNorm` maps values to integers instead of to the
    interval 0-1.

    Mapping to the 0-1 interval could have been done via
    piece-wise linear interpolation, but using integers seems
    simpler, and reduces the number of conversions back and forth
    between integer and floating point.
    """
    def __init__(self, boundaries, ncolors, clip=False):
        """
        *boundaries*
            a monotonically increasing sequence
        *ncolors*
            number of colors in the colormap to be used

        If::

            b[i] <= v < b[i+1]

        then v is mapped to color j;
        as i varies from 0 to len(boundaries)-2,
        j goes from 0 to ncolors-1.

        Out-of-range values are mapped
        to -1 if low and ncolors if high; these are converted
        to valid indices by
        :meth:`Colormap.__call__` .
        If clip == True, out-of-range values
        are mapped to 0 if low and ncolors-1 if high.
        """
        self.clip = clip
        self.vmin = boundaries[0]
        self.vmax = boundaries[-1]
        self.boundaries = np.asarray(boundaries)
        self.N = len(self.boundaries)
        self.Ncmap = ncolors
        if self.N - 1 == self.Ncmap:
            self._interp = False
        else:
            self._interp = True

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        xx, is_scalar = self.process_value(value)
        mask = ma.getmaskarray(xx)
        xx = np.atleast_1d(xx.filled(self.vmax + 1))
        if clip:
            np.clip(xx, self.vmin, self.vmax, out=xx)
            max_col = self.Ncmap - 1
        else:
            max_col = self.Ncmap
        iret = np.zeros(xx.shape, dtype=np.int16)
        for i, b in enumerate(self.boundaries):
            iret[xx >= b] = i
        if self._interp:
            scalefac = float(self.Ncmap - 1) / (self.N - 2)
            iret = (iret * scalefac).astype(np.int16)
        iret[xx < self.vmin] = -1
        iret[xx >= self.vmax] = max_col
        ret = ma.array(iret, mask=mask)
        if is_scalar:
            ret = int(ret[0])  # assume python scalar
        return ret

    def inverse(self, value):
        return ValueError("BoundaryNorm is not invertible")


class NoNorm(Normalize):
    """
    Dummy replacement for Normalize, for the case where we
    want to use indices directly in a
    :class:`~matplotlib.cm.ScalarMappable` .
    """
    def __call__(self, value, clip=None):
        return value

    def inverse(self, value):
        return value


def rgb_to_hsv(arr):
    """
    convert float rgb values (in the range [0, 1]), in a numpy array to hsv
    values.

    Parameters
    ----------
    arr : (..., 3) array-like
       All values must be in the range [0, 1]

    Returns
    -------
    hsv : (..., 3) ndarray
       Colors converted to hsv values in range [0, 1]
    """
    # make sure it is an ndarray
    arr = np.asarray(arr)

    # check length of the last dimension, should be _some_ sort of rgb
    if arr.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {shp} was found.".format(shp=arr.shape))

    in_ndim = arr.ndim
    if arr.ndim == 1:
        arr = np.array(arr, ndmin=2)

    # make sure we don't have an int image
    if arr.dtype.kind in ('iu'):
        arr = arr.astype(np.float32)

    out = np.zeros_like(arr)
    arr_max = arr.max(-1)
    ipos = arr_max > 0
    delta = arr.ptp(-1)
    s = np.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[..., 0] == arr_max) & ipos
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]
    # green is max
    idx = (arr[..., 1] == arr_max) & ipos
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0]) / delta[idx]
    # blue is max
    idx = (arr[..., 2] == arr_max) & ipos
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1]) / delta[idx]

    out[..., 0] = (out[..., 0] / 6.0) % 1.0
    out[..., 1] = s
    out[..., 2] = arr_max

    if in_ndim == 1:
        out.shape = (3,)

    return out


def hsv_to_rgb(hsv):
    """
    convert hsv values in a numpy array to rgb values
    all values assumed to be in range [0, 1]

    Parameters
    ----------
    hsv : (..., 3) array-like
       All values assumed to be in range [0, 1]

    Returns
    -------
    rgb : (..., 3) ndarray
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {shp} was found.".format(shp=hsv.shape))

    # if we got pased a 1D array, try to treat as
    # a single color and reshape as needed
    in_ndim = hsv.ndim
    if in_ndim == 1:
        hsv = np.array(hsv, ndmin=2)

    # make sure we don't have an int image
    if hsv.dtype.kind in ('iu'):
        hsv = hsv.astype(np.float32)

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(np.int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = np.empty_like(hsv)
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b

    if in_ndim == 1:
        rgb.shape = (3, )

    return rgb


class LightSource(object):
    """
    Create a light source coming from the specified azimuth and elevation.
    Angles are in degrees, with the azimuth measured
    clockwise from north and elevation up from the zero plane of the surface.

    The :meth:`shade` is used to produce "shaded" rgb values for a data array.
    :meth:`shade_rgb` can be used to combine an rgb image with
    The :meth:`shade_rgb`
    The :meth:`hillshade` produces an illumination map of a surface.
    """
    def __init__(self, azdeg=315, altdeg=45, hsv_min_val=0, hsv_max_val=1,
                 hsv_min_sat=1, hsv_max_sat=0):
        """
        Specify the azimuth (measured clockwise from south) and altitude
        (measured up from the plane of the surface) of the light source
        in degrees.

        Parameters
        ----------
        azdeg : number, optional
            The azimuth (0-360, degrees clockwise from North) of the light
            source. Defaults to 315 degrees (from the northwest).
        altdeg : number, optional
            The altitude (0-90, degrees up from horizontal) of the light
            source.  Defaults to 45 degrees from horizontal.

        Notes
        -----
        For backwards compatibility, the parameters *hsv_min_val*,
        *hsv_max_val*, *hsv_min_sat*, and *hsv_max_sat* may be supplied at
        initialization as well.  However, these parameters will only be used if
        "blend_mode='hsv'" is passed into :meth:`shade` or :meth:`shade_rgb`.
        See the documentation for :meth:`blend_hsv` for more details.
        """
        self.azdeg = azdeg
        self.altdeg = altdeg
        self.hsv_min_val = hsv_min_val
        self.hsv_max_val = hsv_max_val
        self.hsv_min_sat = hsv_min_sat
        self.hsv_max_sat = hsv_max_sat

    def hillshade(self, elevation, vert_exag=1, dx=1, dy=1, fraction=1.):
        """
        Calculates the illumination intensity for a surface using the defined
        azimuth and elevation for the light source.

        Imagine an artificial sun placed at infinity in some azimuth and
        elevation position illuminating our surface. The parts of the surface
        that slope toward the sun should brighten while those sides facing away
        should become darker.

        Parameters
        ----------
        elevation : array-like
            A 2d array (or equivalent) of the height values used to generate an
            illumination map
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs meters) or to exaggerate
            or de-emphasize topographic effects.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.
        Returns
        -------
        intensity : ndarray
            A 2d array of illumination values between 0-1, where 0 is
            completely in shadow and 1 is completely illuminated.
        """
        # Azimuth is in degrees clockwise from North. Convert to radians
        # counterclockwise from East (mathematical notation).
        az = np.radians(90 - self.azdeg)
        alt = np.radians(self.altdeg)

        # Because most image and raster GIS data has the first row in the array
        # as the "top" of the image, dy is implicitly negative.  This is
        # consistent to what `imshow` assumes, as well.
        dy = -dy

        # Calculate the intensity from the illumination angle
        dy, dx = np.gradient(vert_exag * elevation, dy, dx)
        # The aspect is defined by the _downhill_ direction, thus the negative
        aspect = np.arctan2(-dy, -dx)
        slope = 0.5 * np.pi - np.arctan(np.hypot(dx, dy))
        intensity = (np.sin(alt) * np.sin(slope) +
                     np.cos(alt) * np.cos(slope) * np.cos(az - aspect))

        # Apply contrast stretch
        imin, imax = intensity.min(), intensity.max()
        intensity *= fraction

        # Rescale to 0-1, keeping range before contrast stretch
        # If constant slope, keep relative scaling (i.e. flat should be 0.5,
        # fully occluded 0, etc.)
        if (imax - imin) > 1e-6:
            # Strictly speaking, this is incorrect. Negative values should be
            # clipped to 0 because they're fully occluded. However, rescaling
            # in this manner is consistent with the previous implementation and
            # visually appears better than a "hard" clip.
            intensity -= imin
            intensity /= (imax - imin)
        intensity = np.clip(intensity, 0, 1, intensity)

        return intensity

    def shade(self, data, cmap, norm=None, blend_mode='hsv', vmin=None,
              vmax=None, vert_exag=1, dx=1, dy=1, fraction=1, **kwargs):
        """
        Combine colormapped data values with an illumination intensity map
        (a.k.a.  "hillshade") of the values.

        Parameters
        ----------
        data : array-like
            A 2d array (or equivalent) of the height values used to generate a
            shaded map.
        cmap : `~matplotlib.colors.Colormap` instance
            The colormap used to color the *data* array. Note that this must be
            a `~matplotlib.colors.Colormap` instance.  For example, rather than
            passing in `cmap='gist_earth'`, use
            `cmap=plt.get_cmap('gist_earth')` instead.
        norm : `~matplotlib.colors.Normalize` instance, optional
            The normalization used to scale values before colormapping. If
            None, the input will be linearly scaled between its min and max.
        blend_mode : {'hsv', 'overlay', 'soft'} or callable, optional
            The type of blending used to combine the colormapped data values
            with the illumination intensity.  For backwards compatibility, this
            defaults to "hsv". Note that for most topographic surfaces,
            "overlay" or "soft" appear more visually realistic. If a
            user-defined function is supplied, it is expected to combine an
            MxNx3 RGB array of floats (ranging 0 to 1) with an MxNx1 hillshade
            array (also 0 to 1).  (Call signature `func(rgb, illum, **kwargs)`)
            Additional kwargs supplied to this function will be passed on to
            the *blend_mode* function.
        vmin : scalar or None, optional
            The minimum value used in colormapping *data*. If *None* the
            minimum value in *data* is used. If *norm* is specified, then this
            argument will be ignored.
        vmax : scalar or None, optional
            The maximum value used in colormapping *data*. If *None* the
            maximum value in *data* is used. If *norm* is specified, then this
            argument will be ignored.
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs meters) or to exaggerate
            or de-emphasize topography.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        fraction : number, optional
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.
        Additional kwargs are passed on to the *blend_mode* function.

        Returns
        -------
        rgba : ndarray
            An MxNx4 array of floats ranging between 0-1.
        """
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        if norm is None:
            norm = Normalize(vmin=vmin, vmax=vmax)

        rgb0 = cmap(norm(data))
        rgb1 = self.shade_rgb(rgb0, elevation=data, blend_mode=blend_mode,
                              vert_exag=vert_exag, dx=dx, dy=dy,
                              fraction=fraction, **kwargs)
        # Don't overwrite the alpha channel, if present.
        rgb0[..., :3] = rgb1[..., :3]
        return rgb0

    def shade_rgb(self, rgb, elevation, fraction=1., blend_mode='hsv',
                  vert_exag=1, dx=1, dy=1, **kwargs):
        """
        Take the input RGB array (ny*nx*3) adjust their color values
        to given the impression of a shaded relief map with a
        specified light source using the elevation (ny*nx).
        A new RGB array ((ny*nx*3)) is returned.

        Parameters
        ----------
        rgb : array-like
            An MxNx3 RGB array, assumed to be in the range of 0 to 1.
        elevation : array-like
            A 2d array (or equivalent) of the height values used to generate a
            shaded map.
        fraction : number
            Increases or decreases the contrast of the hillshade.  Values
            greater than one will cause intermediate values to move closer to
            full illumination or shadow (and clipping any values that move
            beyond 0 or 1). Note that this is not visually or mathematically
            the same as vertical exaggeration.
        blend_mode : {'hsv', 'overlay', 'soft'} or callable, optional
            The type of blending used to combine the colormapped data values
            with the illumination intensity.  For backwards compatibility, this
            defaults to "hsv". Note that for most topographic surfaces,
            "overlay" or "soft" appear more visually realistic. If a
            user-defined function is supplied, it is expected to combine an
            MxNx3 RGB array of floats (ranging 0 to 1) with an MxNx1 hillshade
            array (also 0 to 1).  (Call signature `func(rgb, illum, **kwargs)`)
            Additional kwargs supplied to this function will be passed on to
            the *blend_mode* function.
        vert_exag : number, optional
            The amount to exaggerate the elevation values by when calculating
            illumination. This can be used either to correct for differences in
            units between the x-y coordinate system and the elevation
            coordinate system (e.g. decimal degrees vs meters) or to exaggerate
            or de-emphasize topography.
        dx : number, optional
            The x-spacing (columns) of the input *elevation* grid.
        dy : number, optional
            The y-spacing (rows) of the input *elevation* grid.
        Additional kwargs are passed on to the *blend_mode* function.

        Returns
        -------
        shaded_rgb : ndarray
            An MxNx3 array of floats ranging between 0-1.
        """
        # Calculate the "hillshade" intensity.
        intensity = self.hillshade(elevation, vert_exag, dx, dy, fraction)
        intensity = intensity[..., np.newaxis]

        # Blend the hillshade and rgb data using the specified mode
        lookup = {
                'hsv': self.blend_hsv,
                'soft': self.blend_soft_light,
                'overlay': self.blend_overlay,
                }
        if blend_mode in lookup:
            blend = lookup[blend_mode](rgb, intensity, **kwargs)
        else:
            try:
                blend = blend_mode(rgb, intensity, **kwargs)
            except TypeError:
                msg = '"blend_mode" must be callable or one of {0}'
                raise ValueError(msg.format(lookup.keys))

        # Only apply result where hillshade intensity isn't masked
        if hasattr(intensity, 'mask'):
            mask = intensity.mask[..., 0]
            for i in range(3):
                blend[..., i][mask] = rgb[..., i][mask]

        return blend

    def blend_hsv(self, rgb, intensity, hsv_max_sat=None, hsv_max_val=None,
                  hsv_min_val=None, hsv_min_sat=None):
        """
        Take the input data array, convert to HSV values in the given colormap,
        then adjust those color values to give the impression of a shaded
        relief map with a specified light source.  RGBA values are returned,
        which can then be used to plot the shaded image with imshow.

        The color of the resulting image will be darkened by moving the (s,v)
        values (in hsv colorspace) toward (hsv_min_sat, hsv_min_val) in the
        shaded regions, or lightened by sliding (s,v) toward (hsv_max_sat
        hsv_max_val) in regions that are illuminated.  The default extremes are
        chose so that completely shaded points are nearly black (s = 1, v = 0)
        and completely illuminated points are nearly white (s = 0, v = 1).

        Parameters
        ----------
        rgb : ndarray
            An MxNx3 RGB array of floats ranging from 0 to 1 (color image).
        intensity : ndarray
            An MxNx1 array of floats ranging from 0 to 1 (grayscale image).
        hsv_max_sat : number, optional
            The maximum saturation value that the *intensity* map can shift the
            output image to. Defaults to 1.
        hsv_min_sat : number, optional
            The minimum saturation value that the *intensity* map can shift the
            output image to. Defaults to 0.
        hsv_max_val : number, optional
            The maximum value ("v" in "hsv") that the *intensity* map can shift
            the output image to. Defaults to 1.
        hsv_min_val: number, optional
            The minimum value ("v" in "hsv") that the *intensity* map can shift
            the output image to. Defaults to 0.

        Returns
        -------
        rgb : ndarray
            An MxNx3 RGB array representing the combined images.
        """
        # Backward compatibility...
        if hsv_max_sat is None:
            hsv_max_sat = self.hsv_max_sat
        if hsv_max_val is None:
            hsv_max_val = self.hsv_max_val
        if hsv_min_sat is None:
            hsv_min_sat = self.hsv_min_sat
        if hsv_min_val is None:
            hsv_min_val = self.hsv_min_val

        # Expects a 2D intensity array scaled between -1 to 1...
        intensity = intensity[..., 0]
        intensity = 2 * intensity - 1

        # convert to rgb, then rgb to hsv
        hsv = rgb_to_hsv(rgb[:, :, 0:3])

        # modify hsv values to simulate illumination.
        hsv[:, :, 1] = np.where(np.logical_and(np.abs(hsv[:, :, 1]) > 1.e-10,
                                               intensity > 0),
                                ((1. - intensity) * hsv[:, :, 1] +
                                 intensity * hsv_max_sat),
                                hsv[:, :, 1])

        hsv[:, :, 2] = np.where(intensity > 0,
                                ((1. - intensity) * hsv[:, :, 2] +
                                 intensity * hsv_max_val),
                                hsv[:, :, 2])

        hsv[:, :, 1] = np.where(np.logical_and(np.abs(hsv[:, :, 1]) > 1.e-10,
                                               intensity < 0),
                                ((1. + intensity) * hsv[:, :, 1] -
                                 intensity * hsv_min_sat),
                                hsv[:, :, 1])
        hsv[:, :, 2] = np.where(intensity < 0,
                                ((1. + intensity) * hsv[:, :, 2] -
                                 intensity * hsv_min_val),
                                hsv[:, :, 2])
        hsv[:, :, 1:] = np.where(hsv[:, :, 1:] < 0., 0, hsv[:, :, 1:])
        hsv[:, :, 1:] = np.where(hsv[:, :, 1:] > 1., 1, hsv[:, :, 1:])
        # convert modified hsv back to rgb.
        return hsv_to_rgb(hsv)

    def blend_soft_light(self, rgb, intensity):
        """
        Combines an rgb image with an intensity map using "soft light"
        blending.  Uses the "pegtop" formula.

        Parameters
        ----------
        rgb : ndarray
            An MxNx3 RGB array of floats ranging from 0 to 1 (color image).
        intensity : ndarray
            An MxNx1 array of floats ranging from 0 to 1 (grayscale image).

        Returns
        -------
        rgb : ndarray
            An MxNx3 RGB array representing the combined images.
        """
        return 2 * intensity * rgb + (1 - 2 * intensity) * rgb**2

    def blend_overlay(self, rgb, intensity):
        """
        Combines an rgb image with an intensity map using "overlay" blending.

        Parameters
        ----------
        rgb : ndarray
            An MxNx3 RGB array of floats ranging from 0 to 1 (color image).
        intensity : ndarray
            An MxNx1 array of floats ranging from 0 to 1 (grayscale image).

        Returns
        -------
        rgb : ndarray
            An MxNx3 RGB array representing the combined images.
        """
        low = 2 * intensity * rgb
        high = 1 - 2 * (1 - intensity) * (1 - rgb)
        return np.where(rgb <= 0.5, low, high)


def from_levels_and_colors(levels, colors, extend='neither'):
    """
    A helper routine to generate a cmap and a norm instance which
    behave similar to contourf's levels and colors arguments.

    Parameters
    ----------
    levels : sequence of numbers
        The quantization levels used to construct the :class:`BoundaryNorm`.
        Values ``v`` are quantizized to level ``i`` if
        ``lev[i] <= v < lev[i+1]``.
    colors : sequence of colors
        The fill color to use for each level. If `extend` is "neither" there
        must be ``n_level - 1`` colors. For an `extend` of "min" or "max" add
        one extra color, and for an `extend` of "both" add two colors.
    extend : {'neither', 'min', 'max', 'both'}, optional
        The behaviour when a value falls out of range of the given levels.
        See :func:`~matplotlib.pyplot.contourf` for details.

    Returns
    -------
    (cmap, norm) : tuple containing a :class:`Colormap` and a \
                   :class:`Normalize` instance
    """
    colors_i0 = 0
    colors_i1 = None

    if extend == 'both':
        colors_i0 = 1
        colors_i1 = -1
        extra_colors = 2
    elif extend == 'min':
        colors_i0 = 1
        extra_colors = 1
    elif extend == 'max':
        colors_i1 = -1
        extra_colors = 1
    elif extend == 'neither':
        extra_colors = 0
    else:
        raise ValueError('Unexpected value for extend: {0!r}'.format(extend))

    n_data_colors = len(levels) - 1
    n_expected_colors = n_data_colors + extra_colors
    if len(colors) != n_expected_colors:
        raise ValueError('With extend == {0!r} and n_levels == {1!r} expected'
                         ' n_colors == {2!r}. Got {3!r}.'
                         ''.format(extend, len(levels), n_expected_colors,
                                   len(colors)))

    cmap = ListedColormap(colors[colors_i0:colors_i1], N=n_data_colors)

    if extend in ['min', 'both']:
        cmap.set_under(colors[0])
    else:
        cmap.set_under('none')

    if extend in ['max', 'both']:
        cmap.set_over(colors[-1])
    else:
        cmap.set_over('none')

    cmap.colorbar_extend = extend

    norm = BoundaryNorm(levels, ncolors=n_data_colors)
    return cmap, norm
