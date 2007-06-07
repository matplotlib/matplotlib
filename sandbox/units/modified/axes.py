from __future__ import division, generators

import math, sys, warnings

from numerix import absolute, arange, array, asarray, ones, divide,\
     transpose, log, log10, Float, Float32, ravel, zeros,\
     Int16, Int32, Int, Float64, ceil, indices, \
     shape, which, where, sqrt, asum, compress, maximum, minimum, \
     typecode, concatenate, newaxis, reshape

import numerix.ma as ma

import matplotlib.mlab
from artist import Artist, setp
from axis import XAxis, YAxis
from cbook import iterable, is_string_like, flatten, enumerate, \
     allequal, dict_delall, popd, popall, silent_list
from collections import RegularPolyCollection, PolyCollection, LineCollection, QuadMesh
from colors import colorConverter, normalize, Colormap, \
        LinearSegmentedColormap, looks_like_color, is_color_like
import cm
from cm import ScalarMappable
from contour import ContourSet
import _image
from ticker import AutoLocator, LogLocator, NullLocator
from ticker import ScalarFormatter, LogFormatter, LogFormatterExponent, LogFormatterMathtext, NullFormatter

from image import AxesImage
from legend import Legend
from lines import Line2D, lineStyles, lineMarkers

from matplotlib.mlab import meshgrid, detrend_none, detrend_linear, \
     window_none, window_hanning, linspace, prctile
from matplotlib.numerix.mlab import flipud, amin, amax

from matplotlib import rcParams
from patches import Patch, Rectangle, Circle, Polygon, Arrow, Wedge, Shadow, FancyArrow, bbox_artist
import table
from text import Text, TextWithDash, _process_text_args
from transforms import Bbox, Point, Value, Affine, NonseparableTransformation
from transforms import  FuncXY, Func, LOG10, IDENTITY, POLAR
from transforms import get_bbox_transform, unit_bbox, one, origin, zero
from transforms import blend_xy_sep_transform, Interval, identity_transform
from transforms import PBox
from font_manager import FontProperties

from quiver import Quiver, QuiverKey

import matplotlib

if matplotlib._havedate:
    from dates import AutoDateFormatter, AutoDateLocator


def delete_masked_points(*args):
    """
    Find all masked points in a set of arguments, and return
    the arguments with only the unmasked points remaining.

    The overall mask is calculated from any masks that are present.
    If a mask is found, any argument that does not have the same
    dimensions is left unchanged; therefore the argument list may
    include arguments that can take string or array values, for
    example.

    Array arguments must all have the same shape, and must
    be one-dimensional.

    Written as a helper for scatter, but may be more generally
    useful.
    """
    masks = [ma.getmaskarray(x) for x in args if hasattr(x, 'mask')]
    if len(masks) == 0:
        return args
    mask = reduce(ma.mask_or, masks)
    margs = []
    for x in args:
        if shape(x) == shape(mask):
            margs.append(ma.masked_array(x, mask=mask).compressed())
        else:
            margs.append(x)
    return margs

def _process_plot_format(fmt):
    """
    Process a matlab(TM) style color/line style format string.  Return a
    linestyle, color tuple as a result of the processing.  Default
    values are ('-', 'b').  Example format strings include

    'ko'    : black circles
    '.b'    : blue dots
    'r--'   : red dashed lines

    See Line2D.lineStyles and GraphicsContext.colors for all possible
    styles and color format string.

    """

    colors = {
        'b' : 1,
        'g' : 1,
        'r' : 1,
        'c' : 1,
        'm' : 1,
        'y' : 1,
        'k' : 1,
        'w' : 1,
        }


    linestyle = 'None'
    marker = 'None'
    color = rcParams['lines.color']

    # handle the multi char special cases and strip them from the
    # string
    if fmt.find('--')>=0:
        linestyle = '--'
        fmt = fmt.replace('--', '')
    if fmt.find('-.')>=0:
        linestyle = '-.'
        fmt = fmt.replace('-.', '')

    chars = [c for c in fmt]

    for c in chars:
        if lineStyles.has_key(c):
            if linestyle != 'None':
                raise ValueError, 'Illegal format string "%s"; two linestyle symbols' % fmt

            linestyle = c
        elif lineMarkers.has_key(c):
            if marker != 'None':
                raise ValueError, 'Illegal format string "%s"; two marker symbols' % fmt
            marker = c
        elif colors.has_key(c):
            color = c
        else:
            err = 'Unrecognized character %c in format string' % c
            raise ValueError, err

    if linestyle == 'None' and marker == 'None':
        linestyle = rcParams['lines.linestyle']

    return linestyle, marker, color

class _process_plot_var_args:
    """

    Process variable length arguments to the plot command, so that
    plot commands like the following are supported

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of x, y, fmt are allowed
    """

    def __init__(self, command='plot'):
        self.command = command
        self._clear_color_cycle()

    def _clear_color_cycle(self):
        self.colors = ['b','g','r','c','m','y','k']
        # if the default line color is a color format string, move it up
        # in the que
        try: ind = self.colors.index(rcParams['lines.color'])
        except ValueError:
            self.firstColor = rcParams['lines.color']
        else:
            self.colors[0], self.colors[ind] = self.colors[ind], self.colors[0]
            self.firstColor = self.colors[0]

        self.Ncolors = len(self.colors)

        self.count = 0

    def _x_y_units_conversion(self, x, y):
        # convert, if necessary
        if (self._xunits and hasattr(x, 'convert_to')):
            x = x.convert_to(self._xunits)
        if (hasattr(x, 'get_value')):
            x = x.get_value()

        if (self._yunits and hasattr(y, 'convert_to')):
            y = y.convert_to(self._yunits)
        if (hasattr(y, 'get_value')):
            y = y.get_value()
        return (x,y)

    def __call__(self, *args, **kwargs):
        kwargs_copy = kwargs.copy()
        self._xunits = kwargs_copy.pop('xunits', None)
        self._yunits = kwargs_copy.pop('yunits', None)

        ret =  self._grab_next_args(*args, **kwargs_copy)
        return ret

    def set_lineprops(self, line, **kwargs):
        assert self.command == 'plot', 'set_lineprops only works with "plot"'
        for key, val in kwargs.items():
            funcName = "set_%s"%key
            if not hasattr(line,funcName):
                raise TypeError, 'There is no line property "%s"'%key
            func = getattr(line,funcName)
            func(val)

    def set_patchprops(self, fill_poly, **kwargs):
        assert self.command == 'fill', 'set_patchprops only works with "fill"'
        for key, val in kwargs.items():
            funcName = "set_%s"%key
            if not hasattr(fill_poly,funcName):
                raise TypeError, 'There is no patch property "%s"'%key
            func = getattr(fill_poly,funcName)
            func(val)

    def is_filled(self, marker):
        filled = ('o', '^', 'v', '<', '>', 's',
                  'd', 'D', 'h', 'H',
                  'p')
        return marker in filled


    def _plot_1_arg(self, y, **kwargs):
        assert self.command == 'plot', 'fill needs at least 2 arguments'
        if self.count==0:
            color = self.firstColor
        else:
            color = self.colors[int(self.count % self.Ncolors)]

        assert(iterable(y))

        dummy, y = self._x_y_units_conversion(None, y)

        try: N=max(y.shape)
        except AttributeError: N = len(y)
        ret =  Line2D(arange(N), y,
                      color = color,
                      markerfacecolor=color,
                      )
        self.set_lineprops(ret, **kwargs)
        self.count += 1
        return ret

    def _plot_2_args(self, tup2, **kwargs):
        if is_string_like(tup2[1]):

            assert self.command == 'plot', 'fill needs at least 2 non-string arguments'
            y, fmt = tup2

            assert(iterable(y))

            dummy, y = self._x_y_units_conversion(None, y)
            linestyle, marker, color = _process_plot_format(fmt)

            if self.is_filled(marker): mec = None # use default
            else: mec = color                     # use current color
            try: N=max(y.shape)
            except AttributeError: N = len(y)

            ret =  Line2D(xdata=arange(N), ydata=y,
                          color=color, linestyle=linestyle, marker=marker,
                          markerfacecolor=color,
                          markeredgecolor=mec,
                          )
            self.set_lineprops(ret, **kwargs)
            return ret
        else:

            x,y = tup2
            #print self.count, self.Ncolors, self.count % self.Ncolors
            assert(iterable(x))
            assert(iterable(y))
            x,y = self._x_y_units_conversion(x, y)
            if self.command == 'plot':
                c = self.colors[self.count % self.Ncolors]
                ret =  Line2D(x, y,
                              color = c,
                              markerfacecolor = c,
                              )
                self.set_lineprops(ret, **kwargs)
                self.count += 1
            elif self.command == 'fill':
                ret = Polygon( zip(x,y), fill=True, )
                self.set_patchprops(ret, **kwargs)

            return ret

    def _plot_3_args(self, tup3, **kwargs):
        kwargs_copy = kwargs.copy()
        if self.command == 'plot':
            x, y, fmt = tup3

            assert(iterable(x))
            assert(iterable(y))

            x,y = self._x_y_units_conversion(x, y)
            linestyle, marker, color = _process_plot_format(fmt)
            if self.is_filled(marker): mec = None # use default
            else: mec = color                     # use current color

            ret = Line2D(x, y, color=color,
                         linestyle=linestyle, marker=marker,
                         markerfacecolor=color,
                         markeredgecolor=mec,
                         )
            self.set_lineprops(ret, **kwargs_copy)
        if self.command == 'fill':
            x, y, facecolor = tup3
            x,y = self._x_y_units_conversion(x, y)
            ret = Polygon(zip(x,y),
                          facecolor = facecolor,
                          fill=True,
                          )
            self.set_patchprops(ret, **kwargs_copy)
        return ret

    def _grab_next_args(self, *args, **kwargs):

        remaining = args
        while 1:

            if len(remaining)==0: return
            if len(remaining)==1:
                yield self._plot_1_arg(remaining[0], **kwargs)
                remaining = []
                continue
            if len(remaining)==2:
                yield self._plot_2_args(remaining, **kwargs)
                remaining = []
                continue
            if len(remaining)==3:
                if not is_string_like(remaining[2]):
                    raise ValueError, 'third arg must be a format string'
                yield self._plot_3_args(remaining, **kwargs)
                remaining=[]
                continue
            if is_string_like(remaining[2]):
                yield self._plot_3_args(remaining[:3], **kwargs)
                remaining=remaining[3:]
            else:
                yield self._plot_2_args(remaining[:2], **kwargs)
                remaining=remaining[2:]
            #yield self._plot_2_args(remaining[:2])
            #remaining=args[2:]

ValueType=type(zero())
def makeValue(v):
    if type(v) == ValueType:
        return v
    else:
        return Value(v)

def units_conversion(x, units):
    if (units and hasattr(x, 'convert_to')):
        x = x.convert_to(units)
    if (hasattr(x, 'get_value')):
        x = x.get_value()
    return x

class Axes(Artist):
    """
    The Axes contains most of the figure elements: Axis, Tick, Line2D,
    Text, Polygon etc, and sets the coordinate system
    """

    scaled = {IDENTITY : 'linear',
              LOG10 : 'log',
              }
    def __init__(self, fig, rect,
                 axisbg = None, # defaults to rc axes.facecolor
                 frameon = True,
                 sharex=None, # use Axes instance's xaxis info
                 sharey=None, # use Axes instance's yaxis info
                 label='',
                 **kwargs
                 ):
        Artist.__init__(self)
        self._position = map(makeValue, rect)
        self._originalPosition = rect
        self.set_aspect('auto')
        self.set_adjustable('box')
        self.set_anchor('C')

        # unit locator/formatter maps
        self._unit_locator_map = None
        self._unit_formatter_map = None

        # must be set before set_figure
        self._sharex = sharex
        self._sharey = sharey
        # Flag: True if some other Axes instance is sharing our x or y axis
        self._masterx = False
        self._mastery = False
        if sharex: sharex._masterx = True
        if sharey: sharey._mastery = True
        self.set_label(label)
        self.set_figure(fig)

        # this call may differ for non-sep axes, eg polar
        self._init_axis()


        if axisbg is None: axisbg = rcParams['axes.facecolor']
        self._axisbg = axisbg
        self._frameon = frameon
        self._axisbelow = rcParams['axes.axisbelow']

        self._hold = rcParams['axes.hold']
        self._connected = {} # a dict from events to (id, func)
        self.cla()

        # funcs used to format x and y - fall back on major formatters
        self.fmt_xdata = None
        self.fmt_ydata = None

        # unit information
        self._xunits = None
        self._yunits = None

        self.set_cursor_props((1,'k')) # set the cursor properties for axes

        self._cachedRenderer = None
        self.set_navigate(True)
        self.set_navigate_mode(None)

        if len(kwargs): setp(self, **kwargs)

    def _init_axis(self):
        "move this out of __init__ because non-separable axes don't use it"
        self.xaxis = XAxis(self)
        self.yaxis = YAxis(self)

    _default_unit_locator_map = None
    _default_unit_formatter_map = None

    def _set_default_unit_to_locator_map(map_fn):
        Axes._default_unit_locator_map = staticmethod(map_fn)
    def _get_default_unit_to_locator_map():
        return Axes._default_unit_locator_map
    set_default_unit_to_locator_map = \
        staticmethod(_set_default_unit_to_locator_map)
    get_default_unit_to_locator_map = \
        staticmethod(_get_default_unit_to_locator_map)

    def set_unit_to_locator_map(self, map_fn):
        self._unit_locator_map = map_fn
    def get_unit_to_locator_map(self):
        return self._unit_locator_map

    def _set_default_unit_to_formatter_map(map_fn):
        Axes._default_unit_formatter_map = staticmethod(map_fn)
    def _get_default_unit_to_formatter_map():
        return Axes._default_unit_formatter_map
    set_default_unit_to_formatter_map = \
        staticmethod(_set_default_unit_to_formatter_map)
    get_default_unit_to_formatter_map = \
        staticmethod(_get_default_unit_to_formatter_map)

    def set_unit_to_formatter_map(self, map_fn):
        self._unit_formatter_map = map_fn
    def get_unit_to_formatter_map(self):
        return self._unit_formatter_map

    def _set_locators_for_units(self, units, axis):
        fn = self.get_unit_to_locator_map()
        if (not fn):
            fn = Axes.get_default_unit_to_locator_map()
        set_locators = False
        if (fn):
            try:
                locators = fn(units)
                axis.set_major_locator(locators[0])
                axis.set_minor_locator(locators[1])
                set_locators = True
            except: pass
        if (not set_locators):
            axis.set_major_locator(AutoLocator())
            axis.set_minor_locator(NullLocator())

    def _set_formatters_for_units(self, units, axis):
        fn = self.get_unit_to_formatter_map()
        if (not fn):
            fn = Axes.get_default_unit_to_formatter_map()
        set_formatters = False
        if (fn):
            try:
                formatters = fn(units)
                axis.set_major_formatter(formatters[0])
                axis.set_minor_formatter(formatters[1])
                set_formatters = True
            except: pass
        if (not set_formatters):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())

    def _update_units_args(self, kwarg_set):
        "update a set of args to include the default x/y unit settings"
        kwarg_set['xunits'] = kwarg_set.get('xunits', self._xunits)
        kwarg_set['yunits'] = kwarg_set.get('yunits', self._yunits)
        self.set_units(kwarg_set['xunits'], kwarg_set['yunits'])

    def set_units(self, xunits, yunits):
        """
        Set the Axes unit for the x axis and for the y axis

        ACCEPTS: a Unit instance for the x axis and a Unit instance
                 for the y axis
        """
        self.set_xunits(xunits)
        self.set_yunits(yunits)

    def set_xunits(self, xunits):
        """
        Set the Axes unit for the x axis

        ACCEPTS: a Unit instance
        """
        if (self._xunits == xunits):
            return

        self._xunits = xunits
        self._set_locators_for_units(xunits, self.xaxis)
        self._set_formatters_for_units(xunits, self.xaxis)

    def set_yunits(self, yunits):
        """
        Set the Axes unit for the y axis

        ACCEPTS: a Unit instance
        """
        if (self._yunits == yunits):
            return

        self._yunits = yunits
        self._set_locators_for_units(yunits, self.yaxis)
        self._set_formatters_for_units(yunits, self.yaxis)

    def set_figure(self, fig):
        """
        Set the Axes figure

        ACCEPTS: a Figure instance
        """
        Artist.set_figure(self, fig)

        l, b, w, h = self._position
        xmin = fig.bbox.ll().x()
        xmax = fig.bbox.ur().x()
        ymin = fig.bbox.ll().y()
        ymax = fig.bbox.ur().y()
        figw = xmax-xmin
        figh = ymax-ymin
        self.left   =  l*figw
        self.bottom =  b*figh
        self.right  =  (l+w)*figw
        self.top    =  (b+h)*figh


        self.bbox = Bbox( Point(self.left, self.bottom),
                          Point(self.right, self.top ),
                          )
        #these will be updated later as data is added
        self._set_lim_and_transforms()

    def _set_lim_and_transforms(self):
        """
        set the dataLim and viewLim BBox attributes and the
        transData and transAxes Transformation attributes
        """


        if self._sharex is not None:
            left=self._sharex.viewLim.ll().x()
            right=self._sharex.viewLim.ur().x()
        else:
            left=zero()
            right=one()
        if self._sharey is not None:
            bottom=self._sharey.viewLim.ll().y()
            top=self._sharey.viewLim.ur().y()
        else:
            bottom=zero()
            top=one()

        self.viewLim = Bbox(Point(left, bottom), Point(right, top))
        self.dataLim = unit_bbox()


        self.transData = get_bbox_transform(self.viewLim, self.bbox)
        self.transAxes = get_bbox_transform(unit_bbox(), self.bbox)

        if self._sharex:
            self.transData.set_funcx(self._sharex.transData.get_funcx())

        if self._sharey:
            self.transData.set_funcy(self._sharey.transData.get_funcy())

    def get_position(self, original=False):
        'Return the axes rectangle left, bottom, width, height'
        if original:
            return self._originalPosition[:]
        else:
            return [val.get() for val in self._position]

    def set_position(self, pos, which='both'):
        """
        Set the axes position with pos = [left, bottom, width, height]
        in relative 0,1 coords

        There are two position variables: one which is ultimately
        used, but which may be modified by apply_aspect, and a second
        which is the starting point for apply_aspect.

        which = 'active' to change the first;
                'original' to change the second;
                'both' to change both

        ACCEPTS: len(4) sequence of floats
        """
        if which in ('both', 'active'):
            # Change values within self._position--don't replace it.
            for num,val in zip(pos, self._position):
                val.set(num)
        if which in ('both', 'original'):
            self._originalPosition = pos

    def _set_artist_props(self, a):
        'set the boilerplate props for artists added to axes'
        a.set_figure(self.figure)
        if not a.is_transform_set():
            a.set_transform(self.transData)
        a.axes = self

    def cla(self):
        'Clear the current axes'

        self.xaxis.cla()
        self.yaxis.cla()

        self.dataLim.ignore(1)
        if self._sharex is not None:
            self.xaxis.major = self._sharex.xaxis.major
            self.xaxis.minor = self._sharex.xaxis.minor
        if self._sharey is not None:
            self.yaxis.major = self._sharey.yaxis.major
            self.yaxis.minor = self._sharey.yaxis.minor

        self._get_lines = _process_plot_var_args()
        self._get_patches_for_fill = _process_plot_var_args('fill')

        self._gridOn = rcParams['axes.grid']
        self.lines = []
        self.patches = []
        self.texts = []
        self.tables = []
        self.artists = []
        self.images = []
        self.legend_ = None
        self.collections = []  # collection.Collection instances

        self._autoscaleon = True

        self.grid(self._gridOn)
        self.title =  Text(
            x=0.5, y=1.02, text='',
            fontproperties=FontProperties(size=rcParams['axes.titlesize']),
            verticalalignment='bottom',
            horizontalalignment='center',
            )
        self.title.set_transform(self.transAxes)
        self.title.set_clip_box(None)

        self._set_artist_props(self.title)

        self.axesPatch = Rectangle(
            xy=(0,0), width=1, height=1,
            facecolor=self._axisbg,
            edgecolor=rcParams['axes.edgecolor'],
            )
        self.axesPatch.set_figure(self.figure)
        self.axesPatch.set_transform(self.transAxes)
        self.axesPatch.set_linewidth(rcParams['axes.linewidth'])
        self.axesFrame = Line2D((0,1,1,0,0), (0,0,1,1,0),
                            linewidth=rcParams['axes.linewidth'],
                            color=rcParams['axes.edgecolor'])
        self.axesFrame.set_transform(self.transAxes)
        self.axesFrame.set_zorder(2.5)
        self.axison = True

    def clear(self):
        'clear the axes'
        self.cla()

    def ishold(self):
        'return the HOLD status of the axes'
        return self._hold

    def hold(self, b=None):
        """
        HOLD(b=None)

        Set the hold state.  If hold is None (default), toggle the
        hold state.  Else set the hold state to boolean value b.

        Eg
            hold()      # toggle hold
            hold(True)  # hold is on
            hold(False) # hold is off


        When hold is True, subsequent plot commands will be added to
        the current axes.  When hold is False, the current axes and
        figure will be cleared on the next plot command

        """
        if b is None: self._hold = not self._hold
        else: self._hold = b

    def get_aspect(self):
        return self._aspect

    def set_aspect(self, aspect, adjustable=None, anchor=None):
        """
        aspect:
           'auto'   -  automatic; fill position rectangle with data
           'normal' -  same as 'auto'; deprecated
           'equal'  -  same scaling from data to plot units for x and y
            num     -  a circle will be stretched such that the height
                       is num times the width. aspect=1 is the same as
                       aspect='equal'.

        adjustable:
            'box'      - change physical size of axes
            'datalim'  - change xlim or ylim

        anchor:
            'C'     - centered
            'SW'    - lower left corner
            'S'     - middle of bottom edge
            'SE'    - lower right corner
                 etc.

        ACCEPTS: ['auto' | 'equal' | aspect_ratio]
        """
        if aspect in ('normal', 'auto'):
            self._aspect = 'auto'
        elif aspect == 'equal':
            self._aspect = 'equal'
        else:
            self._aspect = float(aspect) # raise ValueError if necessary

        if adjustable is not None:
            self.set_adjustable(adjustable)
        if anchor is not None:
            self.set_anchor(anchor)

    def get_adjustable(self):
        return self._adjustable

    def set_adjustable(self, adjustable):
        """
        ACCEPTS: ['box' | 'datalim']
        """
        if adjustable in ('box', 'datalim'):
            self._adjustable = adjustable
        else:
            raise ValueError('argument must be "box", or "datalim"')

    def get_anchor(self):
        return self._anchor

    def set_anchor(self, anchor):
        """
        ACCEPTS: ['C', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW', 'W']
        """
        if anchor in PBox.coefs.keys() or len(anchor) == 2:
            self._anchor = anchor
        else:
            raise ValueError('argument must be among %s' %
                                ', '.join(PBox.coefs.keys()))


    def apply_aspect(self, data_ratio = None):
        '''
        Use self._aspect and self._adjustable to modify the
        axes box or the view limits.
        The data_ratio kwarg is set to 1 for polar axes.  It is
        used only when _adjustable is 'box'.
        '''

        if self._aspect == 'auto':
            self.set_position( self._originalPosition , 'active')
            return

        if self._aspect == 'equal':
            A = 1
        else:
            A = self._aspect

        #Ensure at drawing time that any Axes involved in axis-sharing
        # does not have its position changed.
        if self._masterx or self._mastery or self._sharex or self._sharey:
            self._adjustable = 'datalim'

        figW,figH = self.get_figure().get_size_inches()
        fig_aspect = figH/figW
        #print 'figW, figH, fig_aspect', figW, figH, fig_aspect
        xmin,xmax = self.get_xlim()
        xsize = max(math.fabs(xmax-xmin), 1e-30)
        ymin,ymax = self.get_ylim()
        ysize = max(math.fabs(ymax-ymin), 1e-30)
        if self._adjustable == 'box':
            if data_ratio is None:
                data_ratio = ysize/xsize
            box_aspect = A * data_ratio
            pb = PBox(self._originalPosition)
            pb1 = pb.shrink_to_aspect(box_aspect, fig_aspect)
            self.set_position(pb1.anchor(self._anchor), 'active')
            return


        l,b,w,h = self.get_position()
        box_aspect = fig_aspect * (h/w)
        data_ratio = box_aspect / A
        #print 'box_aspect, data_ratio, ysize/xsize', box_aspect, data_ratio, ysize/xsize
        y_expander = (data_ratio*xsize/ysize - 1.0)
        #print 'y_expander', y_expander
        # If y_expander > 0, the dy/dx viewLim ratio needs to increase
        if abs(y_expander) < 0.005:
            #print 'good enough already'
            return
        dL = self.dataLim
        xr = 1.05 * dL.width()
        yr = 1.05 * dL.height()
        xmarg = xsize - xr
        ymarg = ysize - yr
        Ysize = data_ratio * xsize
        Xsize = ysize / data_ratio
        Xmarg = Xsize - xr
        Ymarg = Ysize - yr
        xm = 0  # Setting these targets to, e.g., 0.05*xr does not seem to help.
        ym = 0
        #print 'xmin, xmax, ymin, ymax', xmin, xmax, ymin, ymax
        #print 'xsize, Xsize, ysize, Ysize', xsize, Xsize, ysize, Ysize

        changex = ((self._sharey or self._mastery) and not
                            (self._sharex or self._masterx))
        changey = ((self._sharex or self._masterx) and not
                            (self._sharey or self._mastery))
        if changex and changey:
            warnings.warn("adjustable='datalim' cannot work with shared x and y axes")
            return
        if changex:
            adjust_y = False
        else:
            #print 'xmarg, ymarg, Xmarg, Ymarg', xmarg, ymarg, Xmarg, Ymarg
            if xmarg > xm and ymarg > ym:
                adjy = Ymarg > 0 and y_expander < 0
            else:
                adjy = y_expander > 0
            #print 'y_expander, adjy', y_expander, adjy
            adjust_y = changey or adjy  #(Ymarg > xmarg)
        if adjust_y:
            yc = 0.5*(ymin+ymax)
            y0 = yc - Ysize/2.0
            y1 = yc + Ysize/2.0
            self.set_ylim((y0, y1))
            #print 'New y0, y1:', y0, y1
            #print 'New ysize, ysize/xsize', y1-y0, (y1-y0)/xsize
        else:
            xc = 0.5*(xmin+xmax)
            x0 = xc - Xsize/2.0
            x1 = xc + Xsize/2.0
            self.set_xlim((x0, x1))
            #print 'New x0, x1:', x0, x1
            #print 'New xsize, ysize/xsize', x1-x0, ysize/(x1-x0)



    def axis(self, *v, **kwargs):
        '''
        Convenience method for manipulating the x and y view limits
        and the aspect ratio of the plot.

        See docstring for pylab.axis.
        '''
        if len(v)==1 and is_string_like(v[0]):
            s = v[0].lower()
            if s=='on': self.set_axis_on()
            elif s=='off': self.set_axis_off()
            elif s in ('equal', 'tight', 'scaled', 'normal', 'auto', 'image'):
                self.set_autoscale_on(True)
                self.set_aspect('auto')
                self.autoscale_view()
                self.apply_aspect()
                if s=='equal':
                    self.set_aspect('equal', adjustable='datalim')
                elif s == 'scaled':
                    self.set_aspect('equal', adjustable='box', anchor='C')
                elif s=='tight':
                    self.autoscale_view(tight=True)
                    self.set_autoscale_on(False)
                elif s == 'image':
                    self.autoscale_view(tight=True)
                    self.set_autoscale_on(False)
                    self.set_aspect('equal', adjustable='box', anchor='C')

            else:
                raise ValueError('Unrecognized string %s to axis; try on or off' % s)
            xmin, xmax = self.get_xlim()
            ymin, ymax = self.get_ylim()
            return xmin, xmax, ymin, ymax

        try: v[0]
        except IndexError:
            xmin, xmax = self.set_xlim(**kwargs)
            ymin, ymax = self.set_ylim(**kwargs)
            return xmin, xmax, ymin, ymax

        v = v[0]
        if len(v) != 4:
            raise ValueError('v must contain [xmin xmax ymin ymax]')


        self.set_xlim([v[0], v[1]])
        self.set_ylim([v[2], v[3]])

        return v


    def get_child_artists(self):
        """
        Return a list of artists the axes contains.  Deprecated
        """
        artists = [self.title, self.axesPatch, self.xaxis, self.yaxis]
        artists.extend(self.lines)
        artists.extend(self.patches)
        artists.extend(self.texts)
        artists.extend(self.collections)
        if self.legend_ is not None:
            artists.append(self.legend_)
        return silent_list('Artist', artists)

    def get_frame(self):
        'Return the axes Rectangle frame'
        return self.axesPatch

    def get_legend(self):
        'Return the Legend instance, or None if no legend is defined'
        return self.legend_

    def get_images(self):
        'return a list of Axes images contained by the Axes'
        return silent_list('AxesImage', self.images)

    def get_lines(self):
        'Return a list of lines contained by the Axes'
        return silent_list('Line2D', self.lines)

    def get_xaxis(self):
        'Return the XAxis instance'
        return self.xaxis

    def get_xgridlines(self):
        'Get the x grid lines as a list of Line2D instances'
        return silent_list('Line2D xgridline', self.xaxis.get_gridlines())


    def get_xticklines(self):
        'Get the xtick lines as a list of Line2D instances'
        return silent_list('Text xtickline', self.xaxis.get_ticklines())


    def get_yaxis(self):
        'Return the YAxis instance'
        return self.yaxis

    def get_ygridlines(self):
        'Get the y grid lines as a list of Line2D instances'
        return silent_list('Line2D ygridline', self.yaxis.get_gridlines())

    def get_yticklines(self):
        'Get the ytick lines as a list of Line2D instances'
        return silent_list('Line2D ytickline', self.yaxis.get_ticklines())

    #### Adding and tracking artists

    def has_data(self):
        '''Return true if any artists have been added to axes.

        This should not be used to determine whether the dataLim
        need to be updated, and may not actually be useful for
        anything.
        '''
        return (
            len(self.collections)
            len(self.images)
            len(self.lines)
            len(self.patches))>0

    def add_artist(self, a):
        'Add any artist to the axes'
        self.artists.append(a)
        self._set_artist_props(a)

    def add_collection(self, collection, autolim=False):
        'add a Collection instance to Axes'
        self.collections.append(collection)
        self._set_artist_props(collection)
        collection.set_clip_box(self.bbox)
        if autolim:
            self.update_datalim(collection.get_verts(self.transData))

    def add_line(self, l):
        'Add a line to the list of plot lines'
        self._set_artist_props(l)
        l.set_clip_box(self.bbox)
        xdata = l.get_xdata(valid_only=True)
        ydata = l.get_ydata(valid_only=True)

        if l.get_transform() != self.transData:
            xys = self._get_verts_in_data_coords(
                l.get_transform(), zip(xdata, ydata))
            xdata = array([x for x,y in xys])
            ydata = array([y for x,y in xys])

        self.update_datalim_numerix( xdata, ydata )
        label = l.get_label()
        if not label: l.set_label('line%d'%len(self.lines))
        self.lines.append(l)

    def add_patch(self, p):
        """
        Add a patch to the list of Axes patches; the clipbox will be
        set to the Axes clipping box.  If the transform is not set, it
        wil be set to self.transData.
        """

        self._set_artist_props(p)
        p.set_clip_box(self.bbox)
        xys = self._get_verts_in_data_coords(
            p.get_transform(), p.get_verts())
        self.update_datalim(xys)
        self.patches.append(p)

    def add_table(self, tab):
        'Add a table instance to the list of axes tables'
        self._set_artist_props(tab)
        self.tables.append(tab)

    def update_datalim(self, xys):
        'Update the data lim bbox with seq of xy tups or equiv. 2-D array'
        # if no data is set currently, the bbox will ignore its
        # limits and set the bound to be the bounds of the xydata.
        # Otherwise, it will compute the bounds of it's current data
        # and the data in xydata
        xys = asarray(xys)
        self.dataLim.update_numerix_xy(xys, -1)


    def update_datalim_numerix(self, x, y):
        'Update the data lim bbox with seq of xy tups'
        # if no data is set currently, the bbox will ignore it's
        # limits and set the bound to be the bounds of the xydata.
        # Otherwise, it will compute the bounds of it's current data
        # and the data in xydata
        #print type(x), type(y)
        self.dataLim.update_numerix(x, y, -1)

    def _get_verts_in_data_coords(self, trans, xys):
        if trans == self.transData:
            return xys
        # data is not in axis data units.  We must transform it to
        # display and then back to data to get it in data units
        #xys = trans.seq_xy_tups(xys)
        #return [ self.transData.inverse_xy_tup(xy) for xy in xys]
        xys = trans.numerix_xy(asarray(xys))
        return self.transData.inverse_numerix_xy(xys)


    def in_axes(self, xwin, ywin):
        'return True is the point xwin, ywin (display coords) are in the Axes'
        return self.bbox.contains(xwin, ywin)

    def get_autoscale_on(self):
        """
        Get whether autoscaling is applied on plot commands
        """
        return self._autoscaleon

    def set_autoscale_on(self, b):
        """
        Set whether autoscaling is applied on plot commands

        ACCEPTS: True|False
        """
        self._autoscaleon = b


    def autoscale_view(self, tight=False, scalex=True, scaley=True):
        """
        autoscale the view limits using the data limits. You can
        selectively autoscale only a single axis, eg, the xaxis by
        setting scaley to False.  The autoscaling preserves any
        axis direction reversal that has already been done.
        """
        # if image data only just use the datalim

        if not self._autoscaleon: return
        if (tight or (len(self.images)>0 and
                      len(self.lines)==0 and
                      len(self.patches)==0)):

            if scalex: self.set_xlim(self.dataLim.intervalx().get_bounds())

            if scaley: self.set_ylim(self.dataLim.intervaly().get_bounds())
            return

        if scalex:
            xl = self.get_xlim()
            XL = self.xaxis.get_major_locator().autoscale()
            if xl[1] < xl[0]:
                XL = XL[::-1]
            self.set_xlim(XL)
        if scaley:
            yl = self.get_ylim()
            YL = self.yaxis.get_major_locator().autoscale()
            if yl[1] < yl[0]:
                YL = YL[::-1]
            self.set_ylim(YL)
    #### Drawing

    def draw(self, renderer=None, inframe=False):
        "Draw everything (plot lines, axes, labels)"
        if renderer is None:
            renderer = self._cachedRenderer

        if renderer is None:
            raise RuntimeError('No renderer defined')
        if not self.get_visible(): return
        renderer.open_group('axes')
        self.apply_aspect()
        try: self.transData.freeze()  # eval the lazy objects
        except ValueError:
            print >> sys.stderr, 'data freeze value error', self.get_position(), self.dataLim.get_bounds(), self.viewLim.get_bounds()
            raise

        self.transAxes.freeze()  # eval the lazy objects
        if self.axison and self._frameon: self.axesPatch.draw(renderer)
        artists = []

        if len(self.images)==1:
            im = self.images[0]
            artists.append(im)

        elif len(self.images)>1:

            # make a composite image blending alpha
            # list of (_image.Image, ox, oy)


            ims = [(im.make_image(),0,0) for im in self.images if im.get_visible()]


            im = _image.from_images(self.bbox.height(), self.bbox.width(), ims)
            im.is_grayscale = False
            l, b, w, h = self.bbox.get_bounds()
            # composite images need special args so they will not
            # respect z-order for now
            renderer.draw_image(l, b, im, self.bbox)



        artists.extend(self.collections)
        artists.extend(self.patches)
        artists.extend(self.lines)
        artists.extend(self.texts)
        artists.extend(self.artists)
        if self.axison and not inframe:
            if self._axisbelow:
                self.xaxis.set_zorder(0.5)
                self.yaxis.set_zorder(0.5)
            else:
                self.xaxis.set_zorder(2.5)
                self.yaxis.set_zorder(2.5)
            artists.extend([self.xaxis, self.yaxis])
        if not inframe: artists.append(self.title)
        artists.extend(self.tables)
        if self.legend_ is not None:
            artists.append(self.legend_)
        if self.axison and self._frameon:
            artists.append(self.axesFrame)

        # keep track of i to guarantee stable sort for python 2.2
        dsu = [ (a.zorder, i, a) for i, a in enumerate(artists)
                if not a.get_animated()]
        dsu.sort()

        for zorder, i, a in dsu:
            a.draw(renderer)

        self.transData.thaw()  # release the lazy objects
        self.transAxes.thaw()  # release the lazy objects
        renderer.close_group('axes')
        self._cachedRenderer = renderer

    def draw_artist(self, a):
        """
        This method can only be used after an initial draw which
        caches the renderer.  It is used to efficiently update Axes
        data (axis ticks, labels, etc are not updated)
        """
        assert self._cachedRenderer is not None
        a.draw(self._cachedRenderer)

    def redraw_in_frame(self):
        """
        This method can only be used after an initial draw which
        caches the renderer.  It is used to efficiently update Axes
        data (axis ticks, labels, etc are not updated)
        """
        assert self._cachedRenderer is not None
        self.draw(self._cachedRenderer, inframe=True)

    def get_renderer_cache(self):
        return self._cachedRenderer

    def __draw_animate(self):
        # ignore for now; broken
        if self._lastRenderer is None:
            raise RuntimeError('You must first call ax.draw()')
        dsu = [(a.zorder, a) for a in self.animated.keys()]
        dsu.sort()
        renderer = self._lastRenderer
        renderer.blit()
        for tmp, a in dsu:
            a.draw(renderer)

    #### Axes rectangle characteristics

    def get_frame_on(self):
        """
        Get whether the axes rectangle patch is drawn
        """
        return self._frameon

    def set_frame_on(self, b):
        """
        Set whether the axes rectangle patch is drawn

        ACCEPTS: True|False
        """
        self._frameon = b

    def get_axisbelow(self):
        """
        Get whether axist below is true or not
        """
        return self._axisbelow

    def set_axisbelow(self, b):
        """
        Set whether the axis ticks and gridlines are above or below most artists

        ACCEPTS: True|False
        """
        self._axisbelow = b

    def grid(self, b=None, **kwargs):
        """
        GRID(self, b=None, **kwargs)
        Set the axes grids on or off; b is a boolean

        if b is None and len(kwargs)==0, toggle the grid state.  if
        kwargs are supplied, it is assumed that you want a grid and b
        is thus set to True

        kawrgs are used to set the grid line properties, eg

          ax.grid(color='r', linestyle='-', linewidth=2)

        See getp(line) for more information on valid keywords for line
        properties.
        """
        if len(kwargs): b = True
        self.xaxis.grid(b, **kwargs)
        self.yaxis.grid(b, **kwargs)

    def set_axis_off(self):
        """
        turn off the axis

        ACCEPTS: void
        """
        self.axison = False

    def set_axis_on(self):
        """
        turn on the axis

        ACCEPTS: void
        """
        self.axison = True

    def get_axis_bgcolor(self):
        'Return the axis background color'
        return self._axisbg

    def set_axis_bgcolor(self, color):
        """
        set the axes background color

        ACCEPTS: any matplotlib color - see help(colors)
        """

        self._axisbg = color
        self.axesPatch.set_facecolor(color)

    ### data limits, ticks, tick labels, and formatting

    def get_xlim(self):
        'Get the x axis range [xmin, xmax]'
        return self.viewLim.intervalx().get_bounds()


    def set_xlim(self, *args, **kwargs):
        """
        set_xlim(self, *args, **kwargs):

        Set the limits for the xaxis; v = [xmin, xmax]

        set_xlim((valmin, valmax))
        set_xlim(valmin, valmax)
        set_xlim(xmin=1) # xmax unchanged
        set_xlim(xmax=1) # xmin unchanged

        Valid kwargs:

        xmin : the min of the xlim
        xmax : the max of the xlim
        emit : notify observers of lim change


        Returns the current xlimits as a length 2 tuple

        ACCEPTS: len(2) sequence of floats
        """
        # convert x axis units
        kwargs_copy = kwargs.copy()
        self._update_units_args(kwargs_copy)

        vmin, vmax = self.get_xlim()

        xmin = popd(kwargs_copy, 'xmin', None)
        xmax = popd(kwargs_copy, 'xmax', None)
        emit = popd(kwargs_copy, 'emit', False)
        if len(args)!=0 and (xmin is not None or xmax is not None):
            raise TypeError('You cannot pass args and xmin/xmax kwargs')

        if len(args)==0:
            if xmin is not None: vmin = xmin
            if xmax is not None: vmax = xmax
        elif len(args)==1:
            vmin, vmax = args[0]
        elif len(args)==2:
            vmin, vmax = args
        else:
            raise ValueError('args must be length 0, 1 or 2')

        vmin = units_conversion(vmin, self._xunits)
        vmax = units_conversion(vmax, self._xunits)

        if self.transData.get_funcx().get_type()==LOG10 and min(vmin, vmax)<=0:
            raise ValueError('Cannot set nonpositive limits with log transform')

        if abs(vmax - vmin) < 1e-38:
            warnings.warn("vmax too close to vmin; adjusting")
            vmin -= 1e-38
            vmax += 1e-38
        self.viewLim.intervalx().set_bounds(vmin, vmax)
        if emit: self._send_xlim_event()
        return vmin, vmax

    def get_xscale(self):
        'return the xaxis scale string: log or linear'
        return self.scaled[self.transData.get_funcx().get_type()]

    def set_xscale(self, value, basex = 10, subsx=None):
        """
        SET_XSCALE(value, basex=10, subsx=None)

        Set the xscaling: 'log' or 'linear'

        If value is 'log', the additional kwargs have the following meaning

            * basex: base of the logarithm

            * subsx: a sequence of the location of the minor ticks;
              None defaults to autosubs, which depend on the number of
              decades in the plot.  Eg for base 10, subsx=(1,2,5) will
              put minor ticks on 1,2,5,11,12,15,21, ....To turn off
              minor ticking, set subsx=[]

        ACCEPTS: ['log' | 'linear' ]
        """

        #if subsx is None: subsx = range(2, basex)
        assert(value.lower() in ('log', 'linear', ))
        if value == 'log':
            self.xaxis.set_major_locator(LogLocator(basex))
            self.xaxis.set_major_formatter(LogFormatterMathtext(basex))
            self.xaxis.set_minor_locator(LogLocator(basex,subsx))
            self.transData.get_funcx().set_type(LOG10)
            minx, maxx = self.get_xlim()
            if min(minx, maxx)<=0:
                self.autoscale_view()
        elif value == 'linear':
            self.xaxis.set_major_locator(AutoLocator())
            self.xaxis.set_major_formatter(ScalarFormatter())
            self.xaxis.set_minor_locator(NullLocator())
            self.xaxis.set_minor_formatter(NullFormatter())
            self.transData.get_funcx().set_type( IDENTITY )

    def get_xticks(self):
        'Return the x ticks as a list of locations'
        return self.xaxis.get_ticklocs()

    def set_xticks(self, ticks):
        """
        Set the x ticks with list of ticks

        ACCEPTS: sequence of floats
        """
        return self.xaxis.set_ticks(ticks)

    def get_xticklabels(self):
        'Get the xtick labels as a list of Text instances'
        return silent_list('Text xticklabel', self.xaxis.get_ticklabels())

    def set_xticklabels(self, labels, fontdict=None, **kwargs):
        """
        SET_XTICKLABELS(labels, fontdict=None, **kwargs)

        Set the xtick labels with list of strings labels Return a list of axis
        text instances

        ACCEPTS: sequence of strings
        """
        return self.xaxis.set_ticklabels(labels, fontdict, **kwargs)

    def get_ylim(self):
        'Get the y axis range [ymin, ymax]'
        return self.viewLim.intervaly().get_bounds()

    def set_ylim(self, *args, **kwargs):
        """
        set_ylim(self, *args, **kwargs):

        Set the limits for the yaxis; v = [ymin, ymax]

        set_ylim((valmin, valmax))
        set_ylim(valmin, valmax)
        set_ylim(ymin=1) # ymax unchanged
        set_ylim(ymax=1) # ymin unchanged

        Valid kwargs:

        ymin : the min of the ylim
        ymax : the max of the ylim
        emit : notify observers of lim change


        Returns the current ylimits as a length 2 tuple

        ACCEPTS: len(2) sequence of floats
        """
        # convert x axis units
        kwargs_copy = kwargs.copy()
        self._update_units_args(kwargs_copy)

        vmin, vmax = self.get_ylim()

        ymin = popd(kwargs, 'ymin', None)
        ymax = popd(kwargs, 'ymax', None)
        emit = popd(kwargs, 'emit', False)
        if len(args)!=0 and (ymin is not None or ymax is not None):
            raise TypeError('You cannot pass args and ymin/ymax kwargs')

        if len(args)==0:
            if ymin is not None: vmin = ymin
            if ymax is not None: vmax = ymax
        elif len(args)==1:
            vmin, vmax = args[0]
        elif len(args)==2:
            vmin, vmax = args
        else:
            raise ValueError('args must be length 0, 1 or 2')

        vmin = units_conversion(vmin, self._yunits)
        vmax = units_conversion(vmax, self._yunits)

        if self.transData.get_funcy().get_type()==LOG10 and min(vmin, vmax)<=0:
            raise ValueError('Cannot set nonpositive limits with log transform')
        if abs(vmax - vmin) < 1e-38:
            warnings.warn("vmax too close to vmin; adjusting")
            vmin -= 1e-38
            vmax += 1e-38
        self.viewLim.intervaly().set_bounds(vmin, vmax)
        if emit: self._send_ylim_event()
        return vmin, vmax

    def get_yscale(self):
        'return the yaxis scale string: log or linear'
        return self.scaled[self.transData.get_funcy().get_type()]

    def set_yscale(self, value, basey=10, subsy=None):
        """
        SET_YSCALE(value, basey=10, subsy=None)

        Set the yscaling: 'log' or 'linear'

        If value is 'log', the additional kwargs have the following meaning

            * basey: base of the logarithm

            * subsy: a sequence of the location of the minor ticks;
              None defaults to autosubs, which depend on the number of
              decades in the plot.  Eg for base 10, subsy=(1,2,5) will
              put minor ticks on 1,2,5,11,12,15, 21, ....To turn off
              minor ticking, set subsy=[]

        ACCEPTS: ['log' | 'linear']
        """

        #if subsy is None: subsy = range(2, basey)
        assert(value.lower() in ('log', 'linear', ))

        if value == 'log':
            self.yaxis.set_major_locator(LogLocator(basey))
            self.yaxis.set_major_formatter(LogFormatterMathtext(basey))
            self.yaxis.set_minor_locator(LogLocator(basey,subsy))
            self.transData.get_funcy().set_type(LOG10)
            miny, maxy = self.get_ylim()
            if min(miny, maxy)<=0:
                self.autoscale_view()

        elif value == 'linear':
            self.yaxis.set_major_locator(AutoLocator())
            self.yaxis.set_major_formatter(ScalarFormatter())
            self.yaxis.set_minor_locator(NullLocator())
            self.yaxis.set_minor_formatter(NullFormatter())
            self.transData.get_funcy().set_type( IDENTITY )

    def get_yticks(self):
        'Return the y ticks as a list of locations'
        return self.yaxis.get_ticklocs()

    def set_yticks(self, ticks):
        """
        Set the y ticks with list of ticks

        ACCEPTS: sequence of floats
        """
        return self.yaxis.set_ticks(ticks)

    def get_yticklabels(self):
        'Get the ytick labels as a list of Text instances'
        return silent_list('Text yticklabel', self.yaxis.get_ticklabels())

    def set_yticklabels(self, labels, fontdict=None, **kwargs):
        """
        SET_YTICKLABELS(labels, fontdict=None, **kwargs)

        Set the ytick labels with list of strings labels.  Return a list of
        Text instances

        ACCEPTS: sequence of strings
        """
        return self.yaxis.set_ticklabels(labels, fontdict, **kwargs)

    def toggle_log_lineary(self):
        'toggle between log and linear on the y axis'
        funcy = self.transData.get_funcy().get_type()
        if funcy==LOG10: self.set_yscale('linear')
        elif funcy==IDENTITY: self.set_yscale('log')

    def xaxis_date(self, tz=None):
        """Sets up x-axis ticks and labels that treat the x data as dates.

        tz is the time zone to use in labeling dates.  Defaults to rc value.
        """

        locator = AutoDateLocator(tz)
        formatter = AutoDateFormatter(locator)
        self.xaxis.set_major_locator(locator)
        self.xaxis.set_major_formatter(formatter)

    def yaxis_date(self, tz=None):
        """Sets up y-axis ticks and labels that treat the y data as dates.

        tz is the time zone to use in labeling dates.  Defaults to rc value.
        """

        locator = AutoDateLocator(tz)
        formatter = AutoDateFormatter(locator)
        self.yaxis.set_major_locator(locator)
        self.yaxis.set_major_formatter(formatter)

    def format_xdata(self, x):
        """
        Return x string formatted.  This function will use the attribute
        self.fmt_xdata if it is callable, else will fall back on the xaxis
        major formatter
        """
        try: return self.fmt_xdata(x)
        except TypeError:
            func = self.xaxis.get_major_formatter().format_data
            val = func(x)
            return val

    def format_ydata(self, y):
        """
        Return y string formatted.  This function will use the attribute
        self.fmt_ydata if it is callable, else will fall back on the yaxis
        major formatter
        """
        try: return self.fmt_ydata(y)
        except TypeError:
            func = self.yaxis.get_major_formatter().format_data
            val =  func(y)
            return val

    def format_coord(self, x, y):
        'return a format string formatting the x, y coord'

        xs = self.format_xdata(x)
        ys = self.format_ydata(y)
        return  'x=%s, y=%s'%(xs,ys)


    #### Interactive manipulation

    def get_navigate(self):
        """
        Get whether the axes responds to navigation commands
        """
        return self._navigate

    def set_navigate(self, b):
        """
        Set whether the axes responds to navigation toolbar commands

        ACCEPTS: True|False
        """
        self._navigate = b

    def get_navigate_mode(self):
        """
        Get the navigation toolbar button status: 'PAN', 'ZOOM', or None
        """
        return self._navigate_mode

    def set_navigate_mode(self, b):
        """
        Set the navigation toolbar button status;
        this is not a user-API function.

        """
        self._navigate_mode = b

    def get_cursor_props(self):
        """return the cursor props as a linewidth, color tuple where
        linewidth is a float and color is an RGBA tuple"""
        return self._cursorProps

    def set_cursor_props(self, *args):
        """
        Set the cursor property as
        ax.set_cursor_props(linewidth, color)  OR
        ax.set_cursor_props((linewidth, color))

        ACCEPTS: a (float, color) tuple
        """
        if len(args)==1:
            lw, c = args[0]
        elif len(args)==2:
            lw, c = args
        else:
            raise ValueError('args must be a (linewidth, color) tuple')
        c =colorConverter.to_rgba(c)
        self._cursorProps = lw, c

    def _send_xlim_event(self):
        for cid, func in self._connected.get('xlim_changed', []):
            func(self)

    def _send_ylim_event(self):
        for cid, func in self._connected.get('ylim_changed', []):
            func(self)

    def panx(self, numsteps):
        'Pan the x axis numsteps (plus pan right, minus pan left)'
        self.xaxis.pan(numsteps)
        xmin, xmax = self.viewLim.intervalx().get_bounds()
        self._send_xlim_event()

    def pany(self, numsteps):
        'Pan the x axis numsteps (plus pan up, minus pan down)'
        self.yaxis.pan(numsteps)
        self._send_ylim_event()

    def zoomx(self, numsteps):
        'Zoom in on the x xaxis numsteps (plus for zoom in, minus for zoom out)'
        self.xaxis.zoom(numsteps)
        xmin, xmax = self.viewLim.intervalx().get_bounds()
        self._send_xlim_event()

    def zoomy(self, numsteps):
        'Zoom in on the x xaxis numsteps (plus for zoom in, minus for zoom out)'
        self.yaxis.zoom(numsteps)
        self._send_ylim_event()

    _cid = 0
    _events = ('xlim_changed', 'ylim_changed')

    def connect(self, s, func):
        """
        Register observers to be notified when certain events occur.  Register
        with callback functions with the following signatures.  The function
        has the following signature

            func(ax)  # where ax is the instance making the callback.

        The following events can be connected to:

          'xlim_changed','ylim_changed'

        The connection id is is returned - you can use this with
        disconnect to disconnect from the axes event

        """

        if s not in Axes._events:
            raise ValueError('You can only connect to the following axes events: %s' % ', '.join(Axes._events))

        cid = Axes._cid
        self._connected.setdefault(s, []).append((cid, func))
        Axes._cid += 1
        return cid

    def disconnect(self, cid):
        'disconnect from the Axes event.'
        for key, val in self._connected.items():
            for item in val:
                if item[0] == cid:
                    self._connected[key].remove(item)
                    return

    def pick(self, x, y, trans=None, among=None):
        """
        Return the artist under point that is closest to the x, y.  if trans
        is None, x, and y are in window coords, 0,0 = lower left.  Otherwise,
        trans is a matplotlib transform that specifies the coordinate system
        of x, y.

        The selection of artists from amongst which the pick function
        finds an artist can be narrowed using the optional keyword
        argument among. If provided, this should be either a sequence
        of permitted artists or a function taking an artist as its
        argument and returning a true value if and only if that artist
        can be selected.

        Note this algorithm calculates distance to the vertices of the
        polygon, so if you want to pick a patch, click on the edge!
        """
        if trans is not None:
            xywin = trans.xy_tup((x,y))
        else:
            xywin = x,y

        def dist_points(p1, p2):
            'return the distance between two points'
            x1, y1 = p1
            x2, y2 = p2
            return math.sqrt((x1-x2)**2+(y1-y2)**2)

        def dist_x_y(p1, x, y):
            'x and y are arrays; return the distance to the closest point'
            x1, y1 = p1
            return min(sqrt((x-x1)**2+(y-y1)**2))

        def dist(a):
            if isinstance(a, Text):
                bbox = a.get_window_extent()
                l,b,w,h = bbox.get_bounds()
                verts = (l,b), (l,b+h), (l+w,b+h), (l+w, b)
                xt, yt = zip(*verts)
            elif isinstance(a, Patch):
                verts = a.get_verts()
                tverts = a.get_transform().seq_xy_tups(verts)
                xt, yt = zip(*tverts)
            elif isinstance(a, Line2D):
                xdata = a.get_xdata(valid_only = True)
                ydata = a.get_ydata(valid_only = True)
                xt, yt = a.get_transform().numerix_x_y(xdata, ydata)

            return dist_x_y(xywin, asarray(xt), asarray(yt))

        artists = self.lines + self.patches + self.texts
        if callable(among):
            artists = filter(test, artists)
        elif iterable(among):
            amongd = dict([(k,1) for k in among])
            artists = [a for a in artists if a in amongd]
        elif among is None:
            pass
        else:
            raise ValueError('among must be callable or iterable')
        if not len(artists): return None
        ds = [ (dist(a),a) for a in artists]
        ds.sort()
        return ds[0][1]



    #### Labelling

    def set_title(self, label, fontdict=None, **kwargs):
        """
        SET_TITLE(label, fontdict=None, **kwargs):

        Set the title for the axes.  See the text docstring for information
        of how override and the optional args work

        ACCEPTS: str
        """
        default = {
            'fontsize':rcParams['axes.titlesize'],
            'verticalalignment' : 'bottom',
            'horizontalalignment' : 'center'
            }

        self.title.set_text(label)
        self.title.update(default)
        if fontdict is not None: self.title.update(fontdict)
        self.title.update(kwargs)
        return self.title

    def set_xlabel(self, xlabel, fontdict=None, **kwargs):
        """
        SET_XLABEL(xlabel, fontdict=None, **kwargs)

        Set the label for the xaxis.  See the text docstring for information
        of how override and the optional args work.

        ACCEPTS: str
        """

        label = self.xaxis.get_label()
        label.set_text(xlabel)
        if fontdict is not None: label.update(fontdict)
        label.update(kwargs)
        return label

    def set_ylabel(self, ylabel, fontdict=None, **kwargs):
        """
        SET_YLABEL(ylabel, fontdict=None, **kwargs)

        Set the label for the yaxis

        See the text doctstring for information of how override and
        the optional args work

        ACCEPTS: str
        """
        label = self.yaxis.get_label()
        label.set_text(ylabel)
        if fontdict is not None: label.update(fontdict)
        label.update(kwargs)
        return label

    def text(self, x, y, s, fontdict=None,
             withdash=False, **kwargs):
        """
        TEXT(x, y, s, fontdict=None, **kwargs)

        Add text in string s to axis at location x,y (data coords)

          fontdict is a dictionary to override the default text properties.
          If fontdict is None, the defaults are determined by your rc
          parameters.

          withdash=True will create a TextWithDash instance instead
          of a Text instance.

        Individual keyword arguemnts can be used to override any given
        parameter

            text(x, y, s, fontsize=12)

        The default transform specifies that text is in data coords,
        alternatively, you can specify text in axis coords (0,0 lower left and
        1,1 upper right).  The example below places text in the center of the
        axes

            text(0.5, 0.5,'matplotlib',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax.transAxes,
            )

        """
        default = {
            'verticalalignment' : 'bottom',
            'horizontalalignment' : 'left',
            #'verticalalignment' : 'top',
            'transform' : self.transData,
            }

        # At some point if we feel confident that TextWithDash
        # is robust as a drop-in replacement for Text and that
        # the performance impact of the heavier-weight class
        # isn't too significant, it may make sense to eliminate
        # the withdash kwarg and simply delegate whether there's
        # a dash to TextWithDash and dashlength.
        if withdash:
            t = TextWithDash(
                x=x, y=y, text=s,
                )
        else:
            t = Text(
                x=x, y=y, text=s,
                )
        self._set_artist_props(t)

        t.update(default)
        if fontdict is not None: t.update(fontdict)
        t.update(kwargs)
        self.texts.append(t)


        #if t.get_clip_on():  t.set_clip_box(self.bbox)
        if kwargs.has_key('clip_on'):  t.set_clip_box(self.bbox)
        return t


    #### Lines and spans

    def axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        """
        AXHLINE(y=0, xmin=0, xmax=1, **kwargs)

        Axis Horizontal Line

        Draw a horizontal line at y from xmin to xmax.  With the default
        values of xmin=0 and xmax=1, this line will always span the horizontal
        extent of the axes, regardless of the xlim settings, even if you
        change them, eg with the xlim command.  That is, the horizontal extent
        is in axes coords: 0=left, 0.5=middle, 1.0=right but the y location is
        in data coordinates.

        Return value is the Line2D instance.  kwargs are the same as kwargs to
        plot, and can be used to control the line properties.  Eg

          # draw a thick red hline at y=0 that spans the xrange
          axhline(linewidth=4, color='r')

          # draw a default hline at y=1 that spans the xrange
          axhline(y=1)

          # draw a default hline at y=.5 that spans the the middle half of
          # the xrange
          axhline(y=.5, xmin=0.25, xmax=0.75)
        """
        trans = blend_xy_sep_transform( self.transAxes, self.transData)
        l, = self.plot([xmin,xmax], [y,y], transform=trans, scalex=False, **kwargs)
        return l


    def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        """
        AXVLINE(x=0, ymin=0, ymax=1, **kwargs)

        Axis Vertical Line

        Draw a vertical line at x from ymin to ymax.  With the default values
        of ymin=0 and ymax=1, this line will always span the vertical extent
        of the axes, regardless of the xlim settings, even if you change them,
        eg with the xlim command.  That is, the vertical extent is in axes
        coords: 0=bottom, 0.5=middle, 1.0=top but the x location is in data
        coordinates.

        Return value is the Line2D instance.  kwargs are the same as
        kwargs to plot, and can be used to control the line properties.  Eg

            # draw a thick red vline at x=0 that spans the yrange
            l = axvline(linewidth=4, color='r')

            # draw a default vline at x=1 that spans the yrange
            l = axvline(x=1)

            # draw a default vline at x=.5 that spans the the middle half of
            # the yrange
            axvline(x=.5, ymin=0.25, ymax=0.75)

        """

        trans = blend_xy_sep_transform( self.transData, self.transAxes )
        l, = self.plot([x,x], [ymin,ymax] , transform=trans, scaley=False, **kwargs)
        return l


    def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kwargs):
        """
        AXHSPAN(ymin, ymax, xmin=0, xmax=1, **kwargs)

        Axis Horizontal Span.  ycoords are in data units and x
        coords are in axes (relative 0-1) units

        Draw a horizontal span (regtangle) from ymin to ymax.  With the
        default values of xmin=0 and xmax=1, this always span the xrange,
        regardless of the xlim settings, even if you change them, eg with the
        xlim command.  That is, the horizontal extent is in axes coords:
        0=left, 0.5=middle, 1.0=right but the y location is in data
        coordinates.

        kwargs are the kwargs to Patch, eg

          antialiased, aa
          linewidth,   lw
          edgecolor,   ec
          facecolor,   fc

        the terms on the right are aliases

        Return value is the patches.Polygon instance.

            #draws a gray rectangle from y=0.25-0.75 that spans the horizontal
            #extent of the axes
            axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)
        """
        # convert y axis units
        kwargs_copy = kwargs.copy()
        self._update_units_args(kwargs_copy)
        y = units_conversion(y, self._yunits)

        trans = blend_xy_sep_transform( self.transAxes, self.transData  )
        verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)
        p = Polygon(verts, **kwargs_copy)
        p.set_transform(trans)
        self.add_patch(p)
        return p


    def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        """
        AXVSPAN(xmin, xmax, ymin=0, ymax=1, **kwargs)

        axvspan : Axis Vertical Span.  xcoords are in data units and y coords
        are in axes (relative 0-1) units

        Draw a vertical span (regtangle) from xmin to xmax.  With the default
        values of ymin=0 and ymax=1, this always span the yrange, regardless
        of the ylim settings, even if you change them, eg with the ylim
        command.  That is, the vertical extent is in axes coords: 0=bottom,
        0.5=middle, 1.0=top but the y location is in data coordinates.

        kwargs are the kwargs to Patch, eg

          antialiased, aa
          linewidth,   lw
          edgecolor,   ec
          facecolor,   fc

        the terms on the right are aliases

        return value is the patches.Polygon instance.

            # draw a vertical green translucent rectangle from x=1.25 to 1.55 that
            # spans the yrange of the axes
            axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

        """
        # convert x axis units
        kwargs_copy = kwargs.copy()
        self._update_units_args(kwargs_copy)
        x = units_conversion(x, self._xunits)

        trans = blend_xy_sep_transform( self.transData, self.transAxes   )
        verts = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        p = Polxgon(verts, **kwargs_copy)
        p.set_transform(trans)
        self.add_patch(p)

        return p

    def hlines(self, y, xmin, xmax, fmt='k-', **kwargs):
        """
        HLINES(y, xmin, xmax, fmt='k-')

        plot horizontal lines at each y from xmin to xmax.  xmin or xmax can
        be scalars or len(x) numpy arrays.  If they are scalars, then the
        respective values are constant, else the widths of the lines are
        determined by xmin and xmax

        fmt is a plot format string, eg 'g--'

        kwargs are matplotlib.lines.Line2D kwargs

        Returns a list of line instances that were added
        """
        linestyle, marker, color = _process_plot_format(fmt)

        # convert, if necessary
        kwargs_copy = kwargs.copy()
        self._update_units_args(kwargs_copy)
        xmin = units_conversion(xmin, self._xunits)
        xmax = units_conversion(xmax, self._xunits)
        y    = units_conversion(y, self._yunits)

        if not iterable(y): y = [y]
        if not iterable(xmin): xmin = [xmin]
        if not iterable(xmax): xmax = [xmax]
        y = asarray(y)
        xmin = asarray(xmin)
        xmax = asarray(xmax)

        if len(xmin)==1:
            xmin = xmin*ones(y.shape, typecode(y))
        if len(xmax)==1:
            xmax = xmax*ones(y.shape, typecode(y))

        if len(xmin)!=len(y):
            raise ValueError, 'xmin and y are unequal sized sequences'
        if len(xmax)!=len(y):
            raise ValueError, 'xmax and y are unequal sized sequences'

        lines = []
        for (thisY, thisMin, thisMax) in zip(y,xmin,xmax):
            line = Line2D(
                [thisMin, thisMax], [thisY, thisY],
                color=color, linestyle=linestyle, marker=marker,
                **kwargs
                )
            self.add_line( line )
            lines.append(line)
        return lines

    def vlines(self, x, ymin, ymax, fmt='k-', **kwargs):
        """
        VLINES(x, ymin, ymax, color='k')

        Plot vertical lines at each x from ymin to ymax.  ymin or ymax can be
        scalars or len(x) numpy arrays.  If they are scalars, then the
        respective values are constant, else the heights of the lines are
        determined by ymin and ymax


        fmt is a plot format string, eg 'g--'

        kwargs are matplotlib.lines.Line2D kwargs

        Returns a list of lines that were added
        """
        linestyle, marker, color = _process_plot_format(fmt)

        # convert, if necessary
        kwargs_copy = kwargs.copy()
        self._update_units_args(kwargs_copy)
        xunits = kwargs_copy.pop('xunits', None)
        yunits = kwargs_copy.pop('yunits', None)
        ymin = units_conversion(ymin, self._yunits)
        ymax = units_conversion(ymax, self._yunits)
        x    = units_conversion(x, self._xunits)

        if not iterable(x): x = [x]
        if not iterable(ymin): ymin = [ymin]
        if not iterable(ymax): ymax = [ymax]
        x = asarray(x)
        ymin = asarray(ymin)
        ymax = asarray(ymax)

        if len(ymin)==1:
            ymin = ymin*ones(x.shape, typecode(x))
        if len(ymax)==1:
            ymax = ymax*ones(x.shape, typecode(x))


        if len(ymin)!=len(x):
            raise ValueError, 'ymin and x are unequal sized sequences'
        if len(ymax)!=len(x):
            raise ValueError, 'ymax and x are unequal sized sequences'

        Y = transpose(array([ymin, ymax]))
        lines = []
        for thisX, thisY in zip(x,Y):
            line = Line2D(
                [thisX, thisX], thisY,
                color=color, linestyle=linestyle, marker=marker,
                **kwargs
                )
            self.add_line(line)
            lines.append(line)
        return lines



    #### Basic plotting

    def plot(self, *args, **kwargs):
        """
        PLOT(*args, **kwargs)

        Plot lines and/or markers to the Axes.  *args is a variable length
        argument, allowing for multiple x,y pairs with an optional format
        string.  For example, each of the following is legal

            plot(x,y)            # plot x and y using the default line style and color
            plot(x,y, 'bo')      # plot x and y using blue circle markers
            plot(y)              # plot y using x as index array 0..N-1
            plot(y, 'r+')        # ditto, but with red plusses

        An arbitrary number of x, y, fmt groups can be specified, as in

            a.plot(x1, y1, 'g^', x2, y2, 'g-')

        Return value is a list of lines that were added.

        The following line styles are supported:

            -     : solid line
            --    : dashed line
            -.    : dash-dot line
            :     : dotted line
            .     : points
            ,     : pixels
            o     : circle symbols
            ^     : triangle up symbols
            v     : triangle down symbols
            <     : triangle left symbols
            >     : triangle right symbols
            s     : square symbols
            +     : plus symbols
            x     : cross symbols
            D     : diamond symbols
            d     : thin diamond symbols
            1     : tripod down symbols
            2     : tripod up symbols
            3     : tripod left symbols
            4     : tripod right symbols
            h     : hexagon symbols
            H     : rotated hexagon symbols
            p     : pentagon symbols
            |     : vertical line symbols
            _     : horizontal line symbols
            steps : use gnuplot style 'steps' # kwarg only

        The following color strings are supported

            b  : blue
            g  : green
            r  : red
            c  : cyan
            m  : magenta
            y  : yellow
            k  : black
            w  : white

        Line styles and colors are combined in a single format string, as in
        'bo' for blue circles.

        The **kwargs can be used to set line properties (any property that has
        a set_* method).  You can use this to set a line label (for auto
        legends), linewidth, anitialising, marker face color, etc.  Here is an
        example:

            plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
            plot([1,2,3], [1,4,9], 'rs',  label='line 2')
            axis([0, 4, 0, 10])
            legend()

        If you make multiple lines with one plot command, the kwargs apply
        to all those lines, eg

            plot(x1, y1, x2, y2, antialised=False)

        Neither line will be antialiased.

        Additional kwargs scalex and scaley, if defined, are passed on
        to autoscale_view to determine whether the x and y axes are
        autoscaled; default True.  See Axes.autoscale_view for more
        information
        """

        d = kwargs.copy()
        self._update_units_args(d)
        scalex = d.pop('scalex', True)
        scaley = d.pop('scaley', True)
        if not self._hold: self.cla()
        lines = []
        for line in self._get_lines(*args, **d):
            self.add_line(line)
            lines.append(line)
        lines = [line for line in lines] # consume the generator

        self.autoscale_view(scalex=scalex, scaley=scaley)
        return lines

    def plot_date(self, x, y, fmt='bo', tz=None, xdate=True, ydate=False,
                  **kwargs):
        """
        PLOT_DATE(x, y, fmt='bo', tz=None, xdate=True, ydate=False, **kwargs)

        Similar to the plot() command, except the x or y (or both) data
        is considered to be dates, and the axis is labeled accordingly.

        x or y (or both) can be a sequence of dates represented as
        float days since 0001-01-01 UTC.

        fmt is a plot format string.

        tz is the time zone to use in labelling dates.  Defaults to rc value.

        If xdate is True, the x-axis will be labeled with dates.

        If ydate is True, the y-axis will be labeled with dates.

        kwargs are passed on to plot.  See plot for more information.

        See matplotlib.dates for helper functions date2num, num2date
        and drange for help on creating the required floating point dates
        """

        if not matplotlib._havedate:
            raise SystemExit('plot_date: no dates support - dates require python2.3')

        if not self._hold: self.cla()

        ret = self.plot(x, y, fmt, **kwargs)

        if xdate:
            self.xaxis_date(tz)
        if ydate:
            self.yaxis_date(tz)

        self.autoscale_view()

        return ret

    def loglog(self, *args, **kwargs):
        """
        LOGLOG(*args, **kwargs)

        Make a loglog plot with log scaling on the a and y axis.  The args
        to semilog x are the same as the args to plot.  See help plot for
        more info.

        Optional keyword args supported are any of the kwargs
        supported by plot or set_xscale or set_yscale.  Notable, for
        log scaling:

          * basex: base of the x logarithm

          * subsx: the location of the minor ticks; None defaults to
            autosubs, which depend on the number of decades in the
            plot; see set_xscale for details

          * basey: base of the y logarithm

          * subsy: the location of the minor yticks; None defaults to
            autosubs, which depend on the number of decades in the
            plot; see set_yscale for details
        """
        if not self._hold: self.cla()

        dx = {'basex': kwargs.get('basex', 10),
              'subsx': kwargs.get('subsx', None),
              }
        dy = {'basey': kwargs.get('basey', 10),
              'subsy': kwargs.get('subsy', None),
              }

        self.set_xscale('log', **dx)
        self.set_yscale('log', **dy)
        dict_delall( kwargs, ('basex', 'basey', 'subsx', 'subsy'))

        b =  self._hold
        self._hold = True # we've already processed the hold
        l = self.plot(*args, **kwargs)
        self._hold = b    # restore the hold

        return l

    def semilogx(self, *args, **kwargs):
        """
        SEMILOGX(*args, **kwargs)

        Make a semilog plot with log scaling on the x axis.  The args to
        semilog x are the same as the args to plot.  See help plot for more
        info.

        Optional keyword args supported are any of the kwargs supported by
        plot or set_xscale.  Notable, for log scaling:

            * basex: base of the logarithm

            * subsx: the location of the minor ticks; None defaults to
              autosubs, which depend on the number of decades in the
              plot; see set_xscale for details

        """
        if not self._hold: self.cla()
        d = {'basex': kwargs.get('basex', 10),
             'subsx': kwargs.get('subsx', None),
             }

        self.set_xscale('log', **d)
        dict_delall( kwargs, ('basex', 'subsx'))
        b =  self._hold
        self._hold = True # we've already processed the hold
        l = self.plot(*args, **kwargs)
        self._hold = b    # restore the hold
        return l


    def semilogy(self, *args, **kwargs):
        """
        SEMILOGY(*args, **kwargs):

        Make a semilog plot with log scaling on the y axis.  The args to
        semilogy are the same as the args to plot.  See help plot for more
        info.

        Optional keyword args supported are any of the kwargs supported by
        plot or set_yscale.  Notable, for log scaling:

            * basey: base of the logarithm

            * subsy: a sequence of the location of the minor ticks;
              None defaults to autosubs, which depend on the number of
              decades in the plot; see set_yscale for details

        """
        if not self._hold: self.cla()

        d = {'basey': kwargs.get('basey', 10),
             'subsy': kwargs.get('subsy', None),
             }
        self.set_yscale('log', **d)
        dict_delall( kwargs, ('basey', 'subsy'))
        b =  self._hold
        self._hold = True # we've already processed the hold
        l = self.plot(*args, **kwargs)
        self._hold = b    # restore the hold

        return l

    def legend(self, *args, **kwargs):
        """
        LEGEND(*args, **kwargs)

        Place a legend on the current axes at location loc.  Labels are a
        sequence of strings and loc can be a string or an integer specifying
        the legend location

        USAGE:

          Make a legend with existing lines

          >>> legend()

          legend by itself will try and build a legend using the label
          property of the lines/patches/collections.  You can set the label of
          a line by doing plot(x, y, label='my data') or line.set_label('my
          data'). If label is set to '_nolegend_', the item will not be shown
          in legend.

            # automatically generate the legend from labels
            legend( ('label1', 'label2', 'label3') )

            # Make a legend for a list of lines and labels
            legend( (line1, line2, line3), ('label1', 'label2', 'label3') )

            # Make a legend at a given location, using a location argument
            # legend( LABELS, LOC )  or
            # legend( LINES, LABELS, LOC )
            legend( ('label1', 'label2', 'label3'), loc='upper left')
            legend( (line1, line2, line3),  ('label1', 'label2', 'label3'), loc=2)

        The location codes are

          'best' : 0,
          'upper right'  : 1, (default)
          'upper left'   : 2,
          'lower left'   : 3,
          'lower right'  : 4,
          'right'        : 5,
          'center left'  : 6,
          'center right' : 7,
          'lower center' : 8,
          'upper center' : 9,
          'center'       : 10,

        If none of these are suitable, loc can be a 2-tuple giving x,y
        in axes coords, ie,

          loc = 0, 1 is left top
          loc = 0.5, 0.5 is center, center

        and so on.  The following kwargs are supported

          numpoints = 4         # the number of points in the legend line
          prop = FontProperties('smaller')  # the font properties
          pad = 0.2             # the fractional whitespace inside the legend border

          # The kwarg dimensions are in axes coords
          labelsep = 0.005     # the vertical space between the legend entries
          handlelen = 0.05     # the length of the legend lines
          handletextsep = 0.02 # the space between the legend line and legend text
          axespad = 0.02       # the border between the axes and legend edge
          shadow = False       # if True, draw a shadow behind legend
        """

        def get_handles():
            handles = self.lines
            handles.extend(self.patches)
            handles.extend([c for c in self.collections if isinstance(c, LineCollection)])
            return handles


        if len(args)==0:
            handles = []
            labels = []
            for line in get_handles():
                label = line.get_label()
                if label != '_nolegend_':
                    handles.append(line)
                    labels.append(label)
            loc = popd(kwargs, 'loc', 1)

        elif len(args)==1:
            # LABELS
            labels = args[0]
            handles = [h for h, label in zip(get_handles(), labels)]
            loc = popd(kwargs, 'loc', 1)

        elif len(args)==2:
            if is_string_like(args[1]) or isinstance(args[1], int):
                # LABELS, LOC
                labels, loc = args
                handles = [h for h, label in zip(get_handles(), labels)]
            else:
                # LINES, LABELS
                handles, labels = args
                loc = popd(kwargs, 'loc', 1)

        elif len(args)==3:
            # LINES, LABELS, LOC
            handles, labels, loc = args
        else:
            raise RuntimeError('Invalid arguments to legend')


        handles = flatten(handles)
        self.legend_ = Legend(self, handles, labels, loc, **kwargs)
        return self.legend_


    #### Specialized plotting


    def bar(self, left, height, width=0.8, bottom=0,
            color=None, edgecolor=None, yerr=None, xerr=None, ecolor=None, capsize=3,
            align='edge', orientation='vertical'
            ):
        """
        BAR(left, height, width=0.8, bottom=0,
            color=None, edgecolor=None, yerr=None, xerr=None, ecolor=None, capsize=3,
            align='edge', orientation='vertical')

        Make a bar plot with rectangles bounded by

          left, left+width, bottom, bottom+height  (left, right, bottom and top edges)

        left, height, width, and bottom can be either scalars or sequences

        Return value is a list of Rectangle patch instances

        BAR(left, height, width, bottom,
            color, edgecolor, yerr, xerr, ecolor, capsize,
            align, orientation)

            left - the x coordinates of the left sides of the bars

            height - the heights of the bars

        Optional arguments

            width - the widths of the bars

            bottom - the y coordinates of the bottom edges of the bars

            color specifies the colors of the bars

            edgecolor specifies the colors of the bar edges

            xerr and yerr, if not None, will be used to generate errorbars
            on the bar chart

            ecolor specifies the color of any errorbar

            capsize determines the length in points of the error bar caps

            align = 'edge' | 'center'

            orientation = 'vertical' | 'horizontal'

            For vertical bars, 'edge' aligns bars by their left edges in left,
            while 'center' interprets these values as the x coordinates of the bar centers.
            For horizontal bars, 'edge' aligns bars by their bottom edges in bottom,
            while 'center' interprets these values as the y coordinates of the bar centers.

        The optional arguments color, edgecolor, yerr, and xerr can be either
        scalars or sequences of length equal to the number of bars

        This enables you to use bar as the basis for stacked bar
        charts, or candlestick plots
        """
        if not self._hold: self.cla()

        def make_iterable(x):
            if not iterable(x):
                return [x]
            else:
                return x

        # make them safe to take len() of
        left = make_iterable(left)
        height = make_iterable(height)
        width = make_iterable(width)
        bottom = make_iterable(bottom)

        if orientation == 'vertical':
            # size width and bottom according to length of left
            nbars = len(left)
            if len(width) == 1:
                width *= nbars
            if len(bottom) == 1:
                bottom *= nbars
        elif orientation == 'horizontal':
            # size left and height according to length of bottom
            nbars = len(bottom)
            if len(left) == 1:
                left *= nbars
            if len(height) == 1:
                height *= nbars
        else:
            raise ValueError, 'invalid orientation: %s' % orientation

        left = asarray(left)
        height = asarray(height)
        width = asarray(width)
        bottom = asarray(bottom)

        # if color looks like a color string, an RGB tuple or a
        # scalar, then repeat it by nbars
        if (is_string_like(color) or
            (iterable(color) and len(color)==3 and nbars!=3) or
            not iterable(color)):
            color = [color]*nbars

        # if edgecolor looks like a color string, an RGB tuple or a
        # scalar, then repeat it by nbars
        if (is_string_like(edgecolor) or
            (iterable(edgecolor) and len(edgecolor)==3 and nbars!=3) or
            not iterable(edgecolor)):
            edgecolor = [edgecolor]*nbars

        if not iterable(yerr):
            yerr = asarray([yerr]*nbars, Float) # Float converts Nones to NANs
        else:
            yerr = asarray(yerr)
        if not iterable(xerr):
            xerr = asarray([xerr]*nbars, Float)
        else:
            xerr = asarray(xerr)

        if orientation == 'vertical':
            lenarg = 'left'
        elif orientation == 'horizontal':
            lenarg = 'bottom'
        assert len(left)==nbars, 'bar() argument \'left\' must be len(%s) or scalar' % lenarg
        assert len(height)==nbars, 'bar() argument \'height\' must be len(%s) or scalar' % lenarg
        assert len(width)==nbars, 'bar() argument \'width\' must be len(%s) or scalar' % lenarg
        assert len(bottom)==nbars, 'bar() argument \'bottom\' must be len(%s) or scalar' % lenarg
        assert len(color)==nbars, 'bar() argument \'color\' must be len(%s) or scalar' % lenarg
        assert len(edgecolor)==nbars, 'bar() argument \'edgecolor\' must be len(%s) or scalar' % lenarg
        assert len(yerr)==nbars, 'bar() argument \'yerr\' must be len(%s) or scalar' % lenarg
        assert len(xerr)==nbars, 'bar() argument \'xerr\' must be len(%s) or scalar' % lenarg

        patches = []

        if align == 'edge':
            pass
        elif align == 'center':
            if orientation == 'vertical':
                left = left - width/2.
            elif orientation == 'horizontal':
                bottom = bottom - height/2.
        else:
            raise ValueError, 'invalid alignment: %s' % align

        args = zip(left, bottom, width, height, color, edgecolor)
        for l, b, w, h, c, e in args:
            if h<0:
                b += h
                h = abs(h)
            r = Rectangle(
                xy=(l, b), width=w, height=h,
                facecolor=c,
                edgecolor=e,
                )
            self.add_patch(r)
            patches.append(r)

        holdstate = self._hold
        self.hold(True) # ensure hold is on before plotting errorbars

        if xerr is not None or yerr is not None:
            if orientation == 'vertical':
                x, y = left+0.5*width, bottom+height
            elif orientation == 'horizontal':
                x, y = left+width, bottom+0.5*height
            self.errorbar(
                x, y,
                yerr=yerr, xerr=xerr,
                fmt=None, ecolor=ecolor, capsize=capsize)

        self.hold(holdstate) # restore previous hold state

        self.autoscale_view()
        return patches


    def barh(self, bottom, width, height=0.8, left=0,
             color=None, edgecolor=None, xerr=None, yerr=None, ecolor=None, capsize=3,
             align='edge'
             ):
        """
        BARH(bottom, width, height=0.8, left=0,
             color=None, edgecolor=None, xerr=None, yerr=None, ecolor=None, capsize=3,
             align='edge')

        Make a horizontal bar plot with rectangles bounded by

          left, left+width, bottom, bottom+height  (left, right, bottom and top edges)

        bottom, width, height, and left can be either scalars or sequences

        Return value is a list of Rectangle patch instances

        BARH(bottom, width, height, left,
             color, edgecolor, xerr, yerr, ecolor, capsize,
             align)

            bottom - the vertical positions of the bottom edges of the bars

            width - the lengths of the bars

        Optional arguments

            height - the heights (thicknesses) of the bars

            left - the x coordinates of the left edges of the bars

            color specifies the colors of the bars

            edgecolor specifies the colors of the bar edges

            xerr and yerr, if not None, will be used to generate errorbars
            on the bar chart

            ecolor specifies the color of any errorbar

            capsize determines the length in points of the error bar caps

            align = 'edge' | 'center'
            'edge' aligns the horizontal bars by their bottom edges in bottom, while
            'center' interprets these values as the y coordinates of the bar centers.

        The optional arguments color, edgecolor, xerr, and yerr can be either
        scalars or sequences of length equal to the number of bars

        This enables you to use barh as the basis for stacked bar
        charts, or candlestick plots
        """

        patches = self.bar(left=left, height=height, width=width, bottom=bottom,
                           color=color, edgecolor=edgecolor, yerr=yerr, xerr=xerr, ecolor=ecolor, capsize=capsize,
                           align=align, orientation='horizontal'
                           )
        return patches


    def stem(self, x, y, linefmt='b-', markerfmt='bo', basefmt='r-'):
        """
        STEM(x, y, linefmt='b-', markerfmt='bo', basefmt='r-')

        A stem plot plots vertical lines (using linefmt) at each x location
        from the baseline to y, and places a marker there using markerfmt.  A
        horizontal line at 0 is is plotted using basefmt

        Return value is (markerline, stemlines, baseline) .

        See
        http://www.mathworks.com/access/helpdesk/help/techdoc/ref/stem.html
        for details and examples/stem_plot.py for a demo.
        """
        remember_hold=self._hold
        if not self._hold: self.cla()
        self.hold(True)

        markerline, = self.plot(x, y, markerfmt)

        stemlines = []
        for thisx, thisy in zip(x, y):
            l, = self.plot([thisx,thisx], [0, thisy], linefmt)
            stemlines.append(l)

        baseline, = self.plot([amin(x), amax(x)], [0,0], basefmt)

        self.hold(remember_hold)

        return markerline, stemlines, baseline


    def pie(self, x, explode=None, labels=None,
            colors=None,
            autopct=None,
            shadow=False
            ):
        """
        PIE(x, explode=None, labels=None,
            colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'),
            autopct=None, shadow=False)

        Make a pie chart of array x.  The fractional area of each wedge is
        given by x/sum(x).  If sum(x)<=1, then the values of x give the
        fractional area directly and the array will not be normalized.

          - explode, if not None, is a len(x) array which specifies the
            fraction of the radius to offset that wedge.

          - colors is a sequence of matplotlib color args that the pie chart
            will cycle.

          - labels, if not None, is a len(x) list of labels.

          - autopct, if not None, is a string or function used to label the
            wedges with their numeric value.  The label will be placed inside
            the wedge.  If it is a format string, the label will be fmt%pct.
            If it is a function, it will be called

          - shadow, if True, will draw a shadow beneath the pie.

        The pie chart will probably look best if the figure and axes are
        square.  Eg,

          figure(figsize=(8,8))
          ax = axes([0.1, 0.1, 0.8, 0.8])

        Return value:

          If autopct is None, return a list of (patches, texts), where patches
          is a sequence of matplotlib.patches.Wedge instances and texts is a
          list of the label Text instnaces

          If autopct is not None, return (patches, texts, autotexts), where
          patches and texts are as above, and autotexts is a list of text
          instances for the numeric labels
        """
        self.set_frame_on(False)

        x = asarray(x).astype(Float32)

        sx = float(asum(x))
        if sx>1: x = divide(x,sx)

        if labels is None: labels = ['']*len(x)
        if explode is None: explode = [0]*len(x)
        assert(len(x)==len(labels))
        assert(len(x)==len(explode))
        if colors is None: colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')


        center = 0,0
        radius = 1
        theta1 = 0
        i = 0
        texts = []
        slices = []
        autotexts = []
        for frac, label, expl in zip(x,labels, explode):
            x, y = center
            theta2 = theta1 + frac
            thetam = 2*math.pi*0.5*(theta1+theta2)
            x += expl*math.cos(thetam)
            y += expl*math.sin(thetam)

            w = Wedge((x,y), radius, 360.*theta1, 360.*theta2,
                      facecolor=colors[i%len(colors)])
            slices.append(w)
            self.add_patch(w)
            w.set_label(label)

            if shadow:
                # make sure to add a shadow after the call to
                # add_patch so the figure and transform props will be
                # set
                shad = Shadow(w, -0.02, -0.02,
                              #props={'facecolor':w.get_facecolor()}
                              )
                shad.set_zorder(0.9*w.get_zorder())
                self.add_patch(shad)


            xt = x + 1.1*radius*math.cos(thetam)
            yt = y + 1.1*radius*math.sin(thetam)

            t = self.text(xt, yt, label,
                          size=rcParams['xtick.labelsize'],
                          horizontalalignment='center',
                          verticalalignment='center')

            texts.append(t)

            if autopct is not None:
                xt = x + 0.6*radius*math.cos(thetam)
                yt = y + 0.6*radius*math.sin(thetam)
                if is_string_like(autopct):
                    s = autopct%(100.*frac)
                elif callable(autopct):
                    s = autopct(100.*frac)
                else:
                    raise TypeError('autopct must be callable or a format string')

                t = self.text(xt, yt, s,
                              horizontalalignment='center',
                              verticalalignment='center')
                autotexts.append(t)


            theta1 = theta2
            i += 1

        self.set_xlim((-1.25, 1.25))
        self.set_ylim((-1.25, 1.25))
        self.set_xticks([])
        self.set_yticks([])

        if autopct is None: return slices, texts
        else: return slices, texts, autotexts




    def errorbar(self, x, y, yerr=None, xerr=None,
                 fmt='b-', ecolor=None, capsize=3,
                 barsabove=False, **kwargs):
        """
        ERRORBAR(x, y, yerr=None, xerr=None,
                 fmt='b-', ecolor=None, capsize=3, barsabove=False)

        Plot x versus y with error deltas in yerr and xerr.
        Vertical errorbars are plotted if yerr is not None
        Horizontal errorbars are plotted if xerr is not None

        xerr and yerr may be any of:

            a rank-0, Nx1 Numpy array  - symmetric errorbars +/- value

            an N-element list or tuple - symmetric errorbars +/- value

            a rank-1, Nx2 Numpy array  - asymmetric errorbars -column1/+column2

        Alternatively, x, y, xerr, and yerr can all be scalars, which
        plots a single error bar at x, y.

            fmt is the plot format symbol for y.  if fmt is None, just
            plot the errorbars with no line symbols.  This can be useful
            for creating a bar plot with errorbars

            ecolor is a matplotlib color arg which gives the color the
            errorbar lines; if None, use the marker color.

            capsize is the size of the error bar caps in points

            barsabove, if True, will plot the errorbars above the plot symbols
            - default is below

            kwargs are passed on to the plot command for the markers.
              So you can add additional key=value pairs to control the
              errorbar markers.  For example, this code makes big red
              squares with thick green edges

              >>> x,y,yerr = rand(3,10)
              >>> errorbar(x, y, yerr, marker='s',
                           mfc='red', mec='green', ms=20, mew=4)

             mfc, mec, ms and mew are aliases for the longer property
             names, markerfacecolor, markeredgecolor, markersize and
             markeredgewith.


        Return value is a length 2 tuple.  The first element is the
        Line2D instance for the y symbol lines.  The second element is
        a list of error bar lines.
        """
        if not self._hold: self.cla()
        # make sure all the args are iterable arrays
        if not iterable(x): x = asarray([x])
        else: x = asarray(x)

        if not iterable(y): y = asarray([y])
        else: y = asarray(y)

        if xerr is not None:
            if not iterable(xerr): xerr = asarray([xerr])
            else: xerr = asarray(xerr)

        if yerr is not None:
            if not iterable(yerr): yerr = asarray([yerr])
            else: yerr = asarray(yerr)

        l0 = None

        if barsabove and fmt is not None:
            l0, = self.plot(x,y,fmt,**kwargs)

        caplines = []
        barlines = []

        if xerr is not None:
            if len(xerr.shape) == 1:
                left  = x-xerr
                right = x+xerr
            else:
                left  = x-xerr[0]
                right = x+xerr[1]

            barlines.extend( self.hlines(y, x, left, label='_nolegend_' ))
            barlines.extend( self.hlines(y, x, right, label='_nolegend_') )
            caplines.extend( self.plot(left, y, '|', ms=2*capsize, label='_nolegend_') )
            caplines.extend( self.plot(right, y, '|', ms=2*capsize, label='_nolegend_') )

        if yerr is not None:
            if len(yerr.shape) == 1:
                lower = y-yerr
                upper = y+yerr
            else:
                lower = y-yerr[0]
                upper = y+yerr[1]

            barlines.extend( self.vlines(x, y, upper, label='_nolegend_' ) )
            barlines.extend( self.vlines(x, y, lower, label='_nolegend_' ) )
            caplines.extend( self.plot(x, lower, '_', ms=2*capsize, label='_nolegend_') )
            caplines.extend( self.plot(x, upper, '_', ms=2*capsize, label='_nolegend_') )

        if not barsabove and fmt is not None:
            l0, = self.plot(x,y,fmt,**kwargs)

        if ecolor is None and l0 is None:
            ecolor = rcParams['lines.color']
        elif ecolor is None:
            ecolor = l0.get_color()

        for l in barlines:
            l.set_color(ecolor)
        for l in caplines:
            l.set_color(ecolor)
            l.set_markerfacecolor(ecolor)
            l.set_markeredgecolor(ecolor)

        self.autoscale_view()

        ret = silent_list('Line2D errorbar', caplines+barlines)
        return (l0, ret)


    def boxplot(self, x, notch=0, sym='b+', vert=1, whis=1.5,
                positions=None, widths=None):
        """
        boxplot(x, notch=0, sym='+', vert=1, whis=1.5,
                positions=None, widths=None)

        Make a box and whisker plot for each column of x.
        The box extends from the lower to upper quartile values
        of the data, with a line at the median.  The whiskers
        extend from the box to show the range of the data.  Flier
        points are those past the end of the whiskers.

        notch = 0 (default) produces a rectangular box plot.
        notch = 1 will produce a notched box plot

        sym (default 'b+') is the default symbol for flier points.
        Enter an empty string ('') if you don't want to show fliers.

        vert = 1 (default) makes the boxes vertical.
        vert = 0 makes horizontal boxes.  This seems goofy, but
        that's how Matlab did it.

        whis (default 1.5) defines the length of the whiskers as
        a function of the inner quartile range.  They extend to the
        most extreme data point within ( whis*(75%-25%) ) data range.

        positions (default 1,2,...,n) sets the horizontal positions of
        the boxes. The ticks and limits are automatically set to match
        the positions.

        widths is either a scalar or a vector and sets the width of
        each box. The default is 0.5, or 0.15*(distance between extreme
        positions) if that is smaller.

        x is a Numeric array

        Returns a list of the lines added

        """
        if not self._hold: self.cla()
        holdStatus = self._hold
        x = asarray(x)
        whiskers, caps, boxes, medians, fliers = [], [], [], [], []

        # if we've got a vector, reshape it
        rank = len(x.shape)
        if 1 == rank:
            x.shape = -1, 1

        row, col = x.shape

        # get some plot info
        if positions is None:
            positions = range(1, col + 1)
        if widths is None:
            distance = max(positions) - min(positions)
            widths = min(0.15*max(distance,1.0), 0.5)
        if isinstance(widths, float) or isinstance(widths, int):
            widths = ones((col,), 'd') * widths

        # loop through columns, adding each to plot
        self.hold(True)
        for i,pos in enumerate(positions):
            d = x[:,i]
            # get median and quartiles
            q1, med, q3 = prctile(d,[25,50,75])
            # get high extreme
            iq = q3 - q1
            hi_val = q3 + whis*iq
            wisk_hi = compress( d <= hi_val , d )
            if len(wisk_hi) == 0:
                wisk_hi = q3
            else:
                wisk_hi = max(wisk_hi)
            # get low extreme
            lo_val = q1 - whis*iq
            wisk_lo = compress( d >= lo_val, d )
            if len(wisk_lo) == 0:
                wisk_lo = q1
            else:
                wisk_lo = min(wisk_lo)
            # get fliers - if we are showing them
            flier_hi = []
            flier_lo = []
            flier_hi_x = []
            flier_lo_x = []
            if len(sym) != 0:
                flier_hi = compress( d > wisk_hi, d )
                flier_lo = compress( d < wisk_lo, d )
                flier_hi_x = ones(flier_hi.shape[0]) * pos
                flier_lo_x = ones(flier_lo.shape[0]) * pos

            # get x locations for fliers, whisker, whisker cap and box sides
            box_x_min = pos - widths[i] * 0.5
            box_x_max = pos + widths[i] * 0.5

            wisk_x = ones(2) * pos

            cap_x_min = pos - widths[i] * 0.25
            cap_x_max = pos + widths[i] * 0.25
            cap_x = [cap_x_min, cap_x_max]

            # get y location for median
            med_y = [med, med]

            # calculate 'regular' plot
            if notch == 0:
                # make our box vectors
                box_x = [box_x_min, box_x_max, box_x_max, box_x_min, box_x_min ]
                box_y = [q1, q1, q3, q3, q1 ]
                # make our median line vectors
                med_x = [box_x_min, box_x_max]
            # calculate 'notch' plot
            else:
                notch_max = med + 1.57*iq/sqrt(row)
                notch_min = med - 1.57*iq/sqrt(row)
                if notch_max > q3:
                    notch_max = q3
                if notch_min < q1:
                    notch_min = q1
                # make our notched box vectors
                box_x = [box_x_min, box_x_max, box_x_max, cap_x_max, box_x_max, box_x_max, box_x_min, box_x_min, cap_x_min, box_x_min, box_x_min ]
                box_y = [q1, q1, notch_min, med, notch_max, q3, q3, notch_max, med, notch_min, q1]
                # make our median line vectors
                med_x = [cap_x_min, cap_x_max]
                med_y = [med, med]

            # vertical or horizontal plot?
            if vert:
                def doplot(*args):
                    return self.plot(*args)
            else:
                def doplot(*args):
                    shuffled = []
                    for i in range(0, len(args), 3):
                        shuffled.extend([args[i+1], args[i], args[i+2]])
                    return self.plot(*shuffled)

            whiskers.extend(doplot(wisk_x, [q1, wisk_lo], 'b--',
                                   wisk_x, [q3, wisk_hi], 'b--'))
            caps.extend(doplot(cap_x, [wisk_hi, wisk_hi], 'k-',
                               cap_x, [wisk_lo, wisk_lo], 'k-'))
            boxes.extend(doplot(box_x, box_y, 'b-'))
            medians.extend(doplot(med_x, med_y, 'r-'))
            fliers.extend(doplot(flier_hi_x, flier_hi, sym,
                                 flier_lo_x, flier_lo, sym))

        # fix our axes/ticks up a little
        if 1 == vert:
            setticks, setlim = self.set_xticks, self.set_xlim
        else:
            setticks, setlim = self.set_yticks, self.set_ylim

        newlimits = min(positions)-0.5, max(positions)+0.5
        setlim(newlimits)
        setticks(positions)

        # reset hold status
        self.hold(holdStatus)

        return dict(whiskers=whiskers, caps=caps, boxes=boxes,
                    medians=medians, fliers=fliers)

    def scatter(self, x, y, s=20, c='b', marker='o', cmap=None, norm=None,
                    vmin=None, vmax=None, alpha=1.0, linewidths=None,
                    faceted=True, verts=None,
                    **kwargs):
        """
        SCATTER(x, y, s=20, c='b', marker='o', cmap=None, norm=None,
            vmin=None, vmax=None, alpha=1.0, linewidths=None,
            faceted=True, **kwargs)
        Supported function signatures:

            SCATTER(x, y) - make a scatter plot of x vs y

            SCATTER(x, y, s) - make a scatter plot of x vs y with size in area
              given by s

            SCATTER(x, y, s, c) - make a scatter plot of x vs y with size in area
              given by s and colors given by c

            SCATTER(x, y, s, c, **kwargs) - control colormapping and scaling
              with keyword args; see below

        Make a scatter plot of x versus y.  s is a size in points^2 a scalar
        or an array of the same length as x or y.  c is a color and can be a
        single color format string or an length(x) array of intensities which
        will be mapped by the matplotlib.colors.colormap instance cmap

        The marker can be one of

            's' : square
            'o' : circle
            '^' : triangle up
            '>' : triangle right
            'v' : triangle down
            '<' : triangle left
            'd' : diamond
            'p' : pentagram
            'h' : hexagon
            '8' : octagon


        if marker is None and verts is not None, verts is a sequence
        of (x,y) vertices for a custom scatter symbol.  The

        s is a size argument in points squared.

        Any or all of x, y, s, and c may be masked arrays, in which
        case all masks will be combined and only unmasked points
        will be plotted.

        Other keyword args; the color mapping and normalization arguments will
        on be used if c is an array of floats

          * cmap = cm.jet : a cm Colormap instance from matplotlib.cm.
            defaults to rc image.cmap

          * norm = normalize() : matplotlib.colors.normalize is used to
            scale luminance data to 0,1.

          * vmin=None and vmax=None : vmin and vmax are used in conjunction
            with norm to normalize luminance data.  If either are None, the
            min and max of the color array C is used.  Note if you pass a norm
            instance, your settings for vmin and vmax will be ignored

          * alpha =1.0 : the alpha value for the patches

          * linewidths, if None, defaults to (lines.linewidth,).  Note
            that this is a tuple, and if you set the linewidths
            argument you must set it as a sequence of floats, as
            required by RegularPolyCollection -- see
            matplotlib.collections.RegularPolyCollection for details

         * faceted: if True, will use the default edgecolor for the
           markers.  If False, will set the edgecolors to be the same
           as the facecolors
           """
        # convert, if necessary
        kwargs_copy = kwargs.copy()
        self._update_units_args(kwargs_copy)
        xunits = kwargs_copy.pop('xunits', None)
        yunits = kwargs_copy.pop('yunits', None)
        x = units_conversion(x, self._xunits)
        y = units_conversion(y, self._yunits)


        if not self._hold: self.cla()

        syms =  { # a dict from symbol to (numsides, angle)
            's' : (4, math.pi/4.0),  # square
            'o' : (20, 0),           # circle
            '^' : (3,0),             # triangle up
            '>' : (3,math.pi/2.0),   # triangle right
            'v' : (3,math.pi),       # triangle down
            '<' : (3,3*math.pi/2.0), # triangle left
            'd' : (4,0),             # diamond
            'p' : (5,0),             # pentagram
            'h' : (6,0),             # hexagon
            '8' : (8,0),             # octagon
            }

        x, y, s, c = delete_masked_points(x, y, s, c)

        if kwargs_copy.has_key('color'):
            c = kwargs_copy['color']
            kwargs_copy.pop('color')
        if not is_string_like(c) and iterable(c) and len(c)==len(x):
            colors = None
        else:
            colors = ( colorConverter.to_rgba(c, alpha), )


        if not iterable(s):
            scales = (s,)
        else:
            scales = s

        if faceted: edgecolors = None
        else: edgecolors = 'None'

        sym = syms.get(marker)
        if sym is None and verts is None:
            raise ValueError('Unknown marker symbol to scatter')

        if sym is not None:
            numsides, rotation = syms[marker]
            collection = RegularPolyCollection(
                self.figure.dpi,
                numsides, rotation, scales,
                facecolors = colors,
                edgecolors = edgecolors,
                linewidths = linewidths,
                offsets = zip(x,y),
                transOffset = self.transData,
                )
        else:
            verts = asarray(verts)
            # hmm, the scaling is whacked -- how do we want to scale with custom verts?
            scales = asarray(scales)
            #scales = sqrt(scales * self.figure.dpi.get() / 72.)
            if len(scales)==1:
                verts = [scales[0]*verts]
            else:
                # todo -- make this nx friendly
                verts = [verts*s for s in scales]
            collection = PolyCollection(
                verts,
                facecolors = colors,
                edgecolors = edgecolors,
                linewidths = linewidths,
                offsets = zip(x,y),
                transOffset = self.transData,
                )
            collection.set_transform(identity_transform())
        collection.set_alpha(alpha)
        collection.update(kwargs_copy)

        if colors is None:
            if norm is not None: assert(isinstance(norm, normalize))
            if cmap is not None: assert(isinstance(cmap, Colormap))
            collection.set_array(asarray(c))
            collection.set_cmap(cmap)
            collection.set_norm(norm)

            if vmin is not None or vmax is not None:
                collection.set_clim(vmin, vmax)
            else:
                collection.autoscale()

        minx = amin(x)
        maxx = amax(x)
        miny = amin(y)
        maxy = amax(y)

        w = maxx-minx
        h = maxy-miny

        # the pad is a little hack to deal with the fact that we don't
        # want to transform all the symbols whose scales are in points
        # to data coords to get the exact bounding box for efficiency
        # reasons.  It can be done right if this is deemed important
        padx, pady = 0.05*w, 0.05*h
        corners = (minx-padx, miny-pady), (maxx+padx, maxy+pady)
        self.update_datalim( corners)
        self.autoscale_view()

        # add the collection last
        self.add_collection(collection)
        return collection


    def scatter_classic(self, x, y, s=None, c='b'):
        """
        SCATTER_CLASSIC(x, y, s=None, c='b')

        Make a scatter plot of x versus y.  s is a size (in data coords) and
        can be either a scalar or an array of the same length as x or y.  c is
        a color and can be a single color format string or an length(x) array
        of intensities which will be mapped by the colormap jet.

        If size is None a default size will be used
        """
        if not self._hold: self.cla()
        if is_string_like(c):
            c = [c]*len(x)
        elif not iterable(c):
            c = [c]*len(x)
        else:
            norm = normalize()
            norm(c)
            c = cm.jet(c)

        if s is None:
            s = [abs(0.015*(amax(y)-amin(y)))]*len(x)
        elif not iterable(s):
            s = [s]*len(x)

        if len(c)!=len(x):
            raise ValueError, 'c and x are not equal lengths'
        if len(s)!=len(x):
            raise ValueError, 's and x are not equal lengths'

        patches = []
        for thisX, thisY, thisS, thisC in zip(x,y,s,c):
            #print thisX, thisY, thisS, thisC
            circ = Circle( (thisX, thisY),
                           radius=thisS,
                           )
            circ.set_facecolor(thisC)
            self.add_patch(circ)
            patches.append(circ)
        self.autoscale_view()
        return patches

    def arrow(self, x, y, dx, dy, **kwargs):
        """Draws arrow on specified axis from (x,y) to (x+dx,y+dy)."""
        a = FancyArrow(x, y, dx, dy, **kwargs)
        self.add_artist(a)
        return a

    def quiverkey(self, *args, **kw):
        qk = QuiverKey(*args, **kw)
        self.add_artist(qk)
        return qk
    quiverkey.__doc__ = QuiverKey.quiverkey_doc

    def quiver2(self, *args, **kw):
        q = Quiver(self, *args, **kw)
        self.add_collection(q)
        self.update_datalim_numerix(q.X, q.Y)
        self.autoscale_view()
        return q
    quiver2.__doc__ = Quiver.quiver_doc

    def quiver(self, *args, **kw):
        if (len(args) == 3 or len(args) == 5) and not iterable(args[-1]):
            return self.quiver_classic(*args, **kw)
        c = kw.get('color', None)
        if c is not None:
            try:
                if not is_color_like(c):
                    assert shape(asarray(c)) == shape(asarray(args[-1]))
                    return self.quiver_classic(*args, **kw)
            except:
                pass
        return self.quiver2(*args, **kw)
    quiver.__doc__ = Quiver.quiver_doc

    def quiver_classic(self, U, V, *args, **kwargs ):
        """
        QUIVER( X, Y, U, V )
        QUIVER( U, V )
        QUIVER( X, Y, U, V, S)
        QUIVER( U, V, S )
        QUIVER( ..., color=None, width=1.0, cmap=None, norm=None )

        Make a vector plot (U, V) with arrows on a grid (X, Y)

        If X and Y are not specified, U and V must be 2D arrays.  Equally spaced
        X and Y grids are then generated using the meshgrid command.

        color can be a color value or an array of colors, so that the arrows can be
        colored according to another dataset.  If cmap is specified and color is 'length',
        the colormap is used to give a color according to the vector's length.

        If color is a scalar field, the colormap is used to map the scalar to a color
        If a colormap is specified and color is an array of color triplets, then the
        colormap is ignored

        width is a scalar that controls the width of the arrows

        if S is specified it is used to scale the vectors. Use S=0 to disable automatic
        scaling.
        If S!=0, vectors are scaled to fit within the grid and then are multiplied by S.


        """
        msg = '''This version of quiver is obsolete and will be
        phased out; please use the new quiver.
        '''
        warnings.warn(msg, DeprecationWarning)
        if not self._hold: self.cla()
        do_scale = True
        S = 1.0
        if len(args)==0:
            # ( U, V )
            U = asarray(U)
            V = asarray(V)
            X,Y = meshgrid( arange(U.shape[1]), arange(U.shape[0]) )
        elif len(args)==1:
            # ( U, V, S )
            U = asarray(U)
            V = asarray(V)
            X,Y = meshgrid( arange(U.shape[1]), arange(U.shape[0]) )
            S = float(args[0])
            do_scale = ( S != 0.0 )
        elif len(args)==2:
            # ( X, Y, U, V )
            X = asarray(U)
            Y = asarray(V)
            U = asarray(args[0])
            V = asarray(args[1])
        elif len(args)==3:
            # ( X, Y, U, V )
            X = asarray(U)
            Y = asarray(V)
            U = asarray(args[0])
            V = asarray(args[1])
            S = float(args[2])
            do_scale = ( S != 0.0 )

        assert U.shape == V.shape
        assert X.shape == Y.shape
        assert U.shape == X.shape

        U = ravel(U)
        V = ravel(V)
        X = ravel(X)
        Y = ravel(Y)

        arrows = []
        N = sqrt( U**2+V**2 )
        if do_scale:
            Nmax = maximum.reduce(N) or 1 # account for div by zero
            U = U*(S/Nmax)
            V = V*(S/Nmax)
            N = N*Nmax

        alpha = kwargs.get('alpha', 1.0)
        width = kwargs.get('width', .5)
        norm = kwargs.get('norm', None)
        cmap = kwargs.get('cmap', None)
        vmin = kwargs.get('vmin', None)
        vmax = kwargs.get('vmax', None)
        color = kwargs.get('color', None)
        shading = kwargs.get('shading', 'faceted')

        C = None
        if color == 'length' or color is True:
            if color is True:
                warnings.warn('''Use "color='length'",
                not "color=True"''', DeprecationWarning)
            C = N
        elif color is None:
            color = (0,0,0,1)
        else:
            clr = ravel(asarray(color))
            if clr.shape == U.shape:
                C = clr

        I = U.shape[0]
        #arrows = []
        #for i in xrange(I):
        #    arrows.append( FancyArrow(X[i],Y[i],U[i],V[i],0.1*S ).get_verts() )
        arrows = [FancyArrow(X[i],Y[i],U[i],V[i],0.1*S ).get_verts()
                                                    for i in xrange(I)]

        collection = PolyCollection(
            arrows,
            edgecolors = 'None',
            antialiaseds = (1,),
            linewidths = (width,),
            )
        if C is not None:
            collection.set_array( ravel(C) )
            collection.set_cmap(cmap)
            collection.set_norm(norm)
            if norm is not None:
                collection.set_clim( vmin, vmax )
        else:
            collection.set_facecolor(color)
        self.add_collection( collection )

        lims = asarray(arrows)
        _max = maximum.reduce( maximum.reduce( lims ))
        _min = minimum.reduce( minimum.reduce( lims ))
        self.update_datalim( [ tuple(_min), tuple(_max) ] )
        self.autoscale_view()
        return collection





    def fill(self, *args, **kwargs):
        """
        FILL(*args, **kwargs)

        plot filled polygons.  *args is a variable length argument, allowing
        for multiple x,y pairs with an optional color format string; see plot
        for details on the argument parsing.  For example, all of the
        following are legal, assuming ax is an Axes instance:

          ax.fill(x,y)            # plot polygon with vertices at x,y
          ax.fill(x,y, 'b' )      # plot polygon with vertices at x,y in blue

        An arbitrary number of x, y, color groups can be specified, as in
          ax.fill(x1, y1, 'g', x2, y2, 'r')

        Return value is a list of patches that were added

        The same color strings that plot supports are supported by the fill
        format string.

        The kwargs that are can be used to set line properties (any
        property that has a set_* method).  You can use this to set edge
        color, face color, etc.
        """
        d = kwargs.copy()
        self._update_units_args(d)
        if not self._hold: self.cla()
        patches = []
        for poly in self._get_patches_for_fill(*args, **d):
            self.add_patch( poly )
            patches.append( poly )
        self.autoscale_view()
        return patches



    #### plotting z(x,y): imshow, pcolor and relatives, contour

    def imshow(self, X,
               cmap = None,
               norm = None,
               aspect=None,
               interpolation=None,
               alpha=1.0,
               vmin = None,
               vmax = None,
               origin=None,
               extent=None,
               shape=None,
               filternorm=1,
               filterrad=4.0,
               imlim=None):
        """

        IMSHOW(X, cmap=None, norm=None, aspect=None, interpolation=None,
               alpha=1.0, vmin=None, vmax=None, origin=None, extent=None)

        IMSHOW(X) - plot image X to current axes, resampling to scale to axes
                    size (X may be numarray/Numeric array or PIL image)

        IMSHOW(X, **kwargs) - Use keyword args to control image scaling,
        colormapping etc. See below for details


        Display the image in X to current axes.  X may be a float array, a
        UInt8 array or a PIL image. If X is an array, X can have the following
        shapes:

            MxN    : luminance (grayscale, float array only)

            MxNx3  : RGB (float or UInt8 array)

            MxNx4  : RGBA (float or UInt8 array)

        The value for each component of MxNx3 and MxNx4 float arrays should be
        in the range 0.0 to 1.0; MxN float arrays may be normalised.

        A matplotlib.image.AxesImage instance is returned

        The following kwargs are allowed:

          * cmap is a cm colormap instance, eg cm.jet.  If None, default to rc
            image.cmap value (Ignored when X has RGB(A) information)

          * aspect is one of: auto, equal, or a number.  If None, default to rc
            image.aspect value

          * interpolation is one of:

            'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
            'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
            'lanczos', 'blackman'

            if interpolation is None, default to rc
            image.interpolation.  See also th the filternorm and
            filterrad parameters

          * norm is a matplotlib.colors.normalize instance; default is
            normalization().  This scales luminance -> 0-1 (only used for an
            MxN float array).

          * vmin and vmax are used to scale a luminance image to 0-1.  If
            either is None, the min and max of the luminance values will be
            used.  Note if you pass a norm instance, the settings for vmin and
            vmax will be ignored.

          * alpha = 1.0 : the alpha blending value

          * origin is either upper or lower, which indicates where the [0,0]
            index of the array is in the upper left or lower left corner of
            the axes.  If None, default to rc image.origin

          * extent is a data xmin, xmax, ymin, ymax for making image plots
            registered with data plots.  Default is the image dimensions
            in pixels

          * shape is for raw buffer images

          * filternorm is a parameter for the antigrain image resize
            filter.  From the antigrain documentation, if normalize=1,
            the filter normalizes integer values and corrects the
            rounding errors. It doesn't do anything with the source
            floating point values, it corrects only integers according
            to the rule of 1.0 which means that any sum of pixel
            weights must be equal to 1.0.  So, the filter function
            must produce a graph of the proper shape.

         * filterrad: the filter radius for filters that have a radius
           parameter, ie when interpolation is one of: 'sinc',
           'lanczos' or 'blackman'


    """

        if not self._hold: self.cla()

        if norm is not None: assert(isinstance(norm, normalize))
        if cmap is not None: assert(isinstance(cmap, Colormap))
        if aspect is None: aspect = rcParams['image.aspect']
        self.set_aspect(aspect)
        im = AxesImage(self, cmap, norm, interpolation, origin, extent,
                       filternorm=filternorm,
                       filterrad=filterrad)

        im.set_data(X)
        im.set_alpha(alpha)
        #if norm is None and shape is None:
        #    im.set_clim(vmin, vmax)
        if vmin is not None or vmax is not None:
            im.set_clim(vmin, vmax)
        else:
            im.autoscale()

        xmin, xmax, ymin, ymax = im.get_extent()

        corners = (xmin, ymin), (xmax, ymax)
        self.update_datalim(corners)
        if self._autoscaleon:
            self.set_xlim((xmin, xmax))
            self.set_ylim((ymin, ymax))
        self.images.append(im)

        return im



    def pcolor(self, *args, **kwargs):
        """
        PCOLOR(*args, **kwargs)

        Function signatures

          PCOLOR(C) - make a pseudocolor plot of matrix C

          PCOLOR(X, Y, C) - a pseudo color plot of C on the matrices X and Y

          PCOLOR(C, **kwargs) - Use keyword args to control colormapping and
                                scaling; see below

        X,Y and C may be masked arrays.  If either C[i,j], or one
        of the vertices surrounding C[i,j] (X or Y at [i,j],[i+1,j],
        [i,j+1],[i=1,j+1]) is masked, nothing is plotted.

        Optional keyword args are shown with their defaults below (you must
        use kwargs for these):

          * cmap = cm.jet : a cm Colormap instance from matplotlib.cm.
            defaults to cm.jet

          * norm = normalize() : matplotlib.colors.normalize is used to scale
            luminance data to 0,1.

          * vmin=None and vmax=None : vmin and vmax are used in conjunction
            with norm to normalize luminance data.  If either are None, the
            min and max of the color array C is used.  If you pass a norm
            instance, vmin and vmax will be None

          * shading = 'flat' : or 'faceted'.  If 'faceted', a black grid is
            drawn around each rectangle; if 'flat', edge colors are same as
            face colors

          * alpha=1.0 : the alpha blending value

        Return value is a matplotlib.collections.PatchCollection
        object

        Grid Orientation

            The orientation follows the Matlab(TM) convention: an
            array C with shape (nrows, ncolumns) is plotted with
            the column number as X and the row number as Y, increasing
            up; hence it is plotted the way the array would be printed,
            except that the Y axis is reversed.  That is, C is taken
            as C(y,x).

            Similarly for meshgrid:

                x = arange(5)
                y = arange(3)
                X, Y = meshgrid(x,y)

            is equivalent to

                X = array([[0, 1, 2, 3, 4],
                          [0, 1, 2, 3, 4],
                          [0, 1, 2, 3, 4]])

                Y = array([[0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 1],
                          [2, 2, 2, 2, 2]])

            so if you have
                C = rand( len(x), len(y))
            then you need
                pcolor(X, Y, transpose(C))
            or
                pcolor(transpose(C))

        Dimensions

            pcolor differs from Matlab in that Matlab always discards
            the last row and column of C, but matplotlib displays
            the last row and column if X and Y are not specified, or
            if X and Y have one more row and column than C.

        """
        if not self._hold: self.cla()

        alpha = kwargs.get('alpha', 1.0)
        norm = kwargs.get('norm', None)
        cmap = kwargs.get('cmap', None)
        vmin = kwargs.get('vmin', None)
        vmax = kwargs.get('vmax', None)
        shading = kwargs.get('shading', 'faceted')

        if len(args)==1:
            C = args[0]
            numRows, numCols = C.shape
            X, Y = meshgrid(arange(numCols+1), arange(numRows+1) )
        elif len(args)==3:
            X, Y, C = args
        else:
            raise TypeError, 'Illegal arguments to pcolor; see help(pcolor)'

        Nx, Ny = X.shape

        # convert to MA, if necessary.
        C = ma.asarray(C)
        X = ma.asarray(X)
        Y = ma.asarray(Y)
        mask = ma.getmaskarray(X)+ma.getmaskarray(Y)
        xymask = mask[0:-1,0:-1]+mask[1:,1:]+mask[0:-1,1:]+mask[1:,0:-1]
        # don't plot if C or any of the surrounding vertices are masked.
        mask = ma.getmaskarray(C)[0:Nx-1,0:Ny-1]+xymask

        X1 = compress(ravel(mask==0),ravel(ma.filled(X[0:-1,0:-1])))
        Y1 = compress(ravel(mask==0),ravel(ma.filled(Y[0:-1,0:-1])))
        X2 = compress(ravel(mask==0),ravel(ma.filled(X[1:,0:-1])))
        Y2 = compress(ravel(mask==0),ravel(ma.filled(Y[1:,0:-1])))
        X3 = compress(ravel(mask==0),ravel(ma.filled(X[1:,1:])))
        Y3 = compress(ravel(mask==0),ravel(ma.filled(Y[1:,1:])))
        X4 = compress(ravel(mask==0),ravel(ma.filled(X[0:-1,1:])))
        Y4 = compress(ravel(mask==0),ravel(ma.filled(Y[0:-1,1:])))
        npoly = len(X1)
        xy = concatenate((X1[:,newaxis], Y1[:,newaxis],
                             X2[:,newaxis], Y2[:,newaxis],
                             X3[:,newaxis], Y3[:,newaxis],
                             X4[:,newaxis], Y4[:,newaxis]),
                             axis=1)
        verts = reshape(xy, (npoly, 4, 2))

        #verts = zip(zip(X1,Y1),zip(X2,Y2),zip(X3,Y3),zip(X4,Y4))

        C = compress(ravel(mask==0),ravel(ma.filled(C[0:Nx-1,0:Ny-1])))


        if shading == 'faceted':
            edgecolors =  (0,0,0,1),
        else:
            edgecolors = 'None'

        collection = PolyCollection(
            verts,
            edgecolors   = edgecolors,
            antialiaseds = (0,),
            linewidths   = (0.25,),
            )

        collection.set_alpha(alpha)
        collection.set_array(C)
        if norm is not None: assert(isinstance(norm, normalize))
        if cmap is not None: assert(isinstance(cmap, Colormap))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
        else:
            collection.autoscale()
        self.grid(False)

        x = X.compressed()
        y = Y.compressed()
        minx = amin(x)
        maxx = amax(x)
        miny = amin(y)
        maxy = amax(y)

        corners = (minx, miny), (maxx, maxy)
        self.update_datalim( corners)
        self.autoscale_view()

        # add the collection last
        self.add_collection(collection)

        return collection

    def pcolormesh(self, *args, **kwargs):
        """
        PCOLORMESH(*args, **kwargs)

        Function signatures

          PCOLORMESH(C) - make a pseudocolor plot of matrix C

          PCOLORMESH(X, Y, C) - a pseudo color plot of C on the matrices X and Y

          PCOLORMESH(C, **kwargs) - Use keyword args to control colormapping and
                                scaling; see below

        C may be a masked array, but X and Y may not.  Masked array support
        is implemented via cmap and norm; in contrast, pcolor simply does
        not draw quadrilaterals with masked colors or vertices.

        Optional keyword args are shown with their defaults below (you must
        use kwargs for these):

          * cmap = cm.jet : a cm Colormap instance from matplotlib.cm.
            defaults to cm.jet

          * norm = normalize() : matplotlib.colors.normalize is used to scale
            luminance data to 0,1.  Specify it with clip=False if
            C is a masked array.

          * vmin=None and vmax=None : vmin and vmax are used in conjunction
            with norm to normalize luminance data.  If either are None, the
            min and max of the color array C is used.

          * shading = 'flat' : or 'faceted'.  If 'faceted', a black grid is
            drawn around each rectangle; if 'flat', edge colors are same as
            face colors

          * alpha=1.0 : the alpha blending value

        Return value is a matplotlib.collections.PatchCollection
        object

        See pcolor for an explantion of the grid orientation.

        """
        if not self._hold: self.cla()

        alpha = kwargs.get('alpha', 1.0)
        norm = kwargs.get('norm', None)
        cmap = kwargs.get('cmap', None)
        vmin = kwargs.get('vmin', None)
        vmax = kwargs.get('vmax', None)
        shading = kwargs.get('shading', 'faceted')

        if len(args)==1:
            C = args[0]
            numRows, numCols = C.shape
            X, Y = meshgrid(arange(numCols+1), arange(numRows+1) )
        elif len(args)==3:
            X, Y, C = args
        else:
            raise TypeError, 'Illegal arguments to pcolormesh; see help(pcolormesh)'

        Nx, Ny = X.shape

        # convert to one dimensional arrays
        C = ma.ravel(C[0:Nx-1, 0:Ny-1]) # data point in each cell is value at lower left corner
        X = ravel(X)
        Y = ravel(Y)

        coords = zeros(((Nx * Ny), 2), Float32)
        # Numeric and numpy refuse to cast the Float64 arrays
        # to Float32 with simple assignment, so we do it explicitly.
        coords[:, 0] = X.astype(Float32)
        coords[:, 1] = Y.astype(Float32)
        #print coords

        if shading == 'faceted':
            showedges = 1
        else:
            showedges = 0

        collection = QuadMesh(Ny - 1, Nx - 1, coords, showedges)
        collection.set_alpha(alpha)
        collection.set_array(C)
        if norm is not None: assert(isinstance(norm, normalize))
        if cmap is not None: assert(isinstance(cmap, Colormap))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
        else:
            collection.autoscale()

        self.grid(False)

        minx = amin(X)
        maxx = amax(X)
        miny = amin(Y)
        maxy = amax(Y)

        corners = (minx, miny), (maxx, maxy)
        self.update_datalim( corners)
        self.autoscale_view()
        ##################collection.update_scalarmappable()

        # add the collection last
        self.add_collection(collection)
        return collection

    def pcolor_classic(self, *args, **kwargs):
        """
        PCOLOR_CLASSIC(self, *args, **kwargs)

        Function signatures

            pcolor(C) - make a pseudocolor plot of matrix C

            pcolor(X, Y, C) - a pseudo color plot of C on the matrices X and Y

            pcolor(C, cmap=cm.jet) - make a pseudocolor plot of matrix C using
              rectangle patches using a colormap jet.  Colormaps are avalible
              in matplotlib.cm.  You must pass this as a kwarg.

            pcolor(C, norm=normalize()) - the normalization function used
        `     to scale your color data to 0-1.  must be passed as a kwarg.

            pcolor(C, alpha=0.5) - set the alpha of the pseudocolor plot.
              Must be used as a kwarg

        Shading:

            The optional keyword arg shading ('flat' or 'faceted') will
            determine whether a black grid is drawn around each pcolor square.
            Default 'faceteted' e.g., pcolor(C, shading='flat') pcolor(X, Y,
            C, shading='faceted')

        Return value is a list of patch objects.

        Grid orientation

            The behavior of meshgrid in matlab(TM) is a bit counterintuitive for x
            and y arrays.  For example,

                  x = arange(7)
                  y = arange(5)
                  X, Y = meshgrid(x,y)

                  Z = rand( len(x), len(y))
                  pcolor(X, Y, Z)

            will fail in matlab and matplotlib.  You will probably be
            happy with

                pcolor(X, Y, transpose(Z))

            Likewise, for nonsquare Z,

                pcolor(transpose(Z))

            will make the x and y axes in the plot agree with the numrows
            and numcols of Z
        """

        if not self._hold: self.cla()
        shading = kwargs.get('shading', 'faceted')
        cmap = kwargs.get('cmap', cm.get_cmap())
        norm = kwargs.get('norm', normalize())
        alpha = kwargs.get('alpha', 1.0)

        if len(args)==1:
            C = args[0]
            numRows, numCols = C.shape
            X, Y = meshgrid(arange(numCols+1), arange(numRows+1) )
        elif len(args)==3:
            X, Y, C = args
        else:
            raise TypeError, 'Illegal arguments to pcolor; see help(pcolor)'

        Nx, Ny = X.shape

        if isinstance(norm, normalize) and not norm.scaled():
            norm.autoscale(C)

        nc = norm(C)
        RGBA = cmap(nc, alpha)

        patches = []

        for i in range(Nx-1):
            for j in range(Ny-1):
                color = tuple(RGBA[i,j,:3])
                left = X[i,j]
                bottom = Y[i,j]
                width = X[i,j+1]-left
                height = Y[i+1,j]-bottom
                rect = Rectangle(
                    (left, bottom), width, height,
                    )
                rect.set_facecolor(color)
                rect.set_alpha(RGBA[i,j,3])
                if shading == 'faceted':
                    rect.set_linewidth(0.25)
                    rect.set_edgecolor('k')
                else:
                    rect.set_edgecolor(color)
                self.patches.append(rect)
                rect.set_figure(self.figure)
                rect.set_transform(self.transData)
                patches.append(rect)
        self.grid(0)

        x = ravel(X)
        y = ravel(Y)
        minx = amin(x)
        maxx = amax(x)
        miny = amin(y)
        maxy = amax(y)

        corners = (minx, miny), (maxx, maxy)


        self.update_datalim( corners)
        self.autoscale_view()
        return patches


    def contour(self, *args, **kwargs):
        kwargs['filled'] = False
        return ContourSet(self, *args, **kwargs)
    contour.__doc__ = ContourSet.contour_doc

    def contourf(self, *args, **kwargs):
        kwargs['filled'] = True
        return ContourSet(self, *args, **kwargs)
    contourf.__doc__ = ContourSet.contour_doc

    def clabel(self, CS, *args, **kwargs):
        return CS.clabel(*args, **kwargs)
    clabel.__doc__ = ContourSet.clabel.__doc__


    def table(self, **kwargs):
        """
        TABLE(cellText=None, cellColours=None,
              cellLoc='right', colWidths=None,
              rowLabels=None, rowColours=None, rowLoc='left',
              colLabels=None, colColours=None, colLoc='center',
              loc='bottom', bbox=None):

        Add a table to the current axes.  Returns a table instance.  For
        finer grained control over tables, use the Table class and add it
        to the axes with add_table.

        Thanks to John Gill for providing the class and table.
        """
        return table.table(self, **kwargs)


    #### Data analysis

    def hist(self, x, bins=10, normed=0, bottom=0,
             align='edge', orientation='vertical', width=None, **kwargs):
        """
        HIST(x, bins=10, normed=0, bottom=0,
             align='edge', orientation='vertical', width=None, **kwargs)

        Compute the histogram of x.  bins is either an integer number of
        bins or a sequence giving the bins.  x are the data to be binned.

        The return values is (n, bins, patches)

        If normed is true, the first element of the return tuple will
        be the counts normalized to form a probability density, ie,
        n/(len(x)*dbin)

        align = 'edge' | 'center'.  Interprets bins either as edge
        or center values

        orientation = 'horizontal' | 'vertical'.  If horizontal, barh
        will be used and the "bottom" kwarg will be the left edges.

        width: the width of the bars.  If None, automatically compute
        the width.

        kwargs are used to update the properties of the
        hist bars
        """
        if not self._hold: self.cla()
        n, bins = matplotlib.mlab.hist(x, bins, normed)
        if width is None: width = 0.9*(bins[1]-bins[0])
        if orientation == 'horizontal':
            patches = self.barh(bins, n, height=width, left=bottom, align=align)
        elif orientation == 'vertical':
            patches = self.bar(bins, n, width=width, bottom=bottom, align=align)
        else:
            raise ValueError, 'invalid orientation: %s' % orientation
        for p in patches:
            p.update(kwargs)
        return n, bins, silent_list('Patch', patches)

    def psd(self, x, NFFT=256, Fs=2, detrend=detrend_none,
            window=window_hanning, noverlap=0, **kwargs):
        """
        PSD(x, NFFT=256, Fs=2, detrend=detrend_none,
            window=window_hanning, noverlap=0, **kwargs)

        The power spectral density by Welches average periodogram method.  The
        vector x is divided into NFFT length segments.  Each segment is
        detrended by function detrend and windowed by function window.
        noperlap gives the length of the overlap between segments.  The
        absolute(fft(segment))**2 of each segment are averaged to compute Pxx,
        with a scaling to correct for power loss due to windowing.  Fs is the
        sampling frequency.

            NFFT is the length of the fft segment; must be a power of 2

            Fs is the sampling frequency.

            detrend - the function applied to each segment before fft-ing,
              designed to remove the mean or linear trend.  Unlike in matlab,
              where the detrend parameter is a vector, in matplotlib is it a
              function.  The mlab module defines detrend_none, detrend_mean,
              detrend_linear, but you can use a custom function as well.

            window - the function used to window the segments.  window is a
              function, unlike in matlab(TM) where it is a vector.  mlab defines
              window_none, window_hanning, but you can use a custom function
              as well.

            noverlap gives the length of the overlap between segments.

            kwargs are passed to plot to control line props
        Returns the tuple Pxx, freqs

        For plotting, the power is plotted as 10*log10(pxx)) for decibels,
        though pxx itself is returned

        Refs:

          Bendat & Piersol -- Random Data: Analysis and Measurement
          Procedures, John Wiley & Sons (1986)

        """
        if not self._hold: self.cla()
        pxx, freqs = matplotlib.mlab.psd(x, NFFT, Fs, detrend, window, noverlap)
        pxx.shape = len(freqs),

        self.plot(freqs, 10*log10(pxx), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Power Spectrum (dB)')
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly().get_bounds()
        intv = vmax-vmin
        logi = int(log10(intv))
        if logi==0: logi=.1
        step = 10*logi
        #print vmin, vmax, step, intv, math.floor(vmin), math.ceil(vmax)+1
        ticks = arange(math.floor(vmin), math.ceil(vmax)+1, step)
        self.set_yticks(ticks)

        return pxx, freqs


    def csd(self, x, y, NFFT=256, Fs=2, detrend=detrend_none,
            window=window_hanning, noverlap=0, **kwargs):
        """
        CSD(x, y, NFFT=256, Fs=2, detrend=detrend_none,
            window=window_hanning, noverlap=0, **kwargs)

        The cross spectral density Pxy by Welches average periodogram method.
        The vectors x and y are divided into NFFT length segments.  Each
        segment is detrended by function detrend and windowed by function
        window.  The product of the direct FFTs of x and y are averaged over
        each segment to compute Pxy, with a scaling to correct for power loss
        due to windowing.

        See the PSD help for a description of the optional parameters.

        Returns the tuple Pxy, freqs.  Pxy is the cross spectrum (complex
        valued), and 10*log10(|Pxy|) is plotted

        Refs:
          Bendat & Piersol -- Random Data: Analysis and Measurement
            Procedures, John Wiley & Sons (1986)

        """
        if not self._hold: self.cla()
        pxy, freqs = matplotlib.mlab.csd(x, y, NFFT, Fs, detrend, window, noverlap)
        pxy.shape = len(freqs),
        # pxy is complex

        self.plot(freqs, 10*log10(absolute(pxy)), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Cross Spectrum Magnitude (dB)')
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly().get_bounds()

        intv = vmax-vmin
        step = 10*int(log10(intv))

        ticks = arange(math.floor(vmin), math.ceil(vmax)+1, step)
        self.set_yticks(ticks)

        return pxy, freqs

    def cohere(self, x, y, NFFT=256, Fs=2, detrend=detrend_none,
               window=window_hanning, noverlap=0, **kwargs):
        """
        COHERE(x, y, NFFT=256, Fs=2, detrend=detrend_none,
              window=window_hanning, noverlap=0, **kwargs)

        cohere the coherence between x and y.  Coherence is the normalized
        cross spectral density

          Cxy = |Pxy|^2/(Pxx*Pyy)

        The return value is (Cxy, f), where f are the frequencies of the
        coherence vector.

        See the PSD help for a description of the optional parameters.

        kwargs are applied to the lines

        Returns the tuple Cxy, freqs

        Refs: Bendat & Piersol -- Random Data: Analysis and Measurement
          Procedures, John Wiley & Sons (1986)
        """
        if not self._hold: self.cla()
        cxy, freqs = matplotlib.mlab.cohere(x, y, NFFT, Fs, detrend, window, noverlap)

        self.plot(freqs, cxy, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Coherence')
        self.grid(True)

        return cxy, freqs

    def specgram(self, x, NFFT=256, Fs=2, detrend=detrend_none,
                 window=window_hanning, noverlap=128,
                 cmap = None, xextent=None):
        """
        SPECGRAM(x, NFFT=256, Fs=2, detrend=detrend_none,
                 window=window_hanning, noverlap=128,
                 cmap=None, xextent=None)

        Compute a spectrogram of data in x.  Data are split into NFFT length
        segements and the PSD of each section is computed.  The windowing
        function window is applied to each segment, and the amount of overlap
        of each segment is specified with noverlap.

            * cmap is a colormap; if None use default determined by rc

            * xextent is the image extent in the xaxes xextent=xmin, xmax -
              default 0, max(bins), 0, max(freqs) where bins is the return
              value from matplotlib.matplotlib.mlab.specgram

            * See help(psd) for information on the other keyword arguments.

        Return value is (Pxx, freqs, bins, im), where

            bins are the time points the spectrogram is calculated over

            freqs is an array of frequencies

            Pxx is a len(times) x len(freqs) array of power

            im is a matplotlib.image.AxesImage.
    """
        if not self._hold: self.cla()

        Pxx, freqs, bins = matplotlib.mlab.specgram(x, NFFT, Fs, detrend,
             window, noverlap)


        Z = 10*log10(Pxx)
        Z =  flipud(Z)

        if xextent is None: xextent = 0, amax(bins)
        xmin, xmax = xextent
        extent = xmin, xmax, 0, amax(freqs)
        im = self.imshow(Z, cmap, extent=extent)
        self.axis('auto')

        return Pxx, freqs, bins, im

    def spy(self, Z,  marker='s', markersize=10, **kwargs):
        """
        SPY(Z, **kwargs) plots the sparsity pattern of the matrix Z
        using plot markers.

        kwargs give the marker properties - see help(plot) for more
        information on marker properties

        The line handles are returned
        """
        if hasattr(Z, 'tocoo'):
            c = Z.tocoo()
            x = c.row
            y = c.col
            z = c.data
        else:
            x,y,z = matplotlib.mlab.get_xyz_where(Z, Z>0)
        return self.plot(x+0.5,y+0.5, linestyle='None',
                         marker=marker,markersize=markersize, **kwargs)

    def spy2(self, Z):
        """
        SPY2(Z) plots the sparsity pattern of the matrix Z as an image

        The image instance is returned
        """

        #binary colormap min white, max black
        cmapdata = {
             'red'  :  ((0., 1., 1.), (1., 0., 0.)),
             'green':  ((0., 1., 1.), (1., 0., 0.)),
             'blue' :  ((0., 1., 1.), (1., 0., 0.))
             }
        binary =  LinearSegmentedColormap('binary',  cmapdata, 2)
        Z = where(Z>0,1.,0.)
        return self.imshow(transpose(Z), interpolation='nearest', cmap=binary)


class SubplotBase:
    """
    Emulate matlab's(TM) subplot command, creating axes with

      Subplot(numRows, numCols, plotNum)

    where plotNum=1 is the first plot number and increasing plotNums
    fill rows first.  max(plotNum)==numRows*numCols

    You can leave out the commas if numRows<=numCols<=plotNum<10, as
    in

      Subplot(211)    # 2 rows, 1 column, first (upper) plot
    """

    def __init__(self, fig, *args, **kwargs):
        """
        fig is a figure instance

        args is a varargs to specify the subplot

        """

        self.figure = fig

        if len(args)==1:
            s = str(args[0])
            if len(s) != 3:
                raise ValueError('Argument to subplot must be a 3 digits long')
            rows, cols, num = map(int, s)
        elif len(args)==3:
            rows, cols, num = args
        else:
            raise ValueError(  'Illegal argument to subplot')


        total = rows*cols
        num -= 1    # convert from matlab to python indexing ie num in range(0,total)
        if num >= total:
            raise ValueError( 'Subplot number exceeds total subplots')
        self._rows = rows
        self._cols = cols
        self._num = num

        self.update_params()

    def get_geometry(self):
        'get the subplot geometry, eg 2,2,3'
        return self._rows, self._cols, self._num+1

    def change_geometry(self, numrows, numcols, num):
        'change subplot geometry, eg from 1,1,1 to 2,2,3'
        self._rows = numrows
        self._cols = numcols
        self._num = num-1
        self.update_params()
        self.set_position([self.figLeft, self.figBottom,  self.figW, self.figH])

    def update_params(self):
        'update the subplot position from fig.subplotpars'

        rows = self._rows
        cols = self._cols
        num = self._num

        pars = self.figure.subplotpars
        left = pars.left
        right = pars.right
        bottom = pars.bottom
        top = pars.top
        wspace = pars.wspace
        hspace = pars.hspace
        totWidth = right-left
        totHeight = top-bottom

        figH = totHeight/(rows + hspace*(rows-1))
        sepH = hspace*figH

        figW = totWidth/(cols + wspace*(cols-1))
        sepW = wspace*figW

        rowNum, colNum =  divmod(num, cols)

        figBottom = top - (rowNum+1)*figH - rowNum*sepH
        figLeft = left + colNum*(figW + sepW)

        self.figBottom = figBottom
        self.figLeft = figLeft
        self.figW = figW
        self.figH = figH
        self.rowNum = rowNum
        self.colNum = colNum
        self.numRows = rows
        self.numCols = cols

        if 0:
            print 'rcn', rows, cols, num
            print 'lbrt', left, bottom, right, top
            print 'self.figBottom', self.figBottom
            print 'self.figLeft', self.figLeft
            print 'self.figW', self.figW
            print 'self.figH', self.figH
            print 'self.rowNum', self.rowNum
            print 'self.colNum', self.colNum
            print 'self.numRows', self.numRows
            print 'self.numCols', self.numCols


    def is_first_col(self):
        return self.colNum==0

    def is_first_row(self):
        return self.rowNum==0

    def is_last_row(self):
        return self.rowNum==self.numRows-1


    def is_last_col(self):
        return self.colNum==self.numCols-1

class Subplot(SubplotBase, Axes):
    """
    Emulate matlab's(TM) subplot command, creating axes with

      Subplot(numRows, numCols, plotNum)

    where plotNum=1 is the first plot number and increasing plotNums
    fill rows first.  max(plotNum)==numRows*numCols

    You can leave out the commas if numRows<=numCols<=plotNum<10, as
    in

      Subplot(211)    # 2 rows, 1 column, first (upper) plot
    """
    def __init__(self, fig, *args, **kwargs):
        SubplotBase.__init__(self, fig, *args)
        Axes.__init__(self, fig, [self.figLeft, self.figBottom,
                                  self.figW, self.figH], **kwargs)



class PolarAxes(Axes):
    """

    Make a PolarAxes.  The rectangular bounding box of the axes is given by


       PolarAxes(position=[left, bottom, width, height])

    where all the arguments are fractions in [0,1] which specify the
    fraction of the total figure window.

    axisbg is the color of the axis background

    Attributes:
      thetagridlines  : a list of Line2D for the theta grids
      rgridlines      : a list of Line2D for the radial grids
      thetagridlabels : a list of Text for the theta grid labels
      rgridlabels     : a list of Text for the theta grid labels

    """

    RESOLUTION = 200

    def __init__(self, *args, **kwarg):
        Axes.__init__(self, *args, **kwarg)
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.cla()

    def _init_axis(self):
        "nuthin to do"
        pass
    def _set_lim_and_transforms(self):
        """
        set the dataLim and viewLim BBox attributes and the
        transData and transAxes Transformation attributes
        """

        # the lim are theta, r

        self.dataLim = Bbox( Point( Value(5/4.*math.pi), Value(math.sqrt(2))),
                             Point( Value(1/4.*math.pi), Value(math.sqrt(2))))
        self.viewLim = Bbox( Point( Value(5/4.*math.pi), Value(math.sqrt(2))),
                             Point( Value(1/4.*math.pi), Value(math.sqrt(2))))

        self.transData = NonseparableTransformation(self.viewLim, self.bbox,
                                                                  FuncXY(POLAR))
        self.transAxes = get_bbox_transform(unit_bbox(), self.bbox)


    def cla(self):
        'Clear the current axes'

        # init these w/ some arbitrary numbers - they'll be updated as
        # data is added to the axes

        self._get_lines = _process_plot_var_args()
        self._get_patches_for_fill = _process_plot_var_args('fill')

        self._gridOn = rcParams['polaraxes.grid']
        self.thetagridlabels = []
        self.thetagridlines = []
        self.rgridlabels = []
        self.rgridlines = []

        self.lines = []
        self.images = []
        self.patches = []
        self.collections = []
        self.texts = []     # text in axis coords

        self.grid(self._gridOn)
        self.title =  Text(
            x=0.5, y=1.05, text='',
            fontproperties=FontProperties(size=rcParams['axes.titlesize']),
            verticalalignment='bottom',
            horizontalalignment='center',
            )
        self.title.set_transform(self.transAxes)

        self._set_artist_props(self.title)


        self.thetas = linspace(0,2*math.pi, self.RESOLUTION)
        verts = zip(self.thetas, ones(self.RESOLUTION))
        self.axesPatch = Polygon(
            verts,
            facecolor=self._axisbg,
            edgecolor=rcParams['axes.edgecolor'],
            )

        self.axesPatch.set_figure(self.figure)
        self.axesPatch.set_transform(self.transData)
        self.axesPatch.set_linewidth(rcParams['axes.linewidth'])
        self.axison = True

        # we need to set a view and data interval from 0->rmax to make
        # the formatter and locator work correctly
        self.rintv = Interval(Value(0), Value(1))
        self.rintd = Interval(Value(0), Value(1))

        self.rformatter  = ScalarFormatter()
        self.rformatter.set_view_interval(self.rintv)
        self.rformatter.set_data_interval(self.rintd)
        self.rlocator = AutoLocator()
        self.rlocator.set_view_interval(self.rintv)
        self.rlocator.set_data_interval(self.rintd)

        angles = arange(0, 360, 45)
        radii = arange(0.2, 1.1, 0.2)
        self.set_thetagrids(angles)
        self.set_rgrids(radii)

    def grid(self, b):
        'Set the axes grids on or off; b is a boolean'
        self._gridOn = b

    def autoscale_view(self, scalex=True, scaley=True):
        'set the view limits to include all the data in the axes'
        self.rintd.set_bounds(0, self.get_rmax())
        rmin, rmax = self.rlocator.autoscale()
        self.rintv.set_bounds(rmin, rmax)

        self.axesPatch.xy = zip(self.thetas, rmax*ones(self.RESOLUTION))
        val = rmax*math.sqrt(2)
        self.viewLim.intervaly().set_bounds(val, val)

        ticks = self.rlocator()
        self.set_rgrids(ticks)

        for t in self.thetagridlabels:
            t.set_y(1.05*rmax)

        r = linspace(0, rmax, self.RESOLUTION)
        for l in self.thetagridlines:
            l.set_ydata(r)

    def set_rgrids(self, radii, labels=None, angle=22.5, **kwargs):
        """
        set the radial locations and labels of the r grids

        The labels will appear at radial distances radii at angle

        labels, if not None, is a len(radii) list of strings of the
        labels to use at each angle.

        if labels is None, the self.rformatter will be used

        Return value is a list of lines, labels where the lines are
        matplotlib.Line2D instances and the labels are matplotlib.Text
        instances

        ACCEPTS: sequence of floats
        """

        popall(self.rgridlines)
        theta = linspace(0,2*math.pi, self.RESOLUTION)
        ls = rcParams['grid.linestyle']
        color = rcParams['grid.color']
        lw = rcParams['grid.linewidth']

        for r in radii:
            r = ones(self.RESOLUTION)*r
            line = Line2D(theta, r, linestyle=ls, color=color, linewidth=lw)
            line.set_transform(self.transData)
            self.rgridlines.append(line)


        popall(self.rgridlabels)


        color = rcParams['xtick.color']


        props=FontProperties(size=rcParams['xtick.labelsize'])
        if labels is None:
            labels = [self.rformatter(r,0) for r in radii]
        for r,l in zip(radii, labels):
            t = Text(angle/180.*math.pi, r, l,
                     fontproperties=props, color=color,
                     horizontalalignment='center', verticalalignment='center')
            t.set_transform(self.transData)
            t.update(kwargs)
            self._set_artist_props(t)
            t.set_clip_on(False)
            self.rgridlabels.append(t)

        return self.rgridlines, self.rgridlabels


    def set_thetagrids(self, angles, labels=None, fmt='%d', frac = 1.1,
                       **kwargs):
        """
        set the angles at which to place the theta grids (these
        gridlines are equal along the theta dimension).  angles is in
        degrees

        labels, if not None, is a len(angles) list of strings of the
        labels to use at each angle.

        if labels is None, the labels with be fmt%angle

        frac is the fraction of the polar axes radius at which to
        place the label (1 is the edge).Eg 1.05 isd outside the axes
        and 0.95 is inside the axes

        kwargs are optional text properties for the labels

        Return value is a list of lines, labels where the lines are
        matplotlib.Line2D instances and the labels are matplotlib.Text
        instances

        ACCEPTS: sequence of floats
        """
        popall(self.thetagridlines)
        ox, oy = 0,0
        ls = rcParams['grid.linestyle']
        color = rcParams['grid.color']
        lw = rcParams['grid.linewidth']

        r = linspace(0, self.get_rmax(), self.RESOLUTION)
        for a in angles:
            theta = ones(self.RESOLUTION)*a/180.*math.pi
            line = Line2D(theta, r, linestyle=ls, color=color, linewidth=lw)
            line.set_transform(self.transData)
            self.thetagridlines.append(line)

        popall(self.thetagridlabels)

        color = rcParams['xtick.color']

        props=FontProperties(size=rcParams['xtick.labelsize'])
        r = frac*self.get_rmax()
        if labels is None:
            labels = [fmt%a for a in angles]
        for a,l in zip(angles, labels):
            t = Text(a/180.*math.pi, r, l, fontproperties=props, color=color,
                     horizontalalignment='center', verticalalignment='center')
            t.set_transform(self.transData)
            t.update(kwargs)
            self._set_artist_props(t)
            t.set_clip_on(False)
            self.thetagridlabels.append(t)
        return self.thetagridlines, self.thetagridlabels

    def get_rmax(self):
        'get the maximum radius in the view limits dimension'
        vmin, vmax = self.dataLim.intervaly().get_bounds()
        return max(vmin, vmax)

    def draw(self, renderer):
        if not self.get_visible(): return
        renderer.open_group('polar_axes')
        self.apply_aspect(1)
        self.transData.freeze()  # eval the lazy objects
        self.transAxes.freeze()  # eval the lazy objects
        #self._update_axes()
        if self.axison:
            if self._frameon: self.axesPatch.draw(renderer)

        if self._gridOn:
            for l in self.rgridlines+self.thetagridlines:
                l.draw(renderer)

        for t in self.thetagridlabels+self.rgridlabels:
            t.draw(renderer)

        artists = []
        artists.extend(self.lines)
        artists.extend(self.texts)
        artists.extend(self.collections)
        artists.extend(self.patches)
        dsu = [ (a.zorder, a) for a in artists]
        dsu.sort()

        for zorder, a in dsu:
            a.draw(renderer)

        self.title.draw(renderer)
        self.transData.thaw()  # release the lazy objects
        self.transAxes.thaw()  # release the lazy objects
        renderer.close_group('polar_axes')


    def format_coord(self, theta, r):
        'return a format string formatting the coordinate'
        theta /= math.pi
        return 'theta=%1.2fpi, r=%1.3f'%(theta, r)


    def has_data(self):
        'return true if any artists have been added to axes'
        return len(self.lines)+len(self.collections)

    def set_xlabel(self, xlabel, fontdict=None, **kwargs):
        'xlabel not implemented'
        raise NotImplementedError('xlabel not defined for polar axes (yet)')

    def set_ylabel(self, ylabel, fontdict=None, **kwargs):
        'ylabel not implemented'
        raise NotImplementedError('ylabel not defined for polar axes (yet)')


    def set_xlim(self, v, emit=True):
        """
        SET_XLIM(v, emit=True)

        A do nothing impl until we can figure out how to handle interaction
        ACCEPTS: len(2) sequence of floats
        """
        #warnings.warn('Navigation set_ylim not implemented for polar')
        self.viewLim.intervalx().set_bounds(*v)
        if emit: self._send_xlim_event()


    def set_ylim(self, v, emit=True):
        """
        SET_YLIM(v, emit=True)

        ACCEPTS: len(2) sequence of floats
        """
        #warnings.warn('Navigation set_xlim not implemented for polar')
        self.viewLim.intervaly().set_bounds(*v)
        if emit: self._send_ylim_event()

    def get_xscale(self):
        'return the xaxis scale string'
        return 'polar'

    def get_yscale(self):
        'return the yaxis scale string'
        return 'polar'

    def toggle_log_lineary(self):
        'toggle between log and linear axes ignored for polar'
        pass

    def legend(self, *args, **kwargs):
        """
        LEGEND(*args, **kwargs)
        Not implemented for polar yet -- use figlegend
        """
        raise NotImplementedError('legend not implemented for polar yet -- use figlegend')

    def table(self, *args, **kwargs):
        """
        TABLE(*args, **kwargs)
        Not implemented for polar axes
        """
        raise NotImplementedError('table not implemented for polar axes')


class PolarSubplot(SubplotBase, PolarAxes):
    """
    Create a polar subplot with

      PolarSubplot(numRows, numCols, plotNum)

    where plotNum=1 is the first plot number and increasing plotNums
    fill rows first.  max(plotNum)==numRows*numCols

    You can leave out the commas if numRows<=numCols<=plotNum<10, as
    in

      Subplot(211)    # 2 rows, 1 column, first (upper) plot
    """
    def __init__(self, fig, *args, **kwargs):
        SubplotBase.__init__(self, fig, *args)
        PolarAxes.__init__(self, fig, [self.figLeft, self.figBottom, self.figW, self.figH], **kwargs)


"""
# this is some discarded code I was using to find the minimum positive
# data point for some log scaling fixes.  I realized there was a
# cleaner way to do it, but am keeping this around as an example for
# how to get the data out of the axes.  Might want to make something
# like this a method one day, or better yet make get_verts and Artist
# method

            minx, maxx = self.get_xlim()
            if minx<=0 or maxx<=0:
                # find the min pos value in the data
                xs = []
                for line in self.lines:
                    xs.extend(line.get_xdata(valid_only = True))
                for patch in self.patches:
                    xs.extend([x for x,y in patch.get_verts()])
                for collection in self.collections:
                    xs.extend([x for x,y in collection.get_verts()])
                posx = [x for x in xs if x>0]
                if len(posx):

                    minx = min(posx)
                    maxx = max(posx)
                    # warning, probably breaks inverted axis
                    self.set_xlim((0.1*minx, maxx))
"""
