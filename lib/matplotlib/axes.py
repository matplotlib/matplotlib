from __future__ import division, generators
import math, sys, warnings

import numpy as npy

import matplotlib.numerix.npyma as ma

import matplotlib
rcParams = matplotlib.rcParams

from matplotlib import artist as martist
from matplotlib import agg
from matplotlib import axis as maxis
from matplotlib import cbook
from matplotlib import collections as mcoll
from matplotlib import colors as mcolors
from matplotlib import contour as mcontour
from matplotlib import dates as mdates
from matplotlib import font_manager
from matplotlib import image as mimage
from matplotlib import legend as mlegend
from matplotlib import lines as mlines
from matplotlib import mlab
from matplotlib import cm
from matplotlib import patches as mpatches
from matplotlib import quiver as mquiver
from matplotlib import table as mtable
from matplotlib import text as mtext
from matplotlib import ticker as mticker
from matplotlib import transforms as mtrans

iterable = cbook.iterable
is_string_like = cbook.is_string_like



def delete_masked_points(*args):
    """
    Find all masked points in a set of arguments, and return
    the arguments with only the unmasked points remaining.

    The overall mask is calculated from any masks that are present.
    If a mask is found, any argument that does not have the same
    dimensions is left unchanged; therefore the argument list may
    include arguments that can take string or array values, for
    example.

    Array arguments must have the same length; masked arguments must
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
        if (not is_string_like(x)
            and iterable(x)
            and len(x) == len(mask)):
            if (hasattr(x, 'get_compressed_copy')):
                compressed_x = x.get_compressed_copy(mask)
            else:
                compressed_x = ma.masked_array(x, mask=mask).compressed()
            margs.append(compressed_x)
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

    linestyle = None
    marker = None
    color = None

    # Is fmt just a colorspec?
    try:
        color = mcolors.colorConverter.to_rgb(fmt)
        return linestyle, marker, color     # Yes.
    except ValueError:
        pass                                # No, not just a color.

    # handle the multi char special cases and strip them from the
    # string
    if fmt.find('--')>=0:
        linestyle = '--'
        fmt = fmt.replace('--', '')
    if fmt.find('-.')>=0:
        linestyle = '-.'
        fmt = fmt.replace('-.', '')
    if fmt.find(' ')>=0:
        linestyle = 'None'
        fmt = fmt.replace(' ', '')

    chars = [c for c in fmt]

    for c in chars:
        if mlines.lineStyles.has_key(c):
            if linestyle is not None:
                raise ValueError(
                    'Illegal format string "%s"; two linestyle symbols' % fmt)
            linestyle = c
        elif mlines.lineMarkers.has_key(c):
            if marker is not None:
                raise ValueError(
                    'Illegal format string "%s"; two marker symbols' % fmt)
            marker = c
        elif mcolors.colorConverter.colors.has_key(c):
            if color is not None:
                raise ValueError(
                    'Illegal format string "%s"; two color symbols' % fmt)
            color = c
        else:
            raise ValueError(
                'Unrecognized character %c in format string' % c)

    if linestyle is None and marker is None:
        linestyle = rcParams['lines.linestyle']
    if linestyle is None:
        linestyle = 'None'
    if marker is None:
        marker = 'None'

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

    defaultColors = ['b','g','r','c','m','y','k']
    def __init__(self, axes, command='plot'):
        self.axes = axes
        self.command = command
        self._clear_color_cycle()

    def _clear_color_cycle(self):
        self.colors = _process_plot_var_args.defaultColors[:]
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

    def _get_next_cycle_color(self):
        if self.count==0:
            color = self.firstColor
        else:
            color = self.colors[int(self.count % self.Ncolors)]
        self.count += 1
        return color

    def __call__(self, *args, **kwargs):

        if self.axes.xaxis is not None and self.axes.yaxis is not None:
            xunits = kwargs.pop( 'xunits', self.axes.xaxis.units)
            yunits = kwargs.pop( 'yunits', self.axes.yaxis.units)
            if xunits!=self.axes.xaxis.units:
                self.axes.xaxis.set_units(xunits)
            if yunits!=self.axes.yaxis.units:
                self.axes.yaxis.set_units(yunits)

        ret =  self._grab_next_args(*args, **kwargs)
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

    def _xy_from_y(self, y):
        if self.axes.yaxis is not None:
            b = self.axes.yaxis.update_units(y)
            if b: return npy.arange(len(y)), y, False

        y = ma.asarray(y)
        if len(y.shape) == 1:
            y = y[:,npy.newaxis]
        nr, nc = y.shape
        x = npy.arange(nr)
        if len(x.shape) == 1:
            x = x[:,npy.newaxis]
        return x,y, True

    def _xy_from_xy(self, x, y):
        if self.axes.xaxis is not None and self.axes.yaxis is not None:
            bx = self.axes.xaxis.update_units(x)
            by = self.axes.yaxis.update_units(y)
            # right now multicol is not supported if either x or y are
            # unit enabled but this can be fixed..
            if bx or by: return x, y, False

        x = ma.asarray(x)
        y = ma.asarray(y)
        if len(x.shape) == 1:
            x = x[:,npy.newaxis]
        if len(y.shape) == 1:
            y = y[:,npy.newaxis]
        nrx, ncx = x.shape
        nry, ncy = y.shape
        assert nrx == nry, 'Dimensions of x and y are incompatible'
        if ncx == ncy:
            return x, y, True
        if ncx == 1:
            x = npy.repeat(x, ncy, axis=1)
        if ncy == 1:
            y = npy.repeat(y, ncx, axis=1)
        assert x.shape == y.shape, 'Dimensions of x and y are incompatible'
        return x, y, True


    def _plot_1_arg(self, y, **kwargs):
        assert self.command == 'plot', 'fill needs at least 2 arguments'
        ret = []

        x, y, multicol = self._xy_from_y(y)

        if multicol:
            for j in range(y.shape[1]):
                color = self._get_next_cycle_color()
                seg = mlines.Line2D(x, y[:,j],
                             color = color,
                             axes=self.axes,
                          )
                self.set_lineprops(seg, **kwargs)
                ret.append(seg)
        else:
            color = self._get_next_cycle_color()
            seg = mlines.Line2D(x, y,
                         color = color,
                         axes=self.axes,
                         )
            self.set_lineprops(seg, **kwargs)
            ret.append(seg)

        return ret

    def _plot_2_args(self, tup2, **kwargs):
        ret = []
        if is_string_like(tup2[1]):

            assert self.command == 'plot', 'fill needs at least 2 non-string arguments'
            y, fmt = tup2
            x, y, multicol = self._xy_from_y(y)

            linestyle, marker, color = _process_plot_format(fmt)

            def makeline(x, y):
                _color = color
                if _color is None:
                    _color = self._get_next_cycle_color()
                seg = mlines.Line2D(x, y,
                             color=_color,
                             linestyle=linestyle, marker=marker,
                             axes=self.axes,
                             )
                self.set_lineprops(seg, **kwargs)
                ret.append(seg)

            if multicol:
                for j in range(y.shape[1]):
                    makeline(x[:,j], y[:,j])
            else:
                makeline(x, y)

            return ret
        else:

            x, y = tup2
            x, y, multicol = self._xy_from_xy(x, y)

            def makeline(x, y):
                color = self._get_next_cycle_color()
                seg = mlines.Line2D(x, y,
                             color=color,
                             axes=self.axes,
                             )
                self.set_lineprops(seg, **kwargs)
                ret.append(seg)

            def makefill(x, y):
                facecolor = self._get_next_cycle_color()
                seg = mpatches.Polygon(zip(x, y),
                              facecolor = facecolor,
                              fill=True,
                              )
                self.set_patchprops(seg, **kwargs)
                ret.append(seg)


            if self.command == 'plot': func = makeline
            else:                      func = makefill
            if multicol:
                for j in range(y.shape[1]):
                    func(x[:,j], y[:,j])
            else:
                func(x, y)


            return ret

    def _plot_3_args(self, tup3, **kwargs):
        ret = []

        x, y, fmt = tup3
        x, y, multicol = self._xy_from_xy(x, y)

        linestyle, marker, color = _process_plot_format(fmt)

        def makeline(x, y):
            _color = color
            if _color is None:
                _color = self._get_next_cycle_color()
            seg = mlines.Line2D(x, y,
                         color=_color,
                         linestyle=linestyle, marker=marker,
                         axes=self.axes,
                         )
            self.set_lineprops(seg, **kwargs)
            ret.append(seg)

        def makefill(x, y):
            facecolor = color
            seg = mpatches.Polygon(zip(x, y),
                          facecolor = facecolor,
                          fill=True,
                          )
            self.set_patchprops(seg, **kwargs)
            ret.append(seg)

        if self.command == 'plot': func = makeline
        else:                      func = makefill

        if multicol:
            for j in range(y.shape[1]):
                func(x[:,j], y[:,j])
        else:
            func(x, y)
        return ret

    def _grab_next_args(self, *args, **kwargs):

        remaining = args
        while 1:

            if len(remaining)==0: return
            if len(remaining)==1:
                for seg in self._plot_1_arg(remaining[0], **kwargs):
                    yield seg
                remaining = []
                continue
            if len(remaining)==2:
                for seg in self._plot_2_args(remaining, **kwargs):
                    yield seg
                remaining = []
                continue
            if len(remaining)==3:
                if not is_string_like(remaining[2]):
                    raise ValueError, 'third arg must be a format string'
                for seg in self._plot_3_args(remaining, **kwargs):
                    yield seg
                remaining=[]
                continue
            if is_string_like(remaining[2]):
                for seg in self._plot_3_args(remaining[:3], **kwargs):
                    yield seg
                remaining=remaining[3:]
            else:
                for seg in self._plot_2_args(remaining[:2], **kwargs):
                    yield seg
                remaining=remaining[2:]

ValueType=type(mtrans.zero())
def makeValue(v):
    if type(v) == ValueType:
        return v
    else:
        return mtrans.Value(v)


class Axes(martist.Artist):
    """
    The Axes contains most of the figure elements: Axis, Tick, Line2D,
    Text, Polygon etc, and sets the coordinate system

    The Axes instance supports callbacks through a callbacks attribute
    which is a cbook.CallbackRegistry instance.  The events you can
    connect to are 'xlim_changed' and 'ylim_changed' and the callback
    will be called with func(ax) where ax is the Axes instance


    """

    scaled = {mtrans.IDENTITY : 'linear',
              mtrans.LOG10 : 'log',
              }

    def __str__(self):
        return "Axes(%g,%g;%gx%g)"%(self._position[0].get(),self._position[1].get(),
                                    self._position[2].get(),self._position[3].get())
    def __init__(self, fig, rect,
                 axisbg = None, # defaults to rc axes.facecolor
                 frameon = True,
                 sharex=None, # use Axes instance's xaxis info
                 sharey=None, # use Axes instance's yaxis info
                 label='',
                 **kwargs
                 ):
        """

        Build an Axes instance in Figure with
        rect=[left, bottom, width,height in Figure coords

        adjustable: ['box' | 'datalim']
        alpha: the alpha transparency
        anchor: ['C', 'SW', 'S', 'SE', 'E', 'NE', 'N', 'NW', 'W']
        aspect: ['auto' | 'equal' | aspect_ratio]
        autoscale_on: boolean - whether or not to autoscale the viewlim
        axis_bgcolor: any matplotlib color - see help(colors)
        axisbelow: draw the grids and ticks below the other artists
        cursor_props: a (float, color) tuple
        figure: a Figure instance
        frame_on: a boolean - draw the axes frame
        label: the axes label
        navigate: True|False
        navigate_mode: the navigation toolbar button status: 'PAN', 'ZOOM', or None
        position: [left, bottom, width,height in Figure coords
        sharex : an Axes instance to share the x-axis with
        sharey : an Axes instance to share the y-axis with
        title: the title string
        visible: a boolean - whether the axes is visible
        xlabel: the xlabel
        xlim: (xmin, xmax) view limits
        xscale: ['log' | 'linear' ]
        xticklabels: sequence of strings
        xticks: sequence of floats
        ylabel: the ylabel strings
        ylim: (ymin, ymax) view limits
        yscale: ['log' | 'linear']
        yticklabels: sequence of strings
        yticks: sequence of floats

        """
        martist.Artist.__init__(self)
        self._position = map(makeValue, rect)
        self._originalPosition = rect
        self.set_axes(self)
        self.set_aspect('auto')
        self.set_adjustable('box')
        self.set_anchor('C')

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


        self.set_cursor_props((1,'k')) # set the cursor properties for axes

        self._cachedRenderer = None
        self.set_navigate(True)
        self.set_navigate_mode(None)

        if len(kwargs): martist.setp(self, **kwargs)

        if self.xaxis is not None:
            self._xcid = self.xaxis.callbacks.connect('units finalize', self.relim)

        if self.yaxis is not None:
            self._ycid = self.yaxis.callbacks.connect('units finalize', self.relim)


    def get_window_extent(self, *args, **kwargs):
        'get the axes bounding box in display space; args and kwargs are empty'
        return self.bbox

    def _init_axis(self):
        "move this out of __init__ because non-separable axes don't use it"
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)


    def sharex_foreign(self, axforeign):
        """
        You can share your x-axis view limits with another Axes in the
        same Figure by using the sharex and sharey property of the
        Axes.  But this doesn't work for Axes in a different figure.
        This function sets of the callbacks so that when the xaxis of
        this Axes or the Axes in a foreign figure are changed, both
        will be synchronized.

        The connection ids for the self.callbacks and
        axforeign.callbacks cbook.CallbackRegistry instances are
        returned in case you want to disconnect the coupling
        """

        def follow_foreign_xlim(ax):
            xmin, xmax = axforeign.get_xlim()
            # do not emit here or we'll get a ping png effect
            self.set_xlim(xmin, xmax, emit=False)
            self.figure.canvas.draw_idle()

        def follow_self_xlim(ax):
            xmin, xmax = self.get_xlim()
            # do not emit here or we'll get a ping png effect
            axforeign.set_xlim(xmin, xmax, emit=False)
            axforeign.figure.canvas.draw_idle()


        cidForeign = axforeign.callbacks.connect('xlim_changed', follow_foreign_xlim)
        cidSelf = self.callbacks.connect('xlim_changed', follow_self_xlim)
        return cidSelf, cidForeign


    def sharey_foreign(self, axforeign):
        """
        You can share your y-axis view limits with another Axes in the
        same Figure by using the sharey and sharey property of the
        Axes.  But this doesn't work for Axes in a different figure.
        This function sets of the callbacks so that when the yaxis of
        this Axes or the Axes in a foreign figure are changed, both
        will be synchronized.

        The connection ids for the self.callbacks and
        axforeign.callbacks cbook.CallbackRegistry instances are
        returned in case you want to disconnect the coupling
        """

        def follow_foreign_ylim(ax):
            ymin, ymax = axforeign.get_ylim()
            # do not emit here or we'll get a ping png effect
            self.set_ylim(ymin, ymax, emit=False)
            self.figure.canvas.draw_idle()

        def follow_self_ylim(ax):
            ymin, ymax = self.get_ylim()
            # do not emit here or we'll get a ping png effect
            axforeign.set_ylim(ymin, ymax, emit=False)
            axforeign.figure.canvas.draw_idle()


        cidForeign = axforeign.callbacks.connect('ylim_changed', follow_foreign_ylim)
        cidSelf = self.callbacks.connect('ylim_changed', follow_self_ylim)
        return cidSelf, cidForeign

    def set_figure(self, fig):
        """
        Set the Axes figure

        ACCEPTS: a Figure instance
        """
        martist.Artist.set_figure(self, fig)

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


        Bbox = mtrans.Bbox
        Point = mtrans.Point
        self.bbox = Bbox(
            Point(self.left, self.bottom),
            Point(self.right, self.top ),
            )
        #these will be updated later as data is added
        self._set_lim_and_transforms()

    def _set_lim_and_transforms(self):
        """
        set the dataLim and viewLim BBox attributes and the
        transData and transAxes Transformation attributes
        """


        one = mtrans.one
        zero = mtrans.zero
        Point = mtrans.Point
        Bbox = mtrans.Bbox
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
        self.dataLim = mtrans.unit_bbox()

        self.transData = mtrans.get_bbox_transform(
            self.viewLim, self.bbox)
        self.transAxes = mtrans.get_bbox_transform(
            mtrans.unit_bbox(), self.bbox)

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
        self.set_xscale('linear')
        self.set_yscale('linear')

        self.dataLim.ignore(1)
        self.callbacks = cbook.CallbackRegistry(('xlim_changed', 'ylim_changed'))

        if self._sharex is not None:
            self.xaxis.major = self._sharex.xaxis.major
            self.xaxis.minor = self._sharex.xaxis.minor
        if self._sharey is not None:
            self.yaxis.major = self._sharey.yaxis.major
            self.yaxis.minor = self._sharey.yaxis.minor

        self._get_lines = _process_plot_var_args(self)
        self._get_patches_for_fill = _process_plot_var_args(self, 'fill')

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
        props = font_manager.FontProperties(size=rcParams['axes.titlesize'])
        self.title =  mtext.Text(
            x=0.5, y=1.02, text='',
            fontproperties=props,
            verticalalignment='bottom',
            horizontalalignment='center',
            )
        self.title.set_transform(self.transAxes)
        self.title.set_clip_box(None)

        self._set_artist_props(self.title)

        self.axesPatch = mpatches.Rectangle(
            xy=(0,0), width=1, height=1,
            facecolor=self._axisbg,
            edgecolor=rcParams['axes.edgecolor'],
            )
        self.axesPatch.set_figure(self.figure)
        self.axesPatch.set_transform(self.transAxes)
        self.axesPatch.set_linewidth(rcParams['axes.linewidth'])
        self.axesFrame = mlines.Line2D((0,1,1,0,0), (0,0,1,1,0),
                            linewidth=rcParams['axes.linewidth'],
                            color=rcParams['axes.edgecolor'],
                            figure=self.figure)
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
        if anchor in mtrans.PBox.coefs.keys() or len(anchor) == 2:
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
        xmin,xmax = self.get_xbound()
        xsize = max(math.fabs(xmax-xmin), 1e-30)
        ymin,ymax = self.get_ybound()
        ysize = max(math.fabs(ymax-ymin), 1e-30)
        if self._adjustable == 'box':
            if data_ratio is None:
                data_ratio = ysize/xsize
            box_aspect = A * data_ratio
            pb = mtrans.PBox(self._originalPosition)
            pb1 = pb.shrink_to_aspect(box_aspect, fig_aspect)
            self.set_position(pb1.anchor(self._anchor), 'active')
            return


        l,b,w,h = self.get_position(original=True)
        box_aspect = fig_aspect * (h/w)
        data_ratio = box_aspect / A

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
                adjy = ((Ymarg > 0 and y_expander < 0)
                        or (Xmarg < 0 and y_expander > 0))
            else:
                adjy = y_expander > 0
            #print 'y_expander, adjy', y_expander, adjy
            adjust_y = changey or adjy  #(Ymarg > xmarg)
        if adjust_y:
            yc = 0.5*(ymin+ymax)
            y0 = yc - Ysize/2.0
            y1 = yc + Ysize/2.0
            self.set_ybound((y0, y1))
            #print 'New y0, y1:', y0, y1
            #print 'New ysize, ysize/xsize', y1-y0, (y1-y0)/xsize
        else:
            xc = 0.5*(xmin+xmax)
            x0 = xc - Xsize/2.0
            x1 = xc + Xsize/2.0
            self.set_xbound((x0, x1))
            #print 'New x0, x1:', x0, x1
            #print 'New xsize, ysize/xsize', x1-x0, ysize/(x1-x0)



    def axis(self, *v, **kwargs):
        '''
        Convenience method for manipulating the x and y view limits
        and the aspect ratio of the plot.

        kwargs are passed on to set_xlim and set_ylim -- see their
        docstrings for details
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
                    self.set_autoscale_on(False) # Req. by Mark Bakker
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
            emit = kwargs.get('emit', False)
            xmin = kwargs.get('xmin', None)
            xmax = kwargs.get('xmax', None)

            xmin, xmax = self.set_xlim(xmin, xmax, emit)
            ymin = kwargs.get('ymin', None)
            ymax = kwargs.get('ymax', None)
            ymin, ymax = self.set_ylim(ymin, ymax, emit)
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
        raise DeprecationWarning('Use get_children instead')

    def get_frame(self):
        'Return the axes Rectangle frame'
        return self.axesPatch

    def get_legend(self):
        'Return the legend.Legend instance, or None if no legend is defined'
        return self.legend_

    def get_images(self):
        'return a list of Axes images contained by the Axes'
        return cbook.silent_list('AxesImage', self.images)

    def get_lines(self):
        'Return a list of lines contained by the Axes'
        return cbook.silent_list('Line2D', self.lines)

    def get_xaxis(self):
        'Return the XAxis instance'
        return self.xaxis

    def get_xgridlines(self):
        'Get the x grid lines as a list of Line2D instances'
        return cbook.silent_list('Line2D xgridline', self.xaxis.get_gridlines())


    def get_xticklines(self):
        'Get the xtick lines as a list of Line2D instances'
        return cbook.silent_list('Text xtickline', self.xaxis.get_ticklines())


    def get_yaxis(self):
        'Return the YAxis instance'
        return self.yaxis

    def get_ygridlines(self):
        'Get the y grid lines as a list of Line2D instances'
        return cbook.silent_list('Line2D ygridline', self.yaxis.get_gridlines())

    def get_yticklines(self):
        'Get the ytick lines as a list of Line2D instances'
        return cbook.silent_list('Line2D ytickline', self.yaxis.get_ticklines())

    #### Adding and tracking artists

    def has_data(self):
        '''Return true if any artists have been added to axes.

        This should not be used to determine whether the dataLim
        need to be updated, and may not actually be useful for
        anything.
        '''
        return (
            len(self.collections) +
            len(self.images) +
            len(self.lines) +
            len(self.patches))>0

    def add_artist(self, a):
        'Add any artist to the axes'
        a.set_axes(self)
        self.artists.append(a)
        self._set_artist_props(a)
        a._remove_method = lambda h: self.artists.remove(h)

    def add_collection(self, collection, autolim=False):
        'add a Collection instance to Axes'
        label = collection.get_label()
        if not label:
            collection.set_label('collection%d'%len(self.collections))
        self.collections.append(collection)
        self._set_artist_props(collection)
        collection.set_clip_box(self.bbox)
        if autolim:
            self.update_datalim(collection.get_verts(self.transData))
        collection._remove_method = lambda h: self.collections.remove(h)

    def add_line(self, line):
        'Add a line to the list of plot lines'
        self._set_artist_props(line)
        line.set_clip_box(self.bbox)

        self._update_line_limits(line)
        if not line.get_label():
            line.set_label('_line%d'%len(self.lines))
        self.lines.append(line)
        line._remove_method = lambda h: self.lines.remove(h)

    def _update_line_limits(self, line):
        xdata = line.get_xdata(orig=False)
        ydata = line.get_ydata(orig=False)

        if line.get_transform() != self.transData:
            xys = self._get_verts_in_data_coords(
                line.get_transform(), zip(xdata, ydata))
            xdata = npy.array([x for x,y in xys])
            ydata = npy.array([y for x,y in xys])

        self.update_datalim_numerix( xdata, ydata )


    def add_patch(self, p):
        """
        Add a patch to the list of Axes patches; the clipbox will be
        set to the Axes clipping box.  If the transform is not set, it
        wil be set to self.transData.
        """

        self._set_artist_props(p)
        p.set_clip_box(self.bbox)
        self._update_patch_limits(p)
        self.patches.append(p)
        p._remove_method = lambda h: self.patches.remove(h)

    def _update_patch_limits(self, p):
        'update the datalimits for patch p'
        xys = self._get_verts_in_data_coords(
            p.get_transform(), p.get_verts())
        self.update_datalim(xys)


    def add_table(self, tab):
        'Add a table instance to the list of axes tables'
        self._set_artist_props(tab)
        self.tables.append(tab)
        tab._remove_method = lambda h: self.tables.remove(h)

    def relim(self):
        'recompute the datalimits based on current artists'
        self.dataLim.ignore(True)
        for line in self.lines:
            self._update_line_limits(line)

        for p in self.patches:
            self._update_patch_limits(p)

    def update_datalim(self, xys):
        'Update the data lim bbox with seq of xy tups or equiv. 2-D array'
        # if no data is set currently, the bbox will ignore its
        # limits and set the bound to be the bounds of the xydata.
        # Otherwise, it will compute the bounds of it's current data
        # and the data in xydata
        xys = npy.asarray(xys)


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
        xys = trans.numerix_xy(npy.asarray(xys))
        return self.transData.inverse_numerix_xy(xys)

    def _process_unit_info(self, xdata=None, ydata=None, kwargs=None):
        'look for unit kwargs and update the axis instances as necessary'

        if self.xaxis is None or self.yaxis is None: return

        #print 'processing', self.get_geometry()
        if xdata is not None:
            self.xaxis.update_units(xdata)
            #print '\tset from xdata', self.xaxis.units

        if ydata is not None:
            self.yaxis.update_units(ydata)
            #print '\tset from ydata', self.yaxis.units

        # process kwargs 2nd since these will override default units
        if kwargs is not None:
            xunits = kwargs.pop( 'xunits', self.xaxis.units)
            if xunits!=self.xaxis.units:
                #print '\tkw setting xunits', xunits
                self.xaxis.set_units(xunits)

            yunits = kwargs.pop('yunits', self.yaxis.units)
            if yunits!=self.yaxis.units:
                #print '\tkw setting yunits', yunits
                self.yaxis.set_units(yunits)

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

            if scalex: self.set_xbound(self.dataLim.intervalx().get_bounds())

            if scaley: self.set_ybound(self.dataLim.intervaly().get_bounds())
            return

        if scalex:
            xl = self.get_xbound()
            XL = self.xaxis.get_major_locator().autoscale()
            self.set_xbound(XL)
        if scaley:
            ylocator = self.yaxis.get_major_locator()
            yl = self.get_ybound()
            YL = ylocator.autoscale()
            self.set_ybound(YL)
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
        self.transData.freeze()  # eval the lazy objects
        self.transAxes.freeze()
        if self.axison and self._frameon: self.axesPatch.draw(renderer)
        artists = []



        if len(self.images)<=1 or renderer.option_image_nocomposite():
            for im in self.images:
                im.draw(renderer)
        else:
            # make a composite image blending alpha
            # list of (mimage.Image, ox, oy)


            mag = renderer.get_image_magnification()
            ims = [(im.make_image(mag),0,0)
                   for im in self.images if im.get_visible()]


            im = mimage.from_images(self.bbox.height()*mag,
                                    self.bbox.width()*mag,
                                    ims)
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

        Valid Line2D kwargs are
        %(Line2D)s
        """
        if len(kwargs): b = True
        self.xaxis.grid(b, **kwargs)
        self.yaxis.grid(b, **kwargs)
    grid.__doc__ = cbook.dedent(grid.__doc__) % martist.kwdocd

    def ticklabel_format(self, **kwargs):
        """
        Convenience method for manipulating the ScalarFormatter
        used by default for linear axes.

        kwargs:
            style = 'sci' (or 'scientific') or 'plain';
                        plain turns off scientific notation
            axis = 'x', 'y', or 'both'

        Only the major ticks are affected.
        If the method is called when the ScalarFormatter is not
        the one being used, an AttributeError will be raised with
        no additional error message.

        Additional capabilities and/or friendlier error checking may be added.

        """
        style = kwargs.pop('style', '').lower()
        axis = kwargs.pop('axis', 'both').lower()
        if style[:3] == 'sci':
            sb = True
        elif style in ['plain', 'comma']:
            sb = False
            if style == 'plain':
                cb = False
            else:
                cb = True
                raise NotImplementedError, "comma style remains to be added"
        elif style == '':
            sb = None
        else:
            raise ValueError, "%s is not a valid style value"
        if sb is not None:
            if axis == 'both' or axis == 'x':
                self.xaxis.major.formatter.set_scientific(sb)
            if axis == 'both' or axis == 'y':
                self.yaxis.major.formatter.set_scientific(sb)

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

    def invert_xaxis(self):
        "Invert the x-axis."
        left, right = self.get_xlim()
        self.set_xlim(right, left)

    def xaxis_inverted(self):
        'Returns True if the x-axis is inverted.'
        left, right = self.get_xlim()
        return right < left

    def get_xbound(self):
        "Returns the x-axis numerical bounds in the form of lowerBound < upperBound"
        left, right = self.get_xlim()
        if left < right:
           return left, right
        else:
           return right, left

    def set_xbound(self, lower=None, upper=None):
        """Set the lower and upper numerical bounds of the x-axis.
           This method will honor axes inversion regardless of parameter order.
        """
        if upper is None and iterable(lower):
            lower,upper = lower

        old_lower,old_upper = self.get_xbound()

        if lower is None: lower = old_lower
        if upper is None: upper = old_upper

        if self.xaxis_inverted():
            if lower < upper:
                self.set_xlim(upper, lower)
            else:
                self.set_xlim(lower, upper)
        else:
            if lower < upper:
                self.set_xlim(lower, upper)
            else:
                self.set_xlim(upper, lower)

    def get_xlim(self):
        'Get the x axis range [xmin, xmax]'
        return self.viewLim.intervalx().get_bounds()


    def set_xlim(self, xmin=None, xmax=None, emit=True, **kwargs):
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

        if xmax is None and iterable(xmin):
            xmin,xmax = xmin


        self._process_unit_info(xdata=(xmin, xmax))
        if xmin is not None:
            xmin = self.convert_xunits(xmin)
        if xmax is not None:
            xmax = self.convert_xunits(xmax)

        old_xmin,old_xmax = self.get_xlim()
        if xmin is None: xmin = old_xmin
        if xmax is None: xmax = old_xmax

        if (self.transData.get_funcx().get_type()==mtrans.LOG10
            and min(xmin, xmax)<=0):
            raise ValueError('Cannot set nonpositive limits with log transform')

        xmin, xmax = mtrans.nonsingular(xmin, xmax, increasing=False)
        self.viewLim.intervalx().set_bounds(xmin, xmax)
        if emit: self.callbacks.process('xlim_changed', self)

        return xmin, xmax

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
            self.xaxis.set_major_locator(mticker.LogLocator(basex))
            self.xaxis.set_major_formatter(mticker.LogFormatterMathtext(basex))
            self.xaxis.set_minor_locator(mticker.LogLocator(basex,subsx))
            self.transData.get_funcx().set_type(mtrans.LOG10)
            minx, maxx = self.get_xlim()
            if min(minx, maxx)<=0:
                self.autoscale_view()
        elif value == 'linear':
            self.xaxis.set_major_locator(mticker.AutoLocator())
            self.xaxis.set_major_formatter(mticker.ScalarFormatter())
            self.xaxis.set_minor_locator(mticker.NullLocator())
            self.xaxis.set_minor_formatter(mticker.NullFormatter())
            self.transData.get_funcx().set_type( mtrans.IDENTITY )

    def get_xticks(self, minor=False):
        'Return the x ticks as a list of locations'
        return self.xaxis.get_ticklocs(minor=minor)

    def set_xticks(self, ticks, minor=False):
        """
        Set the x ticks with list of ticks

        ACCEPTS: sequence of floats
        """
        return self.xaxis.set_ticks(ticks, minor=minor)

    def get_xmajorticklabels(self):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text xticklabel', self.xaxis.get_majorticklabels())

    def get_xminorticklabels(self):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text xticklabel', self.xaxis.get_minorticklabels())

    def get_xticklabels(self, minor=False):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text xticklabel', self.xaxis.get_ticklabels(minor=minor))

    def set_xticklabels(self, labels, fontdict=None, minor=False, **kwargs):
        """
        set_xticklabels(labels, fontdict=None, minor=False, **kwargs)

        Set the xtick labels with list of strings labels Return a list of axis
        text instances.

        kwargs set the Text properties.  Valid properties are
        %(Text)s

        ACCEPTS: sequence of strings
        """
        return self.xaxis.set_ticklabels(labels, fontdict, minor=minor, **kwargs)
    set_xticklabels.__doc__ = cbook.dedent(set_xticklabels.__doc__) % martist.kwdocd

    def invert_yaxis(self):
        "Invert the y-axis."
        left, right = self.get_ylim()
        self.set_ylim(right, left)

    def yaxis_inverted(self):
        'Returns True if the y-axis is inverted.'
        left, right = self.get_ylim()
        return right < left

    def get_ybound(self):
        "Returns the y-axis numerical bounds in the form of lowerBound < upperBound"
        left, right = self.get_ylim()
        if left < right:
           return left, right
        else:
           return right, left

    def set_ybound(self, lower=None, upper=None):
        """Set the lower and upper numerical bounds of the y-axis.
           This method will honor axes inversion regardless of parameter order.
        """
        if upper is None and iterable(lower):
            lower,upper = lower

        old_lower,old_upper = self.get_ybound()

        if lower is None: lower = old_lower
        if upper is None: upper = old_upper

        if self.yaxis_inverted():
            if lower < upper:
                self.set_ylim(upper, lower)
            else:
                self.set_ylim(lower, upper)
        else:
            if lower < upper:
                self.set_ylim(lower, upper)
            else:
                self.set_ylim(upper, lower)

    def get_ylim(self):
        'Get the y axis range [ymin, ymax]'
        return self.viewLim.intervaly().get_bounds()

    def set_ylim(self, ymin=None, ymax=None, emit=True, **kwargs):
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


        if ymax is None and iterable(ymin):
            ymin,ymax = ymin

        if ymin is not None:
            ymin = self.convert_yunits(ymin)
        if ymax is not None:
            ymax = self.convert_yunits(ymax)

        old_ymin,old_ymax = self.get_ylim()

        if ymin is None: ymin = old_ymin
        if ymax is None: ymax = old_ymax

        if (self.transData.get_funcy().get_type()==mtrans.LOG10
            and min(ymin, ymax)<=0):
            raise ValueError('Cannot set nonpositive limits with log transform')

        ymin, ymax = mtrans.nonsingular(ymin, ymax, increasing=False)
        self.viewLim.intervaly().set_bounds(ymin, ymax)
        if emit: self.callbacks.process('ylim_changed', self)

        return ymin, ymax

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
            self.yaxis.set_major_locator(mticker.LogLocator(basey))
            self.yaxis.set_major_formatter(mticker.LogFormatterMathtext(basey))
            self.yaxis.set_minor_locator(mticker.LogLocator(basey,subsy))
            self.transData.get_funcy().set_type(mtrans.LOG10)
            miny, maxy = self.get_ylim()
            if min(miny, maxy)<=0:
                self.autoscale_view()

        elif value == 'linear':
            self.yaxis.set_major_locator(mticker.AutoLocator())
            self.yaxis.set_major_formatter(mticker.ScalarFormatter())
            self.yaxis.set_minor_locator(mticker.NullLocator())
            self.yaxis.set_minor_formatter(mticker.NullFormatter())
            self.transData.get_funcy().set_type( mtrans.IDENTITY )

    def get_yticks(self, minor=False):
        'Return the y ticks as a list of locations'
        return self.yaxis.get_ticklocs(minor=minor)

    def set_yticks(self, ticks, minor=False):
        """
        Set the y ticks with list of ticks

        ACCEPTS: sequence of floats
        """
        return self.yaxis.set_ticks(ticks, minor=minor)

    def get_ymajorticklabels(self):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text yticklabel', self.yaxis.get_majorticklabels())

    def get_yminorticklabels(self):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text yticklabel', self.yaxis.get_minorticklabels())

    def get_yticklabels(self, minor=False):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text yticklabel', self.yaxis.get_ticklabels(minor=minor))

    def set_yticklabels(self, labels, fontdict=None, minor=False, **kwargs):
        """
        set_yticklabels(labels, fontdict=None, minor=False, **kwargs)

        Set the ytick labels with list of strings labels.  Return a list of
        Text instances.

        kwargs set Text properties for the labels.  Valid properties are
        %(Text)s

        ACCEPTS: sequence of strings
        """
        return self.yaxis.set_ticklabels(labels, fontdict, minor=minor, **kwargs)
    set_yticklabels.__doc__ = cbook.dedent(set_yticklabels.__doc__) % martist.kwdocd

    def toggle_log_lineary(self):
        'toggle between log and linear on the y axis'
        funcy = self.transData.get_funcy().get_type()
        if funcy==mtrans.LOG10: self.set_yscale('linear')
        elif funcy==mtrans.IDENTITY: self.set_yscale('log')

    def xaxis_date(self, tz=None):
        """Sets up x-axis ticks and labels that treat the x data as dates.

        tz is the time zone to use in labeling dates.  Defaults to rc value.
        """

        locator = self.xaxis.get_major_locator()
        if not isinstance(locator, mdates.DateLocator):
            locator = mdates.AutoDateLocator(tz)
            self.xaxis.set_major_locator(locator)

        formatter = self.xaxis.get_major_formatter()
        if not isinstance(formatter, mdates.DateFormatter):
            formatter = mdates.AutoDateFormatter(locator)
            self.xaxis.set_major_formatter(formatter)

    def yaxis_date(self, tz=None):
        """Sets up y-axis ticks and labels that treat the y data as dates.

        tz is the time zone to use in labeling dates.  Defaults to rc value.
        """

        locator = self.yaxis.get_major_locator()
        if not isinstance(locator, mdates.DateLocator):
            locator = mdates.AutoDateLocator(tz)
            self.yaxis.set_major_locator(locator)

        formatter = self.xaxis.get_major_formatter()
        if not isinstance(formatter, mdates.DateFormatter):
            formatter = mdates.AutoDateFormatter(locator)
            self.yaxis.set_major_formatter(formatter)

    def format_xdata(self, x):
        """
        Return x string formatted.  This function will use the attribute
        self.fmt_xdata if it is callable, else will fall back on the xaxis
        major formatter
        """
        try: return self.fmt_xdata(x)
        except TypeError:
            func = self.xaxis.get_major_formatter().format_data_short
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
            func = self.yaxis.get_major_formatter().format_data_short
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
        c =mcolors.colorConverter.to_rgba(c)
        self._cursorProps = lw, c

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
        raise DeprecationWarning('use the callbacks CallbackRegistry instance instead')

    def disconnect(self, cid):
        'disconnect from the Axes event.'
        raise DeprecationWarning('use the callbacks CallbackRegistry instance instead')

    def get_children(self):
        'return a list of child artists'
        children = []
        children.append(self.xaxis)
        children.append(self.yaxis)
        children.extend(self.lines)
        children.extend(self.patches)
        children.extend(self.texts)
        children.extend(self.tables)
        children.extend(self.artists)
        children.extend(self.images)
        if self.legend_ is not None:
            children.append(self.legend_)
        children.extend(self.collections)
        children.append(self.title)
        children.append(self.axesPatch)
        children.append(self.axesFrame)
        return children

    def contains(self,mouseevent):
        """Test whether the mouse event occured in the axes.

        Returns T/F, {}
        """
        if callable(self._contains): return self._contains(self,mouseevent)

        inside = self.bbox.contains(mouseevent.x,mouseevent.y)
        return inside,{}

    def pick(self,*args):
        """
        pick(mouseevent)

        each child artist will fire a pick event if mouseevent is over
        the artist and the artist has picker set
        """
        if len(args)>1:
            raise DeprecationWarning(
                'New pick API implemented -- see API_CHANGES in the src distribution')
        martist.Artist.pick(self,args[0])

    def __pick(self, x, y, trans=None, among=None):
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
            return min(npy.sqrt((x-x1)**2+(y-y1)**2))

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
            elif isinstance(a, mlines.Line2D):
                xdata = a.get_xdata(orig=False)
                ydata = a.get_ydata(orig=False)
                xt, yt = a.get_transform().numerix_x_y(xdata, ydata)

            return dist_x_y(xywin, npy.asarray(xt), npy.asarray(yt))

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

        kwargs are Text properties:
        %(Text)s

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
    set_title.__doc__ = cbook.dedent(set_title.__doc__) % martist.kwdocd

    def set_xlabel(self, xlabel, fontdict=None, **kwargs):
        """
        SET_XLABEL(xlabel, fontdict=None, **kwargs)

        Set the label for the xaxis.  See the text docstring for information
        of how override and the optional args work.

        Valid kwargs are Text properties:
        %(Text)s
        ACCEPTS: str
        """

        label = self.xaxis.get_label()
        label.set_text(xlabel)
        if fontdict is not None: label.update(fontdict)
        label.update(kwargs)
        return label
    set_xlabel.__doc__ = cbook.dedent(set_xlabel.__doc__) % martist.kwdocd

    def set_ylabel(self, ylabel, fontdict=None, **kwargs):
        """
        SET_YLABEL(ylabel, fontdict=None, **kwargs)

        Set the label for the yaxis

        See the text doctstring for information of how override and
        the optional args work

        Valid kwargs are Text properties:
        %(Text)s
        ACCEPTS: str
        """
        label = self.yaxis.get_label()
        label.set_text(ylabel)
        if fontdict is not None: label.update(fontdict)
        label.update(kwargs)
        return label
    set_ylabel.__doc__ = cbook.dedent(set_ylabel.__doc__) % martist.kwdocd

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

        Individual keyword arguments can be used to override any given
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


       You can put a rectangular box around the text instance (eg to
       set a background color) by using the keyword bbox.  bbox is a
       dictionary of patches.Rectangle properties (see help
       for Rectangle for a list of these).  For example

         text(x, y, s, bbox=dict(facecolor='red', alpha=0.5))

       Valid kwargs are Text properties
       %(Text)s
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
            t = mtext.TextWithDash(
                x=x, y=y, text=s,
                )
        else:
            t = mtext.Text(
                x=x, y=y, text=s,
                )
        self._set_artist_props(t)

        t.update(default)
        if fontdict is not None: t.update(fontdict)
        t.update(kwargs)
        self.texts.append(t)
        t._remove_method = lambda h: self.texts.remove(h)


        #if t.get_clip_on():  t.set_clip_box(self.bbox)
        if kwargs.has_key('clip_on'):  t.set_clip_box(self.bbox)
        return t
    text.__doc__ = cbook.dedent(text.__doc__) % martist.kwdocd

    def annotate(self, *args, **kwargs):
        """
        annotate(s, xy,
                 xytext=None,
                 xycoords='data',
                 textcoords='data',
                 arrowprops=None,
                 **props)

        %(Annotation)s
        """
        a = mtext.Annotation(*args, **kwargs)
        a.set_transform(mtrans.identity_transform())
        self._set_artist_props(a)
        if kwargs.has_key('clip_on'):  a.set_clip_box(self.bbox)
        self.texts.append(a)
        return a
    annotate.__doc__ = cbook.dedent(annotate.__doc__) % martist.kwdocd

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

        Valid kwargs are Line2D properties
        %(Line2D)s
        """

        trans = mtrans.blend_xy_sep_transform( self.transAxes, self.transData)
        l, = self.plot([xmin,xmax], [y,y], transform=trans, scalex=False, **kwargs)
        return l

    axhline.__doc__ = cbook.dedent(axhline.__doc__) % martist.kwdocd

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

        Valid kwargs are Line2D properties
        %(Line2D)s
        """

        trans = mtrans.blend_xy_sep_transform( self.transData, self.transAxes )
        l, = self.plot([x,x], [ymin,ymax] , transform=trans, scaley=False, **kwargs)
        return l

    axvline.__doc__ = cbook.dedent(axvline.__doc__) % martist.kwdocd

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

        Valid kwargs are Polygon properties
        %(Polygon)s
        """
        # convert y axis units
        trans = mtrans.blend_xy_sep_transform( self.transAxes, self.transData)
        verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        self.add_patch(p)
        return p
    axhspan.__doc__ = cbook.dedent(axhspan.__doc__) % martist.kwdocd

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

        Valid kwargs are Polygon properties
        %(Polygon)s
        """
        # convert x axis units
        trans = mtrans.blend_xy_sep_transform(self.transData, self.transAxes)
        verts = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        self.add_patch(p)
        return p
    axvspan.__doc__ = cbook.dedent(axvspan.__doc__) % martist.kwdocd


    def hlines(self, y, xmin, xmax, colors='k', linestyle='solid',
                     label='', **kwargs):
        """
        HLINES(y, xmin, xmax, colors='k', linestyle='solid', **kwargs)

        plot horizontal lines at each y from xmin to xmax.  xmin or xmax can
        be scalars or len(x) numpy arrays.  If they are scalars, then the
        respective values are constant, else the widths of the lines are
        determined by xmin and xmax

        colors is a line collections color args, either a single color
        or a len(x) list of colors

        linestyle is one of solid|dashed|dashdot|dotted

        Returns the LineCollection that was added
        """
        if kwargs.get('fmt') is not None:
            raise DeprecationWarning(
                'hlines now uses a collections.LineCollection and not a list of Line2D to draw; see API_CHANGES')

        if not iterable(y): y = [y]
        if not iterable(xmin): xmin = [xmin]
        if not iterable(xmax): xmax = [xmax]
        y = npy.asarray(y)

        if len(xmin)==1:
            xmin = xmin*npy.ones(y.shape, y.dtype)
        if len(xmax)==1:
            xmax = xmax*npy.ones(y.shape, y.dtype)

        xmin = npy.asarray(xmin)
        xmax = npy.asarray(xmax)

        if len(xmin)!=len(y):
            raise ValueError, 'xmin and y are unequal sized sequences'
        if len(xmax)!=len(y):
            raise ValueError, 'xmax and y are unequal sized sequences'

        verts = [ ((thisxmin, thisy), (thisxmax, thisy))
                            for thisxmin, thisxmax, thisy in zip(xmin, xmax, y)]
        coll = mcoll.LineCollection(verts, colors=colors,
                                    linestyle=linestyle, label=label)
        self.add_collection(coll)
        coll.update(kwargs)

        minx = min(xmin.min(), xmax.min())
        maxx = max(xmin.max(), xmax.max())
        miny = y.min()
        maxy = y.max()

        minx, maxx = self.convert_xunits((minx, maxx))
        miny, maxy = self.convert_yunits((miny, maxy))
        corners = (minx, miny), (maxx, maxy)

        self.update_datalim(corners)
        self.autoscale_view()


        return coll
    hlines.__doc__ = cbook.dedent(hlines.__doc__)

    def vlines(self, x, ymin, ymax, colors='k', linestyle='solid',
                     label='', **kwargs):
        """
        VLINES(x, ymin, ymax, color='k')

        Plot vertical lines at each x from ymin to ymax.  ymin or ymax can be
        scalars or len(x) numpy arrays.  If they are scalars, then the
        respective values are constant, else the heights of the lines are
        determined by ymin and ymax


        colors is a line collections color args, either a single color
        or a len(x) list of colors

        linestyle is one of solid|dashed|dashdot|dotted

        Returns the collections.LineCollection that was added

        kwargs are collections.LineCollection properties:
        %(LineCollection)s
        """

        if kwargs.get('fmt') is not None:
            raise DeprecationWarning(
                'vlines now uses a collections.LineCollection and not a list of Line2D to draw; see API_CHANGES')

        self._process_unit_info(xdata=x, ydata=ymin, kwargs=kwargs)

        if not iterable(x): x = [x]
        if not iterable(ymin): ymin = [ymin]
        if not iterable(ymax): ymax = [ymax]
        x = npy.asarray(x)
        ymin = npy.asarray(ymin)
        ymax = npy.asarray(ymax)

        if len(ymin)==1:
            ymin = ymin*npy.ones(x.shape, x.dtype)
        if len(ymax)==1:
            ymax = ymax*npy.ones(x.shape, x.dtype)

        if len(ymin)!=len(x):
            raise ValueError, 'ymin and x are unequal sized sequences'
        if len(ymax)!=len(x):
            raise ValueError, 'ymax and x are unequal sized sequences'

        Y = npy.array([ymin, ymax]).T

        verts = [ ((thisx, thisymin), (thisx, thisymax))
                                    for thisx, (thisymin, thisymax) in zip(x,Y)]
        #print 'creating line collection'
        coll = mcoll.LineCollection(verts, colors=colors,
                              linestyle=linestyle, label=label)
        self.add_collection(coll)
        coll.update(kwargs)

        minx = x.min()
        maxx = x.max()
        miny = min(ymin.min(), ymax.min())
        maxy = max(ymin.max(), ymax.max())
        minx, maxx = self.convert_xunits((minx, maxx))
        miny, maxy = self.convert_yunits((miny, maxy))
        corners = (minx, miny), (maxx, maxy)
        self.update_datalim(corners)
        self.autoscale_view()

        return coll
    vlines.__doc__ = cbook.dedent(vlines.__doc__) % martist.kwdocd

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

        If x and/or y is 2-Dimensional, then the corresponding columns
        will be plotted.

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

        The following color abbreviations are supported

            b  : blue
            g  : green
            r  : red
            c  : cyan
            m  : magenta
            y  : yellow
            k  : black
            w  : white

        In addition, you can specify colors in many weird and
        wonderful ways, including full names 'green', hex strings
        '#008000', RGB or RGBA tuples (0,1,0,1) or grayscale
        intensities as a string '0.8'.  Of these, the string
        specifications can be used in place of a fmt group, but the
        tuple forms can be used only as kwargs.

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

        The kwargs are Line2D properties:
        %(Line2D)s

        kwargs scalex and scaley, if defined, are passed on
        to autoscale_view to determine whether the x and y axes are
        autoscaled; default True.  See Axes.autoscale_view for more
        information
        """

        scalex = kwargs.pop( 'scalex', True)
        scaley = kwargs.pop( 'scaley', True)

        if not self._hold: self.cla()
        lines = []

        for line in self._get_lines(*args, **kwargs):
            self.add_line(line)
            lines.append(line)


        self.autoscale_view(scalex=scalex, scaley=scaley)
        return lines

    plot.__doc__ = cbook.dedent(plot.__doc__) % martist.kwdocd

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

        Note if you are using custom date tickers and formatters, it
        may be necessary to set the formatters/locators after the call
        to plot_date since plot_date will set the default tick locator
        to ticker.AutoDateLocator (if the tick locator is not already set to
        a ticker.DateLocator instance) and the default tick formatter to
        AutoDateFormatter (if the tick formatter is not already set to
        a DateFormatter instance).

        Valid kwargs are Line2D properties:
        %(Line2D)s


        See dates for helper functions date2num, num2date
        and drange for help on creating the required floating point dates
        """

        if not self._hold: self.cla()

        ret = self.plot(x, y, fmt, **kwargs)

        if xdate:
            self.xaxis_date(tz)
        if ydate:
            self.yaxis_date(tz)

        self.autoscale_view()

        return ret
    plot_date.__doc__ = cbook.dedent(plot_date.__doc__) % martist.kwdocd


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

        The remaining valid kwargs are Line2D properties:
        %(Line2D)s
        """
        if not self._hold: self.cla()

        dx = {'basex': kwargs.pop('basex', 10),
              'subsx': kwargs.pop('subsx', None),
              }
        dy = {'basey': kwargs.pop('basey', 10),
              'subsy': kwargs.pop('subsy', None),
              }

        self.set_xscale('log', **dx)
        self.set_yscale('log', **dy)

        b =  self._hold
        self._hold = True # we've already processed the hold
        l = self.plot(*args, **kwargs)
        self._hold = b    # restore the hold

        return l
    loglog.__doc__ = cbook.dedent(loglog.__doc__) % martist.kwdocd

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

        The remaining valid kwargs are Line2D properties:
        %(Line2D)s
        """
        if not self._hold: self.cla()
        d = {'basex': kwargs.pop( 'basex', 10),
             'subsx': kwargs.pop( 'subsx', None),
             }

        self.set_xscale('log', **d)
        b =  self._hold
        self._hold = True # we've already processed the hold
        l = self.plot(*args, **kwargs)
        self._hold = b    # restore the hold
        return l
    semilogx.__doc__ = cbook.dedent(semilogx.__doc__) % martist.kwdocd

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

        The remaining valid kwargs are Line2D properties:
        %(Line2D)s

        """
        if not self._hold: self.cla()
        d = {'basey': kwargs.pop('basey', 10),
             'subsy': kwargs.pop('subsy', None),
             }
        self.set_yscale('log', **d)
        b =  self._hold
        self._hold = True # we've already processed the hold
        l = self.plot(*args, **kwargs)
        self._hold = b    # restore the hold

        return l
    semilogy.__doc__ = cbook.dedent(semilogy.__doc__) % martist.kwdocd

    def acorr(self, x, **kwargs):
        """
        ACORR(x, normed=False, detrend=mlab.detrend_none, usevlines=False,
              maxlags=None, **kwargs)
        Plot the autocorrelation of x.  If normed=True, normalize the
        data but the autocorrelation at 0-th lag.  x is detrended by
        the detrend callable (default no normalization.

        data are plotted as plot(lags, c, **kwargs)

        return value is lags, c, line where lags are a length
        2*maxlags+1 lag vector, c is the 2*maxlags+1 auto correlation
        vector, and line is a Line2D instance returned by plot.  The
        default linestyle is None and the default marker is 'o',
        though these can be overridden with keyword args.  The cross
        correlation is performed with numpy correlate with
        mode=2.

        If usevlines is True, Axes.vlines rather than Axes.plot is used
        to draw vertical lines from the origin to the acorr.
        Otherwise the plotstyle is determined by the kwargs, which are
        Line2D properties.  If usevlines, the return value is lags, c,
        linecol, b where linecol is the collections.LineCollection and b is the x-axis

        if usevlines=True, kwargs are passed onto Axes.vlines
        if usevlines=False, kwargs are passed onto Axes.plot

        maxlags is a positive integer detailing the number of lags to show.
        The default value of None will return all (2*len(x)-1) lags.

        See the respective function for documentation on valid kwargs
        """
        return self.xcorr(x, x, **kwargs)
    acorr.__doc__ = cbook.dedent(acorr.__doc__) % martist.kwdocd

    def xcorr(self, x, y, normed=False, detrend=mlab.detrend_none, usevlines=False,
              maxlags=None, **kwargs):
        """
        XCORR(x, y, normed=False, detrend=mlab.detrend_none, usevlines=False, **kwargs):

        Plot the cross correlation between x and y.  If normed=True,
        normalize the data but the cross correlation at 0-th lag.  x
        and y are detrended by the detrend callable (default no
        normalization.  x and y must be equal length

        data are plotted as plot(lags, c, **kwargs)

        return value is lags, c, line where lags are a length
        2*maxlags+1 lag vector, c is the 2*maxlags+1 auto correlation
        vector, and line is a Line2D instance returned by plot.  The
        default linestyle is None and the default marker is 'o',
        though these can be overridden with keyword args.  The cross
        correlation is performed with numpy correlate with
        mode=2.

        If usevlines is True, Axes.vlines rather than Axes.plot is used
        to draw vertical lines from the origin to the acorr.
        Otherwise the plotstyle is determined by the kwargs, which are
        Line2D properties.  If usevlines, the return value is lags, c,
        linecol, b where linecol is the collections.LineCollection and b is the x-axis

        if usevlines=True, kwargs are passed onto Axes.vlines
        if usevlines=False, kwargs are passed onto Axes.plot

        maxlags is a positive integer detailing the number of lags to show.
        The default value of None will return all (2*len(x)-1) lags.

        See the respective function for documentation on valid kwargs
        """

        Nx = len(x)
        if Nx!=len(y):
            raise ValueError('x and y must be equal length')

        x = detrend(npy.asarray(x))
        y = detrend(npy.asarray(y))

        c = npy.correlate(x, y, mode=2)

        if normed: c/= npy.sqrt(npy.dot(x,x) * npy.dot(y,y))

        if maxlags is None: maxlags = Nx - 1

        if maxlags >= Nx or maxlags < 1:
            raise ValueError('maglags must be None or strictly positive < %d'%Nx)

        lags = npy.arange(-maxlags,maxlags+1)
        c = c[Nx-1-maxlags:Nx+maxlags]


        if usevlines:
            a = self.vlines(lags, [0], c, **kwargs)
            b = self.axhline(**kwargs)
        else:

            kwargs.setdefault('marker', 'o')
            kwargs.setdefault('linestyle', 'None')
            a, = self.plot(lags, c, **kwargs)
            b = None
        return lags, c, a, b
    xcorr.__doc__ = cbook.dedent(xcorr.__doc__) % martist.kwdocd

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

          'best'         : 0,
          'upper right'  : 1,
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

        and so on.  The following kwargs are supported:

        isaxes=True           # whether this is an axes legend
        numpoints = 4         # the number of points in the legend line
        prop = FontProperties(size='smaller')  # the font property
        pad = 0.2             # the fractional whitespace inside the legend border
        markerscale = 0.6     # the relative size of legend markers vs. original
        shadow                # if True, draw a shadow behind legend
        labelsep = 0.005     # the vertical space between the legend entries
        handlelen = 0.05     # the length of the legend lines
        handletextsep = 0.02 # the space between the legend line and legend text
        axespad = 0.02       # the border between the axes and legend edge
        """

        def get_handles():
            handles = self.lines[:]
            handles.extend(self.patches)
            handles.extend([c for c in self.collections
                            if isinstance(c, mcoll.LineCollection)])
            handles.extend([c for c in self.collections
                            if isinstance(c, mcoll.RegularPolyCollection)])
            return handles

        if len(args)==0:
            handles = []
            labels = []
            for handle in get_handles():
                label = handle.get_label()
                if label is not None and label != '' and not label.startswith('_'):
                    handles.append(handle)
                    labels.append(label)
            if len(handles) == 0:
                warnings.warn("No labeled objects found. Use label='...' kwarg on individual plots.")
                return None

        elif len(args)==1:
            # LABELS
            labels = args[0]
            handles = [h for h, label in zip(get_handles(), labels)]

        elif len(args)==2:
            if is_string_like(args[1]) or isinstance(args[1], int):
                # LABELS, LOC
                labels, loc = args
                handles = [h for h, label in zip(get_handles(), labels)]
                kwargs['loc'] = loc
            else:
                # LINES, LABELS
                handles, labels = args

        elif len(args)==3:
            # LINES, LABELS, LOC
            handles, labels, loc = args
            kwargs['loc'] = loc
        else:
            raise TypeError('Invalid arguments to legend')


        handles = cbook.flatten(handles)
        self.legend_ = mlegend.Legend(self, handles, labels, **kwargs)
        return self.legend_


    #### Specialized plotting

    def step(self, x, y, *args, **kwargs):
        '''
        step(x, y, *args, **kwargs)

        x and y must be 1-D sequences, and it is assumed, but not checked,
        that x is uniformly increasing.

        Make a step plot. The args and keyword args to step are the same
        as the args to plot. See help plot for more info.

        Additional keyword args for step:

        * where: can be 'pre', 'post' or 'mid'; if 'pre', the
                    interval from x[i] to x[i+1] has level y[i];
                    if 'post', that interval has level y[i+1];
                    and if 'mid', the jumps in y occur half-way
                    between the x-values.  Default is 'pre'.
        '''

        where = kwargs.pop('where', 'pre')

        if not cbook.iterable(x):
            x = ma.array([x], dtype=npy.float_)
        if not cbook.iterable(y):
            y = ma.array([y], dtype=npy.float_)

        if where=='pre':
            x2 = ma.zeros((2*len(x)-1,), npy.float_)
            y2 = ma.zeros((2*len(y)-1,), npy.float_)

            x2[0::2], x2[1::2] = x, x[:-1]
            y2[0::2], y2[1:-1:2] = y, y[1:]

        elif where=='post':
            x2 = ma.zeros((2*len(x)-1,), npy.float_)
            y2 = ma.zeros((2*len(y)-1,), npy.float_)

            x2[::2], x2[1:-1:2] = x, x[1:]
            y2[0::2], y2[1::2] = y, y[:-1]

        elif where=='mid':
            x2 = ma.zeros((2*len(x),), npy.float_)
            y2 = ma.zeros((2*len(y),), npy.float_)

            x2[1:-1:2] = 0.5*(x[:-1]+x[1:])
            x2[2::2] = 0.5*(x[:-1]+x[1:])
            x2[0], x2[-1] = x[0], x[-1]

            y2[0::2], y2[1::2] = y, y

        return self.plot(x2, y2, *args, **kwargs)


    def bar(self, left, height, width=0.8, bottom=None,
            color=None, edgecolor=None, linewidth=None,
            yerr=None, xerr=None, ecolor=None, capsize=3,
            align='edge', orientation='vertical', log=False,
            **kwargs
            ):
        """
        BAR(left, height, width=0.8, bottom=0,
            color=None, edgecolor=None, linewidth=None,
            yerr=None, xerr=None, ecolor=None, capsize=3,
            align='edge', orientation='vertical', log=False)

        Make a bar plot with rectangles bounded by

          left, left+width, bottom, bottom+height
                (left, right, bottom and top edges)

        left, height, width, and bottom can be either scalars or sequences

        Return value is a list of Rectangle patch instances

            left - the x coordinates of the left sides of the bars

            height - the heights of the bars

        Optional arguments:

            width - the widths of the bars

            bottom - the y coordinates of the bottom edges of the bars

            color - the colors of the bars

            edgecolor - the colors of the bar edges

            linewidth - width of bar edges; None means use default
                linewidth; 0 means don't draw edges.

            xerr and yerr, if not None, will be used to generate errorbars
            on the bar chart

            ecolor specifies the color of any errorbar

            capsize (default 3) determines the length in points of the error
            bar caps

            align = 'edge' (default) | 'center'

            orientation = 'vertical' | 'horizontal'

            log = False | True - False (default) leaves the orientation
                    axis as-is; True sets it to log scale

        For vertical bars, align='edge' aligns bars by their left edges in
        left, while 'center' interprets these values as the x coordinates of
        the bar centers. For horizontal bars, 'edge' aligns bars by their
        bottom edges in bottom, while 'center' interprets these values as the
        y coordinates of the bar centers.

        The optional arguments color, edgecolor, linewidth, xerr, and yerr can
        be either scalars or sequences of length equal to the number of bars.
        This enables you to use bar as the basis for stacked bar charts, or
        candlestick plots.

        Optional kwargs:
        %(Rectangle)s
        """
        if not self._hold: self.cla()

        def make_iterable(x):
            if not iterable(x):
                return [x]
            else:
                return x

        # make them safe to take len() of
        _left = left
        left = make_iterable(left)
        height = make_iterable(height)
        width = make_iterable(width)
        _bottom = bottom
        bottom = make_iterable(bottom)
        linewidth = make_iterable(linewidth)

        adjust_ylim = False
        adjust_xlim = False
        if orientation == 'vertical':
            self._process_unit_info(xdata=left, ydata=height, kwargs=kwargs)
            if log:
                self.set_yscale('log')
            # size width and bottom according to length of left
            if _bottom is None:
                if self.get_yscale() == 'log':
                    bottom = [1e-100]
                    adjust_ylim = True
                else:
                    bottom = [0]
            nbars = len(left)
            if len(width) == 1:
                width *= nbars
            if len(bottom) == 1:
                bottom *= nbars
        elif orientation == 'horizontal':
            self._process_unit_info(xdata=width, ydata=bottom, kwargs=kwargs)
            if log:
                self.set_xscale('log')
            # size left and height according to length of bottom
            if _left is None:
                if self.get_xscale() == 'log':
                    left = [1e-100]
                    adjust_xlim = True
                else:
                    left = [0]
            nbars = len(bottom)
            if len(left) == 1:
                left *= nbars
            if len(height) == 1:
                height *= nbars
        else:
            raise ValueError, 'invalid orientation: %s' % orientation


        # do not convert to array here as unit info is lost
        #left = npy.asarray(left)
        #height = npy.asarray(height)
        #width = npy.asarray(width)
        #bottom = npy.asarray(bottom)

        if len(linewidth) == 1: linewidth = linewidth * nbars

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

        if yerr is not None:
            if not iterable(yerr):
                yerr = [yerr]*nbars

        if xerr is not None:
            if not iterable(xerr):
                xerr = [xerr]*nbars

        assert len(left)==nbars, "argument 'left' must be %d or scalar" % nbars
        assert len(height)==nbars, "argument 'height' must be %d or scalar" % nbars
        assert len(width)==nbars, "argument 'width' must be %d or scalar" % nbars
        assert len(bottom)==nbars, "argument 'bottom' must be %d or scalar" % nbars
        assert len(color)==nbars, "argument 'color' must be %d or scalar" % nbars
        assert len(edgecolor)==nbars, "argument 'edgecolor' must be %d or scalar" % nbars
        assert len(linewidth)==nbars, "argument 'linewidth' must be %d or scalar" % nbars

        if yerr is not None and len(yerr)!=nbars:
            raise ValueError("bar() argument 'yerr' must be len(%s) or scalar" % nbars)
        if xerr is not None and len(xerr)!=nbars:
            raise ValueError("bar() argument 'xerr' must be len(%s) or scalar" % nbars)

        patches = []


        if align == 'edge':
            pass
        elif align == 'center':
            if orientation == 'vertical':
                left = [left[i] - width[i]/2. for i in range(len(left))]
            elif orientation == 'horizontal':
                bottom = [bottom[i] - height[i]/2. for i in range(len(bottom))]

        else:
            raise ValueError, 'invalid alignment: %s' % align

        args = zip(left, bottom, width, height, color, edgecolor, linewidth)
        for l, b, w, h, c, e, lw in args:
            if h<0:
                b += h
                h = abs(h)
            if w<0:
                l += w
                w = abs(w)
            r = mpatches.Rectangle(
                xy=(l, b), width=w, height=h,
                facecolor=c,
                edgecolor=e,
                linewidth=lw,
                )
            r.update(kwargs)
            self.add_patch(r)
            patches.append(r)

        holdstate = self._hold
        self.hold(True) # ensure hold is on before plotting errorbars

        if xerr is not None or yerr is not None:
            if orientation == 'vertical':
                # using list comps rather than arrays to preserve unit info
                x = [l+0.5*w for l, w in zip(left, width)]
                y = [b+h for b,h in zip(bottom, height)]

            elif orientation == 'horizontal':
                # using list comps rather than arrays to preserve unit info
                x = [l+w for l,w in zip(left, width)]
                y = [b+0.5*h for b,h in zip(bottom, height)]

            self.errorbar(
                x, y,
                yerr=yerr, xerr=xerr,
                fmt=None, ecolor=ecolor, capsize=capsize)

        self.hold(holdstate) # restore previous hold state

        if adjust_xlim:
            xmin, xmax = self.dataLim.intervalx().get_bounds()
            xmin = npy.amin(width)
            if xerr is not None:
                xmin = xmin - npy.amax(xerr)
            xmin = max(xmin*0.9, 1e-100)
            self.dataLim.intervalx().set_bounds(xmin, xmax)
        if adjust_ylim:
            ymin, ymax = self.dataLim.intervaly().get_bounds()
            ymin = npy.amin(height)
            if yerr is not None:
                ymin = ymin - npy.amax(yerr)
            ymin = max(ymin*0.9, 1e-100)
            self.dataLim.intervaly().set_bounds(ymin, ymax)
        self.autoscale_view()
        return patches
    bar.__doc__ = cbook.dedent(bar.__doc__) % martist.kwdocd

    def barh(self, bottom, width, height=0.8, left=None, **kwargs):
        """
        BARH(bottom, width, height=0.8, left=0, **kwargs)

        Make a horizontal bar plot with rectangles bounded by

          left, left+width, bottom, bottom+height
                (left, right, bottom and top edges)

        bottom, width, height, and left can be either scalars or sequences

        Return value is a list of Rectangle patch instances

            bottom - the vertical positions of the bottom edges of the bars

            width - the lengths of the bars

        Optional arguments:

            height - the heights (thicknesses) of the bars

            left - the x coordinates of the left edges of the bars

            color - the colors of the bars

            edgecolor - the colors of the bar edges

            linewidth - width of bar edges; None means use default
                linewidth; 0 means don't draw edges.

            xerr and yerr, if not None, will be used to generate errorbars
            on the bar chart

            ecolor specifies the color of any errorbar

            capsize (default 3) determines the length in points of the error
            bar caps

            align = 'edge' (default) | 'center'

            log = False | True - False (default) leaves the horizontal
                    axis as-is; True sets it to log scale

        Setting align='edge' aligns bars by their bottom edges in bottom,
        while 'center' interprets these values as the y coordinates of the bar
        centers.

        The optional arguments color, edgecolor, linewidth, xerr, and yerr can
        be either scalars or sequences of length equal to the number of bars.
        This enables you to use barh as the basis for stacked bar charts, or
        candlestick plots.

        Optional kwargs:
        %(Rectangle)s
        """

        patches = self.bar(left=left, height=height, width=width, bottom=bottom,
                           orientation='horizontal', **kwargs)
        return patches

    barh.__doc__ = cbook.dedent(barh.__doc__) % martist.kwdocd

    def broken_barh(self, xranges, yrange, **kwargs):
        """
        A collection of horizontal bars spanning yrange with a sequence of
        xranges

        xranges : sequence of (xmin, xwidth)
        yrange  : (ymin, ywidth)

        kwargs are collections.BrokenBarHCollection properties
        %(BrokenBarHCollection)s

        these can either be a single argument, ie facecolors='black'
        or a sequence of arguments for the various bars, ie
        facecolors='black', 'red', 'green'

        """
        col = mcoll.BrokenBarHCollection(xranges, yrange, **kwargs)
        self.add_collection(col, autolim=True)
        self.autoscale_view()

        return col

    broken_barh.__doc__ = cbook.dedent(broken_barh.__doc__) % martist.kwdocd

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

        baseline, = self.plot([npy.amin(x), npy.amax(x)], [0,0], basefmt)

        self.hold(remember_hold)

        return markerline, stemlines, baseline


    def pie(self, x, explode=None, labels=None,
            colors=None,
            autopct=None,
            pctdistance=0.6,
            shadow=False,
            labeldistance=1.1,
            ):
        """
        PIE(x, explode=None, labels=None,
            colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'),
            autopct=None, pctdistance=0.6, labeldistance=1.1, shadow=False)

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

          - pctdistance is the ratio between the center of each pie slice
            and the start of the text generated by autopct.  Ignored if autopct
            is None; default is 0.6.

          - labeldistance is the radial distance at which the pie labels are drawn

          - shadow, if True, will draw a shadow beneath the pie.

        The pie chart will probably look best if the figure and axes are
        square.  Eg,

          figure(figsize=(8,8))
          ax = axes([0.1, 0.1, 0.8, 0.8])

        Return value:

          If autopct is None, return a list of (patches, texts), where patches
          is a sequence of mpatches.Wedge instances and texts is a
          list of the label Text instnaces

          If autopct is not None, return (patches, texts, autotexts), where
          patches and texts are as above, and autotexts is a list of text
          instances for the numeric labels
        """
        self.set_frame_on(False)

        x = npy.asarray(x).astype(npy.float32)

        sx = float(x.sum())
        if sx>1: x = npy.divide(x,sx)

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
        for frac, label, expl in cbook.safezip(x,labels, explode):
            x, y = center
            theta2 = theta1 + frac
            thetam = 2*math.pi*0.5*(theta1+theta2)
            x += expl*math.cos(thetam)
            y += expl*math.sin(thetam)

            w = mpatches.Wedge((x,y), radius, 360.*theta1, 360.*theta2,
                      facecolor=colors[i%len(colors)])
            slices.append(w)
            self.add_patch(w)
            w.set_label(label)

            if shadow:
                # make sure to add a shadow after the call to
                # add_patch so the figure and transform props will be
                # set
                shad = mpatches.Shadow(w, -0.02, -0.02,
                              #props={'facecolor':w.get_facecolor()}
                              )
                shad.set_zorder(0.9*w.get_zorder())
                self.add_patch(shad)


            xt = x + labeldistance*radius*math.cos(thetam)
            yt = y + labeldistance*radius*math.sin(thetam)

            t = self.text(xt, yt, label,
                          size=rcParams['xtick.labelsize'],
                          horizontalalignment='center',
                          verticalalignment='center')

            texts.append(t)

            if autopct is not None:
                xt = x + pctdistance*radius*math.cos(thetam)
                yt = y + pctdistance*radius*math.sin(thetam)
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
                 fmt='-', ecolor=None, capsize=3,
                 barsabove=False, lolims=False, uplims=False,
                 xlolims=False, xuplims=False, **kwargs):
        """
        ERRORBAR(x, y, yerr=None, xerr=None,
                 fmt='b-', ecolor=None, capsize=3, barsabove=False,
                 lolims=False, uplims=False,
                 xlolims=False, xuplims=False)

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

            lolims, uplims, xlolims, xuplims: These arguments can be used
             to indicate that a value gives only upper/lower limits. In
             that case a caret symbol is used to indicate this. lims-arguments
             may be of the same type as xerr and yerr.

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

        valid kwargs for the marker properties are
        %(Line2D)s

        Return value is a length 3 tuple.  The first element is the
        Line2D instance for the y symbol lines.  The second element is
        a list of error bar cap lines, the third element is a list of
        line collections for the horizontal and vertical error ranges
        """

        self._process_unit_info(xdata=x, ydata=y, kwargs=kwargs)
        if not self._hold: self.cla()

        # make sure all the args are iterable; use lists not arrays to preserve units
        if not iterable(x):
            x = [x]

        if not iterable(y):
            y = [y]

        if xerr is not None:
            if not iterable(xerr):
                xerr = [xerr]*len(x)

        if yerr is not None:
            if not iterable(yerr):
                yerr = [yerr]*len(y)

        l0 = None

        if barsabove and fmt is not None:
            l0, = self.plot(x,y,fmt,**kwargs)

        barcols = []
        caplines = []

        lines_kw = {'label':'_nolegend_'}
        if 'linewidth' in kwargs:
            lines_kw['linewidth']=kwargs['linewidth']
        if 'lw' in kwargs:
            lines_kw['lw']=kwargs['lw']
        if 'transform' in kwargs:
            lines_kw['transform'] = kwargs['transform']

        # arrays fine here, they are booleans and hence not units
        if not iterable(lolims):
            lolims = npy.asarray([lolims]*len(x), bool)
        else: lolims = npy.asarray(lolims, bool)

        if not iterable(uplims): uplims = npy.array([uplims]*len(x), bool)
        else: uplims = npy.asarray(uplims, bool)

        if not iterable(xlolims): xlolims = npy.array([xlolims]*len(x), bool)
        else: xlolims = npy.asarray(xlolims, bool)

        if not iterable(xuplims): xuplims = npy.array([xuplims]*len(x), bool)
        else: xuplims = npy.asarray(xuplims, bool)

        def xywhere(xs, ys, mask):
            """
            return xs[mask], ys[mask] where mask is True but xs and
            ys are not arrays
            """
            assert len(xs)==len(ys)
            assert len(xs)==len(mask)
            xs = [thisx for thisx, b in zip(xs, mask) if b]
            ys = [thisy for thisy, b in zip(ys, mask) if b]
            return xs, ys


        if capsize > 0:
            plot_kw = {
                'ms':2*capsize,
                'label':'_nolegend_'}
            if 'markeredgewidth' in kwargs:
                plot_kw['markeredgewidth']=kwargs['markeredgewidth']
            if 'mew' in kwargs:
                plot_kw['mew']=kwargs['mew']
            if 'transform' in kwargs:
                plot_kw['transform'] = kwargs['transform']

        if xerr is not None:
            if iterable(xerr) and len(xerr)==2 and iterable(xerr[0]) and iterable(xerr[1]):
                # using list comps rather than arrays to preserve units
                left  = [thisx-thiserr for (thisx, thiserr) in cbook.safezip(x,xerr[0])]
                right  = [thisx+thiserr for (thisx, thiserr) in cbook.safezip(x,xerr[1])]
            else:
                # using list comps rather than arrays to preserve units
                left  = [thisx-thiserr for (thisx, thiserr) in cbook.safezip(x,xerr)]
                right  = [thisx+thiserr for (thisx, thiserr) in cbook.safezip(x,xerr)]

            barcols.append( self.hlines(y, left, right, **lines_kw ) )
            if capsize > 0:
                if xlolims.any():
                    # can't use numpy logical indexing since left and
                    # y are lists
                    leftlo, ylo = xywhere(left, y, xlolims)

                    caplines.extend( self.plot(leftlo, ylo, ls='None', marker=mlines.CARETLEFT, **plot_kw) )
                    xlolims = ~xlolims
                    leftlo, ylo = xywhere(left, y, xlolims)
                    caplines.extend( self.plot(leftlo, ylo, 'k|', **plot_kw) )
                else:
                    caplines.extend( self.plot(left, y, 'k|', **plot_kw) )

                if xuplims.any():

                    rightup, yup = xywhere(right, y, xuplims)
                    caplines.extend( self.plot(rightup,  yup, ls='None', marker=mlines.CARETRIGHT, **plot_kw) )
                    xuplims = ~xuplims
                    rightup, yup = xywhere(right, y, xuplims)
                    caplines.extend( self.plot(rightup,  yup, 'k|', **plot_kw) )
                else:
                    caplines.extend( self.plot(right, y, 'k|', **plot_kw) )

        if yerr is not None:
            if iterable(yerr) and len(yerr)==2 and iterable(yerr[0]) and iterable(yerr[1]):
                # using list comps rather than arrays to preserve units
                lower  = [thisy-thiserr for (thisy, thiserr) in cbook.safezip(y,yerr[0])]
                upper  = [thisy+thiserr for (thisy, thiserr) in cbook.safezip(y,yerr[1])]
            else:
                # using list comps rather than arrays to preserve units
                lower  = [thisy-thiserr for (thisy, thiserr) in cbook.safezip(y,yerr)]
                upper  = [thisy+thiserr for (thisy, thiserr) in cbook.safezip(y,yerr)]

            barcols.append( self.vlines(x, lower, upper, **lines_kw) )
            if capsize > 0:

                if lolims.any():
                    xlo, lowerlo = xywhere(x, lower, lolims)
                    caplines.extend( self.plot(xlo, lowerlo, ls='None', marker=mlines.CARETDOWN, **plot_kw) )
                    lolims = ~lolims
                    xlo, lowerlo = xywhere(x, lower, lolims)
                    caplines.extend( self.plot(xlo, lowerlo, 'k_', **plot_kw) )
                else:
                    caplines.extend( self.plot(x, lower, 'k_', **plot_kw) )

                if uplims.any():
                    xup, upperup = xywhere(x, upper, uplims)

                    caplines.extend( self.plot(xup, upperup, ls='None', marker=mlines.CARETUP, **plot_kw) )
                    uplims = ~uplims
                    xup, upperup = xywhere(x, upper, uplims)
                    caplines.extend( self.plot(xup, upperup, 'k_', **plot_kw) )
                else:
                    caplines.extend( self.plot(x, upper, 'k_', **plot_kw) )

        if not barsabove and fmt is not None:
            l0, = self.plot(x,y,fmt,**kwargs)

        if ecolor is None:
            if l0 is None:
                ecolor = self._get_lines._get_next_cycle_color()
            else:
                ecolor = l0.get_color()

        for l in barcols:
            l.set_color(ecolor)
        for l in caplines:
            l.set_color(ecolor)

        self.autoscale_view()
        return (l0, caplines, barcols)
    errorbar.__doc__ = cbook.dedent(errorbar.__doc__) % martist.kwdocd

    def boxplot(self, x, notch=0, sym='b+', vert=1, whis=1.5,
                positions=None, widths=None):
        """
        boxplot(x, notch=0, sym='+', vert=1, whis=1.5,
                positions=None, widths=None)

        Make a box and whisker plot for each column of x or
        each vector in sequence x.
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

        x is an array or a sequence of vectors.

        Returns a list of the lines added.

        """
        if not self._hold: self.cla()
        holdStatus = self._hold
        whiskers, caps, boxes, medians, fliers = [], [], [], [], []

        # convert x to a list of vectors
        if hasattr(x, 'shape'):
            if len(x.shape) == 1:
                if hasattr(x[0], 'shape'):
                    x = list(x)
                else:
                    x = [x,]
            elif len(x.shape) == 2:
                nr, nc = x.shape
                if nr == 1:
                    x = [x]
                elif nc == 1:
                    x = [x.ravel()]
                else:
                    x = [x[:,i] for i in range(nc)]
            else:
                raise ValueError, "input x can have no more than 2 dimensions"
        if not hasattr(x[0], '__len__'):
            x = [x]
        col = len(x)

        # get some plot info
        if positions is None:
            positions = range(1, col + 1)
        if widths is None:
            distance = max(positions) - min(positions)
            widths = min(0.15*max(distance,1.0), 0.5)
        if isinstance(widths, float) or isinstance(widths, int):
            widths = npy.ones((col,), float) * widths

        # loop through columns, adding each to plot
        self.hold(True)
        for i,pos in enumerate(positions):
            d = npy.ravel(x[i])
            row = len(d)
            # get median and quartiles
            q1, med, q3 = mlab.prctile(d,[25,50,75])
            # get high extreme
            iq = q3 - q1
            hi_val = q3 + whis*iq
            wisk_hi = npy.compress( d <= hi_val , d )
            if len(wisk_hi) == 0:
                wisk_hi = q3
            else:
                wisk_hi = max(wisk_hi)
            # get low extreme
            lo_val = q1 - whis*iq
            wisk_lo = npy.compress( d >= lo_val, d )
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
                flier_hi = npy.compress( d > wisk_hi, d )
                flier_lo = npy.compress( d < wisk_lo, d )
                flier_hi_x = npy.ones(flier_hi.shape[0]) * pos
                flier_lo_x = npy.ones(flier_lo.shape[0]) * pos

            # get x locations for fliers, whisker, whisker cap and box sides
            box_x_min = pos - widths[i] * 0.5
            box_x_max = pos + widths[i] * 0.5

            wisk_x = npy.ones(2) * pos

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
                notch_max = med + 1.57*iq/npy.sqrt(row)
                notch_min = med - 1.57*iq/npy.sqrt(row)
                if notch_max > q3:
                    notch_max = q3
                if notch_min < q1:
                    notch_min = q1
                # make our notched box vectors
                box_x = [box_x_min, box_x_max, box_x_max, cap_x_max, box_x_max,
                         box_x_max, box_x_min, box_x_min, cap_x_min, box_x_min,
                         box_x_min ]
                box_y = [q1, q1, notch_min, med, notch_max, q3, q3, notch_max,
                         med, notch_min, q1]
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

            SCATTER(x, y, **kwargs)
            SCATTER(x, y, s, **kwargs)
            SCATTER(x, y, s, c, **kwargs)

        Make a scatter plot of x versus y, where x, y are 1-D sequences
        of the same length, N.

        Arguments s and c can also be given as kwargs; this is encouraged
        for readability.

            s is a size in points^2.  It is a scalar
              or an array of the same length as x and y.

            c is a color and can be a single color format string,
              or a sequence of color specifications of length N,
              or a sequence of N numbers to be mapped to colors
              using the cmap and norm specified via kwargs (see below).
              Note that c should not be a single numeric RGB or RGBA
              sequence because that is indistinguishable from an array
              of values to be colormapped. c can be a 2-D array in which
              the rows are RGB or RGBA, however.

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

        If marker is None and verts is not None, verts is a sequence
        of (x,y) vertices for a custom scatter symbol.

        s is a size argument in points squared.

        Any or all of x, y, s, and c may be masked arrays, in which
        case all masks will be combined and only unmasked points
        will be plotted.

        Other keyword args; the color mapping and normalization arguments will
        be used only if c is an array of floats

          * cmap = cm.jet : a colors.Colormap instance from cm.
            defaults to rc image.cmap

          * norm = colors.Normalize() : colors.Normalize instance
            is used to scale luminance data to 0,1.

          * vmin=None and vmax=None : vmin and vmax are used in conjunction
            with norm to normalize luminance data.  If either are None, the
            min and max of the color array C is used.  Note if you pass a norm
            instance, your settings for vmin and vmax will be ignored

          * alpha =1.0 : the alpha value for the patches

          * linewidths, if None, defaults to (lines.linewidth,).  Note
            that this is a tuple, and if you set the linewidths
            argument you must set it as a sequence of floats, as
            required by RegularPolyCollection -- see
            collections.RegularPolyCollection for details

         * faceted: if True, will use the default edgecolor for the
           markers.  If False, will set the edgecolors to be the same
           as the facecolors.
           This kwarg is deprecated;
           please use the edgecolors kwarg instead:
               shading='flat'    --> edgecolors='None'
               shading='faceted  --> edgecolors=None
           edgecolors also can be any mpl color or sequence of colors.

           Optional kwargs control the PatchCollection properties:
        %(PatchCollection)s

        A Collection instance is returned
        """

        if not self._hold: self.cla()

        syms =  { # a dict from symbol to (numsides, angle)
            's' : (4,math.pi/4.0,0),   # square
            'o' : (20,0,0),            # circle
            '^' : (3,0,0),             # triangle up
            '>' : (3,math.pi/2.0,0),   # triangle right
            'v' : (3,math.pi,0),       # triangle down
            '<' : (3,3*math.pi/2.0,0), # triangle left
            'd' : (4,0,0),             # diamond
            'p' : (5,0,0),             # pentagram
            'h' : (6,0,0),             # hexagon
            '8' : (8,0,0),             # octagon
            '+' : (4,0,2),             # plus
            'x' : (4,math.pi/4.0,2)    # cross
            }

        self._process_unit_info(xdata=x, ydata=y, kwargs=kwargs)

        x, y, s, c = delete_masked_points(x, y, s, c)

        # The inherent ambiguity is resolved in favor of color
        # mapping, not interpretation as rgb or rgba.

        if not is_string_like(c):
            sh = npy.shape(c)
            if len(sh) == 1 and sh[0] == len(x):
                colors = None  # use cmap, norm after collection is created
            else:
                colors = mcolors.colorConverter.to_rgba_list(c, alpha)
        else:
            colors = mcolors.colorConverter.to_rgba_list(c, alpha)

        if not iterable(s):
            scales = (s,)
        else:
            scales = s

        if faceted: edgecolors = None
        else: edgecolors = 'None'

        sym = None
        symstyle = 0

        # to be API compatible
        if marker is None and not (verts is None):
            marker = (verts, 0)
            verts = None

        if is_string_like(marker):
            # the standard way to define symbols using a string character
            sym = syms.get(marker)
            if sym is None and verts is None:
                raise ValueError('Unknown marker symbol to scatter')
            numsides, rotation, symstyle = syms[marker]

        elif iterable(marker):
            # accept marker to be:
            #    (numsides, style, [angle])
            # or
            #    (verts[], style, [angle])

            if len(marker)<2 or len(marker)>3:
                raise ValueError('Cannot create markersymbol from marker')

            if cbook.is_numlike(marker[0]):
                # (numsides, style, [angle])

                if len(marker)==2:
                    numsides, rotation = marker[0], 0.
                elif len(marker)==3:
                    numsides, rotation = marker[0], marker[2]
                sym = True

                if marker[1] in (1,2):
                    symstyle = marker[1]

            else:
                verts = npy.asarray(marker[0])

        if sym is not None:
            if symstyle==0:

                collection = mcoll.RegularPolyCollection(
                    self.figure.dpi,
                    numsides, rotation, scales,
                    facecolors = colors,
                    edgecolors = edgecolors,
                    linewidths = linewidths,
                    offsets = zip(x,y),
                    transOffset = self.transData,
                    )
            elif symstyle==1:
                collection = mcoll.StarPolygonCollection(
                    self.figure.dpi,
                    numsides, rotation, scales,
                    facecolors = colors,
                    edgecolors = edgecolors,
                    linewidths = linewidths,
                    offsets = zip(x,y),
                    transOffset = self.transData,
                    )
            elif symstyle==2:
                collection = mcoll.AsteriskPolygonCollection(
                    self.figure.dpi,
                    numsides, rotation, scales,
                    facecolors = colors,
                    edgecolors = edgecolors,
                    linewidths = linewidths,
                    offsets = zip(x,y),
                    transOffset = self.transData,
                    )
        else:
            # rescale verts
            rescale = npy.sqrt(max(verts[:,0]**2+verts[:,1]**2))
            verts /= rescale

            scales = npy.asarray(scales)
            scales = npy.sqrt(scales * self.figure.dpi.get() / 72.)
            if len(scales)==1:
                verts = [scales[0]*verts]
            else:
                # todo -- make this nx friendly
                verts = [verts*s for s in scales]
            collection = mcoll.PolyCollection(
                verts,
                facecolors = colors,
                edgecolors = edgecolors,
                linewidths = linewidths,
                offsets = zip(x,y),
                transOffset = self.transData,
                )
            collection.set_transform(mtrans.identity_transform())
        collection.set_alpha(alpha)
        collection.update(kwargs)

        if colors is None:
            if norm is not None: assert(isinstance(norm, mcolors.Normalize))
            if cmap is not None: assert(isinstance(cmap, mcolors.Colormap))
            collection.set_array(npy.asarray(c))
            collection.set_cmap(cmap)
            collection.set_norm(norm)

            if vmin is not None or vmax is not None:
                collection.set_clim(vmin, vmax)
            else:
                collection.autoscale_None()

        temp_x = x
        temp_y = y

        minx = npy.amin(temp_x)
        maxx = npy.amax(temp_x)
        miny = npy.amin(temp_y)
        maxy = npy.amax(temp_y)

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

    scatter.__doc__ = cbook.dedent(scatter.__doc__) % martist.kwdocd

    def arrow(self, x, y, dx, dy, **kwargs):
        """
        Draws arrow on specified axis from (x,y) to (x+dx,y+dy).

        Optional kwargs control the arrow properties:
        %(FancyArrow)s
        """
        a = mpatches.FancyArrow(x, y, dx, dy, **kwargs)
        self.add_artist(a)
        return a
    arrow.__doc__ = cbook.dedent(arrow.__doc__) % martist.kwdocd

    def quiverkey(self, *args, **kw):
        qk = mquiver.QuiverKey(*args, **kw)
        self.add_artist(qk)
        return qk
    quiverkey.__doc__ = mquiver.QuiverKey.quiverkey_doc

    def quiver(self, *args, **kw):
        q = mquiver.Quiver(self, *args, **kw)
        self.add_collection(q)
        self.update_datalim_numerix(q.X, q.Y)
        self.autoscale_view()
        return q
    quiver.__doc__ = mquiver.Quiver.quiver_doc

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

        If you would like to fill below a curve, eg shade a region
        between 0 and y along x, use mlab.poly_between, eg

          xs, ys = poly_between(x, 0, y)
          axes.fill(xs, ys, facecolor='red', alpha=0.5)

        See examples/fill_between.py for more examples.

        kwargs control the Polygon properties:
        %(Polygon)s
        """
        if not self._hold: self.cla()

        patches = []
        for poly in self._get_patches_for_fill(*args, **kwargs):
            self.add_patch( poly )
            patches.append( poly )
        self.autoscale_view()
        return patches
    fill.__doc__ = cbook.dedent(fill.__doc__) % martist.kwdocd
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
               imlim=None,
               **kwargs):
        """

        IMSHOW(X, cmap=None, norm=None, aspect=None, interpolation=None,
               alpha=1.0, vmin=None, vmax=None, origin=None, extent=None)

        IMSHOW(X) - plot image X to current axes, resampling to scale to axes
                    size (X may be numpy array or PIL image)

        IMSHOW(X, **kwargs) - Use keyword args to control image scaling,
        colormapping etc. See below for details


        Display the image in X to current axes.  X may be a float array, a
        uint8 array or a PIL image. If X is an array, X can have the following
        shapes:

            MxN    : luminance (grayscale, float array only)

            MxNx3  : RGB (float or uint8 array)

            MxNx4  : RGBA (float or uint8 array)

        The value for each component of MxNx3 and MxNx4 float arrays should be
        in the range 0.0 to 1.0; MxN float arrays may be normalised.

        A image.AxesImage instance is returned

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

          * norm is a mcolors.Normalize instance; default is
            normalization().  This scales luminance -> 0-1 (only used for an
            MxN float array).

          * vmin and vmax are used to scale a luminance image to 0-1.  If
            either is None, the min and max of the luminance values will be
            used.  Note if you pass a norm instance, the settings for vmin and
            vmax will be ignored.

          * alpha = 1.0 : the alpha blending value

          * origin is 'upper' or 'lower', to place the [0,0]
            index of the array in the upper left or lower left corner of
            the axes.  If None, default to rc image.origin

          * extent is (left, right, bottom, top) data values of the
            axes.  The default assigns zero-based row, column indices
            to the x, y centers of the pixels.

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

        Additional kwargs are martist properties
        """

        if not self._hold: self.cla()

        if norm is not None: assert(isinstance(norm, mcolors.Normalize))
        if cmap is not None: assert(isinstance(cmap, mcolors.Colormap))
        if aspect is None: aspect = rcParams['image.aspect']
        self.set_aspect(aspect)
        im = mimage.AxesImage(self, cmap, norm, interpolation, origin, extent,
                       filternorm=filternorm,
                       filterrad=filterrad, **kwargs)

        im.set_data(X)
        im.set_alpha(alpha)
        self._set_artist_props(im)
        #if norm is None and shape is None:
        #    im.set_clim(vmin, vmax)
        if vmin is not None or vmax is not None:
            im.set_clim(vmin, vmax)
        else:
            im.autoscale_None()

        xmin, xmax, ymin, ymax = im.get_extent()

        corners = (xmin, ymin), (xmax, ymax)
        self.update_datalim(corners)
        if self._autoscaleon:
            self.set_xlim((xmin, xmax))
            self.set_ylim((ymin, ymax))
        self.images.append(im)

        return im


    def _pcolorargs(self, funcname, *args):
        if len(args)==1:
            C = args[0]
            numRows, numCols = C.shape
            X, Y = npy.meshgrid(npy.arange(numCols+1), npy.arange(numRows+1) )
        elif len(args)==3:
            X, Y, C = args
        else:
            raise TypeError(
                'Illegal arguments to %s; see help(%s)' % (funcname, funcname))

        Nx = X.shape[-1]
        Ny = Y.shape[0]
        if len(X.shape) <> 2 or X.shape[0] == 1:
            x = X.reshape(1,Nx)
            X = x.repeat(Ny, axis=0)
        if len(Y.shape) <> 2 or Y.shape[1] == 1:
            y = Y.reshape(Ny, 1)
            Y = y.repeat(Nx, axis=1)
        if X.shape != Y.shape:
            raise TypeError(
                'Incompatible X, Y inputs to %s; see help(%s)' % (funcname, funcname))
        return X, Y, C

    def pcolor(self, *args, **kwargs):
        """
        pcolor(*args, **kwargs): pseudocolor plot of a 2-D array

        Function signatures

          pcolor(C, **kwargs)
          pcolor(X, Y, C, **kwargs)

        C is the array of color values

        X and Y, if given, specify the (x,y) coordinates of the colored
        quadrilaterals; the quadrilateral for C[i,j] has corners at
        (X[i,j],Y[i,j]), (X[i,j+1],Y[i,j+1]), (X[i+1,j],Y[i+1,j]),
        (X[i+1,j+1],Y[i+1,j+1]).  Ideally the dimensions of X and Y
        should be one greater than those of C; if the dimensions are the
        same, then the last row and column of C will be ignored.

        Note that the the column index corresponds to the x-coordinate,
        and the row index corresponds to y; for details, see
        the "Grid Orientation" section below.

        If either or both of X and Y are 1-D arrays or column vectors,
        they will be expanded as needed into the appropriate 2-D arrays,
        making a rectangular grid.

        X,Y and C may be masked arrays.  If either C[i,j], or one
        of the vertices surrounding C[i,j] (X or Y at [i,j],[i+1,j],
        [i,j+1],[i=1,j+1]) is masked, nothing is plotted.

        Optional keyword args are shown with their defaults below (you must
        use kwargs for these):

          * cmap = cm.jet : a cm Colormap instance from cm

          * norm = Normalize() : mcolors.Normalize instance
            is used to scale luminance data to 0,1.

          * vmin=None and vmax=None : vmin and vmax are used in conjunction
            with norm to normalize luminance data.  If either are None, the
            min and max of the color array C is used.  If you pass a norm
            instance, vmin and vmax will be None

          * shading = 'flat' : or 'faceted'.  If 'faceted', a black grid is
            drawn around each rectangle; if 'flat', edges are not drawn.
            Default is 'flat', contrary to Matlab(TM).
            This kwarg is deprecated;
            please use the edgecolors kwarg instead:
                shading='flat'    --> edgecolors='None'
                shading='faceted  --> edgecolors='k'
            edgecolors can also be None to specify the rcParams
            default, or any mpl color or sequence of colors.

          * alpha=1.0 : the alpha blending value

        Return value is a mcoll.PatchCollection
        object

        Grid Orientation

            The orientation follows the Matlab(TM) convention: an
            array C with shape (nrows, ncolumns) is plotted with
            the column number as X and the row number as Y, increasing
            up; hence it is plotted the way the array would be printed,
            except that the Y axis is reversed.  That is, C is taken
            as C(y,x).

            Similarly for meshgrid:

                x = npy.arange(5)
                y = npy.arange(3)
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
                pcolor(X, Y, C.T)
            or
                pcolor(C.T)

        Dimensions

            Matlab pcolor always discards
            the last row and column of C, but matplotlib displays
            the last row and column if X and Y are not specified, or
            if X and Y have one more row and column than C.


        kwargs can be used to control the PolyCollection properties:
        %(PolyCollection)s
        """

        if not self._hold: self.cla()

        alpha = kwargs.pop('alpha', 1.0)
        norm = kwargs.pop('norm', None)
        cmap = kwargs.pop('cmap', None)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        shading = kwargs.pop('shading', 'flat')

        X, Y, C = self._pcolorargs('pcolor', *args)
        Ny, Nx = X.shape

        # convert to MA, if necessary.
        C = ma.asarray(C)
        X = ma.asarray(X)
        Y = ma.asarray(Y)
        mask = ma.getmaskarray(X)+ma.getmaskarray(Y)
        xymask = mask[0:-1,0:-1]+mask[1:,1:]+mask[0:-1,1:]+mask[1:,0:-1]
        # don't plot if C or any of the surrounding vertices are masked.
        mask = ma.getmaskarray(C)[0:Ny-1,0:Nx-1]+xymask

        newaxis = npy.newaxis
        compress = npy.compress

        ravelmask = (mask==0).ravel()
        X1 = compress(ravelmask, ma.filled(X[0:-1,0:-1]).ravel())
        Y1 = compress(ravelmask, ma.filled(Y[0:-1,0:-1]).ravel())
        X2 = compress(ravelmask, ma.filled(X[1:,0:-1]).ravel())
        Y2 = compress(ravelmask, ma.filled(Y[1:,0:-1]).ravel())
        X3 = compress(ravelmask, ma.filled(X[1:,1:]).ravel())
        Y3 = compress(ravelmask, ma.filled(Y[1:,1:]).ravel())
        X4 = compress(ravelmask, ma.filled(X[0:-1,1:]).ravel())
        Y4 = compress(ravelmask, ma.filled(Y[0:-1,1:]).ravel())
        npoly = len(X1)

        xy = npy.concatenate((X1[:,newaxis], Y1[:,newaxis],
                             X2[:,newaxis], Y2[:,newaxis],
                             X3[:,newaxis], Y3[:,newaxis],
                             X4[:,newaxis], Y4[:,newaxis]),
                             axis=1)
        verts = xy.reshape((npoly, 4, 2))

        #verts = zip(zip(X1,Y1),zip(X2,Y2),zip(X3,Y3),zip(X4,Y4))

        C = compress(ravelmask, ma.filled(C[0:Ny-1,0:Nx-1]).ravel())


        if shading == 'faceted':
            edgecolors =  (0,0,0,1),
        else:
            edgecolors = 'None'
        kwargs.setdefault('edgecolors', edgecolors)
        kwargs.setdefault('antialiaseds', (0,))
        kwargs.setdefault('linewidths', (0.25,))

        collection = mcoll.PolyCollection(verts, **kwargs)

        collection.set_alpha(alpha)
        collection.set_array(C)
        if norm is not None: assert(isinstance(norm, mcolors.Normalize))
        if cmap is not None: assert(isinstance(cmap, mcolors.Colormap))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
        else:
            collection.autoscale_None()
        self.grid(False)

        x = X.compressed()
        y = Y.compressed()
        minx = npy.amin(x)
        maxx = npy.amax(x)
        miny = npy.amin(y)
        maxy = npy.amax(y)

        corners = (minx, miny), (maxx, maxy)
        self.update_datalim( corners)
        self.autoscale_view()
        self.add_collection(collection)
        return collection
    pcolor.__doc__ = cbook.dedent(pcolor.__doc__) % martist.kwdocd

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

          * cmap = cm.jet : a cm Colormap instance from cm.

          * norm = Normalize() : colors.Normalize instance
            is used to scale luminance data to 0,1.  Instantiate it
            with clip=False if C is a masked array.

          * vmin=None and vmax=None : vmin and vmax are used in conjunction
            with norm to normalize luminance data.  If either are None, the
            min and max of the color array C is used.

          * shading = 'flat' : or 'faceted'.  If 'faceted', a black grid is
            drawn around each rectangle; if 'flat', edges are not drawn.
            Default is 'flat', contrary to Matlab(TM).
            This kwarg is deprecated;
            please use the edgecolors kwarg instead:
                shading='flat'    --> edgecolors='None'
                shading='faceted  --> edgecolors='k'
            More flexible specification of edgecolors, as in pcolor,
            is not presently supported.

          * alpha=1.0 : the alpha blending value

        Return value is a collections.PatchCollection
        object

        See pcolor for an explantion of the grid orientation and the
        expansion of 1-D X and/or Y to 2-D arrays.

        kwargs can be used to control the collections.QuadMesh polygon
        collection properties:

        %(QuadMesh)s
        """
        if not self._hold: self.cla()

        alpha = kwargs.pop('alpha', 1.0)
        norm = kwargs.pop('norm', None)
        cmap = kwargs.pop('cmap', None)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        shading = kwargs.pop('shading', 'flat')
        edgecolors = kwargs.pop('edgecolors', 'None')

        X, Y, C = self._pcolorargs('pcolormesh', *args)
        Ny, Nx = X.shape

        # convert to one dimensional arrays
        C = ma.ravel(C[0:Ny-1, 0:Nx-1]) # data point in each cell is value at lower left corner
        X = X.ravel()
        Y = Y.ravel()

        coords = npy.zeros(((Nx * Ny), 2), dtype=float)
        coords[:, 0] = X
        coords[:, 1] = Y

        if shading == 'faceted' or edgecolors != 'None':
            showedges = 1
        else:
            showedges = 0

        collection = mcoll.QuadMesh(
            Nx - 1, Ny - 1, coords, showedges)  # kwargs are not used
        collection.set_alpha(alpha)
        collection.set_array(C)
        if norm is not None: assert(isinstance(norm, mcolors.Normalize))
        if cmap is not None: assert(isinstance(cmap, mcolors.Colormap))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
        else:
            collection.autoscale_None()

        self.grid(False)

        minx = npy.amin(X)
        maxx = npy.amax(X)
        miny = npy.amin(Y)
        maxy = npy.amax(Y)

        corners = (minx, miny), (maxx, maxy)
        self.update_datalim( corners)
        self.autoscale_view()
        self.add_collection(collection)
        return collection
    pcolormesh.__doc__ = cbook.dedent(pcolormesh.__doc__) % martist.kwdocd

    def pcolorfast(self, *args, **kwargs):
        """
        Experimental; this is a version of pcolor that
        does not draw lines, that provides the fastest
        possible rendering with the Agg backend, and that
        can handle any quadrilateral grid.

        pcolor(*args, **kwargs): pseudocolor plot of a 2-D array

        Function signatures

          pcolor(C, **kwargs)
          pcolor(xr, yr, C, **kwargs)
          pcolor(x, y, C, **kwargs)
          pcolor(X, Y, C, **kwargs)

        C is the 2D array of color values corresponding to quadrilateral
        cells. Let (nr, nc) be its shape.  C may be a masked array.

        pcolor(C, **kwargs) is equivalent to
        pcolor([0,nc], [0,nr], C, **kwargs)

        xr, yr specify the ranges of x and y corresponding to the rectangular
        region bounding C.  If xr = [x0, x1] and yr = [y0,y1] then
        x goes from x0 to x1 as the second index of C goes from 0 to nc,
        etc.  (x0, y0) is the outermost corner of cell (0,0), and (x1, y1)
        is the outermost corner of cell (nr-1, nc-1).  All cells are
        rectangles of the same size.  This is the fastest version.

        x, y are 1D arrays of length nc+1 and nr+1, respectively, giving
        the x and y boundaries of the cells.  Hence the cells are
        rectangular but the grid may be nonuniform.  The speed is
        intermediate.  (The grid is checked, and if found to be
        uniform the fast version is used.)

        X and Y are 2D arrays with shape (nr+1, nc+1) that specify
        the (x,y) coordinates of the corners of the colored
        quadrilaterals; the quadrilateral for C[i,j] has corners at
        (X[i,j],Y[i,j]), (X[i,j+1],Y[i,j+1]), (X[i+1,j],Y[i+1,j]),
        (X[i+1,j+1],Y[i+1,j+1]).  The cells need not be rectangular.
        This is the most general, but the slowest to render.  It may
        produce faster and more compact output using ps, pdf, and
        svg backends, however.

        Note that the the column index corresponds to the x-coordinate,
        and the row index corresponds to y; for details, see
        the "Grid Orientation" section below.

        Optional keyword args are shown with their defaults below (you must
        use kwargs for these):

          * cmap = cm.jet : a cm Colormap instance from cm

          * norm = Normalize() : mcolors.Normalize instance
            is used to scale luminance data to 0,1.

          * vmin=None and vmax=None : vmin and vmax are used in conjunction
            with norm to normalize luminance data.  If either are None, the
            min and max of the color array C is used.  If you pass a norm
            instance, vmin and vmax will be None

          * alpha=1.0 : the alpha blending value

        Return value is an image if a regular or rectangular grid
        is specified, and a QuadMesh collection in the general
        quadrilateral case.

        """

        if not self._hold: self.cla()

        alpha = kwargs.pop('alpha', 1.0)
        norm = kwargs.pop('norm', None)
        cmap = kwargs.pop('cmap', None)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        if norm is not None: assert(isinstance(norm, mcolors.Normalize))
        if cmap is not None: assert(isinstance(cmap, mcolors.Colormap))

        C = args[-1]
        nr, nc = C.shape
        if len(args) == 1:
            style = "image"
            x = [0, nc+1]
            y = [0, nr+1]
        elif len(args) == 3:
            x, y = args[:2]
            x = npy.asarray(x)
            y = npy.asarray(y)
            if x.ndim == 1 and y.ndim == 1:
                if x.size == 2 and y.size == 2:
                    style = "image"
                else:
                    dx = npy.diff(x)
                    dy = npy.diff(y)
                    if (npy.ptp(dx) < 0.01*npy.abs(dx.mean()) and
                        npy.ptp(dy) < 0.01*npy.abs(dy.mean())):
                        style = "image"
                    style = "pcolorimage"
            elif x.ndim == 2 and y.ndim == 2:
                style = "quadmesh"
            else:
                raise TypeError("arguments do not match valid signatures")
        else:
            raise TypeError("need 1 argument or 3 arguments")

        if style == "quadmesh":

            # convert to one dimensional arrays
            # This should also be moved to the QuadMesh class
            C = ma.ravel(C) # data point in each cell is value at lower left corner
            X = x.ravel()
            Y = y.ravel()
            Nx = nc+1
            Ny = nr+1

            # The following needs to be cleaned up; the renderer
            # requires separate contiguous arrays for X and Y,
            # but the QuadMesh class requires the 2D array.
            coords = npy.empty(((Nx * Ny), 2), npy.float64)
            coords[:, 0] = X
            coords[:, 1] = Y

            # The QuadMesh class can also be changed to
            # handle relevant superclass kwargs; the initializer
            # should do much more than it does now.
            collection = mcoll.QuadMesh(nc, nr, coords, 0)
            collection.set_alpha(alpha)
            collection.set_array(C)
            collection.set_cmap(cmap)
            collection.set_norm(norm)
            self.add_collection(collection)
            xl, xr, yb, yt = X.min(), X.max(), Y.min(), Y.max()
            ret = collection

        else:
            # One of the image styles:
            xl, xr, yb, yt = x[0], x[-1], y[0], y[-1]
        if style == "image":

            im = mimage.AxesImage(self, cmap, norm,
                                        interpolation='nearest',
                                        origin='lower',
                                        extent=(xl, xr, yb, yt),
                                         **kwargs)
            im.set_data(C)
            im.set_alpha(alpha)
            self.images.append(im)
            ret = im

        if style == "pcolorimage":
            im = mimage.PcolorImage(self, x, y, C,
                                    cmap=cmap,
                                    norm=norm,
                                    alpha=alpha,
                                    **kwargs)
            self.images.append(im)
            ret = im

        self._set_artist_props(ret)
        if vmin is not None or vmax is not None:
            ret.set_clim(vmin, vmax)
        else:
            ret.autoscale_None()
        self.update_datalim(npy.array([[xl, yb], [xr, yt]]))
        self.autoscale_view(tight=True)
        return ret




    def contour(self, *args, **kwargs):
        kwargs['filled'] = False
        return mcontour.ContourSet(self, *args, **kwargs)
    contour.__doc__ = mcontour.ContourSet.contour_doc

    def contourf(self, *args, **kwargs):
        kwargs['filled'] = True
        return mcontour.ContourSet(self, *args, **kwargs)
    contourf.__doc__ = mcontour.ContourSet.contour_doc

    def clabel(self, CS, *args, **kwargs):
        return CS.clabel(*args, **kwargs)
    clabel.__doc__ = mcontour.ContourSet.clabel.__doc__


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

        kwargs control the Table properties:
        %(Table)s
        """
        return mtable.table(self, **kwargs)
    table.__doc__ = cbook.dedent(table.__doc__) % martist.kwdocd

    def twinx(self):
        """
        ax = twinx()

        create a twin of Axes for generating a plot with a sharex
        x-axis but independent y axis.  The y-axis of self will have
        ticks on left and the returned axes will have ticks on the
        right
        """

        ax2 = self.figure.add_axes(self.get_position(), sharex=self, frameon=False)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        self.yaxis.tick_left()
        return ax2

    def twiny(self):
        """
        ax = twiny()

        create a twin of Axes for generating a plot with a shared
        y-axis but independent x axis.  The x-axis of self will have
        ticks on bottom and the returned axes will have ticks on the
        top
        """

        ax2 = self.figure.add_axes(self.get_position(), sharey=self, frameon=False)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        self.xaxis.tick_bottom()
        return ax2


    #### Data analysis


    def hist(self, x, bins=10, normed=0, bottom=None,
             align='edge', orientation='vertical', width=None,
             log=False, **kwargs):
        """
        HIST(x, bins=10, normed=0, bottom=None,
             align='edge', orientation='vertical', width=None,
             log=False, **kwargs)

        Compute the histogram of x.  bins is either an integer number of
        bins or a sequence giving the bins.  x are the data to be binned.

        The return values is (n, bins, patches)

        If normed is true, the first element of the return tuple will
        be the counts normalized to form a probability density, ie,
        n/(len(x)*dbin).  In a probability density, the integral of
        the histogram should be one (we assume equally spaced bins);
        you can verify that with

          # trapezoidal integration of the probability density function
          pdf, bins, patches = ax.hist(...)
          print npy.trapz(pdf, bins)

        align = 'edge' | 'center'.  Interprets bins either as edge
        or center values

        orientation = 'horizontal' | 'vertical'.  If horizontal, barh
        will be used and the "bottom" kwarg will be the left edges.

        width: the width of the bars.  If None, automatically compute
        the width.

        log: if True, the histogram axis will be set to a log scale

        kwargs are used to update the properties of the
        hist Rectangles:
        %(Rectangle)s
        """
        if not self._hold: self.cla()
        n, bins = npy.histogram(x, bins, range=None, normed=normed)
        if width is None: width = 0.9*(bins[1]-bins[0])
        if orientation == 'horizontal':
            patches = self.barh(bins, n, height=width, left=bottom,
                                align=align, log=log)
        elif orientation == 'vertical':
            patches = self.bar(bins, n, width=width, bottom=bottom,
                                align=align, log=log)
        else:
            raise ValueError, 'invalid orientation: %s' % orientation
        for p in patches:
            p.update(kwargs)
        return n, bins, cbook.silent_list('Patch', patches)
    hist.__doc__ = cbook.dedent(hist.__doc__) % martist.kwdocd

    def psd(self, x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, **kwargs):
        """
        PSD(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, **kwargs)

        The power spectral density by Welches average periodogram method.  The
        vector x is divided into NFFT length segments.  Each segment is
        detrended by function detrend and windowed by function window.
        noperlap gives the length of the overlap between segments.  The
        absolute(fft(segment))**2 of each segment are averaged to compute Pxx,
        with a scaling to correct for power loss due to windowing.  Fs is the
        sampling frequency.

            * NFFT is the length of the fft segment; must be a power of 2

            * Fs is the sampling frequency.

            * Fc is the center frequency of x (defaults to 0), which offsets
              the yextents of the image to reflect the frequency range used
              when a signal is acquired and then filtered and downsampled to
              baseband.

            * detrend - the function applied to each segment before fft-ing,
              designed to remove the mean or linear trend.  Unlike in matlab,
              where the detrend parameter is a vector, in matplotlib is it a
              function.  The mlab module defines detrend_none, detrend_mean,
              detrend_linear, but you can use a custom function as well.

            * window - the function used to window the segments.  window is a
              function, unlike in matlab(TM) where it is a vector.  mlab defines
              window_none, window_hanning, but you can use a custom function
              as well.

            * noverlap gives the length of the overlap between segments.

        Returns the tuple Pxx, freqs

        For plotting, the power is plotted as 10*npy.log10(pxx) for decibels,
        though pxx itself is returned

        Refs:

          Bendat & Piersol -- Random Data: Analysis and Measurement
          Procedures, John Wiley & Sons (1986)

        kwargs control the Line2D properties:
        %(Line2D)s
        """
        if not self._hold: self.cla()
        pxx, freqs = mlab.psd(x, NFFT, Fs, detrend, window, noverlap)
        pxx.shape = len(freqs),
        freqs += Fc

        self.plot(freqs, 10*npy.log10(pxx), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Power Spectrum (dB)')
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly().get_bounds()
        intv = vmax-vmin
        logi = int(npy.log10(intv))
        if logi==0: logi=.1
        step = 10*logi
        #print vmin, vmax, step, intv, math.floor(vmin), math.ceil(vmax)+1
        ticks = npy.arange(math.floor(vmin), math.ceil(vmax)+1, step)
        self.set_yticks(ticks)

        return pxx, freqs
    psd.__doc__ = cbook.dedent(psd.__doc__) % martist.kwdocd

    def csd(self, x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, **kwargs):
        """
        CSD(x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
            window=window_hanning, noverlap=0, **kwargs)

        The cross spectral density Pxy by Welches average periodogram method.
        The vectors x and y are divided into NFFT length segments.  Each
        segment is detrended by function detrend and windowed by function
        window.  The product of the direct FFTs of x and y are averaged over
        each segment to compute Pxy, with a scaling to correct for power loss
        due to windowing.

        See the PSD help for a description of the optional parameters.

        Returns the tuple Pxy, freqs.  Pxy is the cross spectrum (complex
        valued), and 10*npy.log10(|Pxy|) is plotted

        Refs:
          Bendat & Piersol -- Random Data: Analysis and Measurement
            Procedures, John Wiley & Sons (1986)

        kwargs control the Line2D properties:
        %(Line2D)s
        """
        if not self._hold: self.cla()
        pxy, freqs = mlab.csd(x, y, NFFT, Fs, detrend, window, noverlap)
        pxy.shape = len(freqs),
        # pxy is complex
        freqs += Fc

        self.plot(freqs, 10*npy.log10(npy.absolute(pxy)), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Cross Spectrum Magnitude (dB)')
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly().get_bounds()

        intv = vmax-vmin
        step = 10*int(npy.log10(intv))

        ticks = npy.arange(math.floor(vmin), math.ceil(vmax)+1, step)
        self.set_yticks(ticks)

        return pxy, freqs
    csd.__doc__ = cbook.dedent(csd.__doc__) % martist.kwdocd

    def cohere(self, x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=0, **kwargs):
        """
        COHERE(x, y, NFFT=256, Fs=2, Fc=0, detrend = mlab.detrend_none,
              window = mlab.window_hanning, noverlap=0, **kwargs)

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

        kwargs control the Line2D properties of the coherence plot:
        %(Line2D)s
        """
        if not self._hold: self.cla()
        cxy, freqs = mlab.cohere(x, y, NFFT, Fs, detrend, window, noverlap)
        freqs += Fc

        self.plot(freqs, cxy, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Coherence')
        self.grid(True)

        return cxy, freqs
    cohere.__doc__ = cbook.dedent(cohere.__doc__) % martist.kwdocd

    def specgram(self, x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
                 window=mlab.window_hanning, noverlap=128,
                 cmap = None, xextent=None):
        """
        SPECGRAM(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
                 window = mlab.window_hanning, noverlap=128,
                 cmap=None, xextent=None)

        Compute a spectrogram of data in x.  Data are split into NFFT length
        segements and the PSD of each section is computed.  The windowing
        function window is applied to each segment, and the amount of overlap
        of each segment is specified with noverlap.

            * cmap is a colormap; if None use default determined by rc

            * xextent is the image extent in the xaxes xextent=xmin, xmax -
              default 0, max(bins), 0, max(freqs) where bins is the return
              value from mlab.specgram

            * See help(psd) for information on the other keyword arguments.

        Return value is (Pxx, freqs, bins, im), where

            bins are the time points the spectrogram is calculated over

            freqs is an array of frequencies

            Pxx is a len(times) x len(freqs) array of power

            im is a image.AxesImage.

        Note: If x is real (i.e. non-complex) only the positive spectrum is
        shown.  If x is complex both positive and negative parts of the
        spectrum are shown.
        """
        if not self._hold: self.cla()

        Pxx, freqs, bins = mlab.specgram(x, NFFT, Fs, detrend,
             window, noverlap)


        Z = 10*npy.log10(Pxx)
        Z =  npy.flipud(Z)

        if xextent is None: xextent = 0, npy.amax(bins)
        xmin, xmax = xextent
        freqs += Fc
        extent = xmin, xmax, freqs[0], freqs[-1]
        im = self.imshow(Z, cmap, extent=extent)
        self.axis('auto')

        return Pxx, freqs, bins, im

    def spy(self, Z, precision=None, marker=None, markersize=None,
                                    aspect='equal', **kwargs):
        """
        spy(Z) plots the sparsity pattern of the 2-D array Z

        If precision is None, any non-zero value will be plotted;
        else, values of absolute(Z)>precision will be plotted.

        The array will be plotted as it would be printed, with
        the first index (row) increasing down and the second
        index (column) increasing to the right.

        By default aspect is 'equal' so that each array element
        occupies a square space; set the aspect kwarg to 'auto'
        to allow the plot to fill the plot box, or to any scalar
        number to specify the aspect ratio of an array element
        directly.

        Two plotting styles are available: image or marker. Both
        are available for full arrays, but only the marker style
        works for scipy.sparse.spmatrix instances.

        If marker and markersize are None, an image will be
        returned and any remaining kwargs are passed to imshow;
        else, a Line2D object will be returned with the value
        of marker determining the marker type, and any remaining
        kwargs passed to the axes plot method.

        If marker and markersize are None, useful kwargs include:
            cmap
            alpha
        See documentation for imshow() for details.
        For controlling colors, e.g. cyan background and red marks, use:
            cmap = mcolors.ListedColormap(['c','r'])

        If marker or markersize is not None, useful kwargs include:
            marker
            markersize
            color
        See documentation for plot() for details.

        Useful values for marker include:
            's'  square (default)
            'o'  circle
            '.'  point
            ','  pixel

        """
        if marker is None and markersize is None:
            if hasattr(Z, 'tocoo'):
                raise TypeError, "Image mode does not support scipy.sparse arrays"
            Z = npy.asarray(Z)
            if precision is None: mask = Z!=0.
            else:                 mask = npy.absolute(Z)>precision

            if 'cmap' not in kwargs:
                kwargs['cmap'] = mcolors.ListedColormap(['w', 'k'], name='binary')
            nr, nc = Z.shape
            extent = [-0.5, nc-0.5, nr-0.5, -0.5]
            ret = self.imshow(mask, interpolation='nearest', aspect=aspect,
                                extent=extent, origin='upper', **kwargs)
        else:
            if hasattr(Z, 'tocoo'):
                c = Z.tocoo()
                y = c.row
                x = c.col
                z = c.data
            else:
                Z = npy.asarray(Z)
                if precision is None: mask = Z!=0.
                else:                 mask = npy.absolute(Z)>precision
                y,x,z = mlab.get_xyz_where(mask, mask)
            if marker is None: marker = 's'
            if markersize is None: markersize = 10
            lines = self.plot(x, y, linestyle='None',
                         marker=marker, markersize=markersize, **kwargs)
            nr, nc = Z.shape
            self.set_xlim(xmin=-0.5, xmax=nc-0.5)
            self.set_ylim(ymin=nr-0.5, ymax=-0.5)
            self.set_aspect(aspect)
            ret = lines
        self.title.set_y(1.05)
        self.xaxis.tick_top()
        self.xaxis.set_ticks_position('both')
        self.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9,
                                                 steps=[1, 2, 5, 10],
                                                 integer=True))
        self.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9,
                                                 steps=[1, 2, 5, 10],
                                                 integer=True))
        return ret

    def matshow(self, Z, **kwargs):
        '''
        Plot a matrix as an image.

        The matrix will be shown the way it would be printed,
        with the first row at the top.  Row and column numbering
        is zero-based.

        Argument:
            Z   anything that can be interpreted as a 2-D array

        kwargs: all are passed to imshow.  matshow sets defaults
        for extent, origin, interpolation, and aspect; use care
        in overriding the extent and origin kwargs, because they
        interact.  (Also, if you want to change them, you probably
        should be using imshow directly in your own version of
        matshow.)

        Returns: an image.AxesImage instance

        '''
        Z = npy.asarray(Z)
        nr, nc = Z.shape
        extent = [-0.5, nc-0.5, nr-0.5, -0.5]
        kw = {'extent': extent,
              'origin': 'upper',
              'interpolation': 'nearest',
              'aspect': 'equal'}          # (already the imshow default)
        kw.update(kwargs)
        im = self.imshow(Z, **kw)
        self.title.set_y(1.05)
        self.xaxis.tick_top()
        self.xaxis.set_ticks_position('both')
        self.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9,
                                                 steps=[1, 2, 5, 10],
                                                 integer=True))
        self.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9,
                                                 steps=[1, 2, 5, 10],
                                                 integer=True))
        return im

class SubplotBase:
    """
    Base class for subplots, which are Axes instances with additional
    methods to facilitate generating and manipulating a set of Axes
    within a figure.
    """

    def __init__(self, fig, *args):
        """
        fig is a figure instance

        args is numRows, numCols, plotNum
            where the array of subplots in the figure has dimensions
            numRows, numCols, and where plotNum is the number of the
            subplot being created.  plotNum starts at 1 in the upper
            right corner and increases to the right.

            If numRows<=numCols<=plotNum<10, args can be the decimal
            integer numRows*100 + numCols*10 + plotNum.
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

    def label_outer(self):
        """
        set the visible property on ticklabels so xticklabels are
        visible only if the subplot is in the last row and yticklabels
        are visible only if the subplot is in the first column
        """
        lastrow = self.is_last_row()
        firstcol = self.is_first_col()
        for label in self.get_xticklabels():
            label.set_visible(lastrow)

        for label in self.get_yticklabels():
            label.set_visible(firstcol)

class Subplot(SubplotBase, Axes):
    """
    subplot class for Cartesian Axes

    This is not normally instantiated by the user; instead,
    use the Figure.add_subplot method.
    """
    def __str__(self):
        return "Subplot(%g,%g)"%(self.bottom.get(),self.left.get())

    def __init__(self, fig, *args, **kwargs):
        """
        See SubplotBase and Axes base class documentation for args and kwargs
        """
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

    RESOLUTION = 100

    def __init__(self, *args, **kwarg):
        """
        See Axes base class for args and kwargs documentation
        """
        Axes.__init__(self, *args, **kwarg)
        self.set_aspect('equal', adjustable='box', anchor='C')
        self.cla()
    def _init_axis(self):
        "nuthin to do"
        self.xaxis = None
        self.yaxis = None


    def _set_lim_and_transforms(self):
        """
        set the dataLim and viewLim BBox attributes and the
        transData and transAxes Transformation attributes
        """

        # the lim are theta, r

        Bbox = mtrans.Bbox
        Value = mtrans.Value
        Point = mtrans.Point
        self.dataLim = Bbox( Point( Value(5/4.*math.pi), Value(math.sqrt(2))),
                             Point( Value(1/4.*math.pi), Value(math.sqrt(2))))
        self.viewLim = Bbox( Point( Value(5/4.*math.pi), Value(math.sqrt(2))),
                             Point( Value(1/4.*math.pi), Value(math.sqrt(2))))

        self.transData = mtrans.NonseparableTransformation(
            self.viewLim, self.bbox,
            mtrans.FuncXY(mtrans.POLAR))
        self.transAxes = mtrans.get_bbox_transform(
            mtrans.unit_bbox(), self.bbox)


    def contains(self,mouseevent):
        """Test whether the mouse event occured in the axes.

        Returns T/F, {}
        """
        if callable(self._contains): return self._contains(self,mouseevent)

        x,y = self.axes.transAxes.inverse_xy_tup((mouseevent.x,mouseevent.y))
        #print "Polar: x,y = ",x,y
        inside = (x-0.5)**2 + (y-0.5)**2 <= 0.25
        return inside,{}

    def cla(self):
        'Clear the current axes'

        # init these w/ some arbitrary numbers - they'll be updated as
        # data is added to the axes

        self._get_lines = _process_plot_var_args(self)
        self._get_patches_for_fill = _process_plot_var_args(self, 'fill')

        self._gridOn = rcParams['polaraxes.grid']
        self.thetagridlabels = []
        self.thetagridlines = []
        self.rgridlabels = []
        self.rgridlines = []

        self.lines = []
        self.images = []
        self.patches = []
        self.artists = []
        self.collections = []
        self.texts = []     # text in axis coords
        self.legend_ = None

        self.grid(self._gridOn)
        props = font_manager.FontProperties(size=rcParams['axes.titlesize'])
        self.title =  mtext.Text(
            x=0.5, y=1.05, text='',
            fontproperties=props,
            verticalalignment='bottom',
            horizontalalignment='center',
            )
        self.title.set_transform(self.transAxes)

        self._set_artist_props(self.title)


        self.thetas = npy.linspace(0, 2*math.pi, self.RESOLUTION)

        verts = zip(self.thetas, npy.ones(self.RESOLUTION))
        self.axesPatch = mpatches.Polygon(
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
        Value = mtrans.Value
        Interval = mtrans.Interval
        self.rintv = Interval(Value(0), Value(1))
        self.rintd = Interval(Value(0), Value(1))

        self.rformatter  = mticker.ScalarFormatter()
        self.rformatter.set_view_interval(self.rintv)
        self.rformatter.set_data_interval(self.rintd)

        class RadialLocator(mticker.AutoLocator):
            'enforce strictly positive radial ticks'

            def __call__(self):
                ticks = mticker.AutoLocator.__call__(self)
                return [t for t in ticks if t>0]

        self.rlocator = RadialLocator()
        self.rlocator.set_view_interval(self.rintv)
        self.rlocator.set_data_interval(self.rintd)


        angles = npy.arange(0, 360, 45)
        radii = npy.arange(0.2, 1.1, 0.2)
        self.set_thetagrids(angles)
        self.set_rgrids(radii)

    def get_children(self):
        'return a list of child artists'
        children = []
        children.extend(self.rgridlines)
        children.extend(self.rgridlabels)
        children.extend(self.thetagridlines)
        children.extend(self.thetagridlabels)
        children.extend(self.lines)
        children.extend(self.patches)
        children.extend(self.texts)
        children.extend(self.artists)
        children.extend(self.images)
        if self.legend_ is not None:
            children.append(self.legend_)
        children.extend(self.collections)
        children.append(self.title)
        children.append(self.axesPatch)
        return children


    def set_rmax(self, rmax):
        self.rintv.set_bounds(0, rmax)
        self.regrid(rmax)

    def grid(self, b):
        'Set the axes grids on or off; b is a boolean'
        self._gridOn = b

    def regrid(self, rmax):
        rmax = float(rmax)
        self.axesPatch.xy = zip(self.thetas, rmax*npy.ones(self.RESOLUTION))

        val = rmax*math.sqrt(2)
        self.viewLim.intervaly().set_bounds(val, val)

        ticks = self.rlocator()
        self.set_rgrids(ticks)
        self.rformatter.set_locs(ticks)

        for t in self.thetagridlabels:
            t.set_y(1.05*rmax)

        r = npy.linspace(0, rmax, self.RESOLUTION)
        for l in self.thetagridlines:
            l.set_ydata(r)

    def autoscale_view(self, scalex=True, scaley=True):
        'set the view limits to include all the data in the axes'
        self.rintd.set_bounds(0, self.get_rmax())
        rmin, rmax = self.rlocator.autoscale()
        self.rintv.set_bounds(rmin, rmax)
        self.regrid(rmax)

    def set_rgrids(self, radii, labels=None, angle=22.5, rpad=0.05, **kwargs):
        """
        set the radial locations and labels of the r grids

        The labels will appear at radial distances radii at angle

        labels, if not None, is a len(radii) list of strings of the
        labels to use at each angle.

        if labels is None, the self.rformatter will be used

        rpad is a fraction of the max of radii which will pad each of
        the radial labels in the radial direction.

        Return value is a list of lines, labels where the lines are
        lines.Line2D instances and the labels are text.Text
        instances

        kwargs control the rgrid Text label properties:
        %(Text)s

        ACCEPTS: sequence of floats
        """


        radii = npy.asarray(radii)
        rmin = radii.min()
        if rmin<=0:
            raise ValueError('radial grids must be strictly positive')

        rpad = rpad * max(radii)
        cbook.popall(self.rgridlines)

        theta = npy.linspace(0., 2*math.pi, self.RESOLUTION)
        ls = rcParams['grid.linestyle']
        color = rcParams['grid.color']
        lw = rcParams['grid.linewidth']

        rmax = self.get_rmax()
        for r in radii:
            r = npy.ones(self.RESOLUTION)*r
            line = mlines.Line2D(theta, r, linestyle=ls, color=color, linewidth=lw,
                          figure=self.figure)
            line.set_transform(self.transData)
            self.rgridlines.append(line)

        cbook.popall(self.rgridlabels)


        color = rcParams['xtick.color']


        props = font_manager.FontProperties(size=rcParams['xtick.labelsize'])
        if labels is None:
            labels = [self.rformatter(r,0) for r in radii]
        for r,l in zip(radii, labels):
            t = mtext.Text(angle/180.*math.pi, r+rpad, l,
                     fontproperties=props, color=color,
                     horizontalalignment='center', verticalalignment='center')
            t.set_transform(self.transData)
            t.update(kwargs)
            self._set_artist_props(t)
            t.set_clip_on(False)
            self.rgridlabels.append(t)

        return self.rgridlines, self.rgridlabels
    set_rgrids.__doc__ = cbook.dedent(set_rgrids.__doc__) % martist.kwdocd

    def set_thetagrids(self, angles, labels=None, fmt='%d', frac = 1.1,
                       **kwargs):
        """
        set the angles at which to place the theta grids (these
        gridlines are equal along the theta dimension).  angles is in
        degrees

        labels, if not None, is a len(angles) list of strings of the
        labels to use at each angle.

        if labels is None, the labels with be fmt%%angle

        frac is the fraction of the polar axes radius at which to
        place the label (1 is the edge).Eg 1.05 isd outside the axes
        and 0.95 is inside the axes

        Return value is a list of lines, labels where the lines are
        lines.Line2D instances and the labels are Text
        instances:

        kwargs are optional text properties for the labels
        %(Text)s
        ACCEPTS: sequence of floats
        """
        cbook.popall(self.thetagridlines)
        ox, oy = 0,0
        ls = rcParams['grid.linestyle']
        color = rcParams['grid.color']
        lw = rcParams['grid.linewidth']

        rmax = self.get_rmax()
        r = npy.linspace(0., rmax, self.RESOLUTION)
        for a in angles:
            theta = npy.ones(self.RESOLUTION)*a/180.*math.pi
            line = mlines.Line2D(
                theta, r, linestyle=ls, color=color, linewidth=lw,
                figure=self.figure)
            line.set_transform(self.transData)
            self.thetagridlines.append(line)

        cbook.popall(self.thetagridlabels)

        color = rcParams['xtick.color']

        props = font_manager.FontProperties(size=rcParams['xtick.labelsize'])
        r = frac*rmax
        if labels is None:
            labels = [fmt%a for a in angles]
        for a,l in zip(angles, labels):
            t = mtext.Text(a/180.*math.pi, r, l, fontproperties=props, color=color,
                     horizontalalignment='center', verticalalignment='center')
            t.set_transform(self.transData)
            t.update(kwargs)
            self._set_artist_props(t)
            t.set_clip_on(False)
            self.thetagridlabels.append(t)
        return self.thetagridlines, self.thetagridlabels
    set_thetagrids.__doc__ = cbook.dedent(set_thetagrids.__doc__) % martist.kwdocd

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

        verts = self.axesPatch.get_verts()
        tverts = self.transData.seq_xy_tups(verts)

        #for i,v,t in zip(range(len(verts)), verts, tverts):
        #    print i,v,t



        l,b,w,h = self.figure.bbox.get_bounds()
        clippath = agg.path_storage()
        for i, xy in enumerate(tverts):
            x,y = xy
            y = h-y
            if i==0: clippath.move_to(x, y)
            else:    clippath.line_to(x, y)
        clippath.close_polygon()

        #self._update_axes()
        if self.axison:
            if self._frameon: self.axesPatch.draw(renderer)

        if self._gridOn:
            for l in self.rgridlines:
                l.set_clip_path(clippath)
                l.draw(renderer)

            for l in self.thetagridlines:
                l.set_clip_path(clippath)
                l.draw(renderer)

        for a in self.lines:# + self.patches:
            a.set_clip_path(clippath)

        artists = []
        artists.extend(self.lines)
        artists.extend(self.texts)
        artists.extend(self.collections)
        artists.extend(self.patches)
        artists.extend(self.artists)

        dsu = [ (a.zorder, a) for a in artists]
        dsu.sort()

        for zorder, a in dsu:
            a.draw(renderer)


        for t in self.thetagridlabels+self.rgridlabels:
            t.draw(renderer)

        if self.legend_ is not None:
            self.legend_.draw(renderer)

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

    def set_xlim(self, xmin=None, xmax=None, emit=True, **kwargs):
        'xlim not implemented'
        raise NotImplementedError('xlim not meaningful for polar axes')

    def set_ylim(self, ymin=None, ymax=None, emit=True, **kwargs):
        'ylim not implemented'
        raise NotImplementedError('ylim not meaningful for polar axes')

    def get_xscale(self):
        'return the xaxis scale string'
        return 'polar'

    def get_yscale(self):
        'return the yaxis scale string'
        return 'polar'

    def toggle_log_lineary(self):
        'toggle between log and linear axes ignored for polar'
        pass


    def table(self, *args, **kwargs):
        """
        TABLE(*args, **kwargs)
        Not implemented for polar axes
        """
        raise NotImplementedError('table not implemented for polar axes')



class PolarSubplot(SubplotBase, PolarAxes):
    """
    subplot class for Polar Axes

    This is not normally instantiated by the user; instead,
    use the Figure.add_subplot(..., polar=True) method.
    """
    def __str__(self):
        return "PolarSubplot(%gx%g)"%(self.figW,self.figH)
    def __init__(self, fig, *args, **kwargs):
        SubplotBase.__init__(self, fig, *args)
        PolarAxes.__init__(
            self, fig,
            [self.figLeft, self.figBottom, self.figW, self.figH], **kwargs)



martist.kwdocd['Axes'] = martist.kwdocd['Subplot'] = martist.kwdoc(Axes)
"""
# this is some discarded code I was using to find the minimum positive
# data point for some log scaling fixes.  I realized there was a
# cleaner way to do it, but am keeping this around as an example for
# how to get the data out of the axes.  Might want to make something
# like this a method one day, or better yet make get_verts an Artist
# method

            minx, maxx = self.get_xlim()
            if minx<=0 or maxx<=0:
                # find the min pos value in the data
                xs = []
                for line in self.lines:
                    xs.extend(line.get_xdata(orig=False))
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


