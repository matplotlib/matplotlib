from __future__ import division, generators
import math, sys, warnings, datetime, new

import numpy as np
from numpy import ma

import matplotlib
rcParams = matplotlib.rcParams

import matplotlib.artist as martist
import matplotlib.axis as maxis
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.quiver as mquiver
import matplotlib.scale as mscale
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

iterable = cbook.iterable
is_string_like = cbook.is_string_like


def _process_plot_format(fmt):
    """
    Process a matlab(TM) style color/line style format string.  Return a
    (*linestyle*, *color*) tuple as a result of the processing.  Default
    values are ('-', 'b').  Example format strings include:

    * 'ko': black circles
    * '.b': blue dots
    * 'r--': red dashed lines

    .. seealso::
        :func:`~matplotlib.Line2D.lineStyles` and
        :func:`~matplotlib.pyplot.colors`:
            for all possible styles and color format string.
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
        if c in mlines.lineStyles:
            if linestyle is not None:
                raise ValueError(
                    'Illegal format string "%s"; two linestyle symbols' % fmt)
            linestyle = c
        elif c in mlines.lineMarkers:
            if marker is not None:
                raise ValueError(
                    'Illegal format string "%s"; two marker symbols' % fmt)
            marker = c
        elif c in mcolors.colorConverter.colors:
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

def set_default_color_cycle(clist):
    """
    Change the default cycle of colors that will be used by the plot
    command.  This must be called before creating the
    :class:`Axes` to which it will apply; it will
    apply to all future axes.

    *clist* is a sequence of mpl color specifiers

    """
    _process_plot_var_args.defaultColors = clist[:]
    rcParams['lines.color'] = clist[0]

class _process_plot_var_args:
    """

    Process variable length arguments to the plot command, so that
    plot commands like the following are supported::

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of *x*, *y*, *fmt* are allowed
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

    def set_color_cycle(self, clist):
        self.colors = clist[:]
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
            if b: return np.arange(len(y)), y, False

        if not ma.isMaskedArray(y):
            y = np.asarray(y)
        if len(y.shape) == 1:
            y = y[:,np.newaxis]
        nr, nc = y.shape
        x = np.arange(nr)
        if len(x.shape) == 1:
            x = x[:,np.newaxis]
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
            x = x[:,np.newaxis]
        if len(y.shape) == 1:
            y = y[:,np.newaxis]
        nrx, ncx = x.shape
        nry, ncy = y.shape
        assert nrx == nry, 'Dimensions of x and y are incompatible'
        if ncx == ncy:
            return x, y, True
        if ncx == 1:
            x = np.repeat(x, ncy, axis=1)
        if ncy == 1:
            y = np.repeat(y, ncx, axis=1)
        assert x.shape == y.shape, 'Dimensions of x and y are incompatible'
        return x, y, True


    def _plot_1_arg(self, y, **kwargs):
        assert self.command == 'plot', 'fill needs at least 2 arguments'
        ret = []

        x, y, multicol = self._xy_from_y(y)

        if multicol:
            for j in xrange(y.shape[1]):
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

            assert self.command == 'plot', ('fill needs at least 2 non-string '
                                            'arguments')
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
                for j in xrange(y.shape[1]):
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
                x = self.axes.convert_xunits(x)
                y = self.axes.convert_yunits(y)
                facecolor = self._get_next_cycle_color()
                seg = mpatches.Polygon(np.hstack(
                                    (x[:,np.newaxis],y[:,np.newaxis])),
                              facecolor = facecolor,
                              fill=True,
                              closed=closed
                              )
                self.set_patchprops(seg, **kwargs)
                ret.append(seg)

            if self.command == 'plot':
                func = makeline
            else:
                closed = kwargs.get('closed', True)
                func = makefill
            if multicol:
                for j in xrange(y.shape[1]):
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
            x = self.axes.convert_xunits(x)
            y = self.axes.convert_yunits(y)
            seg = mpatches.Polygon(np.hstack(
                                    (x[:,np.newaxis],y[:,np.newaxis])),
                          facecolor = facecolor,
                          fill=True,
                          closed=closed
                          )
            self.set_patchprops(seg, **kwargs)
            ret.append(seg)

        if self.command == 'plot':
            func = makeline
        else:
            closed = kwargs.get('closed', True)
            func = makefill

        if multicol:
            for j in xrange(y.shape[1]):
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


class Axes(martist.Artist):
    """
    The :class:`Axes` contains most of the figure elements:
    :class:`~matplotlib.axis.Axis`, :class:`~matplotlib.axis.Tick`,
    :class:`~matplotlib.lines.Line2D`, :class:`~matplotlib.text.Text`,
    :class:`~matplotlib.patches.Polygon`, etc., and sets the
    coordinate system.

    The :class:`Axes` instance supports callbacks through a callbacks
    attribute which is a :class:`~matplotlib.cbook.CallbackRegistry`
    instance.  The events you can connect to are 'xlim_changed' and
    'ylim_changed' and the callback will be called with func(*ax*)
    where *ax* is the :class:`Axes` instance.
    """
    name = "rectilinear"

    _shared_x_axes = cbook.Grouper()
    _shared_y_axes = cbook.Grouper()

    def __str__(self):
        return "Axes(%g,%g;%gx%g)" % tuple(self._position.bounds)
    def __init__(self, fig, rect,
                 axisbg = None, # defaults to rc axes.facecolor
                 frameon = True,
                 sharex=None, # use Axes instance's xaxis info
                 sharey=None, # use Axes instance's yaxis info
                 label='',
                 **kwargs
                 ):
        """
        Build an :class:`Axes` instance in
        :class:`~matplotlib.figure.Figure` *fig* with
        *rect=[left, bottom, width, height]* in
        :class:`~matplotlib.figure.Figure` coordinates

        Optional keyword arguments:

          ================   =========================================
          Keyword            Description
          ================   =========================================
          *adjustable*       [ 'box' | 'datalim' ]
          *alpha*            float: the alpha transparency
          *anchor*           [ 'C', 'SW', 'S', 'SE', 'E', 'NE', 'N',
                               'NW', 'W' ]
          *aspect*           [ 'auto' | 'equal' | aspect_ratio ]
          *autoscale_on*     [ *True* | *False* ] whether or not to
                             autoscale the *viewlim*
          *axis_bgcolor*     any matplotlib color, see
                             :func:`~matplotlib.pyplot.colors`
          *axisbelow*        draw the grids and ticks below the other
                             artists
          *cursor_props*     a (*float*, *color*) tuple
          *figure*           a :class:`~matplotlib.figure.Figure`
                             instance
          *frame_on*         a boolean - draw the axes frame
          *label*            the axes label
          *navigate*         [ *True* | *False* ]
          *navigate_mode*    [ 'PAN' | 'ZOOM' | None ] the navigation
                             toolbar button status
          *position*         [left, bottom, width, height] in
                             class:`~matplotlib.figure.Figure` coords
          *sharex*           an class:`~matplotlib.axes.Axes` instance
                             to share the x-axis with
          *sharey*           an class:`~matplotlib.axes.Axes` instance
                             to share the y-axis with
          *title*            the title string
          *visible*          [ *True* | *False* ] whether the axes is
                             visible
          *xlabel*           the xlabel
          *xlim*             (*xmin*, *xmax*) view limits
          *xscale*           [%(scale)s]
          *xticklabels*      sequence of strings
          *xticks*           sequence of floats
          *ylabel*           the ylabel strings
          *ylim*             (*ymin*, *ymax*) view limits
          *yscale*           [%(scale)s]
          *yticklabels*      sequence of strings
          *yticks*           sequence of floats
          ================   =========================================
        """ % {'scale': ' | '.join([repr(x) for x in mscale.get_scale_names()])}
        martist.Artist.__init__(self)
        if isinstance(rect, mtransforms.Bbox):
            self._position = rect
        else:
            self._position = mtransforms.Bbox.from_bounds(*rect)
        self._originalPosition = self._position.frozen()
        self.set_axes(self)
        self.set_aspect('auto')
        self._adjustable = 'box'
        self.set_anchor('C')
        self._sharex = sharex
        self._sharey = sharey
        if sharex is not None:
            self._shared_x_axes.join(self, sharex)
            if sharex._adjustable == 'box':
                sharex._adjustable = 'datalim'
                #warnings.warn(
                #    'shared axes: "adjustable" is being changed to "datalim"')
            self._adjustable = 'datalim'
        if sharey is not None:
            self._shared_y_axes.join(self, sharey)
            if sharey._adjustable == 'box':
                sharey._adjustable = 'datalim'
                #warnings.warn(
                #    'shared axes: "adjustable" is being changed to "datalim"')
            self._adjustable = 'datalim'
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
            self._xcid = self.xaxis.callbacks.connect('units finalize',
                                                      self.relim)

        if self.yaxis is not None:
            self._ycid = self.yaxis.callbacks.connect('units finalize',
                                                      self.relim)

    def get_window_extent(self, *args, **kwargs):
        '''
        get the axes bounding box in display space; *args* and
        *kwargs* are empty
        '''
        return self.bbox

    def _init_axis(self):
        "move this out of __init__ because non-separable axes don't use it"
        self.xaxis = maxis.XAxis(self)
        self.yaxis = maxis.YAxis(self)
        self._update_transScale()

    def set_figure(self, fig):
        """
        Set the class:`~matplotlib.axes.Axes` figure

        accepts a class:`~matplotlib.figure.Figure` instance
        """
        martist.Artist.set_figure(self, fig)

        self.bbox = mtransforms.TransformedBbox(self._position, fig.transFigure)
        #these will be updated later as data is added
        self.dataLim = mtransforms.Bbox.unit()
        self.viewLim = mtransforms.Bbox.unit()
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        self._set_lim_and_transforms()

    def _set_lim_and_transforms(self):
        """
        set the *dataLim* and *viewLim*
        :class:`~matplotlib.transforms.Bbox` attributes and the
        *transScale*, *transData*, *transLimits* and *transAxes*
        transformations.
        """
        self.transAxes = mtransforms.BboxTransformTo(self.bbox)

        # Transforms the x and y axis separately by a scale factor
        # It is assumed that this part will have non-linear components
        self.transScale = mtransforms.TransformWrapper(
            mtransforms.IdentityTransform())

        # An affine transformation on the data, generally to limit the
        # range of the axes
        self.transLimits = mtransforms.BboxTransformFrom(
            mtransforms.TransformedBbox(self.viewLim, self.transScale))

        # The parentheses are important for efficiency here -- they
        # group the last two (which are usually affines) separately
        # from the first (which, with log-scaling can be non-affine).
        self.transData = self.transScale + (self.transLimits + self.transAxes)

        self._xaxis_transform = mtransforms.blended_transform_factory(
                self.axes.transData, self.axes.transAxes)
        self._yaxis_transform = mtransforms.blended_transform_factory(
                self.axes.transAxes, self.axes.transData)

    def get_xaxis_transform(self):
        """
        Get the transformation used for drawing x-axis labels, ticks
        and gridlines.  The x-direction is in data coordinates and the
        y-direction is in axis coordinates.

        .. note::
            This transformation is primarily used by the
            :class:`~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.
        """
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad_points):
        """
        Get the transformation used for drawing x-axis labels, which
        will add the given amount of padding (in points) between the
        axes and the label.  The x-direction is in data coordinates
        and the y-direction is in axis coordinates.  Returns a
        3-tuple of the form::

          (transform, valign, halign)

        where *valign* and *halign* are requested alignments for the
        text.

        .. note::
            This transformation is primarily used by the
            :class:`~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.
        """
        return (self._xaxis_transform +
                mtransforms.ScaledTranslation(0, -1 * pad_points / 72.0,
                                              self.figure.dpi_scale_trans),
                "top", "center")

    def get_xaxis_text2_transform(self, pad_points):
        """
        Get the transformation used for drawing the secondary x-axis
        labels, which will add the given amount of padding (in points)
        between the axes and the label.  The x-direction is in data
        coordinates and the y-direction is in axis coordinates.
        Returns a 3-tuple of the form::

          (transform, valign, halign)

        where *valign* and *halign* are requested alignments for the
        text.

        .. note::
            This transformation is primarily used by the
            :class:`~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.
        """
        return (self._xaxis_transform +
                mtransforms.ScaledTranslation(0, pad_points / 72.0,
                                              self.figure.dpi_scale_trans),
                "bottom", "center")

    def get_yaxis_transform(self):
        """
        Get the transformation used for drawing y-axis labels, ticks
        and gridlines.  The x-direction is in axis coordinates and the
        y-direction is in data coordinates.

        .. note::
            This transformation is primarily used by the
            :class:`~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.
        """
        return self._yaxis_transform

    def get_yaxis_text1_transform(self, pad_points):
        """
        Get the transformation used for drawing y-axis labels, which
        will add the given amount of padding (in points) between the
        axes and the label.  The x-direction is in axis coordinates
        and the y-direction is in data coordinates.  Returns a 3-tuple
        of the form::

          (transform, valign, halign)

        where *valign* and *halign* are requested alignments for the
        text.

        .. note::
            This transformation is primarily used by the
            :class:`~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.
        """
        return (self._yaxis_transform +
                mtransforms.ScaledTranslation(-1 * pad_points / 72.0, 0,
                                               self.figure.dpi_scale_trans),
                "center", "right")

    def get_yaxis_text2_transform(self, pad_points):
        """
        Get the transformation used for drawing the secondary y-axis
        labels, which will add the given amount of padding (in points)
        between the axes and the label.  The x-direction is in axis
        coordinates and the y-direction is in data coordinates.
        Returns a 3-tuple of the form::

          (transform, valign, halign)

        where *valign* and *halign* are requested alignments for the
        text.

        .. note::

            This transformation is primarily used by the
            :class:`~matplotlib.axis.Axis` class, and is meant to be
            overridden by new kinds of projections that may need to
            place axis elements in different locations.
        """
        return (self._yaxis_transform +
                mtransforms.ScaledTranslation(pad_points / 72.0, 0,
                                               self.figure.dpi_scale_trans),
                "center", "left")

    def _update_transScale(self):
        self.transScale.set(
            mtransforms.blended_transform_factory(
                self.xaxis.get_transform(), self.yaxis.get_transform()))
        if hasattr(self, "lines"):
            for line in self.lines:
                line._transformed_path.invalidate()

    def get_position(self, original=False):
        'Return the a copy of the axes rectangle as a Bbox'
        if original:
            return self._originalPosition.frozen()
        else:
            return self._position.frozen()


    def set_position(self, pos, which='both'):
        """
        Set the axes position with::

          pos = [left, bottom, width, height]

        in relative 0,1 coords, or *pos* can be a
        :class:`~matplotlib.transforms.Bbox`

        There are two position variables: one which is ultimately
        used, but which may be modified by :meth:`apply_aspect`, and a
        second which is the starting point for :meth:`apply_aspect`.


        Optional keyword arguments:
          *which*

            ==========   ====================
            value        description
            ==========   ====================
            'active'     to change the first
            'original'   to change the second
            'both'       to change both
            ==========   ====================

        """
        if not isinstance(pos, mtransforms.BboxBase):
            pos = mtransforms.Bbox.from_bounds(*pos)
        if which in ('both', 'active'):
            self._position.set(pos)
        if which in ('both', 'original'):
            self._originalPosition.set(pos)

    def reset_position(self):
        'Make the original position the active position'
        pos = self.get_position(original=True)
        self.set_position(pos, which='active')

    def _set_artist_props(self, a):
        'set the boilerplate props for artists added to axes'
        a.set_figure(self.figure)
        if not a.is_transform_set():
            a.set_transform(self.transData)

        a.set_axes(self)

    def _gen_axes_patch(self):
        """
        Returns the patch used to draw the background of the axes.  It
        is also used as the clipping path for any data elements on the
        axes.

        In the standard axes, this is a rectangle, but in other
        projections it may not be.

        .. note::
            Intended to be overridden by new projection types.
        """
        return mpatches.Rectangle((0.0, 0.0), 1.0, 1.0)

    def cla(self):
        'Clear the current axes'
        # Note: this is called by Axes.__init__()
        self.xaxis.cla()
        self.yaxis.cla()

        self.ignore_existing_data_limits = True
        self.callbacks = cbook.CallbackRegistry(('xlim_changed',
                                                 'ylim_changed'))

        if self._sharex is not None:
            # major and minor are class instances with
            # locator and formatter attributes
            self.xaxis.major = self._sharex.xaxis.major
            self.xaxis.minor = self._sharex.xaxis.minor
            x0, x1 = self._sharex.get_xlim()
            self.set_xlim(x0, x1, emit=False)
            self.xaxis.set_scale(self._sharex.xaxis.get_scale())
        else:
            self.xaxis.set_scale('linear')

        if self._sharey is not None:
            self.yaxis.major = self._sharey.yaxis.major
            self.yaxis.minor = self._sharey.yaxis.minor
            y0, y1 = self._sharey.get_ylim()
            self.set_ylim(y0, y1, emit=False)
            self.yaxis.set_scale(self._sharey.yaxis.get_scale())
        else:
            self.yaxis.set_scale('linear')

        self._autoscaleon = True
        self._update_transScale()         # needed?

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

        self.grid(self._gridOn)
        props = font_manager.FontProperties(size=rcParams['axes.titlesize'])


        self.titleOffsetTrans = mtransforms.ScaledTranslation(
            0.0, 5.0 / 72.0, self.figure.dpi_scale_trans)
        self.title =  mtext.Text(
            x=0.5, y=1.0, text='',
            fontproperties=props,
            verticalalignment='bottom',
            horizontalalignment='center',
            )
        self.title.set_transform(self.transAxes + self.titleOffsetTrans)
        self.title.set_clip_box(None)

        self._set_artist_props(self.title)

        # the patch draws the background of the axes.  we want this to
        # be below the other artists; the axesPatch name is
        # deprecated.  We use the frame to draw the edges so we are
        # setting the edgecolor to None
        self.patch = self.axesPatch = self._gen_axes_patch()
        self.patch.set_figure(self.figure)
        self.patch.set_facecolor(self._axisbg)
        self.patch.set_edgecolor('None')
        self.patch.set_linewidth(0)
        self.patch.set_transform(self.transAxes)

        # the frame draws the border around the axes and we want this
        # above.  this is a place holder for a more sophisticated
        # artist that might just draw a left, bottom frame, or a
        # centered frame, etc the axesFrame name is deprecated
        self.frame = self.axesFrame = self._gen_axes_patch()
        self.frame.set_figure(self.figure)
        self.frame.set_facecolor('none')
        self.frame.set_edgecolor(rcParams['axes.edgecolor'])
        self.frame.set_linewidth(rcParams['axes.linewidth'])
        self.frame.set_transform(self.transAxes)
        self.frame.set_zorder(2.5)
        self.axison = True

        self.xaxis.set_clip_path(self.patch)
        self.yaxis.set_clip_path(self.patch)

        self._shared_x_axes.clean()
        self._shared_y_axes.clean()

    def clear(self):
        'clear the axes'
        self.cla()

    def set_color_cycle(self, clist):
        """
        Set the color cycle for any future plot commands on this Axes.

        clist is a list of mpl color specifiers.
        """
        self._get_lines.set_color_cycle(clist)


    def ishold(self):
        'return the HOLD status of the axes'
        return self._hold

    def hold(self, b=None):
        """
        call signature::

          hold(b=None)

        Set the hold state.  If *hold* is *None* (default), toggle the
        *hold* state.  Else set the *hold* state to boolean value *b*.

        Examples:

        * toggle hold:
          >>> hold()
        * turn hold on:
          >>> hold(True)
        * turn hold off
          >>> hold(False)


        When hold is True, subsequent plot commands will be added to
        the current axes.  When hold is False, the current axes and
        figure will be cleared on the next plot command

        """
        if b is None:
            self._hold = not self._hold
        else:
            self._hold = b

    def get_aspect(self):
        return self._aspect

    def set_aspect(self, aspect, adjustable=None, anchor=None):
        """
        *aspect*

          ========   ================================================
          value      description
          ========   ================================================
          'auto'     automatic; fill position rectangle with data
          'normal'   same as 'auto'; deprecated
          'equal'    same scaling from data to plot units for x and y
           num       a circle will be stretched such that the height
                     is num times the width. aspect=1 is the same as
                     aspect='equal'.
          ========   ================================================

        *adjustable*

          =========   ============================
          value       description
          =========   ============================
          'box'       change physical size of axes
          'datalim'   change xlim or ylim
          =========   ============================

        *anchor*

          =====   =====================
          value   description
          =====   =====================
          'C'     centered
          'SW'    lower left corner
          'S'     middle of bottom edge
          'SE'    lower right corner
          etc.
          =====   =====================

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
        ACCEPTS: [ 'box' | 'datalim' ]
        """
        if adjustable in ('box', 'datalim'):
            if self in self._shared_x_axes or self in self._shared_y_axes:
                if adjustable == 'box':
                    raise ValueError(
                        'adjustable must be "datalim" for shared axes')
            self._adjustable = adjustable
        else:
            raise ValueError('argument must be "box", or "datalim"')

    def get_anchor(self):
        return self._anchor

    def set_anchor(self, anchor):
        """
        *anchor*

          =====  ============
          value  description
          =====  ============
          'C'    Center
          'SW'   bottom left
          'S'    bottom
          'SE'   bottom right
          'E'    right
          'NE'   top right
          'N'    top
          'NW'   top left
          'W'    left
          =====  ============

        """
        if anchor in mtransforms.Bbox.coefs.keys() or len(anchor) == 2:
            self._anchor = anchor
        else:
            raise ValueError('argument must be among %s' %
                                ', '.join(mtransforms.BBox.coefs.keys()))

    def get_data_ratio(self):
        """
        Returns the aspect ratio of the raw data.

        This method is intended to be overridden by new projection
        types.
        """
        xmin,xmax = self.get_xbound()
        xsize = max(math.fabs(xmax-xmin), 1e-30)
        ymin,ymax = self.get_ybound()
        ysize = max(math.fabs(ymax-ymin), 1e-30)
        return ysize/xsize

    def apply_aspect(self, position=None):
        '''
        Use :meth:`_aspect` and :meth:`_adjustable` to modify the
        axes box or the view limits.
        '''
        if position is None:
            position = self.get_position(original=True)

        aspect = self.get_aspect()
        if aspect == 'auto':
            self.set_position( position , which='active')
            return

        if aspect == 'equal':
            A = 1
        else:
            A = aspect

        #Ensure at drawing time that any Axes involved in axis-sharing
        # does not have its position changed.
        if self in self._shared_x_axes or self in self._shared_y_axes:
            if self._adjustable == 'box':
                self._adjustable = 'datalim'
                warnings.warn(
                    'shared axes: "adjustable" is being changed to "datalim"')

        figW,figH = self.get_figure().get_size_inches()
        fig_aspect = figH/figW
        if self._adjustable == 'box':
            box_aspect = A * self.get_data_ratio()
            pb = position.frozen()
            pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
            self.set_position(pb1.anchored(self.get_anchor(), pb), 'active')
            return

        # reset active to original in case it had been changed
        # by prior use of 'box'
        self.set_position(position, which='active')

        xmin,xmax = self.get_xbound()
        xsize = max(math.fabs(xmax-xmin), 1e-30)
        ymin,ymax = self.get_ybound()
        ysize = max(math.fabs(ymax-ymin), 1e-30)

        l,b,w,h = position.bounds
        box_aspect = fig_aspect * (h/w)
        data_ratio = box_aspect / A

        y_expander = (data_ratio*xsize/ysize - 1.0)
        #print 'y_expander', y_expander
        # If y_expander > 0, the dy/dx viewLim ratio needs to increase
        if abs(y_expander) < 0.005:
            #print 'good enough already'
            return
        dL = self.dataLim
        xr = 1.05 * dL.width
        yr = 1.05 * dL.height
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

        changex = (self in self._shared_y_axes
                   and self not in self._shared_x_axes)
        changey = (self in self._shared_x_axes
                   and self not in self._shared_y_axes)
        if changex and changey:
            warnings.warn("adjustable='datalim' cannot work with shared "
                          "x and y axes")
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

        *kwargs* are passed on to :meth:`set_xlim` and
        :meth:`set_ylim`
        '''
        if len(v)==1 and is_string_like(v[0]):
            s = v[0].lower()
            if s=='on': self.set_axis_on()
            elif s=='off': self.set_axis_off()
            elif s in ('equal', 'tight', 'scaled', 'normal', 'auto', 'image'):
                self.set_autoscale_on(True)
                self.set_aspect('auto')
                self.autoscale_view()
                # self.apply_aspect()
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
                raise ValueError('Unrecognized string %s to axis; '
                                 'try on or off' % s)
            xmin, xmax = self.get_xlim()
            ymin, ymax = self.get_ylim()
            return xmin, xmax, ymin, ymax

        try: v[0]
        except IndexError:
            emit = kwargs.get('emit', True)
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
        Return a list of artists the axes contains.

        .. deprecated:: 0.98
        """
        raise DeprecationWarning('Use get_children instead')

    def get_frame(self):
        'Return the axes Rectangle frame'
        warnings.warn('use ax.patch instead', DeprecationWarning)
        return self.patch

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
        '''Return *True* if any artists have been added to axes.

        This should not be used to determine whether the *dataLim*
        need to be updated, and may not actually be useful for
        anything.
        '''
        return (
            len(self.collections) +
            len(self.images) +
            len(self.lines) +
            len(self.patches))>0

    def add_artist(self, a):
        'Add any :class:`~matplotlib.artist.Artist` to the axes'
        a.set_axes(self)
        self.artists.append(a)
        self._set_artist_props(a)
        a.set_clip_path(self.patch)
        a._remove_method = lambda h: self.artists.remove(h)

    def add_collection(self, collection, autolim=True):
        '''
        add a :class:`~matplotlib.collections.Collection` instance
        to the axes
        '''
        label = collection.get_label()
        if not label:
            collection.set_label('collection%d'%len(self.collections))
        self.collections.append(collection)
        self._set_artist_props(collection)
        collection.set_clip_path(self.patch)
        if autolim:
            if collection._paths and len(collection._paths):
                self.update_datalim(collection.get_datalim(self.transData))

        collection._remove_method = lambda h: self.collections.remove(h)

    def add_line(self, line):
        '''
        Add a :class:`~matplotlib.lines.Line2D` to the list of plot
        lines
        '''
        self._set_artist_props(line)
        line.set_clip_path(self.patch)

        self._update_line_limits(line)
        if not line.get_label():
            line.set_label('_line%d'%len(self.lines))
        self.lines.append(line)
        line._remove_method = lambda h: self.lines.remove(h)

    def _update_line_limits(self, line):
        p = line.get_path()
        if p.vertices.size > 0:
            self.dataLim.update_from_path(p, self.ignore_existing_data_limits,
                                            updatex=line.x_isdata,
                                            updatey=line.y_isdata)
            self.ignore_existing_data_limits = False

    def add_patch(self, p):
        """
        Add a :class:`~matplotlib.patches.Patch` *p* to the list of
        axes patches; the clipbox will be set to the Axes clipping
        box.  If the transform is not set, it will be set to
        :attr:`transData`.
        """

        self._set_artist_props(p)
        p.set_clip_path(self.patch)
        self._update_patch_limits(p)
        self.patches.append(p)
        p._remove_method = lambda h: self.patches.remove(h)

    def _update_patch_limits(self, patch):
        'update the data limits for patch *p*'
        # hist can add zero height Rectangles, which is useful to keep
        # the bins, counts and patches lined up, but it throws off log
        # scaling.  We'll ignore rects with zero height or width in
        # the auto-scaling

        if (isinstance(patch, mpatches.Rectangle) and
                    (patch.get_width()==0 or patch.get_height()==0)):
            return
        vertices = patch.get_path().vertices
        if vertices.size > 0:
            xys = patch.get_patch_transform().transform(vertices)
            if patch.get_data_transform() != self.transData:
                transform = (patch.get_data_transform() +
                                    self.transData.inverted())
                xys = transform.transform(xys)
            self.update_datalim(xys, updatex=patch.x_isdata,
                                     updatey=patch.y_isdata)


    def add_table(self, tab):
        '''
        Add a :class:`~matplotlib.tables.Table` instance to the
        list of axes tables
        '''
        self._set_artist_props(tab)
        self.tables.append(tab)
        tab.set_clip_path(self.patch)
        tab._remove_method = lambda h: self.tables.remove(h)

    def relim(self):
        'recompute the data limits based on current artists'
        # Collections are deliberately not supported (yet); see
        # the TODO note in artists.py.
        self.dataLim.ignore(True)
        self.ignore_existing_data_limits = True
        for line in self.lines:
            self._update_line_limits(line)

        for p in self.patches:
            self._update_patch_limits(p)

    def update_datalim(self, xys, updatex=True, updatey=True):
        'Update the data lim bbox with seq of xy tups or equiv. 2-D array'
        # if no data is set currently, the bbox will ignore its
        # limits and set the bound to be the bounds of the xydata.
        # Otherwise, it will compute the bounds of it's current data
        # and the data in xydata

        if iterable(xys) and not len(xys): return
        if not ma.isMaskedArray(xys):
            xys = np.asarray(xys)
        self.dataLim.update_from_data_xy(xys, self.ignore_existing_data_limits,
                                           updatex=updatex, updatey=updatey)
        self.ignore_existing_data_limits = False

    def update_datalim_numerix(self, x, y):
        'Update the data lim bbox with seq of xy tups'
        # if no data is set currently, the bbox will ignore it's
        # limits and set the bound to be the bounds of the xydata.
        # Otherwise, it will compute the bounds of it's current data
        # and the data in xydata
        if iterable(x) and not len(x): return
        self.dataLim.update_from_data(x, y, self.ignore_existing_data_limits)
        self.ignore_existing_data_limits = False

    def update_datalim_bounds(self, bounds):
        '''
        Update the datalim to include the given
        :class:`~matplotlib.transforms.Bbox` *bounds*
        '''
        self.dataLim.set(mtransforms.Bbox.union([self.dataLim, bounds]))

    def _process_unit_info(self, xdata=None, ydata=None, kwargs=None):
        'look for unit *kwargs* and update the axis instances as necessary'

        if self.xaxis is None or self.yaxis is None: return

        #print 'processing', self.get_geometry()
        if xdata is not None:
            # we only need to update if there is nothing set yet.
            if not self.xaxis.have_units():
               self.xaxis.update_units(xdata)
            #print '\tset from xdata', self.xaxis.units

        if ydata is not None:
            # we only need to update if there is nothing set yet.
            if not self.yaxis.have_units():
               self.yaxis.update_units(ydata)
            #print '\tset from ydata', self.yaxis.units

        # process kwargs 2nd since these will override default units
        if kwargs is not None:
            xunits = kwargs.pop( 'xunits', self.xaxis.units)
            if xunits!=self.xaxis.units:
                #print '\tkw setting xunits', xunits
                self.xaxis.set_units(xunits)
                # If the units being set imply a different converter,
                # we need to update.
                if xdata is not None:
                    self.xaxis.update_units(xdata)

            yunits = kwargs.pop('yunits', self.yaxis.units)
            if yunits!=self.yaxis.units:
                #print '\tkw setting yunits', yunits
                self.yaxis.set_units(yunits)
                # If the units being set imply a different converter,
                # we need to update.
                if ydata is not None:
                    self.yaxis.update_units(ydata)

    def in_axes(self, mouseevent):
        '''
        return *True* if the given *mouseevent* (in display coords)
        is in the Axes
        '''
        return self.patch.contains(mouseevent)[0]

    def get_autoscale_on(self):
        """
        Get whether autoscaling is applied on plot commands
        """
        return self._autoscaleon

    def set_autoscale_on(self, b):
        """
        Set whether autoscaling is applied on plot commands

        accepts: [ *True* | *False* ]
        """
        self._autoscaleon = b

    def autoscale_view(self, tight=False, scalex=True, scaley=True):
        """
        autoscale the view limits using the data limits. You can
        selectively autoscale only a single axis, eg, the xaxis by
        setting *scaley* to *False*.  The autoscaling preserves any
        axis direction reversal that has already been done.
        """
        # if image data only just use the datalim
        if not self._autoscaleon: return
        if scalex:
            xshared = self._shared_x_axes.get_siblings(self)
            dl = [ax.dataLim for ax in xshared]
            bb = mtransforms.BboxBase.union(dl)
            x0, x1 = bb.intervalx
        if scaley:
            yshared = self._shared_y_axes.get_siblings(self)
            dl = [ax.dataLim for ax in yshared]
            bb = mtransforms.BboxBase.union(dl)
            y0, y1 = bb.intervaly
        if (tight or (len(self.images)>0 and
                      len(self.lines)==0 and
                      len(self.patches)==0)):
            if scalex:
                self.set_xbound(x0, x1)
            if scaley:
                self.set_ybound(y0, y1)
            return

        if scalex:
            XL = self.xaxis.get_major_locator().view_limits(x0, x1)
            self.set_xbound(XL)
        if scaley:
            YL = self.yaxis.get_major_locator().view_limits(y0, y1)
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

        # the patch draws the background rectangle -- the frame below
        # will draw the edges
        if self.axison and self._frameon:
            self.patch.draw(renderer)

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


            l, b, r, t = self.bbox.extents
            width = mag*((round(r) + 0.5) - (round(l) - 0.5))
            height = mag*((round(t) + 0.5) - (round(b) - 0.5))
            im = mimage.from_images(height,
                                    width,
                                    ims)

            im.is_grayscale = False
            l, b, w, h = self.bbox.bounds
            # composite images need special args so they will not
            # respect z-order for now
            renderer.draw_image(
                round(l), round(b), im, self.bbox,
                self.patch.get_path(),
                self.patch.get_transform())

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

        # the frame draws the edges around the axes patch -- we
        # decouple these so the patch can be in the background and the
        # frame in the foreground.
        if self.axison and self._frameon:
            artists.append(self.frame)


        dsu = [ (a.zorder, i, a) for i, a in enumerate(artists)
                if not a.get_animated() ]
        dsu.sort()

        for zorder, i, a in dsu:
            a.draw(renderer)

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

        ACCEPTS: [ *True* | *False* ]
        """
        self._frameon = b

    def get_axisbelow(self):
        """
        Get whether axis below is true or not
        """
        return self._axisbelow

    def set_axisbelow(self, b):
        """
        Set whether the axis ticks and gridlines are above or below most artists

        ACCEPTS: [ *True* | *False* ]
        """
        self._axisbelow = b

    def grid(self, b=None, **kwargs):
        """
        call signature::

          grid(self, b=None, **kwargs)

        Set the axes grids on or off; *b* is a boolean

        If *b* is *None* and ``len(kwargs)==0``, toggle the grid state.  If
        *kwargs* are supplied, it is assumed that you want a grid and *b*
        is thus set to *True*

        *kawrgs* are used to set the grid line properties, eg::

          ax.grid(color='r', linestyle='-', linewidth=2)

        Valid :class:`~matplotlib.lines.Line2D` kwargs are

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

        Optional keyword arguments:

          ============   =====================================
          Keyword        Description
          ============   =====================================
          *style*        [ 'sci' (or 'scientific') | 'plain' ]
                         plain turns off scientific notation
          *scilimits*    (m, n), pair of integers; if *style*
                         is 'sci', scientific notation will
                         be used for numbers outside the range
                         10`-m`:sup: to 10`n`:sup:.
                         Use (0,0) to include all numbers.
          *axis*         [ 'x' | 'y' | 'both' ]
          ============   =====================================

        Only the major ticks are affected.
        If the method is called when the
        :class:`~matplotlib.ticker.ScalarFormatter` is not the
        :class:`~matplotlib.ticker.Formatter` being used, an
        :exc:`AttributeError` will be raised.

        """
        style = kwargs.pop('style', '').lower()
        scilimits = kwargs.pop('scilimits', None)
        if scilimits is not None:
            try:
                m, n = scilimits
                m+n+1  # check that both are numbers
            except (ValueError, TypeError):
                raise ValueError("scilimits must be a sequence of 2 integers")
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
        try:
            if sb is not None:
                if axis == 'both' or axis == 'x':
                    self.xaxis.major.formatter.set_scientific(sb)
                if axis == 'both' or axis == 'y':
                    self.yaxis.major.formatter.set_scientific(sb)
            if scilimits is not None:
                if axis == 'both' or axis == 'x':
                    self.xaxis.major.formatter.set_powerlimits(scilimits)
                if axis == 'both' or axis == 'y':
                    self.yaxis.major.formatter.set_powerlimits(scilimits)
        except AttributeError:
            raise AttributeError(
                "This method only works with the ScalarFormatter.")

    def set_axis_off(self):
        """turn off the axis"""
        self.axison = False

    def set_axis_on(self):
        """turn on the axis"""
        self.axison = True

    def get_axis_bgcolor(self):
        'Return the axis background color'
        return self._axisbg

    def set_axis_bgcolor(self, color):
        """
        set the axes background color

        ACCEPTS: any matplotlib color - see
        :func:`~matplotlib.pyplot.colors`
        """

        self._axisbg = color
        self.patch.set_facecolor(color)

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
        """
        Returns the x-axis numerical bounds where::

          lowerBound < upperBound

        """
        left, right = self.get_xlim()
        if left < right:
            return left, right
        else:
            return right, left

    def set_xbound(self, lower=None, upper=None):
        """
        Set the lower and upper numerical bounds of the x-axis.
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
        """
        Get the x-axis range [*xmin*, *xmax*]
        """
        return tuple(self.viewLim.intervalx)

    def set_xlim(self, xmin=None, xmax=None, emit=True, **kwargs):
        """
        call signature::

          set_xlim(self, *args, **kwargs)

        Set the limits for the xaxis

        Returns the current xlimits as a length 2 tuple: [*xmin*, *xmax*]

        Examples::

          set_xlim((valmin, valmax))
          set_xlim(valmin, valmax)
          set_xlim(xmin=1) # xmax unchanged
          set_xlim(xmax=1) # xmin unchanged

        Keyword arguments:

          *ymin*: scalar
            the min of the ylim
          *ymax*: scalar
            the max of the ylim
          *emit*: [ True | False ]
            notify observers of lim change

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

        xmin, xmax = mtransforms.nonsingular(xmin, xmax, increasing=False)
        xmin, xmax = self.xaxis.limit_range_for_scale(xmin, xmax)

        self.viewLim.intervalx = (xmin, xmax)

        if emit:
            self.callbacks.process('xlim_changed', self)
            # Call all of the other x-axes that are shared with this one
            for other in self._shared_x_axes.get_siblings(self):
                if other is not self:
                    other.set_xlim(self.viewLim.intervalx, emit=False)
                    if (other.figure != self.figure and
                        other.figure.canvas is not None):
                        other.figure.canvas.draw_idle()

        return xmin, xmax

    def get_xscale(self):
        'return the xaxis scale string: %s' % (
            ", ".join(mscale.get_scale_names()))
        return self.xaxis.get_scale()

    def set_xscale(self, value, **kwargs):
        """
        call signature::

          set_xscale(value)

        Set the scaling of the x-axis: %(scale)s

        ACCEPTS: [%(scale)s]

        Different kwargs are accepted, depending on the scale:
        %(scale_docs)s
        """
        self.xaxis.set_scale(value, **kwargs)
        self.autoscale_view()
        self._update_transScale()

    set_xscale.__doc__ = cbook.dedent(set_xscale.__doc__) % {
        'scale': ' | '.join([repr(x) for x in mscale.get_scale_names()]),
        'scale_docs': mscale.get_scale_docs().strip()}

    def get_xticks(self, minor=False):
        'Return the x ticks as a list of locations'
        return self.xaxis.get_ticklocs(minor=minor)

    def set_xticks(self, ticks, minor=False):
        """
        Set the x ticks with list of *ticks*

        ACCEPTS: sequence of floats
        """
        return self.xaxis.set_ticks(ticks, minor=minor)

    def get_xmajorticklabels(self):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text xticklabel',
                                 self.xaxis.get_majorticklabels())

    def get_xminorticklabels(self):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text xticklabel',
                                 self.xaxis.get_minorticklabels())

    def get_xticklabels(self, minor=False):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text xticklabel',
                                 self.xaxis.get_ticklabels(minor=minor))

    def set_xticklabels(self, labels, fontdict=None, minor=False, **kwargs):
        """
        call signature::

          set_xticklabels(labels, fontdict=None, minor=False, **kwargs)

        Set the xtick labels with list of strings *labels*. Return a
        list of axis text instances.

        *kwargs* set the :class:`~matplotlib.text.Text` properties.
        Valid properties are
        %(Text)s

        ACCEPTS: sequence of strings
        """
        return self.xaxis.set_ticklabels(labels, fontdict,
                                         minor=minor, **kwargs)
    set_xticklabels.__doc__ = cbook.dedent(
        set_xticklabels.__doc__) % martist.kwdocd

    def invert_yaxis(self):
        "Invert the y-axis."
        left, right = self.get_ylim()
        self.set_ylim(right, left)

    def yaxis_inverted(self):
        'Returns True if the y-axis is inverted.'
        left, right = self.get_ylim()
        return right < left

    def get_ybound(self):
        "Return y-axis numerical bounds in the form of lowerBound < upperBound"
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
        """
        Get the y-axis range [*ymin*, *ymax*]
        """
        return tuple(self.viewLim.intervaly)

    def set_ylim(self, ymin=None, ymax=None, emit=True, **kwargs):
        """
        call signature::

          set_ylim(self, *args, **kwargs):

        Set the limits for the yaxis; v = [ymin, ymax]::

          set_ylim((valmin, valmax))
          set_ylim(valmin, valmax)
          set_ylim(ymin=1) # ymax unchanged
          set_ylim(ymax=1) # ymin unchanged

        Keyword arguments:

          *ymin*: scalar
            the min of the ylim
          *ymax*: scalar
            the max of the ylim
          *emit*: [ True | False ]
            notify observers of lim change

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

        ymin, ymax = mtransforms.nonsingular(ymin, ymax, increasing=False)
        ymin, ymax = self.yaxis.limit_range_for_scale(ymin, ymax)
        self.viewLim.intervaly = (ymin, ymax)

        if emit:
            self.callbacks.process('ylim_changed', self)
            # Call all of the other y-axes that are shared with this one
            for other in self._shared_y_axes.get_siblings(self):
                if other is not self:
                    other.set_ylim(self.viewLim.intervaly, emit=False)
                    if (other.figure != self.figure and
                        other.figure.canvas is not None):
                        other.figure.canvas.draw_idle()
        return ymin, ymax

    def get_yscale(self):
        'return the xaxis scale string: %s' % (
                ", ".join(mscale.get_scale_names()))
        return self.yaxis.get_scale()

    def set_yscale(self, value, **kwargs):
        """
        call signature::

          set_yscale(value)

        Set the scaling of the y-axis: %(scale)s

        ACCEPTS: [%(scale)s]

        Different kwargs are accepted, depending on the scale:
        %(scale_docs)s
        """
        self.yaxis.set_scale(value, **kwargs)
        self.autoscale_view()
        self._update_transScale()

    set_yscale.__doc__ = cbook.dedent(set_yscale.__doc__) % {
        'scale': ' | '.join([repr(x) for x in mscale.get_scale_names()]),
        'scale_docs': mscale.get_scale_docs().strip()}

    def get_yticks(self, minor=False):
        'Return the y ticks as a list of locations'
        return self.yaxis.get_ticklocs(minor=minor)

    def set_yticks(self, ticks, minor=False):
        """
        Set the y ticks with list of *ticks*

        ACCEPTS: sequence of floats

        Keyword arguments:

          *minor*: [ False | True ]
            Sets the minor ticks if True
        """
        return self.yaxis.set_ticks(ticks, minor=minor)

    def get_ymajorticklabels(self):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text yticklabel',
                                 self.yaxis.get_majorticklabels())

    def get_yminorticklabels(self):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text yticklabel',
                                 self.yaxis.get_minorticklabels())

    def get_yticklabels(self, minor=False):
        'Get the xtick labels as a list of Text instances'
        return cbook.silent_list('Text yticklabel',
                                 self.yaxis.get_ticklabels(minor=minor))

    def set_yticklabels(self, labels, fontdict=None, minor=False, **kwargs):
        """
        call signature::

          set_yticklabels(labels, fontdict=None, minor=False, **kwargs)

        Set the ytick labels with list of strings *labels*.  Return a list of
        :class:`~matplotlib.text.Text` instances.

        *kwargs* set :class:`~matplotlib.text.Text` properties for the labels.
        Valid properties are
        %(Text)s

        ACCEPTS: sequence of strings
        """
        return self.yaxis.set_ticklabels(labels, fontdict,
                                         minor=minor, **kwargs)
    set_yticklabels.__doc__ = cbook.dedent(
        set_yticklabels.__doc__) % martist.kwdocd

    def xaxis_date(self, tz=None):
        """Sets up x-axis ticks and labels that treat the x data as dates.

        *tz* is the time zone to use in labeling dates.  Defaults to rc value.
        """

        xmin, xmax = self.dataLim.intervalx
        if xmin==0.:
            # no data has been added - let's set the default datalim.
            # We should probably use a better proxy for the datalim
            # have been updated than the ignore setting
            dmax = today = datetime.date.today()
            dmin = today-datetime.timedelta(days=10)
            self._process_unit_info(xdata=(dmin, dmax))
            dmin, dmax = self.convert_xunits([dmin, dmax])
            self.viewLim.intervalx = dmin, dmax
            self.dataLim.intervalx = dmin, dmax

        locator = self.xaxis.get_major_locator()
        if not isinstance(locator, mdates.DateLocator):
            locator = mdates.AutoDateLocator(tz)
            self.xaxis.set_major_locator(locator)

        # the autolocator uses the viewlim to pick the right date
        # locator, but it may not have correct viewlim before an
        # autoscale.  If the viewlim is still zero..1, set it to the
        # datalim and the autoscaler will update it on request
        if self.viewLim.intervalx[0]==0.:
            self.viewLim.intervalx = tuple(self.dataLim.intervalx)
        locator.refresh()

        formatter = self.xaxis.get_major_formatter()
        if not isinstance(formatter, mdates.DateFormatter):
            formatter = mdates.AutoDateFormatter(locator, tz)
            self.xaxis.set_major_formatter(formatter)

    def yaxis_date(self, tz=None):
        """Sets up y-axis ticks and labels that treat the y data as dates.

        *tz* is the time zone to use in labeling dates.  Defaults to rc value.
        """
        ymin, ymax = self.dataLim.intervaly
        if ymin==0.:
            # no data has been added - let's set the default datalim.
            # We should probably use a better proxy for the datalim
            # have been updated than the ignore setting
            dmax = today = datetime.date.today()
            dmin = today-datetime.timedelta(days=10)
            self._process_unit_info(ydata=(dmin, dmax))

            dmin, dmax = self.convert_yunits([dmin, dmax])
            self.viewLim.intervaly = dmin, dmax
            self.dataLim.intervaly = dmin, dmax


        locator = self.yaxis.get_major_locator()
        if not isinstance(locator, mdates.DateLocator):
            locator = mdates.AutoDateLocator(tz)
            self.yaxis.set_major_locator(locator)

        # the autolocator uses the viewlim to pick the right date
        # locator, but it may not have correct viewlim before an
        # autoscale.  If the viewlim is still zero..1, set it to the
        # datalim and the autoscaler will update it on request
        if self.viewLim.intervaly[0]==0.:
            self.viewLim.intervaly = tuple(self.dataLim.intervaly)
        locator.refresh()

        formatter = self.xaxis.get_major_formatter()
        if not isinstance(formatter, mdates.DateFormatter):
            formatter = mdates.AutoDateFormatter(locator, tz)
            self.yaxis.set_major_formatter(formatter)

    def format_xdata(self, x):
        """
        Return *x* string formatted.  This function will use the attribute
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
        Return y string formatted.  This function will use the
        :attr:`fmt_ydata` attribute if it is callable, else will fall
        back on the yaxis major formatter
        """
        try: return self.fmt_ydata(y)
        except TypeError:
            func = self.yaxis.get_major_formatter().format_data_short
            val =  func(y)
            return val

    def format_coord(self, x, y):
        'return a format string formatting the *x*, *y* coord'
        if x is None:
            x = '???'
        if y is None:
            y = '???'
        xs = self.format_xdata(x)
        ys = self.format_ydata(y)
        return  'x=%s, y=%s'%(xs,ys)

    #### Interactive manipulation

    def can_zoom(self):
        """
        Return *True* if this axes support the zoom box
        """
        return True

    def get_navigate(self):
        """
        Get whether the axes responds to navigation commands
        """
        return self._navigate

    def set_navigate(self, b):
        """
        Set whether the axes responds to navigation toolbar commands

        ACCEPTS: [ True | False ]
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

        .. warning::
            this is not a user-API function.

        """
        self._navigate_mode = b

    def start_pan(self, x, y, button):
        """
        Called when a pan operation has started.

        *x*, *y* are the mouse coordinates in display coords.
        button is the mouse button number:

        * 1: LEFT
        * 2: MIDDLE
        * 3: RIGHT

        .. note::
            Intended to be overridden by new projection types.
        """
        self._pan_start = cbook.Bunch(
            lim           = self.viewLim.frozen(),
            trans         = self.transData.frozen(),
            trans_inverse = self.transData.inverted().frozen(),
            bbox          = self.bbox.frozen(),
            x             = x,
            y             = y
            )

    def end_pan(self):
        """
        Called when a pan operation completes (when the mouse button
        is up.)

        .. note::
            Intended to be overridden by new projection types.
        """
        del self._pan_start

    def drag_pan(self, button, key, x, y):
        """
        Called when the mouse moves during a pan operation.

        *button* is the mouse button number:

        * 1: LEFT
        * 2: MIDDLE
        * 3: RIGHT

        *key* is a "shift" key

        *x*, *y* are the mouse coordinates in display coords.

        .. note::
            Intended to be overridden by new projection types.
        """
        def format_deltas(key, dx, dy):
            if key=='control':
                if(abs(dx)>abs(dy)):
                    dy = dx
                else:
                    dx = dy
            elif key=='x':
                dy = 0
            elif key=='y':
                dx = 0
            elif key=='shift':
                if 2*abs(dx) < abs(dy):
                    dx=0
                elif 2*abs(dy) < abs(dx):
                    dy=0
                elif(abs(dx)>abs(dy)):
                    dy=dy/abs(dy)*abs(dx)
                else:
                    dx=dx/abs(dx)*abs(dy)
            return (dx,dy)

        p = self._pan_start
        dx = x - p.x
        dy = y - p.y
        if dx == 0 and dy == 0:
            return
        if button == 1:
            dx, dy = format_deltas(key, dx, dy)
            result = p.bbox.translated(-dx, -dy) \
                .transformed(p.trans_inverse)
        elif button == 3:
            try:
                dx = -dx / float(self.bbox.width)
                dy = -dy / float(self.bbox.height)
                dx, dy = format_deltas(key, dx, dy)
                if self.get_aspect() != 'auto':
                    dx = 0.5 * (dx + dy)
                    dy = dx

                alpha = np.power(10.0, (dx, dy))
                start = p.trans_inverse.transform_point((p.x, p.y))
                lim_points = p.lim.get_points()
                result = start + alpha * (lim_points - start)
                result = mtransforms.Bbox(result)
            except OverflowError:
                warnings.warn('Overflow while panning')
                return

        self.set_xlim(*result.intervalx)
        self.set_ylim(*result.intervaly)

    def get_cursor_props(self):
        """
        return the cursor propertiess as a (*linewidth*, *color*)
        tuple, where *linewidth* is a float and *color* is an RGBA
        tuple
        """
        return self._cursorProps

    def set_cursor_props(self, *args):
        """
        Set the cursor property as::

          ax.set_cursor_props(linewidth, color)

        or::

          ax.set_cursor_props((linewidth, color))

        ACCEPTS: a (*float*, *color*) tuple
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
        has the following signature::

            func(ax)  # where ax is the instance making the callback.

        The following events can be connected to:

          'xlim_changed','ylim_changed'

        The connection id is is returned - you can use this with
        disconnect to disconnect from the axes event

        """
        raise DeprecationWarning('use the callbacks CallbackRegistry instance '
                                 'instead')

    def disconnect(self, cid):
        'disconnect from the Axes event.'
        raise DeprecationWarning('use the callbacks CallbackRegistry instance '
                                 'instead')

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
        children.append(self.patch)
        children.append(self.frame)
        return children

    def contains(self,mouseevent):
        """Test whether the mouse event occured in the axes.

        Returns T/F, {}
        """
        if callable(self._contains): return self._contains(self,mouseevent)

        return self.patch.contains(mouseevent)

    def pick(self, *args):
        """
        call signature::

            pick(mouseevent)

        each child artist will fire a pick event if mouseevent is over
        the artist and the artist has picker set
        """
        if len(args)>1:
            raise DeprecationWarning('New pick API implemented -- '
                                     'see API_CHANGES in the src distribution')
        martist.Artist.pick(self,args[0])

    def __pick(self, x, y, trans=None, among=None):
        """
        Return the artist under point that is closest to the *x*, *y*.
        If *trans* is *None*, *x*, and *y* are in window coords,
        (0,0 = lower left).  Otherwise, *trans* is a
        :class:`~matplotlib.transforms.Transform` that specifies the
        coordinate system of *x*, *y*.

        The selection of artists from amongst which the pick function
        finds an artist can be narrowed using the optional keyword
        argument *among*. If provided, this should be either a sequence
        of permitted artists or a function taking an artist as its
        argument and returning a true value if and only if that artist
        can be selected.

        Note this algorithm calculates distance to the vertices of the
        polygon, so if you want to pick a patch, click on the edge!
        """
        # MGDTODO: Needs updating
        if trans is not None:
            xywin = trans.transform_point((x,y))
        else:
            xywin = x,y

        def dist_points(p1, p2):
            'return the distance between two points'
            x1, y1 = p1
            x2, y2 = p2
            return math.sqrt((x1-x2)**2+(y1-y2)**2)

        def dist_x_y(p1, x, y):
            '*x* and *y* are arrays; return the distance to the closest point'
            x1, y1 = p1
            return min(np.sqrt((x-x1)**2+(y-y1)**2))

        def dist(a):
            if isinstance(a, Text):
                bbox = a.get_window_extent()
                l,b,w,h = bbox.bounds
                verts = (l,b), (l,b+h), (l+w,b+h), (l+w, b)
                xt, yt = zip(*verts)
            elif isinstance(a, Patch):
                path = a.get_path()
                tverts = a.get_transform().transform_path(path)
                xt, yt = zip(*tverts)
            elif isinstance(a, mlines.Line2D):
                xdata = a.get_xdata(orig=False)
                ydata = a.get_ydata(orig=False)
                xt, yt = a.get_transform().numerix_x_y(xdata, ydata)

            return dist_x_y(xywin, np.asarray(xt), np.asarray(yt))

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

    def get_title(self):
        """
        Get the title text string.
        """
        return self.title.get_text()

    def set_title(self, label, fontdict=None, **kwargs):
        """
        call signature::

          set_title(label, fontdict=None, **kwargs):

        Set the title for the axes.

        kwargs are Text properties:
        %(Text)s

        ACCEPTS: str

        .. seealso::
            :meth:`text`:
                for information on how override and the optional args work
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

    def get_xlabel(self):
        """
        Get the xlabel text string.
        """
        label = self.xaxis.get_label()
        return label.get_text()

    def set_xlabel(self, xlabel, fontdict=None, **kwargs):
        """
        call signature::

          set_xlabel(xlabel, fontdict=None, **kwargs)

        Set the label for the xaxis.

        Valid kwargs are Text properties:
        %(Text)s
        ACCEPTS: str

        .. seealso::
            :meth:`text`:
                for information on how override and the optional args work
        """

        label = self.xaxis.get_label()
        label.set_text(xlabel)
        if fontdict is not None: label.update(fontdict)
        label.update(kwargs)
        return label
    set_xlabel.__doc__ = cbook.dedent(set_xlabel.__doc__) % martist.kwdocd

    def get_ylabel(self):
        """
        Get the ylabel text string.
        """
        label = self.yaxis.get_label()
        return label.get_text()

    def set_ylabel(self, ylabel, fontdict=None, **kwargs):
        """
        call signature::

          set_ylabel(ylabel, fontdict=None, **kwargs)

        Set the label for the yaxis

        Valid kwargs are Text properties:
        %(Text)s
        ACCEPTS: str

        .. seealso::
            :meth:`text`:
                for information on how override and the optional args work
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
        call signature::

          text(x, y, s, fontdict=None, **kwargs)

        Add text in string *s* to axis at location *x*, *y*, data
        coordinates.

        Keyword arguments:

          *fontdict*:
            A dictionary to override the default text properties.
            If *fontdict* is *None*, the defaults are determined by your rc
            parameters.

          *withdash*: [ False | True ]
            Creates a :class:`~matplotlib.text.TextWithDash` instance
            instead of a :class:`~matplotlib.text.Text` instance.

        Individual keyword arguments can be used to override any given
        parameter::

            text(x, y, s, fontsize=12)

        The default transform specifies that text is in data coords,
        alternatively, you can specify text in axis coords (0,0 is
        lower-left and 1,1 is upper-right).  The example below places
        text in the center of the axes::

            text(0.5, 0.5,'matplotlib',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax.transAxes)

       You can put a rectangular box around the text instance (eg. to
       set a background color) by using the keyword *bbox*.  *bbox* is
       a dictionary of :class:`matplotlib.patches.Rectangle`
       properties.  For example::

         text(x, y, s, bbox=dict(facecolor='red', alpha=0.5))

       Valid kwargs are :class:`matplotlib.text.Text` properties:

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
        if 'clip_on' in kwargs:  t.set_clip_box(self.bbox)
        return t
    text.__doc__ = cbook.dedent(text.__doc__) % martist.kwdocd

    def annotate(self, *args, **kwargs):
        """
        call signature::

          annotate(s, xy, xytext=None, xycoords='data',
                   textcoords='data', arrowprops=None, **kwargs)

        Keyword arguments:

        %(Annotation)s

        .. plot:: mpl_examples/pylab_examples/annotation_demo2.py
        """
        a = mtext.Annotation(*args, **kwargs)
        a.set_transform(mtransforms.IdentityTransform())
        self._set_artist_props(a)
        if kwargs.has_key('clip_on'):  a.set_clip_path(self.patch)
        self.texts.append(a)
        return a
    annotate.__doc__ = cbook.dedent(annotate.__doc__) % martist.kwdocd

    #### Lines and spans

    def axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        """
        call signature::

          axhline(y=0, xmin=0, xmax=1, **kwargs)

        Axis Horizontal Line

        Draw a horizontal line at *y* from *xmin* to *xmax*.  With the
        default values of *xmin* = 0 and *xmax* = 1, this line will
        always span the horizontal extent of the axes, regardless of
        the xlim settings, even if you change them, eg. with the
        :meth:`set_xlim` command.  That is, the horizontal extent is
        in axes coords: 0=left, 0.5=middle, 1.0=right but the *y*
        location is in data coordinates.

        Return value is the :class:`~matplotlib.lines.Line2D`
        instance.  kwargs are the same as kwargs to plot, and can be
        used to control the line properties.  Eg.,

        * draw a thick red hline at *y* = 0 that spans the xrange

            >>> axhline(linewidth=4, color='r')

        * draw a default hline at *y* = 1 that spans the xrange

            >>> axhline(y=1)

        * draw a default hline at *y* = .5 that spans the the middle half of
          the xrange

            >>> axhline(y=.5, xmin=0.25, xmax=0.75)

        Valid kwargs are :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        .. seealso::
            :meth:`axhspan`:
                for example plot and source code
        """

        ymin, ymax = self.get_ybound()

        # We need to strip away the units for comparison with
        # non-unitized bounds
        yy = self.convert_yunits( y )
        scaley = (yy<ymin) or (yy>ymax)

        trans = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)
        l = mlines.Line2D([xmin,xmax], [y,y], transform=trans, **kwargs)
        l.x_isdata = False
        self.add_line(l)
        self.autoscale_view(scalex=False, scaley=scaley)
        return l

    axhline.__doc__ = cbook.dedent(axhline.__doc__) % martist.kwdocd

    def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        """
        call signature::

          axvline(x=0, ymin=0, ymax=1, **kwargs)

        Axis Vertical Line

        Draw a vertical line at *x* from *ymin* to *ymax*.  With the
        default values of *ymin* = 0 and *ymax* = 1, this line will
        always span the vertical extent of the axes, regardless of the
        xlim settings, even if you change them, eg. with the
        :meth:`set_xlim` command.  That is, the vertical extent is in
        axes coords: 0=bottom, 0.5=middle, 1.0=top but the *x* location
        is in data coordinates.

        Return value is the :class:`~matplotlib.lines.Line2D`
        instance.  kwargs are the same as kwargs to plot, and can be
        used to control the line properties.  Eg.,

        * draw a thick red vline at *x* = 0 that spans the yrange

            >>> axvline(linewidth=4, color='r')

        * draw a default vline at *x* = 1 that spans the yrange

            >>> axvline(x=1)

        * draw a default vline at *x* = .5 that spans the the middle half of
          the yrange

            >>> axvline(x=.5, ymin=0.25, ymax=0.75)

        Valid kwargs are :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        .. seealso::
            :meth:`axhspan`:
                for example plot and source code
        """

        xmin, xmax = self.get_xbound()

        # We need to strip away the units for comparison with
        # non-unitized bounds
        xx = self.convert_xunits( x )
        scalex = (xx<xmin) or (xx>xmax)

        trans = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)
        l = mlines.Line2D([x,x], [ymin,ymax] , transform=trans, **kwargs)
        l.y_isdata = False
        self.add_line(l)
        self.autoscale_view(scalex=scalex, scaley=False)
        return l

    axvline.__doc__ = cbook.dedent(axvline.__doc__) % martist.kwdocd

    def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kwargs):
        """
        call signature::

          axhspan(ymin, ymax, xmin=0, xmax=1, **kwargs)

        Axis Horizontal Span.

        *y* coords are in data units and *x* coords are in axes (relative
        0-1) units.

        Draw a horizontal span (rectangle) from *ymin* to *ymax*.
        With the default values of *xmin* = 0 and *xmax* = 1, this
        always spans the xrange, regardless of the xlim settings, even
        if you change them, eg. with the :meth:`set_xlim` command.
        That is, the horizontal extent is in axes coords: 0=left,
        0.5=middle, 1.0=right but the *y* location is in data
        coordinates.

        Return value is a :class:`matplotlib.patches.Polygon`
        instance.

        Examples:

        * draw a gray rectangle from *y* = 0.25-0.75 that spans the
          horizontal extent of the axes

            >>> axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)

        Valid kwargs are :class:`~matplotlib.patches.Polygon` properties:

        %(Polygon)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/axhspan_demo.py

        """
        trans = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)

        # process the unit information
        self._process_unit_info( [xmin, xmax], [ymin, ymax], kwargs=kwargs )

        # first we need to strip away the units
        xmin, xmax = self.convert_xunits( [xmin, xmax] )
        ymin, ymax = self.convert_yunits( [ymin, ymax] )

        verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        p.x_isdata = False
        self.add_patch(p)
        return p
    axhspan.__doc__ = cbook.dedent(axhspan.__doc__) % martist.kwdocd

    def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        """
        call signature::

          axvspan(xmin, xmax, ymin=0, ymax=1, **kwargs)

        Axis Vertical Span.

        *x* coords are in data units and *y* coords are in axes (relative
        0-1) units.

        Draw a vertical span (rectangle) from *xmin* to *xmax*.  With
        the default values of *ymin* = 0 and *ymax* = 1, this always
        spans the yrange, regardless of the ylim settings, even if you
        change them, eg. with the :meth:`set_ylim` command.  That is,
        the vertical extent is in axes coords: 0=bottom, 0.5=middle,
        1.0=top but the *y* location is in data coordinates.

        Return value is the :class:`matplotlib.patches.Polygon`
        instance.

        Examples:

        * draw a vertical green translucent rectangle from x=1.25 to 1.55 that
          spans the yrange of the axes

            >>> axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

        Valid kwargs are :class:`~matplotlib.patches.Polygon`
        properties:

        %(Polygon)s

        .. seealso::
            :meth:`axhspan`:
                for example plot and source code
        """
        trans = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)

        # process the unit information
        self._process_unit_info( [xmin, xmax], [ymin, ymax], kwargs=kwargs )

        # first we need to strip away the units
        xmin, xmax = self.convert_xunits( [xmin, xmax] )
        ymin, ymax = self.convert_yunits( [ymin, ymax] )

        verts = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        p.y_isdata = False
        self.add_patch(p)
        return p
    axvspan.__doc__ = cbook.dedent(axvspan.__doc__) % martist.kwdocd


    def hlines(self, y, xmin, xmax, colors='k', linestyles='solid',
                     label='', **kwargs):
        """
        call signature::

          hlines(y, xmin, xmax, colors='k', linestyles='solid', **kwargs)

        Plot horizontal lines at each *y* from *xmin* to *xmax*.

        Returns the :class:`~matplotlib.collections.LineCollection`
        that was added.

        Required arguments:

          *y*:
            a 1-D numpy array or iterable.

          *xmin* and *xmax*:
            can be scalars or ``len(x)`` numpy arrays.  If they are
            scalars, then the respective values are constant, else the
            widths of the lines are determined by *xmin* and *xmax*.

        Optional keyword arguments:

          *colors*:
            a line collections color argument, either a single color
            or a ``len(y)`` list of colors

          *linestyles*:
            [ 'solid' | 'dashed' | 'dashdot' | 'dotted' ]

        **Example:**

        .. plot:: mpl_examples/pylab_examples/hline_demo.py
        """
        if kwargs.get('fmt') is not None:
            raise DeprecationWarning('hlines now uses a '
                                     'collections.LineCollection and not a '
                                     'list of Line2D to draw; see API_CHANGES')

        # We do the conversion first since not all unitized data is uniform
        y = self.convert_yunits( y )
        xmin = self.convert_xunits( xmin )
        xmax = self.convert_xunits( xmax )

        if not iterable(y): y = [y]
        if not iterable(xmin): xmin = [xmin]
        if not iterable(xmax): xmax = [xmax]

        y = np.asarray(y)
        xmin = np.asarray(xmin)
        xmax = np.asarray(xmax)

        if len(xmin)==1:
            xmin = np.resize( xmin, y.shape )
        if len(xmax)==1:
            xmax = np.resize( xmax, y.shape )

        if len(xmin)!=len(y):
            raise ValueError, 'xmin and y are unequal sized sequences'
        if len(xmax)!=len(y):
            raise ValueError, 'xmax and y are unequal sized sequences'

        verts = [ ((thisxmin, thisy), (thisxmax, thisy))
                            for thisxmin, thisxmax, thisy in zip(xmin, xmax, y)]
        coll = mcoll.LineCollection(verts, colors=colors,
                                    linestyles=linestyles, label=label)
        self.add_collection(coll)
        coll.update(kwargs)

        minx = min(xmin.min(), xmax.min())
        maxx = max(xmin.max(), xmax.max())
        miny = y.min()
        maxy = y.max()

        corners = (minx, miny), (maxx, maxy)

        self.update_datalim(corners)
        self.autoscale_view()


        return coll
    hlines.__doc__ = cbook.dedent(hlines.__doc__)

    def vlines(self, x, ymin, ymax, colors='k', linestyles='solid',
                     label='', **kwargs):
        """
        call signature::

          vlines(x, ymin, ymax, color='k', linestyles='solid')

        Plot vertical lines at each *x* from *ymin* to *ymax*.  *ymin*
        or *ymax* can be scalars or len(*x*) numpy arrays.  If they are
        scalars, then the respective values are constant, else the
        heights of the lines are determined by *ymin* and *ymax*.

        *colors*
          a line collections color args, either a single color
          or a len(*x*) list of colors

        *linestyles*

          one of [ 'solid' | 'dashed' | 'dashdot' | 'dotted' ]

        Returns the :class:`matplotlib.collections.LineCollection`
        that was added.

        kwargs are :class:`~matplotlib.collections.LineCollection` properties:

        %(LineCollection)s
        """

        if kwargs.get('fmt') is not None:
            raise DeprecationWarning('vlines now uses a '
                                     'collections.LineCollection and not a '
                                     'list of Line2D to draw; see API_CHANGES')

        self._process_unit_info(xdata=x, ydata=ymin, kwargs=kwargs)

        # We do the conversion first since not all unitized data is uniform
        x = self.convert_xunits( x )
        ymin = self.convert_yunits( ymin )
        ymax = self.convert_yunits( ymax )

        if not iterable(x): x = [x]
        if not iterable(ymin): ymin = [ymin]
        if not iterable(ymax): ymax = [ymax]

        x = np.asarray(x)
        ymin = np.asarray(ymin)
        ymax = np.asarray(ymax)
        if len(ymin)==1:
            ymin = np.resize( ymin, x.shape )
        if len(ymax)==1:
            ymax = np.resize( ymax, x.shape )

        if len(ymin)!=len(x):
            raise ValueError, 'ymin and x are unequal sized sequences'
        if len(ymax)!=len(x):
            raise ValueError, 'ymax and x are unequal sized sequences'

        Y = np.array([ymin, ymax]).T

        verts = [ ((thisx, thisymin), (thisx, thisymax))
                                    for thisx, (thisymin, thisymax) in zip(x,Y)]
        #print 'creating line collection'
        coll = mcoll.LineCollection(verts, colors=colors,
                                    linestyles=linestyles, label=label)
        self.add_collection(coll)
        coll.update(kwargs)

        minx = min( x )
        maxx = max( x )

        miny = min( min(ymin), min(ymax) )
        maxy = max( max(ymin), max(ymax) )

        corners = (minx, miny), (maxx, maxy)
        self.update_datalim(corners)
        self.autoscale_view()

        return coll
    vlines.__doc__ = cbook.dedent(vlines.__doc__) % martist.kwdocd

    #### Basic plotting
    def plot(self, *args, **kwargs):
        """
        Plot lines and/or markers to the
        :class:`~matplotlib.axes.Axes`.  *args* is a variable length
        argument, allowing for multiple *x*, *y* pairs with an
        optional format string.  For example, each of the following is
        legal::

            plot(x, y)         # plot x and y using default line style and color
            plot(x, y, 'bo')   # plot x and y using blue circle markers
            plot(y)            # plot y using x as index array 0..N-1
            plot(y, 'r+')      # ditto, but with red plusses

        If *x* and/or *y* is 2-dimensional, then the corresponding columns
        will be plotted.

        An arbitrary number of *x*, *y*, *fmt* groups can be
        specified, as in::

            a.plot(x1, y1, 'g^', x2, y2, 'g-')

        Return value is a list of lines that were added.

        The following format string characters are accepted to control
        the line style or marker:

        ================    ===============================
        character           description
        ================    ===============================
        '-'                 solid line style
        '--'                dashed line style
        '-.'                dash-dot line style
        ':'                 dotted line style
        '.'                 point marker
        ','                 pixel marker
        'o'                 circle marker
        'v'                 triangle_down marker
        '^'                 triangle_up marker
        '<'                 triangle_left marker
        '>'                 triangle_right marker
        '1'                 tri_down marker
        '2'                 tri_up marker
        '3'                 tri_left marker
        '4'                 tri_right marker
        's'                 square marker
        'p'                 pentagon marker
        '*'                 star marker
        'h'                 hexagon1 marker
        'H'                 hexagon2 marker
        '+'                 plus marker
        'x'                 x marker
        'D'                 diamond marker
        'd'                 thin_diamond marker
        '|'                 vline marker
        '_'                 hline marker
        ================    ===============================


        The following color abbreviations are supported:

        ==========  ========
        character   color
        ==========  ========
        'b'         blue
        'g'         green
        'r'         red
        'c'         cyan
        'm'         magenta
        'y'         yellow
        'k'         black
        'w'         white
        ==========  ========

        In addition, you can specify colors in many weird and
        wonderful ways, including full names (``'green'``), hex
        strings (``'#008000'``), RGB or RGBA tuples (``(0,1,0,1)``) or
        grayscale intensities as a string (``'0.8'``).  Of these, the
        string specifications can be used in place of a ``fmt`` group,
        but the tuple forms can be used only as ``kwargs``.

        Line styles and colors are combined in a single format string, as in
        ``'bo'`` for blue circles.

        The *kwargs* can be used to set line properties (any property that has
        a ``set_*`` method).  You can use this to set a line label (for auto
        legends), linewidth, anitialising, marker face color, etc.  Here is an
        example::

            plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
            plot([1,2,3], [1,4,9], 'rs',  label='line 2')
            axis([0, 4, 0, 10])
            legend()

        If you make multiple lines with one plot command, the kwargs
        apply to all those lines, e.g.::

            plot(x1, y1, x2, y2, antialised=False)

        Neither line will be antialiased.

        You do not need to use format strings, which are just
        abbreviations.  All of the line properties can be controlled
        by keyword arguments.  For example, you can set the color,
        marker, linestyle, and markercolor with::

            plot(x, y, color='green', linestyle='dashed', marker='o',
                 markerfacecolor='blue', markersize=12).  See
                 :class:`~matplotlib.lines.Line2D` for details.

        The kwargs are :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        kwargs *scalex* and *scaley*, if defined, are passed on to
        :meth:`~matplotlib.axes.Axes.autoscale_view` to determine
        whether the *x* and *y* axes are autoscaled; the default is
        *True*.
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
        call signature::

          plot_date(x, y, fmt='bo', tz=None, xdate=True, ydate=False, **kwargs)

        Similar to the :func:`~matplotlib.pyplot.plot` command, except
        the *x* or *y* (or both) data is considered to be dates, and the
        axis is labeled accordingly.

        *x* and/or *y* can be a sequence of dates represented as float
        days since 0001-01-01 UTC.

        Keyword arguments:

          *fmt*: string
            The plot format string.

          *tz*: [ None | timezone string ]
            The time zone to use in labeling dates. If *None*, defaults to rc
            value.

          *xdate*: [ True | False ]
            If *True*, the *x*-axis will be labeled with dates.

          *ydate*: [ False | True ]
            If *True*, the *y*-axis will be labeled with dates.

        Note if you are using custom date tickers and formatters, it
        may be necessary to set the formatters/locators after the call
        to :meth:`plot_date` since :meth:`plot_date` will set the
        default tick locator to
        :class:`matplotlib.ticker.AutoDateLocator` (if the tick
        locator is not already set to a
        :class:`matplotlib.ticker.DateLocator` instance) and the
        default tick formatter to
        :class:`matplotlib.ticker.AutoDateFormatter` (if the tick
        formatter is not already set to a
        :class:`matplotlib.ticker.DateFormatter` instance).

        Valid kwargs are :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        .. seealso::
            :mod:`~matplotlib.dates`:
                for helper functions

            :func:`~matplotlib.dates.date2num`,
            :func:`~matplotlib.dates.num2date` and
            :func:`~matplotlib.dates.drange`:
                for help on creating the required floating point
                dates.
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
        call signature::

          loglog(*args, **kwargs)

        Make a plot with log scaling on the *x* and *y* axis.

        :func:`~matplotlib.pyplot.loglog` supports all the keyword
        arguments of :func:`~matplotlib.pyplot.plot` and
        :meth:`matplotlib.axes.Axes.set_xscale` /
        :meth:`matplotlib.axes.Axes.set_yscale`.

        Notable keyword arguments:

          *basex*/*basey*: scalar > 1
            base of the *x*/*y* logarithm

          *subsx*/*subsy*: [ None | sequence ]
            the location of the minor *x*/*y* ticks; *None* defaults
            to autosubs, which depend on the number of decades in the
            plot; see :meth:`matplotlib.axes.Axes.set_xscale` /
            :meth:`matplotlib.axes.Axes.set_yscale` for details

        The remaining valid kwargs are
        :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/log_demo.py

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
        call signature::

          semilogx(*args, **kwargs)

        Make a plot with log scaling on the *x* axis.

        :func:`semilogx` supports all the keyword arguments of
        :func:`~matplotlib.pyplot.plot` and
        :meth:`matplotlib.axes.Axes.set_xscale`.

        Notable keyword arguments:

          *basex*: scalar > 1
            base of the *x* logarithm

          *subsx*: [ None | sequence ]
            The location of the minor xticks; *None* defaults to
            autosubs, which depend on the number of decades in the
            plot; see :meth:`~matplotlib.axes.Axes.set_xscale` for
            details.

        The remaining valid kwargs are
        :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        .. seealso::
            :meth:`loglog`:
                For example code and figure
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
        call signature::

          semilogy(*args, **kwargs)

        Make a plot with log scaling on the *y* axis.

        :func:`semilogy` supports all the keyword arguments of
        :func:`~matplotlib.pylab.plot` and
        :meth:`matplotlib.axes.Axes.set_yscale`.

        Notable keyword arguments:

          *basey*: scalar > 1
            Base of the *y* logarithm

          *subsy*: [ None | sequence ]
            The location of the minor yticks; *None* defaults to
            autosubs, which depend on the number of decades in the
            plot; see :meth:`~matplotlib.axes.Axes.set_yscale` for
            details.

        The remaining valid kwargs are
        :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        .. seealso::
            :meth:`loglog`:
                For example code and figure
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
        call signature::

            acorr(x, normed=False, detrend=mlab.detrend_none, usevlines=False,
                  maxlags=None, **kwargs)

        Plot the autocorrelation of *x*.  If *normed* = *True*,
        normalize the data by the autocorrelation at 0-th lag.  *x* is
        detrended by the *detrend* callable (default no normalization).

        Data are plotted as ``plot(lags, c, **kwargs)``

        Return value is a tuple (*lags*, *c*, *line*) where:

          - *lags* are a length 2*maxlags+1 lag vector

          - *c* is the 2*maxlags+1 auto correlation vector

          - *line* is a :class:`~matplotlib.lines.Line2D` instance
            returned by :meth:`plot`

        The default *linestyle* is None and the default *marker* is
        ``'o'``, though these can be overridden with keyword args.
        The cross correlation is performed with
        :func:`numpy.correlate` with *mode* = 2.

        If *usevlines* is *True*, :meth:`~matplotlib.axes.Axes.vlines`
        rather than :meth:`~matplotlib.axes.Axes.plot` is used to draw
        vertical lines from the origin to the acorr.  Otherwise, the
        plot style is determined by the kwargs, which are
        :class:`~matplotlib.lines.Line2D` properties.

        *maxlags* is a positive integer detailing the number of lags
        to show.  The default value of *None* will return all
        :math:`2 \mathrm{len}(x) - 1` lags.

        The return value is a tuple (*lags*, *c*, *linecol*, *b*)
        where

        - *linecol* is the
          :class:`~matplotlib.collections.LineCollection`

        - *b* is the *x*-axis.

        .. seealso::
            :meth:`~matplotlib.axes.Axes.plot` or
            :meth:`~matplotlib.axes.Axes.vlines`: For documentation on
            valid kwargs.

        **Example:**

        :func:`~matplotlib.pyplot.xcorr` above, and
        :func:`~matplotlib.pyplot.acorr` below.

        **Example:**

        .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
        """
        return self.xcorr(x, x, **kwargs)
    acorr.__doc__ = cbook.dedent(acorr.__doc__) % martist.kwdocd

    def xcorr(self, x, y, normed=False, detrend=mlab.detrend_none,
              usevlines=False, maxlags=None, **kwargs):
        """
        call signature::

          xcorr(x, y, normed=False, detrend=mlab.detrend_none,
                usevlines=False, **kwargs):

        Plot the cross correlation between *x* and *y*.  If *normed* =
        *True*, normalize the data by the cross correlation at 0-th
        lag.  *x* and y are detrended by the *detrend* callable
        (default no normalization).  *x* and *y* must be equal length.

        Data are plotted as ``plot(lags, c, **kwargs)``

        Return value is a tuple (*lags*, *c*, *line*) where:

          - *lags* are a length ``2*maxlags+1`` lag vector

          - *c* is the ``2*maxlags+1`` auto correlation vector

          - *line* is a :class:`~matplotlib.lines.Line2D` instance
             returned by :func:`~matplotlib.pyplot.plot`.

        The default *linestyle* is *None* and the default *marker* is
        'o', though these can be overridden with keyword args.  The
        cross correlation is performed with :func:`numpy.correlate`
        with *mode* = 2.

        If *usevlines* is *True*:

           :func:`~matplotlib.pyplot.vlines`
           rather than :func:`~matplotlib.pyplot.plot` is used to draw
           vertical lines from the origin to the xcorr.  Otherwise the
           plotstyle is determined by the kwargs, which are
           :class:`~matplotlib.lines.Line2D` properties.

           The return value is a tuple (*lags*, *c*, *linecol*, *b*)
           where *linecol* is the
           :class:`matplotlib.collections.LineCollection` instance and
           *b* is the *x*-axis.

        *maxlags* is a positive integer detailing the number of lags to show.
        The default value of *None* will return all ``(2*len(x)-1)`` lags.

        **Example:**

        :func:`~matplotlib.pyplot.xcorr` above, and
        :func:`~matplotlib.pyplot.acorr` below.

        **Example:**

        .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
        """

        Nx = len(x)
        if Nx!=len(y):
            raise ValueError('x and y must be equal length')

        x = detrend(np.asarray(x))
        y = detrend(np.asarray(y))

        c = np.correlate(x, y, mode=2)

        if normed: c/= np.sqrt(np.dot(x,x) * np.dot(y,y))

        if maxlags is None: maxlags = Nx - 1

        if maxlags >= Nx or maxlags < 1:
            raise ValueError('maglags must be None or strictly '
                             'positive < %d'%Nx)

        lags = np.arange(-maxlags,maxlags+1)
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
        call signature::

          legend(*args, **kwargs)

        Place a legend on the current axes at location *loc*.  Labels are a
        sequence of strings and *loc* can be a string or an integer specifying
        the legend location.

        To make a legend with existing lines::

          legend()

        :meth:`legend` by itself will try and build a legend using the label
        property of the lines/patches/collections.  You can set the label of
        a line by doing::

          plot(x, y, label='my data')

        or::

          line.set_label('my data').

        If label is set to '_nolegend_', the item will not be shown in
        legend.

        To automatically generate the legend from labels::

          legend( ('label1', 'label2', 'label3') )

        To make a legend for a list of lines and labels::

          legend( (line1, line2, line3), ('label1', 'label2', 'label3') )

        To make a legend at a given location, using a location argument::

          legend( ('label1', 'label2', 'label3'), loc='upper left')

        or::

          legend( (line1, line2, line3),  ('label1', 'label2', 'label3'), loc=2)

        The location codes are

          ===============   =============
          Location String   Location Code
          ===============   =============
          'best'            0
          'upper right'     1
          'upper left'      2
          'lower left'      3
          'lower right'     4
          'right'           5
          'center left'     6
          'center right'    7
          'lower center'    8
          'upper center'    9
          'center'          10
          ===============   =============

        If none of these are locations are suitable, loc can be a 2-tuple
        giving x,y in axes coords, ie::

          loc = 0, 1 # left top
          loc = 0.5, 0.5 # center

        Keyword arguments:

          *isaxes*: [ True | False ]
            Indicates that this is an axes legend

          *numpoints*: integer
            The number of points in the legend line, default is 4

          *prop*: [ None | FontProperties ]
            A :class:`matplotlib.font_manager.FontProperties`
            instance, or *None* to use rc settings.

          *pad*: [ None | scalar ]
            The fractional whitespace inside the legend border, between 0 and 1.
            If *None*, use rc settings.

          *markerscale*: [ None | scalar ]
            The relative size of legend markers vs. original. If *None*, use rc
            settings.

          *shadow*: [ None | False | True ]
            If *True*, draw a shadow behind legend. If *None*, use rc settings.

          *labelsep*: [ None | scalar ]
            The vertical space between the legend entries. If *None*, use rc
            settings.

          *handlelen*: [ None | scalar ]
            The length of the legend lines. If *None*, use rc settings.

          *handletextsep*: [ None | scalar ]
            The space between the legend line and legend text. If *None*, use rc
            settings.

          *axespad*: [ None | scalar ]
            The border between the axes and legend edge. If *None*, use rc
            settings.

        **Example:**

        .. plot:: mpl_examples/api/legend_demo.py
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
                if (label is not None and
                    label != '' and not label.startswith('_')):
                    handles.append(handle)
                    labels.append(label)
            if len(handles) == 0:
                warnings.warn("No labeled objects found. "
                              "Use label='...' kwarg on individual plots.")
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
        call signature::

          step(x, y, *args, **kwargs)

        Make a step plot. Additional keyword args to :func:`step` are the same
        as those for :func:`~matplotlib.pyplot.plot`.

        *x* and *y* must be 1-D sequences, and it is assumed, but not checked,
        that *x* is uniformly increasing.

        Keyword arguments:

        *where*: [ 'pre' | 'post' | 'mid'  ]
          If 'pre', the interval from x[i] to x[i+1] has level y[i]

          If 'post', that interval has level y[i+1]

          If 'mid', the jumps in *y* occur half-way between the
          *x*-values.
        '''

        where = kwargs.pop('where', 'pre')
        if where not in ('pre', 'post', 'mid'):
            raise ValueError("'where' argument to step must be "
                             "'pre', 'post' or 'mid'")
        kwargs['linestyle'] = 'steps-' + where

        return self.plot(x, y, *args, **kwargs)


    def bar(self, left, height, width=0.8, bottom=None,
            color=None, edgecolor=None, linewidth=None,
            yerr=None, xerr=None, ecolor=None, capsize=3,
            align='edge', orientation='vertical', log=False,
            **kwargs
            ):
        """
        call signature::

          bar(left, height, width=0.8, bottom=0,
              color=None, edgecolor=None, linewidth=None,
              yerr=None, xerr=None, ecolor=None, capsize=3,
              align='edge', orientation='vertical', log=False)

        Make a bar plot with rectangles bounded by:

          *left*, *left* + *width*, *bottom*, *bottom* + *height*
                (left, right, bottom and top edges)

        *left*, *height*, *width*, and *bottom* can be either scalars
        or sequences

        Return value is a list of
        :class:`matplotlib.patches.Rectangle` instances.

        Required arguments:

          ========   ===============================================
          Argument   Description
          ========   ===============================================
          *left*     the x coordinates of the left sides of the bars
          *height*   the heights of the bars
          ========   ===============================================

        Optional keyword arguments:

          ===============   ==========================================
          Keyword           Description
          ===============   ==========================================
          *width*           the widths of the bars
          *bottom*          the y coordinates of the bottom edges of
                            the bars
          *color*           the colors of the bars
          *edgecolor*       the colors of the bar edges
          *linewidth*       width of bar edges; None means use default
                            linewidth; 0 means don't draw edges.
          *xerr*            if not None, will be used to generate
                            errorbars on the bar chart
          *yerr*            if not None, will be used to generate
                            errorbars on the bar chart
          *ecolor*          specifies the color of any errorbar
          *capsize*         (default 3) determines the length in
                            points of the error bar caps
          *align*           'edge' (default) | 'center'
          *orientation*     'vertical' | 'horizontal'
          *log*             [False|True] False (default) leaves the
                            orientation axis as-is; True sets it to
                            log scale
          ===============   ==========================================

        For vertical bars, *align* = 'edge' aligns bars by their left
        edges in left, while *align* = 'center' interprets these
        values as the *x* coordinates of the bar centers. For
        horizontal bars, *align* = 'edge' aligns bars by their bottom
        edges in bottom, while *align* = 'center' interprets these
        values as the *y* coordinates of the bar centers.

        The optional arguments *color*, *edgecolor*, *linewidth*,
        *xerr*, and *yerr* can be either scalars or sequences of
        length equal to the number of bars.  This enables you to use
        bar as the basis for stacked bar charts, or candlestick plots.

        Other optional kwargs:

        %(Rectangle)s

        **Example:** A stacked bar chart.

        .. plot:: mpl_examples/pylab_examples/bar_stacked.py
        """
        if not self._hold: self.cla()

        label = kwargs.pop('label', '')
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
        #left = np.asarray(left)
        #height = np.asarray(height)
        #width = np.asarray(width)
        #bottom = np.asarray(bottom)

        if len(linewidth) < nbars:
            linewidth *= nbars

        if color is None:
            color = [None] * nbars
        else:
            color = list(mcolors.colorConverter.to_rgba_array(color))
            if len(color) < nbars:
                color *= nbars

        if edgecolor is None:
            edgecolor = [None] * nbars
        else:
            edgecolor = list(mcolors.colorConverter.to_rgba_array(edgecolor))
            if len(edgecolor) < nbars:
                edgecolor *= nbars

        if yerr is not None:
            if not iterable(yerr):
                yerr = [yerr]*nbars

        if xerr is not None:
            if not iterable(xerr):
                xerr = [xerr]*nbars

        # FIXME: convert the following to proper input validation
        # raising ValueError; don't use assert for this.
        assert len(left)==nbars, "argument 'left' must be %d or scalar" % nbars
        assert len(height)==nbars, ("argument 'height' must be %d or scalar" %
                                    nbars)
        assert len(width)==nbars, ("argument 'width' must be %d or scalar" %
                                   nbars)
        assert len(bottom)==nbars, ("argument 'bottom' must be %d or scalar" %
                                    nbars)

        if yerr is not None and len(yerr)!=nbars:
            raise ValueError(
                "bar() argument 'yerr' must be len(%s) or scalar" % nbars)
        if xerr is not None and len(xerr)!=nbars:
            raise ValueError(
                "bar() argument 'xerr' must be len(%s) or scalar" % nbars)

        patches = []

        # lets do some conversions now since some types cannot be
        # subtracted uniformly
        if self.xaxis is not None:
            xconv = self.xaxis.converter
            if xconv is not None:
                units = self.xaxis.get_units()
                left = xconv.convert( left, units )
                width = xconv.convert( width, units )

        if self.yaxis is not None:
            yconv = self.yaxis.converter
            if yconv is not None :
                units = self.yaxis.get_units()
                bottom = yconv.convert( bottom, units )
                height = yconv.convert( height, units )

        if align == 'edge':
            pass
        elif align == 'center':
            if orientation == 'vertical':
                left = [left[i] - width[i]/2. for i in xrange(len(left))]
            elif orientation == 'horizontal':
                bottom = [bottom[i] - height[i]/2. for i in xrange(len(bottom))]

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
                label=label
                )
            label = '_nolegend_'
            r.update(kwargs)
            #print r.get_label(), label, 'label' in kwargs
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
            xmin, xmax = self.dataLim.intervalx
            xmin = np.amin(width[width!=0]) # filter out the 0 width rects
            if xerr is not None:
                xmin = xmin - np.amax(xerr)
            xmin = max(xmin*0.9, 1e-100)
            self.dataLim.intervalx = (xmin, xmax)

        if adjust_ylim:
            ymin, ymax = self.dataLim.intervaly
            ymin = np.amin(height[height!=0]) # filter out the 0 height rects
            if yerr is not None:
                ymin = ymin - np.amax(yerr)
            ymin = max(ymin*0.9, 1e-100)
            self.dataLim.intervaly = (ymin, ymax)
        self.autoscale_view()
        return patches
    bar.__doc__ = cbook.dedent(bar.__doc__) % martist.kwdocd

    def barh(self, bottom, width, height=0.8, left=None, **kwargs):
        """
        call signature::

          barh(bottom, width, height=0.8, left=0, **kwargs)

        Make a horizontal bar plot with rectangles bounded by:

          *left*, *left* + *width*, *bottom*, *bottom* + *height*
                (left, right, bottom and top edges)

        *bottom*, *width*, *height*, and *left* can be either scalars
        or sequences

        Return value is a list of
        :class:`matplotlib.patches.Rectangle` instances.

        Required arguments:

          ========   ======================================================
          Argument   Description
          ========   ======================================================
          *bottom*   the vertical positions of the bottom edges of the bars
          *width*    the lengths of the bars
          ========   ======================================================

        Optional keyword arguments:

          ===============   ==========================================
          Keyword           Description
          ===============   ==========================================
          *height*          the heights (thicknesses) of the bars
          *left*            the x coordinates of the left edges of the
                            bars
          *color*           the colors of the bars
          *edgecolor*       the colors of the bar edges
          *linewidth*       width of bar edges; None means use default
                            linewidth; 0 means don't draw edges.
          *xerr*            if not None, will be used to generate
                            errorbars on the bar chart
          *yerr*            if not None, will be used to generate
                            errorbars on the bar chart
          *ecolor*          specifies the color of any errorbar
          *capsize*         (default 3) determines the length in
                            points of the error bar caps
          *align*           'edge' (default) | 'center'
          *log*             [False|True] False (default) leaves the
                            horizontal axis as-is; True sets it to log
                            scale
          ===============   ==========================================

        Setting *align* = 'edge' aligns bars by their bottom edges in
        bottom, while *align* = 'center' interprets these values as
        the *y* coordinates of the bar centers.

        The optional arguments *color*, *edgecolor*, *linewidth*,
        *xerr*, and *yerr* can be either scalars or sequences of
        length equal to the number of bars.  This enables you to use
        barh as the basis for stacked bar charts, or candlestick
        plots.

        other optional kwargs:

        %(Rectangle)s
        """

        patches = self.bar(left=left, height=height, width=width, bottom=bottom,
                           orientation='horizontal', **kwargs)
        return patches

    barh.__doc__ = cbook.dedent(barh.__doc__) % martist.kwdocd

    def broken_barh(self, xranges, yrange, **kwargs):
        """
        call signature::

          broken_barh(self, xranges, yrange, **kwargs)

        A collection of horizontal bars spanning *yrange* with a sequence of
        *xranges*.

        Required arguments:

          =========   ==============================
          Argument    Description
          =========   ==============================
          *xranges*   sequence of (*xmin*, *xwidth*)
          *yrange*    sequence of (*ymin*, *ywidth*)
          =========   ==============================

        kwargs are
        :class:`matplotlib.collections.BrokenBarHCollection`
        properties:

        %(BrokenBarHCollection)s

        these can either be a single argument, ie::

          facecolors = 'black'

        or a sequence of arguments for the various bars, ie::

          facecolors = ('black', 'red', 'green')

        **Example:**

        .. plot:: mpl_examples/pylab_examples/broken_barh.py
        """
        col = mcoll.BrokenBarHCollection(xranges, yrange, **kwargs)
        self.add_collection(col, autolim=True)
        self.autoscale_view()

        return col

    broken_barh.__doc__ = cbook.dedent(broken_barh.__doc__) % martist.kwdocd

    def stem(self, x, y, linefmt='b-', markerfmt='bo', basefmt='r-'):
        """
        call signature::

          stem(x, y, linefmt='b-', markerfmt='bo', basefmt='r-')

        A stem plot plots vertical lines (using *linefmt*) at each *x*
        location from the baseline to *y*, and places a marker there
        using *markerfmt*.  A horizontal line at 0 is is plotted using
        *basefmt*.

        Return value is a tuple (*markerline*, *stemlines*,
        *baseline*).

        .. seealso::
            `this document`__ for details

            :file:`examples/pylab_examples/stem_plot.py`:
                for a demo

        __ http://www.mathworks.com/access/helpdesk/help/techdoc/ref/stem.html

        """
        remember_hold=self._hold
        if not self._hold: self.cla()
        self.hold(True)

        markerline, = self.plot(x, y, markerfmt)

        stemlines = []
        for thisx, thisy in zip(x, y):
            l, = self.plot([thisx,thisx], [0, thisy], linefmt)
            stemlines.append(l)

        baseline, = self.plot([np.amin(x), np.amax(x)], [0,0], basefmt)

        self.hold(remember_hold)

        return markerline, stemlines, baseline


    def pie(self, x, explode=None, labels=None, colors=None,
            autopct=None, pctdistance=0.6, shadow=False,
            labeldistance=1.1):
        r"""
        call signature::

          pie(x, explode=None, labels=None,
              colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'),
              autopct=None, pctdistance=0.6, labeldistance=1.1, shadow=False)

        Make a pie chart of array *x*.  The fractional area of each
        wedge is given by x/sum(x).  If sum(x) <= 1, then the values
        of x give the fractional area directly and the array will not
        be normalized.

        Keyword arguments:

          *explode*: [ None | len(x) sequence ]
            If not *None*, is a len(*x*) array which specifies the
            fraction of the radius with which to offset each wedge.

          *colors*: [ None | color sequence ]
            A sequence of matplotlib color args through which the pie chart
            will cycle.

          *labels*: [ None | len(x) sequence of strings ]
            A sequence of strings providing the labels for each wedge

          *autopct*: [ None | format string | format function ]
            If not *None*, is a string or function used to label the
            wedges with their numeric value.  The label will be placed inside
            the wedge.  If it is a format string, the label will be ``fmt%pct``.
            If it is a function, it will be called.

          *pctdistance*: scalar
            The ratio between the center of each pie slice and the
            start of the text generated by *autopct*.  Ignored if
            *autopct* is *None*; default is 0.6.

          *labeldistance*: scalar
            The radial distance at which the pie labels are drawn

          *shadow*: [ False | True ]
            Draw a shadow beneath the pie.

        The pie chart will probably look best if the figure and axes are
        square.  Eg.::

          figure(figsize=(8,8))
          ax = axes([0.1, 0.1, 0.8, 0.8])

        Return value:
          If *autopct* is None, return the tuple (*patches*, *texts*):

            - *patches* is a sequence of
              :class:`matplotlib.patches.Wedge` instances

            - *texts* is a list of the label
              :class:`matplotlib.text.Text` instances.

          If *autopct* is not *None*, return the tuple (*patches*,
          *texts*, *autotexts*), where *patches* and *texts* are as
          above, and *autotexts* is a list of
          :class:`~matplotlib.text.Text` instances for the numeric
          labels.
        """
        self.set_frame_on(False)

        x = np.asarray(x).astype(np.float32)

        sx = float(x.sum())
        if sx>1: x = np.divide(x,sx)

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
            label_alignment = xt > 0 and 'left' or 'right'

            t = self.text(xt, yt, label,
                          size=rcParams['xtick.labelsize'],
                          horizontalalignment=label_alignment,
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
                    raise TypeError(
                        'autopct must be callable or a format string')

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
                 fmt='-', ecolor=None, elinewidth=None, capsize=3,
                 barsabove=False, lolims=False, uplims=False,
                 xlolims=False, xuplims=False, **kwargs):
        """
        call signature::

          errorbar(x, y, yerr=None, xerr=None,
                   fmt='-', ecolor=None, elinewidth=None, capsize=3,
                   barsabove=False, lolims=False, uplims=False,
                   xlolims=False, xuplims=False)

        Plot *x* versus *y* with error deltas in *yerr* and *xerr*.
        Vertical errorbars are plotted if *yerr* is not *None*.
        Horizontal errorbars are plotted if *xerr* is not *None*.

        *x*, *y*, *xerr*, and *yerr* can all be scalars, which plots a
        single error bar at *x*, *y*.

        Optional keyword arguments:

          *xerr*/*yerr*: [ scalar | N, Nx1, Nx2 array-like ]
            If a scalar number, len(N) array-like object, or an Nx1 array-like
            object, errorbars are drawn +/- value.

            If a rank-1, Nx2 Numpy array, errorbars are drawn at -column1 and
            +column2

          *fmt*: '-'
            The plot format symbol for *y*. If *fmt* is *None*, just plot the
            errorbars with no line symbols.  This can be useful for creating a
            bar plot with errorbars.

          *ecolor*: [ None | mpl color ]
            a matplotlib color arg which gives the color the errorbar lines; if
            *None*, use the marker color.

          *elinewidth*: scalar
            the linewidth of the errorbar lines. If *None*, use the linewidth.

          *capsize*: scalar
            the size of the error bar caps in points

          *barsabove*: [ True | False ]
            if *True*, will plot the errorbars above the plot
            symbols. Default is below.

          *lolims*/*uplims*/*xlolims*/*xuplims*: [ False | True ]
            These arguments can be used to indicate that a value gives
            only upper/lower limits. In that case a caret symbol is
            used to indicate this. lims-arguments may be of the same
            type as *xerr* and *yerr*.

        All other keyword arguments are passed on to the plot command for the
        markers, so you can add additional key=value pairs to control the
        errorbar markers.  For example, this code makes big red squares with
        thick green edges::

          x,y,yerr = rand(3,10)
          errorbar(x, y, yerr, marker='s',
                   mfc='red', mec='green', ms=20, mew=4)

        where *mfc*, *mec*, *ms* and *mew* are aliases for the longer
        property names, *markerfacecolor*, *markeredgecolor*, *markersize*
        and *markeredgewith*.

        valid kwargs for the marker properties are

        %(Line2D)s

        Return value is a length 3 tuple.  The first element is the
        :class:`~matplotlib.lines.Line2D` instance for the *y* symbol
        lines.  The second element is a list of error bar cap lines,
        the third element is a list of
        :class:`~matplotlib.collections.LineCollection` instances for
        the horizontal and vertical error ranges.

        **Example:**

        .. plot:: mpl_examples/pylab_examples/errorbar_demo.py

        """

        self._process_unit_info(xdata=x, ydata=y, kwargs=kwargs)
        if not self._hold: self.cla()

        # make sure all the args are iterable; use lists not arrays to
        # preserve units
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
        if elinewidth:
            lines_kw['linewidth'] = elinewidth
        else:
            if 'linewidth' in kwargs:
                lines_kw['linewidth']=kwargs['linewidth']
            if 'lw' in kwargs:
                lines_kw['lw']=kwargs['lw']
        if 'transform' in kwargs:
            lines_kw['transform'] = kwargs['transform']

        # arrays fine here, they are booleans and hence not units
        if not iterable(lolims):
            lolims = np.asarray([lolims]*len(x), bool)
        else: lolims = np.asarray(lolims, bool)

        if not iterable(uplims): uplims = np.array([uplims]*len(x), bool)
        else: uplims = np.asarray(uplims, bool)

        if not iterable(xlolims): xlolims = np.array([xlolims]*len(x), bool)
        else: xlolims = np.asarray(xlolims, bool)

        if not iterable(xuplims): xuplims = np.array([xuplims]*len(x), bool)
        else: xuplims = np.asarray(xuplims, bool)

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
            if (iterable(xerr) and len(xerr)==2 and
                iterable(xerr[0]) and iterable(xerr[1])):
                # using list comps rather than arrays to preserve units
                left  = [thisx-thiserr for (thisx, thiserr)
                         in cbook.safezip(x,xerr[0])]
                right  = [thisx+thiserr for (thisx, thiserr)
                          in cbook.safezip(x,xerr[1])]
            else:
                # using list comps rather than arrays to preserve units
                left  = [thisx-thiserr for (thisx, thiserr)
                         in cbook.safezip(x,xerr)]
                right  = [thisx+thiserr for (thisx, thiserr)
                          in cbook.safezip(x,xerr)]

            barcols.append( self.hlines(y, left, right, **lines_kw ) )
            if capsize > 0:
                if xlolims.any():
                    # can't use numpy logical indexing since left and
                    # y are lists
                    leftlo, ylo = xywhere(left, y, xlolims)

                    caplines.extend(
                        self.plot(leftlo, ylo, ls='None',
                                  marker=mlines.CARETLEFT, **plot_kw) )
                    xlolims = ~xlolims
                    leftlo, ylo = xywhere(left, y, xlolims)
                    caplines.extend( self.plot(leftlo, ylo, 'k|', **plot_kw) )
                else:
                    caplines.extend( self.plot(left, y, 'k|', **plot_kw) )

                if xuplims.any():

                    rightup, yup = xywhere(right, y, xuplims)
                    caplines.extend(
                        self.plot(rightup,  yup, ls='None',
                                  marker=mlines.CARETRIGHT, **plot_kw) )
                    xuplims = ~xuplims
                    rightup, yup = xywhere(right, y, xuplims)
                    caplines.extend( self.plot(rightup,  yup, 'k|', **plot_kw) )
                else:
                    caplines.extend( self.plot(right, y, 'k|', **plot_kw) )

        if yerr is not None:
            if (iterable(yerr) and len(yerr)==2 and
                iterable(yerr[0]) and iterable(yerr[1])):
                # using list comps rather than arrays to preserve units
                lower  = [thisy-thiserr for (thisy, thiserr)
                          in cbook.safezip(y,yerr[0])]
                upper  = [thisy+thiserr for (thisy, thiserr)
                          in cbook.safezip(y,yerr[1])]
            else:
                # using list comps rather than arrays to preserve units
                lower  = [thisy-thiserr for (thisy, thiserr)
                          in cbook.safezip(y,yerr)]
                upper  = [thisy+thiserr for (thisy, thiserr)
                          in cbook.safezip(y,yerr)]

            barcols.append( self.vlines(x, lower, upper, **lines_kw) )
            if capsize > 0:

                if lolims.any():
                    xlo, lowerlo = xywhere(x, lower, lolims)
                    caplines.extend(
                        self.plot(xlo, lowerlo, ls='None',
                                  marker=mlines.CARETDOWN, **plot_kw) )
                    lolims = ~lolims
                    xlo, lowerlo = xywhere(x, lower, lolims)
                    caplines.extend( self.plot(xlo, lowerlo, 'k_', **plot_kw) )
                else:
                    caplines.extend( self.plot(x, lower, 'k_', **plot_kw) )

                if uplims.any():
                    xup, upperup = xywhere(x, upper, uplims)

                    caplines.extend(
                        self.plot(xup, upperup, ls='None',
                                  marker=mlines.CARETUP, **plot_kw) )
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
        call signature::

          boxplot(x, notch=0, sym='+', vert=1, whis=1.5,
                  positions=None, widths=None)

        Make a box and whisker plot for each column of *x* or each
        vector in sequence *x*.  The box extends from the lower to
        upper quartile values of the data, with a line at the median.
        The whiskers extend from the box to show the range of the
        data.  Flier points are those past the end of the whiskers.

        - *notch* = 0 (default) produces a rectangular box plot.
        - *notch* = 1 will produce a notched box plot

        *sym* (default 'b+') is the default symbol for flier points.
        Enter an empty string ('') if you don't want to show fliers.

        - *vert* = 1 (default) makes the boxes vertical.
        - *vert* = 0 makes horizontal boxes.  This seems goofy, but
          that's how Matlab did it.

        *whis* (default 1.5) defines the length of the whiskers as
        a function of the inner quartile range.  They extend to the
        most extreme data point within ( ``whis*(75%-25%)`` ) data range.

        *positions* (default 1,2,...,n) sets the horizontal positions of
        the boxes. The ticks and limits are automatically set to match
        the positions.

        *widths* is either a scalar or a vector and sets the width of
        each box. The default is 0.5, or ``0.15*(distance between extreme
        positions)`` if that is smaller.

        *x* is an array or a sequence of vectors.

        Returns a dictionary mapping each component of the boxplot
        to a list of the :class:`matplotlib.lines.Line2D`
        instances created.

        **Example:**

        .. plot:: pyplots/boxplot_demo.py
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
                    x = [x[:,i] for i in xrange(nc)]
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
            widths = np.ones((col,), float) * widths

        # loop through columns, adding each to plot
        self.hold(True)
        for i,pos in enumerate(positions):
            d = np.ravel(x[i])
            row = len(d)
            # get median and quartiles
            q1, med, q3 = mlab.prctile(d,[25,50,75])
            # get high extreme
            iq = q3 - q1
            hi_val = q3 + whis*iq
            wisk_hi = np.compress( d <= hi_val , d )
            if len(wisk_hi) == 0:
                wisk_hi = q3
            else:
                wisk_hi = max(wisk_hi)
            # get low extreme
            lo_val = q1 - whis*iq
            wisk_lo = np.compress( d >= lo_val, d )
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
                flier_hi = np.compress( d > wisk_hi, d )
                flier_lo = np.compress( d < wisk_lo, d )
                flier_hi_x = np.ones(flier_hi.shape[0]) * pos
                flier_lo_x = np.ones(flier_lo.shape[0]) * pos

            # get x locations for fliers, whisker, whisker cap and box sides
            box_x_min = pos - widths[i] * 0.5
            box_x_max = pos + widths[i] * 0.5

            wisk_x = np.ones(2) * pos

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
                notch_max = med + 1.57*iq/np.sqrt(row)
                notch_min = med - 1.57*iq/np.sqrt(row)
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
                    for i in xrange(0, len(args), 3):
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
        call signatures::

          scatter(x, y, s=20, c='b', marker='o', cmap=None, norm=None,
                  vmin=None, vmax=None, alpha=1.0, linewidths=None,
                  verts=None, **kwargs)

        Make a scatter plot of *x* versus *y*, where *x*, *y* are 1-D
        sequences of the same length, *N*.

        Keyword arguments:

          *s*:
            size in points^2.  It is a scalar or an array of the same
            length as *x* and *y*.

          *c*:
            a color. *c* can be a single color format string, or a
            sequence of color specifications of length *N*, or a
            sequence of *N* numbers to be mapped to colors using the
            *cmap* and *norm* specified via kwargs (see below). Note
            that *c* should not be a single numeric RGB or RGBA
            sequence because that is indistinguishable from an array
            of values to be colormapped.  *c* can be a 2-D array in
            which the rows are RGB or RGBA, however.

          *marker*:
            can be one of:

            =====   ==============
            Value   Description
            =====   ==============
            's'     square
            'o'     circle
            '^'     triangle up
            '>'     triangle right
            'v'     triangle down
            '<'     triangle left
            'd'     diamond
            'p'     pentagram
            'h'     hexagon
            '8'     octagon
            '+'     plus
            'x'     cross
            =====   ==============

            The marker can also be a tuple (*numsides*, *style*,
            *angle*), which will create a custom, regular symbol.

              *numsides*:
                the number of sides

              *style*:
                the style of the regular symbol:

                =====   =============================================
                Value   Description
                =====   =============================================
                0       a regular polygon
                1       a star-like symbol
                2       an asterisk
                3       a circle (*numsides* and *angle* is ignored)
                =====   =============================================

              *angle*:
                the angle of rotation of the symbol

            Finally, *marker* can be (*verts*, 0): *verts* is a
            sequence of (*x*, *y*) vertices for a custom scatter
            symbol.  Alternatively, use the kwarg combination
            *marker* = *None*, *verts* = *verts*.

        Any or all of *x*, *y*, *s*, and *c* may be masked arrays, in
        which case all masks will be combined and only unmasked points
        will be plotted.

        Other keyword arguments: the color mapping and normalization
        arguments will be used only if *c* is an array of floats.

          *cmap*: [ None | Colormap ]
            A :class:`matplotlib.colors.Colormap` instance. If *None*,
            defaults to rc ``image.cmap``. *cmap* is only used if *c*
            is an array of floats.

          *norm*: [ None | Normalize ]
            A :class:`matplotlib.colors.Normalize` instance is used to
            scale luminance data to 0, 1. If *None*, use the default
            :func:`normalize`. *norm* is only used if *c* is an array
            of floats.

          *vmin*/*vmax*:
            *vmin* and *vmax* are used in conjunction with norm to
            normalize luminance data.  If either are None, the min and
            max of the color array *C* is used.  Note if you pass a
            *norm* instance, your settings for *vmin* and *vmax* will
            be ignored.

          *alpha*: 0 <= scalar <= 1
            The alpha value for the patches

          *linewidths*: [ None | scalar | sequence ]
            If *None*, defaults to (lines.linewidth,).  Note that this
            is a tuple, and if you set the linewidths argument you
            must set it as a sequence of floats, as required by
            :class:`~matplotlib.collections.RegularPolyCollection`.

        Optional kwargs control the
        :class:`~matplotlib.collections.Collection` properties; in
        particular:

          *edgecolors*:
            'none' to plot faces with no outlines

          *facecolors*:
            'none' to plot unfilled outlines

        Here are the standard descriptions of all the
        :class:`~matplotlib.collections.Collection` kwargs:

        %(Collection)s

        A :class:`~matplotlib.collections.Collection` instance is
        returned.
        """

        if not self._hold: self.cla()

        syms =  { # a dict from symbol to (numsides, angle)
            's' : (4,math.pi/4.0,0),   # square
            'o' : (20,3,0),            # circle
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

        x, y, s, c = cbook.delete_masked_points(x, y, s, c)


        if is_string_like(c) or cbook.is_sequence_of_strings(c):
            colors = mcolors.colorConverter.to_rgba_array(c, alpha)
        else:
            sh = np.shape(c)
            # The inherent ambiguity is resolved in favor of color
            # mapping, not interpretation as rgb or rgba:
            if len(sh) == 1 and sh[0] == len(x):
                colors = None  # use cmap, norm after collection is created
            else:
                colors = mcolors.colorConverter.to_rgba_array(c, alpha)

        if not iterable(s):
            scales = (s,)
        else:
            scales = s

        if faceted:
            edgecolors = None
        else:
            edgecolors = 'none'
            warnings.warn(
                '''replace "faceted=False" with "edgecolors='none'"''',
                DeprecationWarning)   #2008/04/18

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
                verts = np.asarray(marker[0])

        if sym is not None:
            if symstyle==0:
                collection = mcoll.RegularPolyCollection(
                    numsides, rotation, scales,
                    facecolors = colors,
                    edgecolors = edgecolors,
                    linewidths = linewidths,
                    offsets = zip(x,y),
                    transOffset = self.transData,
                    )
            elif symstyle==1:
                collection = mcoll.StarPolygonCollection(
                    numsides, rotation, scales,
                    facecolors = colors,
                    edgecolors = edgecolors,
                    linewidths = linewidths,
                    offsets = zip(x,y),
                    transOffset = self.transData,
                    )
            elif symstyle==2:
                collection = mcoll.AsteriskPolygonCollection(
                    numsides, rotation, scales,
                    facecolors = colors,
                    edgecolors = edgecolors,
                    linewidths = linewidths,
                    offsets = zip(x,y),
                    transOffset = self.transData,
                    )
            elif symstyle==3:
                collection = mcoll.CircleCollection(
                    scales,
                    facecolors = colors,
                    edgecolors = edgecolors,
                    linewidths = linewidths,
                    offsets = zip(x,y),
                    transOffset = self.transData,
                    )
        else:
            rescale = np.sqrt(max(verts[:,0]**2+verts[:,1]**2))
            verts /= rescale

            collection = mcoll.PolyCollection(
                (verts,), scales,
                facecolors = colors,
                edgecolors = edgecolors,
                linewidths = linewidths,
                offsets = zip(x,y),
                transOffset = self.transData,
                )
            collection.set_transform(mtransforms.IdentityTransform())
        collection.set_alpha(alpha)
        collection.update(kwargs)

        if colors is None:
            if norm is not None: assert(isinstance(norm, mcolors.Normalize))
            if cmap is not None: assert(isinstance(cmap, mcolors.Colormap))
            collection.set_array(np.asarray(c))
            collection.set_cmap(cmap)
            collection.set_norm(norm)

            if vmin is not None or vmax is not None:
                collection.set_clim(vmin, vmax)
            else:
                collection.autoscale_None()

        temp_x = x
        temp_y = y

        minx = np.amin(temp_x)
        maxx = np.amax(temp_x)
        miny = np.amin(temp_y)
        maxy = np.amax(temp_y)

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

    def hexbin(self, x, y, C = None, gridsize = 100, bins = None,
                    xscale = 'linear', yscale = 'linear',
                    cmap=None, norm=None, vmin=None, vmax=None,
                    alpha=1.0, linewidths=None, edgecolors='none',
                    reduce_C_function = np.mean,
                    **kwargs):
        """
        call signature::

          hexbin(x, y, C = None, gridsize = 100, bins = None,
                 xscale = 'linear', yscale = 'linear',
                 cmap=None, norm=None, vmin=None, vmax=None,
                 alpha=1.0, linewidths=None, edgecolors='none'
                 reduce_C_function = np.mean,
                 **kwargs)

        Make a hexagonal binning plot of *x* versus *y*, where *x*,
        *y* are 1-D sequences of the same length, *N*. If *C* is None
        (the default), this is a histogram of the number of occurences
        of the observations at (x[i],y[i]).

        If *C* is specified, it specifies values at the coordinate
        (x[i],y[i]). These values are accumulated for each hexagonal
        bin and then reduced according to *reduce_C_function*, which
        defaults to numpy's mean function (np.mean). (If *C* is
        specified, it must also be a 1-D sequence of the same length
        as *x* and *y*.)

        *x*, *y* and/or *C* may be masked arrays, in which case only
        unmasked points will be plotted.

        Optional keyword arguments:

          *gridsize*: [ 100 | integer ]
            The number of hexagons in the *x*-direction, default is
            100. The corresponding number of hexagons in the
            *y*-direction is chosen such that the hexagons are
            approximately regular. Alternatively, gridsize can be a
            tuple with two elements specifying the number of hexagons
            in the *x*-direction and the *y*-direction.

          *bins*: [ None | 'log' | integer | sequence ]
            If *None*, no binning is applied; the color of each hexagon
            directly corresponds to its count value.

            If 'log', use a logarithmic scale for the color
            map. Internally, :math:`log_{10}(i+1)` is used to
            determine the hexagon color.

            If an integer, divide the counts in the specified number
            of bins, and color the hexagons accordingly.

            If a sequence of values, the values of the lower bound of
            the bins to be used.

          *xscale*: [ 'linear' | 'log' ]
            Use a linear or log10 scale on the horizontal axis.

          *scale*: [ 'linear' | 'log' ]
            Use a linear or log10 scale on the vertical axis.

        Other keyword arguments controlling color mapping and normalization
        arguments:

          *cmap*: [ None | Colormap ]
            a :class:`matplotlib.cm.Colormap` instance. If *None*,
            defaults to rc ``image.cmap``.

          *norm*: [ None | Normalize ]
            :class:`matplotlib.colors.Normalize` instance is used to
            scale luminance data to 0,1.

          *vmin*/*vmax*: scalar
            *vmin* and *vmax* are used in conjunction with *norm* to normalize
            luminance data.  If either are *None*, the min and max of the color
            array *C* is used.  Note if you pass a norm instance, your settings
            for *vmin* and *vmax* will be ignored.

          *alpha*: scalar
            the alpha value for the patches

          *linewidths*: [ None | scalar ]
            If *None*, defaults to rc lines.linewidth. Note that this
            is a tuple, and if you set the linewidths argument you
            must set it as a sequence of floats, as required by
            :class:`~matplotlib.collections.RegularPolyCollection`.

        Other keyword arguments controlling the Collection properties:

          *edgecolors*: [ None | mpl color | color sequence ]
            If 'none', draws the edges in the same color as the fill color.
            This is the default, as it avoids unsightly unpainted pixels
            between the hexagons.

            If *None*, draws the outlines in the default color.

            If a matplotlib color arg or sequence of rgba tuples, draws the
            outlines in the specified color.

        Here are the standard descriptions of all the
        :class:`~matplotlib.collections.Collection` kwargs:

        %(Collection)s

        The return value is a
        :class:`~matplotlib.collections.PolyCollection` instance; use
        :meth:`~matplotlib.collection.PolyCollection.get_array` on
        this :class:`~matplotlib.collections.PolyCollection` to get
        the counts in each hexagon.

        **Example:**

        .. plot:: mpl_examples/pylab_examples/hexbin_demo.py
        """

        if not self._hold: self.cla()

        self._process_unit_info(xdata=x, ydata=y, kwargs=kwargs)

        x, y, C = cbook.delete_masked_points(x, y, C)

        # Set the size of the hexagon grid
        if iterable(gridsize):
            nx, ny = gridsize
        else:
            nx = gridsize
            ny = int(nx/math.sqrt(3))
        # Count the number of data in each hexagon
        x = np.array(x, float)
        y = np.array(y, float)
        if xscale=='log':
            x = np.log10(x)
        if yscale=='log':
            y = np.log10(y)
        xmin = np.amin(x)
        xmax = np.amax(x)
        ymin = np.amin(y)
        ymax = np.amax(y)
        # In the x-direction, the hexagons exactly cover the region from
        # xmin to xmax. Need some padding to avoid roundoff errors.
        padding = 1.e-9 * (xmax - xmin)
        xmin -= padding
        xmax += padding
        sx = (xmax-xmin) / nx
        sy = (ymax-ymin) / ny
        x = (x-xmin)/sx
        y = (y-ymin)/sy
        ix1 = np.round(x).astype(int)
        iy1 = np.round(y).astype(int)
        ix2 = np.floor(x).astype(int)
        iy2 = np.floor(y).astype(int)

        nx1 = nx + 1
        ny1 = ny + 1
        nx2 = nx
        ny2 = ny
        n = nx1*ny1+nx2*ny2

        d1 = (x-ix1)**2 + 3.0 * (y-iy1)**2
        d2 = (x-ix2-0.5)**2 + 3.0 * (y-iy2-0.5)**2
        bdist = (d1<d2)

        if C is None:
            accum = np.zeros(n)
            # Create appropriate views into "accum" array.
            lattice1 = accum[:nx1*ny1]
            lattice2 = accum[nx1*ny1:]
            lattice1.shape = (nx1,ny1)
            lattice2.shape = (nx2,ny2)

            for i in xrange(len(x)):
                if bdist[i]:
                    lattice1[ix1[i], iy1[i]]+=1
                else:
                    lattice2[ix2[i], iy2[i]]+=1
        else:
            # create accumulation arrays
            lattice1 = np.empty((nx1,ny1),dtype=object)
            for i in xrange(nx1):
                for j in xrange(ny1):
                    lattice1[i,j] = []
            lattice2 = np.empty((nx2,ny2),dtype=object)
            for i in xrange(nx2):
                for j in xrange(ny2):
                    lattice2[i,j] = []

            for i in xrange(len(x)):
                if bdist[i]:
                    lattice1[ix1[i], iy1[i]].append( C[i] )
                else:
                    lattice2[ix2[i], iy2[i]].append( C[i] )

            for i in xrange(nx1):
                for j in xrange(ny1):
                    vals = lattice1[i,j]
                    if len(vals):
                        lattice1[i,j] = reduce_C_function( vals )
                    else:
                        lattice1[i,j] = np.nan
            for i in xrange(nx2):
                for j in xrange(ny2):
                    vals = lattice2[i,j]
                    if len(vals):
                        lattice2[i,j] = reduce_C_function( vals )
                    else:
                        lattice2[i,j] = np.nan

            accum = np.hstack((
                lattice1.astype(float).ravel(), lattice2.astype(float).ravel()))
            good_idxs = ~np.isnan(accum)

        px = xmin + sx * np.array([ 0.5, 0.5, 0.0, -0.5, -0.5,  0.0])
        py = ymin + sy * np.array([-0.5, 0.5, 1.0,  0.5, -0.5, -1.0]) / 3.0

        polygons = np.zeros((6, n, 2), float)
        polygons[:,:nx1*ny1,0] = np.repeat(np.arange(nx1), ny1)
        polygons[:,:nx1*ny1,1] = np.tile(np.arange(ny1), nx1)
        polygons[:,nx1*ny1:,0] = np.repeat(np.arange(nx2) + 0.5, ny2)
        polygons[:,nx1*ny1:,1] = np.tile(np.arange(ny2), nx2) + 0.5

        if C is not None:
            # remove accumulation bins with no data
            polygons = polygons[:,good_idxs,:]
            accum = accum[good_idxs]

        polygons = np.transpose(polygons, axes=[1,0,2])
        polygons[:,:,0] *= sx
        polygons[:,:,1] *= sy
        polygons[:,:,0] += px
        polygons[:,:,1] += py

        if xscale=='log':
            polygons[:,:,0] = 10**(polygons[:,:,0])
            xmin = 10**xmin
            xmax = 10**xmax
            self.set_xscale('log')
        if yscale=='log':
            polygons[:,:,1] = 10**(polygons[:,:,1])
            ymin = 10**ymin
            ymax = 10**ymax
            self.set_yscale('log')

        if edgecolors=='none':
            edgecolors = 'face'
        collection = mcoll.PolyCollection(
            polygons,
            edgecolors = edgecolors,
            linewidths = linewidths,
            transOffset = self.transData,
            )

        # Transform accum if needed
        if bins=='log':
            accum = np.log10(accum+1)
        elif bins!=None:
            if not iterable(bins):
                minimum, maximum = min(accum), max(accum)
                bins-=1 # one less edge than bins
                bins = minimum + (maximum-minimum)*np.arange(bins)/bins
            bins = np.sort(bins)
            accum = bins.searchsorted(accum)

        if norm is not None: assert(isinstance(norm, mcolors.Normalize))
        if cmap is not None: assert(isinstance(cmap, mcolors.Colormap))
        collection.set_array(accum)
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection.set_alpha(alpha)
        collection.update(kwargs)

        if vmin is not None or vmax is not None:
            collection.set_clim(vmin, vmax)
        else:
            collection.autoscale_None()

        corners = ((xmin, ymin), (xmax, ymax))
        self.update_datalim( corners)
        self.autoscale_view()

        # add the collection last
        self.add_collection(collection)
        return collection

    hexbin.__doc__ = cbook.dedent(hexbin.__doc__) % martist.kwdocd


    def arrow(self, x, y, dx, dy, **kwargs):
        """
        call signature::

           arrow(x, y, dx, dy, **kwargs)

        Draws arrow on specified axis from (*x*, *y*) to (*x* + *dx*,
        *y* + *dy*).

        Optional kwargs control the arrow properties:
        %(FancyArrow)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/arrow_demo.py
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
        if not self._hold: self.cla()
        q = mquiver.Quiver(self, *args, **kw)
        self.add_collection(q, False)
        self.update_datalim(q.XY)
        self.autoscale_view()
        return q
    quiver.__doc__ = mquiver.Quiver.quiver_doc

    def barbs(self, *args, **kw):
        """
        %(barbs_doc)s
        **Example:**

        .. plot:: mpl_examples/pylab_examples/barb_demo.py
        """
        if not self._hold: self.cla()
        b = mquiver.Barbs(self, *args, **kw)
        self.add_collection(b)
        self.update_datalim(b.get_offsets())
        self.autoscale_view()
        return b
    barbs.__doc__ = cbook.dedent(barbs.__doc__) % {
        'barbs_doc': mquiver.Barbs.barbs_doc}

    def fill(self, *args, **kwargs):
        """
        call signature::

          fill(*args, **kwargs)

        Plot filled polygons.  *args* is a variable length argument,
        allowing for multiple *x*, *y* pairs with an optional color
        format string; see :func:`~matplotlib.pyplot.plot` for details
        on the argument parsing.  For example, to plot a polygon with
        vertices at *x*, *y* in blue.::

          ax.fill(x,y, 'b' )

        An arbitrary number of *x*, *y*, *color* groups can be specified::

          ax.fill(x1, y1, 'g', x2, y2, 'r')

        Return value is a list of :class:`~matplotlib.patches.Patch`
        instances that were added.

        The same color strings that :func:`~matplotlib.pyplot.plot`
        supports are supported by the fill format string.

        If you would like to fill below a curve, eg. shade a region
        between 0 and *y* along *x*, use :meth:`fill_between`

        The *closed* kwarg will close the polygon when *True* (default).

        kwargs control the Polygon properties:

        %(Polygon)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/fill_demo.py

        """
        if not self._hold: self.cla()

        patches = []
        for poly in self._get_patches_for_fill(*args, **kwargs):
            self.add_patch( poly )
            patches.append( poly )
        self.autoscale_view()
        return patches
    fill.__doc__ = cbook.dedent(fill.__doc__) % martist.kwdocd

    def fill_between(self, x, y1, y2=0, where=None, **kwargs):
        """
        call signature::

          fill_between(x, y1, y2=0, where=None, **kwargs)

        Create a :class:`~matplotlib.collections.PolyCollection`
        filling the regions between *y1* and *y2* where
        ``where==True``

        *x*
          an N length np array of the x data

        *y1*
          an N length scalar or np array of the x data

        *y2*
          an N length scalar or np array of the x data

        *where*
           if None, default to fill between everywhere.  If not None,
           it is a a N length numpy boolean array and the fill will
           only happen over the regions where ``where==True``

        *kwargs*
          keyword args passed on to the :class:`PolyCollection`

        kwargs control the Polygon properties:

        %(PolyCollection)s

        .. plot:: mpl_examples/pylab_examples/fill_between.py
        """
        # Handle united data, such as dates
        self._process_unit_info(xdata=x, ydata=y1, kwargs=kwargs)
        self._process_unit_info(ydata=y2)

        # Convert the arrays so we can work with them
        x = np.asarray(self.convert_xunits(x))
        y1 = np.asarray(self.convert_yunits(y1))
        y2 = np.asarray(self.convert_yunits(y2))

        if not cbook.iterable(y1):
            y1 = np.ones_like(x)*y1

        if not cbook.iterable(y2):
            y2 = np.ones_like(x)*y2

        if where is None:
            where = np.ones(len(x), np.bool)

        where = np.asarray(where)
        assert( (len(x)==len(y1)) and (len(x)==len(y2)) and len(x)==len(where))

        polys = []
        for ind0, ind1 in mlab.contiguous_regions(where):
            theseverts = []
            xslice = x[ind0:ind1]
            y1slice = y1[ind0:ind1]
            y2slice = y2[ind0:ind1]

            if not len(xslice):
                continue

            N = len(xslice)
            X = np.zeros((2*N+2, 2), np.float)

            # the purpose of the next two lines is for when y2 is a
            # scalar like 0 and we want the fill to go all the way
            # down to 0 even if none of the y1 sample points do
            X[0] = xslice[0], y2slice[0]
            X[N+1] = xslice[-1], y2slice[-1]

            X[1:N+1,0] = xslice
            X[1:N+1,1] = y1slice
            X[N+2:,0] = xslice[::-1]
            X[N+2:,1] = y2slice[::-1]

            polys.append(X)

        collection = mcoll.PolyCollection(polys, **kwargs)

        # now update the datalim and autoscale
        XY1 = np.array([x[where], y1[where]]).T
        XY2 = np.array([x[where], y2[where]]).T
        self.dataLim.update_from_data_xy(XY1, self.ignore_existing_data_limits,
                                         updatex=True, updatey=True)

        self.dataLim.update_from_data_xy(XY2, self.ignore_existing_data_limits,
                                         updatex=False, updatey=True)
        self.add_collection(collection)
        self.autoscale_view()
        return collection
    fill_between.__doc__ = cbook.dedent(fill_between.__doc__) % martist.kwdocd

    #### plotting z(x,y): imshow, pcolor and relatives, contour

    def imshow(self, X, cmap=None, norm=None, aspect=None,
               interpolation=None, alpha=1.0, vmin=None, vmax=None,
               origin=None, extent=None, shape=None, filternorm=1,
               filterrad=4.0, imlim=None, resample=None, url=None, **kwargs):
        """
        call signature::

          imshow(X, cmap=None, norm=None, aspect=None, interpolation=None,
                 alpha=1.0, vmin=None, vmax=None, origin=None, extent=None,
                 **kwargs)

        Display the image in *X* to current axes.  *X* may be a float
        array, a uint8 array or a PIL image. If *X* is an array, *X*
        can have the following shapes:

        * MxN -- luminance (grayscale, float array only)
        * MxNx3 -- RGB (float or uint8 array)
        * MxNx4 -- RGBA (float or uint8 array)

        The value for each component of MxNx3 and MxNx4 float arrays should be
        in the range 0.0 to 1.0; MxN float arrays may be normalised.

        An :class:`matplotlib.image.AxesImage` instance is returned.

        Keyword arguments:

          *cmap*: [ None | Colormap ]
            A :class:`matplotlib.cm.Colormap` instance, eg. cm.jet.
            If *None*, default to rc ``image.cmap`` value.

            *cmap* is ignored when *X* has RGB(A) information

          *aspect*: [ None | 'auto' | 'equal' | scalar ]
            If 'auto', changes the image aspect ratio to match that of the axes

            If 'equal', and *extent* is *None*, changes the axes
            aspect ratio to match that of the image. If *extent* is
            not *None*, the axes aspect ratio is changed to match that
            of the extent.

            If *None*, default to rc ``image.aspect`` value.

          *interpolation*:

            Acceptable values are *None*, 'nearest', 'bilinear',
              'bicubic', 'spline16', 'spline36', 'hanning', 'hamming',
              'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian',
              'bessel', 'mitchell', 'sinc', 'lanczos',


            If *interpolation* is *None*, default to rc
            ``image.interpolation``. See also the *filternorm* and
            *filterrad* parameters

          *norm*: [ None | Normalize ]
            An :class:`matplotlib.colors.Normalize` instance; if
            *None*, default is ``normalization()``.  This scales
            luminance -> 0-1

            *norm* is only used for an MxN float array.

          *vmin*/*vmax*: [ None | scalar ]
            Used to scale a luminance image to 0-1.  If either is
            *None*, the min and max of the luminance values will be
            used.  Note if *norm* is not *None*, the settings for
            *vmin* and *vmax* will be ignored.

          *alpha*: scalar
            The alpha blending value, between 0 (transparent) and 1 (opaque)

          *origin*: [ None | 'upper' | 'lower' ]
            Place the [0,0] index of the array in the upper left or lower left
            corner of the axes. If *None*, default to rc ``image.origin``.

          *extent*: [ None | scalars (left, right, bottom, top) ]
            Eata values of the axes.  The default assigns zero-based row,
            column indices to the *x*, *y* centers of the pixels.

          *shape*: [ None | scalars (columns, rows) ]
            For raw buffer images

          *filternorm*:
            A parameter for the antigrain image resize filter.  From the
            antigrain documentation, if *filternorm* = 1, the filter normalizes
            integer values and corrects the rounding errors. It doesn't do
            anything with the source floating point values, it corrects only
            integers according to the rule of 1.0 which means that any sum of
            pixel weights must be equal to 1.0.  So, the filter function must
            produce a graph of the proper shape.

          *filterrad*:
            The filter radius for filters that have a radius
            parameter, i.e. when interpolation is one of: 'sinc',
            'lanczos' or 'blackman'

        Additional kwargs are :class:`~matplotlib.artist.Artist` properties:

        %(Artist)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/image_demo.py
        """

        if not self._hold: self.cla()

        if norm is not None: assert(isinstance(norm, mcolors.Normalize))
        if cmap is not None: assert(isinstance(cmap, mcolors.Colormap))
        if aspect is None: aspect = rcParams['image.aspect']
        self.set_aspect(aspect)
        im = mimage.AxesImage(self, cmap, norm, interpolation, origin, extent,
                       filternorm=filternorm,
                       filterrad=filterrad, resample=resample, **kwargs)

        im.set_data(X)
        im.set_alpha(alpha)
        self._set_artist_props(im)
        im.set_clip_path(self.patch)
        #if norm is None and shape is None:
        #    im.set_clim(vmin, vmax)
        if vmin is not None or vmax is not None:
            im.set_clim(vmin, vmax)
        else:
            im.autoscale_None()
        im.set_url(url)

        xmin, xmax, ymin, ymax = im.get_extent()

        corners = (xmin, ymin), (xmax, ymax)
        self.update_datalim(corners)
        if self._autoscaleon:
            self.set_xlim((xmin, xmax))
            self.set_ylim((ymin, ymax))
        self.images.append(im)

        return im
    imshow.__doc__ = cbook.dedent(imshow.__doc__) % martist.kwdocd


    def _pcolorargs(self, funcname, *args):
        if len(args)==1:
            C = args[0]
            numRows, numCols = C.shape
            X, Y = np.meshgrid(np.arange(numCols+1), np.arange(numRows+1) )
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
                'Incompatible X, Y inputs to %s; see help(%s)' % (
                funcname, funcname))
        return X, Y, C

    def pcolor(self, *args, **kwargs):
        """
        call signatures::

          pcolor(C, **kwargs)
          pcolor(X, Y, C, **kwargs)

        Create a pseudocolor plot of a 2-D array.

        *C* is the array of color values.

        *X* and *Y*, if given, specify the (*x*, *y*) coordinates of
        the colored quadrilaterals; the quadrilateral for C[i,j] has
        corners at::

          (X[i,   j],   Y[i,   j]),
          (X[i,   j+1], Y[i,   j+1]),
          (X[i+1, j],   Y[i+1, j]),
          (X[i+1, j+1], Y[i+1, j+1]).

        Ideally the dimensions of *X* and *Y* should be one greater
        than those of *C*; if the dimensions are the same, then the
        last row and column of *C* will be ignored.

        Note that the the column index corresponds to the
        *x*-coordinate, and the row index corresponds to *y*; for
        details, see the :ref:`Grid Orientation
        <axes-pcolor-grid-orientation>` section below.

        If either or both of *X* and *Y* are 1-D arrays or column vectors,
        they will be expanded as needed into the appropriate 2-D arrays,
        making a rectangular grid.

        *X*, *Y* and *C* may be masked arrays.  If either C[i, j], or one
        of the vertices surrounding C[i,j] (*X* or *Y* at [i, j], [i+1, j],
        [i, j+1],[i+1, j+1]) is masked, nothing is plotted.

        Keyword arguments:

          *cmap*: [ None | Colormap ]
            A :class:`matplotlib.cm.Colormap` instance. If *None*, use
            rc settings.

          norm: [ None | Normalize ]
            An :class:`matplotlib.colors.Normalize` instance is used
            to scale luminance data to 0,1. If *None*, defaults to
            :func:`normalize`.

          *vmin*/*vmax*: [ None | scalar ]
            *vmin* and *vmax* are used in conjunction with *norm* to
            normalize luminance data.  If either are *None*, the min
            and max of the color array *C* is used.  If you pass a
            *norm* instance, *vmin* and *vmax* will be ignored.

          *shading*: [ 'flat' | 'faceted' ]
            If 'faceted', a black grid is drawn around each rectangle; if
            'flat', edges are not drawn. Default is 'flat', contrary to
            Matlab(TM).

            This kwarg is deprecated; please use 'edgecolors' instead:
              * shading='flat' -- edgecolors='None'
              * shading='faceted  -- edgecolors='k'

          *edgecolors*: [ None | 'None' | color | color sequence]
            If *None*, the rc setting is used by default.

            If 'None', edges will not be visible.

            An mpl color or sequence of colors will set the edge color

          *alpha*: 0 <= scalar <= 1
            the alpha blending value

        Return value is a :class:`matplotlib.collection.Collection`
        instance.

        .. _axes-pcolor-grid-orientation:

        The grid orientation follows the Matlab(TM) convention: an
        array *C* with shape (*nrows*, *ncolumns*) is plotted with
        the column number as *X* and the row number as *Y*, increasing
        up; hence it is plotted the way the array would be printed,
        except that the *Y* axis is reversed.  That is, *C* is taken
        as *C*(*y*, *x*).

        Similarly for :func:`~matplotlib.pyplot.meshgrid`::

          x = np.arange(5)
          y = np.arange(3)
          X, Y = meshgrid(x,y)

        is equivalent to:

          X = array([[0, 1, 2, 3, 4],
                     [0, 1, 2, 3, 4],
                     [0, 1, 2, 3, 4]])

          Y = array([[0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2]])

        so if you have::

          C = rand( len(x), len(y))

        then you need::

          pcolor(X, Y, C.T)

        or::

          pcolor(C.T)

        Matlab :func:`pcolor` always discards the last row and column
        of *C*, but matplotlib displays the last row and column if *X* and
        *Y* are not specified, or if *X* and *Y* have one more row and
        column than *C*.

        kwargs can be used to control the
        :class:`~matplotlib.collection.PolyCollection` properties:

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

        newaxis = np.newaxis
        compress = np.compress

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

        xy = np.concatenate((X1[:,newaxis], Y1[:,newaxis],
                             X2[:,newaxis], Y2[:,newaxis],
                             X3[:,newaxis], Y3[:,newaxis],
                             X4[:,newaxis], Y4[:,newaxis],
                             X1[:,newaxis], Y1[:,newaxis]),
                             axis=1)
        verts = xy.reshape((npoly, 5, 2))

        #verts = zip(zip(X1,Y1),zip(X2,Y2),zip(X3,Y3),zip(X4,Y4))

        C = compress(ravelmask, ma.filled(C[0:Ny-1,0:Nx-1]).ravel())


        if shading == 'faceted':
            edgecolors = (0,0,0,1),
            linewidths = (0.25,)
        else:
            edgecolors = 'face'
            linewidths = (1.0,)
        kwargs.setdefault('edgecolors', edgecolors)
        kwargs.setdefault('antialiaseds', (0,))
        kwargs.setdefault('linewidths', linewidths)

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
        minx = np.amin(x)
        maxx = np.amax(x)
        miny = np.amin(y)
        maxy = np.amax(y)

        corners = (minx, miny), (maxx, maxy)
        self.update_datalim( corners)
        self.autoscale_view()
        self.add_collection(collection)
        return collection
    pcolor.__doc__ = cbook.dedent(pcolor.__doc__) % martist.kwdocd

    def pcolormesh(self, *args, **kwargs):
        """
        call signatures::

          pcolormesh(C)
          pcolormesh(X, Y, C)
          pcolormesh(C, **kwargs)

        *C* may be a masked array, but *X* and *Y* may not.  Masked
        array support is implemented via *cmap* and *norm*; in
        contrast, :func:`~matplotlib.pyplot.pcolor` simply does not
        draw quadrilaterals with masked colors or vertices.

        Keyword arguments:

          *cmap*: [ None | Colormap ]
            A :class:`matplotlib.cm.Colormap` instance. If None, use
            rc settings.

          *norm*: [ None | Normalize ]
            A :class:`matplotlib.colors.Normalize` instance is used to
            scale luminance data to 0,1. If None, defaults to
            :func:`normalize`.

          *vmin*/*vmax*: [ None | scalar ]
            *vmin* and *vmax* are used in conjunction with *norm* to
            normalize luminance data.  If either are *None*, the min
            and max of the color array *C* is used.  If you pass a
            *norm* instance, *vmin* and *vmax* will be ignored.

          *shading*: [ 'flat' | 'faceted' ]
            If 'faceted', a black grid is drawn around each rectangle; if
            'flat', edges are not drawn. Default is 'flat', contrary to
            Matlab(TM).

            This kwarg is deprecated; please use 'edgecolors' instead:
              * shading='flat' -- edgecolors='None'
              * shading='faceted  -- edgecolors='k'

          *edgecolors*: [ None | 'None' | color | color sequence]
            If None, the rc setting is used by default.

            If 'None', edges will not be visible.

            An mpl color or sequence of colors will set the edge color

          *alpha*: 0 <= scalar <= 1
            the alpha blending value

        Return value is a :class:`matplotlib.collection.QuadMesh`
        object.

        kwargs can be used to control the
        :class:`matplotlib.collections.QuadMesh`
        properties:

        %(QuadMesh)s

        .. seealso::
            :func:`~matplotlib.pyplot.pcolor`:
                For an explanation of the grid orientation and the
                expansion of 1-D *X* and/or *Y* to 2-D arrays.
        """
        if not self._hold: self.cla()

        alpha = kwargs.pop('alpha', 1.0)
        norm = kwargs.pop('norm', None)
        cmap = kwargs.pop('cmap', None)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        shading = kwargs.pop('shading', 'flat')
        edgecolors = kwargs.pop('edgecolors', 'None')
        antialiased = kwargs.pop('antialiased', False)

        X, Y, C = self._pcolorargs('pcolormesh', *args)
        Ny, Nx = X.shape

        # convert to one dimensional arrays
        C = ma.ravel(C[0:Ny-1, 0:Nx-1]) # data point in each cell is value at
                                        # lower left corner
        X = X.ravel()
        Y = Y.ravel()

        coords = np.zeros(((Nx * Ny), 2), dtype=float)
        coords[:, 0] = X
        coords[:, 1] = Y

        if shading == 'faceted' or edgecolors != 'None':
            showedges = 1
        else:
            showedges = 0

        collection = mcoll.QuadMesh(
            Nx - 1, Ny - 1, coords, showedges,
            antialiased=antialiased)  # kwargs are not used
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

        minx = np.amin(X)
        maxx = np.amax(X)
        miny = np.amin(Y)
        maxy = np.amax(Y)

        corners = (minx, miny), (maxx, maxy)
        self.update_datalim( corners)
        self.autoscale_view()
        self.add_collection(collection)
        return collection
    pcolormesh.__doc__ = cbook.dedent(pcolormesh.__doc__) % martist.kwdocd

    def pcolorfast(self, *args, **kwargs):
        """
        pseudocolor plot of a 2-D array

        Experimental; this is a version of pcolor that
        does not draw lines, that provides the fastest
        possible rendering with the Agg backend, and that
        can handle any quadrilateral grid.

        Call signatures::

          pcolor(C, **kwargs)
          pcolor(xr, yr, C, **kwargs)
          pcolor(x, y, C, **kwargs)
          pcolor(X, Y, C, **kwargs)

        C is the 2D array of color values corresponding to quadrilateral
        cells. Let (nr, nc) be its shape.  C may be a masked array.

        ``pcolor(C, **kwargs)`` is equivalent to
        ``pcolor([0,nc], [0,nr], C, **kwargs)``

        *xr*, *yr* specify the ranges of *x* and *y* corresponding to the
        rectangular region bounding *C*.  If::

            xr = [x0, x1]

        and::

            yr = [y0,y1]

        then *x* goes from *x0* to *x1* as the second index of *C* goes
        from 0 to *nc*, etc.  (*x0*, *y0*) is the outermost corner of
        cell (0,0), and (*x1*, *y1*) is the outermost corner of cell
        (*nr*-1, *nc*-1).  All cells are rectangles of the same size.
        This is the fastest version.

        *x*, *y* are 1D arrays of length *nc* +1 and *nr* +1, respectively,
        giving the x and y boundaries of the cells.  Hence the cells are
        rectangular but the grid may be nonuniform.  The speed is
        intermediate.  (The grid is checked, and if found to be
        uniform the fast version is used.)

        *X* and *Y* are 2D arrays with shape (*nr* +1, *nc* +1) that specify
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

        Optional keyword arguments:

          *cmap*: [ None | Colormap ]
            A cm Colormap instance from cm. If None, use rc settings.
          *norm*: [ None | Normalize ]
            An mcolors.Normalize instance is used to scale luminance data to
            0,1. If None, defaults to normalize()
          *vmin*/*vmax*: [ None | scalar ]
            *vmin* and *vmax* are used in conjunction with norm to normalize
            luminance data.  If either are *None*, the min and max of the color
            array *C* is used.  If you pass a norm instance, *vmin* and *vmax*
            will be *None*.
          *alpha*: 0 <= scalar <= 1
            the alpha blending value

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
            x = [0, nc]
            y = [0, nr]
        elif len(args) == 3:
            x, y = args[:2]
            x = np.asarray(x)
            y = np.asarray(y)
            if x.ndim == 1 and y.ndim == 1:
                if x.size == 2 and y.size == 2:
                    style = "image"
                else:
                    dx = np.diff(x)
                    dy = np.diff(y)
                    if (np.ptp(dx) < 0.01*np.abs(dx.mean()) and
                        np.ptp(dy) < 0.01*np.abs(dy.mean())):
                        style = "image"
                    else:
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
            C = ma.ravel(C) # data point in each cell is value
                            # at lower left corner
            X = x.ravel()
            Y = y.ravel()
            Nx = nc+1
            Ny = nr+1

            # The following needs to be cleaned up; the renderer
            # requires separate contiguous arrays for X and Y,
            # but the QuadMesh class requires the 2D array.
            coords = np.empty(((Nx * Ny), 2), np.float64)
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
        self.update_datalim(np.array([[xl, yb], [xr, yt]]))
        self.autoscale_view(tight=True)
        return ret

    def contour(self, *args, **kwargs):
        if not self._hold: self.cla()
        kwargs['filled'] = False
        return mcontour.ContourSet(self, *args, **kwargs)
    contour.__doc__ = mcontour.ContourSet.contour_doc

    def contourf(self, *args, **kwargs):
        if not self._hold: self.cla()
        kwargs['filled'] = True
        return mcontour.ContourSet(self, *args, **kwargs)
    contourf.__doc__ = mcontour.ContourSet.contour_doc

    def clabel(self, CS, *args, **kwargs):
        return CS.clabel(*args, **kwargs)
    clabel.__doc__ = mcontour.ContourSet.clabel.__doc__

    def table(self, **kwargs):
        """
        call signature::

          table(cellText=None, cellColours=None,
                cellLoc='right', colWidths=None,
                rowLabels=None, rowColours=None, rowLoc='left',
                colLabels=None, colColours=None, colLoc='center',
                loc='bottom', bbox=None):

        Add a table to the current axes.  Returns a
        :class:`matplotlib.table.Table` instance.  For finer grained
        control over tables, use the :class:`~matplotlib.table.Table`
        class and add it to the axes with
        :meth:`~matplotlib.axes.Axes.add_table`.

        Thanks to John Gill for providing the class and table.

        kwargs control the :class:`~matplotlib.table.Table`
        properties:

        %(Table)s
        """
        return mtable.table(self, **kwargs)
    table.__doc__ = cbook.dedent(table.__doc__) % martist.kwdocd

    def twinx(self):
        """
        call signature::

          ax = twinx()

        create a twin of Axes for generating a plot with a sharex
        x-axis but independent y axis.  The y-axis of self will have
        ticks on left and the returned axes will have ticks on the
        right
        """

        ax2 = self.figure.add_axes(self.get_position(True), sharex=self,
            frameon=False)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        self.yaxis.tick_left()
        return ax2

    def twiny(self):
        """
        call signature::

          ax = twiny()

        create a twin of Axes for generating a plot with a shared
        y-axis but independent x axis.  The x-axis of self will have
        ticks on bottom and the returned axes will have ticks on the
        top
        """

        ax2 = self.figure.add_axes(self.get_position(True), sharey=self,
            frameon=False)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        self.xaxis.tick_bottom()
        return ax2

    def get_shared_x_axes(self):
        'Return a copy of the shared axes Grouper object for x axes'
        return self._shared_x_axes

    def get_shared_y_axes(self):
        'Return a copy of the shared axes Grouper object for y axes'
        return self._shared_y_axes

    #### Data analysis

    def hist(self, x, bins=10, range=None, normed=False, cumulative=False,
             bottom=None, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, log=False, **kwargs):
        """
        call signature::

          hist(x, bins=10, range=None, normed=False, cumulative=False,
               bottom=None, histtype='bar', align='mid',
               orientation='vertical', rwidth=None, log=False, **kwargs)

        Compute and draw the histogram of *x*. The return value is a
        tuple (*n*, *bins*, *patches*) or ([*n0*, *n1*, ...], *bins*,
        [*patches0*, *patches1*,...]) if the input contains multiple
        data.

        Keyword arguments:

          *bins*:
            Either an integer number of bins or a sequence giving the
            bins.  *x* are the data to be binned. *x* can be an array,
            a 2D array with multiple data in its columns, or a list of
            arrays with data of different length.  Note, if *bins*
            is an integer input argument=numbins, *bins* + 1 bin edges
            will be returned, compatible with the semantics of
            :func:`numpy.histogram` with the *new* = True argument.
            Unequally spaced bins are supported if *bins* is a sequence.

          *range*:
            The lower and upper range of the bins. Lower and upper outliers
            are ignored. If not provided, *range* is (x.min(), x.max()).
            Range has no effect if *bins* is a sequence.

            If *bins* is a sequence or *range* is specified, autoscaling is
            set off (*autoscale_on* is set to *False*) and the xaxis limits
            are set to encompass the full specified bin range.

          *normed*:
            If *True*, the first element of the return tuple will
            be the counts normalized to form a probability density, i.e.,
            ``n/(len(x)*dbin)``.  In a probability density, the integral of
            the histogram should be 1; you can verify that with a
            trapezoidal integration of the probability density function::

              pdf, bins, patches = ax.hist(...)
              print np.sum(pdf * np.diff(bins))

          *cumulative*:
            If *True*, then a histogram is computed where each bin
            gives the counts in that bin plus all bins for smaller values.
            The last bin gives the total number of datapoints.  If *normed*
            is also *True* then the histogram is normalized such that the
            last bin equals 1. If *cumulative* evaluates to less than 0
            (e.g. -1), the direction of accumulation is reversed.  In this
            case, if *normed* is also *True*, then the histogram is normalized
            such that the first bin equals 1.

          *histtype*: [ 'bar' | 'barstacked' | 'step' | 'stepfilled' ]
            The type of histogram to draw.

              - 'bar' is a traditional bar-type histogram.  If multiple data
                are given the bars are aranged side by side.

              - 'barstacked' is a bar-type histogram where multiple
                data are stacked on top of each other.

              - 'step' generates a lineplot that is by default
                unfilled.

              - 'stepfilled' generates a lineplot that is by default
                filled.

          *align*: ['left' | 'mid' | 'right' ]
            Controls how the histogram is plotted.

              - 'left': bars are centered on the left bin edges.

              - 'mid': bars are centered between the bin edges.

              - 'right': bars are centered on the right bin edges.

          *orientation*: [ 'horizontal' | 'vertical' ]
            If 'horizontal', :func:`~matplotlib.pyplot.barh` will be
            used for bar-type histograms and the *bottom* kwarg will be
            the left edges.

          *rwidth*:
            The relative width of the bars as a fraction of the bin
            width.  If *None*, automatically compute the width. Ignored
            if *histtype* = 'step' or 'stepfilled'.

          *log*:
            If *True*, the histogram axis will be set to a log scale.
            If *log* is *True* and *x* is a 1D array, empty bins will
            be filtered out and only the non-empty (*n*, *bins*,
            *patches*) will be returned.

        kwargs are used to update the properties of the hist
        :class:`~matplotlib.patches.Rectangle` instances:

        %(Rectangle)s

        You can use labels for your histogram, and only the first
        :class:`~matplotlib.patches.Rectangle` gets the label (the
        others get the magic string '_nolegend_'.  This will make the
        histograms work in the intuitive way for bar charts::

            ax.hist(10+2*np.random.randn(1000), label='men')
            ax.hist(12+3*np.random.randn(1000), label='women', alpha=0.5)
            ax.legend()

        **Example:**

        .. plot:: mpl_examples/pylab_examples/histogram_demo.py
        """
        if not self._hold: self.cla()

        # NOTE: the range keyword overwrites the built-in func range !!!
        #       needs to be fixed in  with numpy                     !!!

        if kwargs.get('width') is not None:
            raise DeprecationWarning(
                'hist now uses the rwidth to give relative width '
                'and not absolute width')

        try:
            # make sure a copy is created: don't use asarray
            x = np.transpose(np.array(x))
            if len(x.shape)==1:
                x.shape = (1,x.shape[0])
            elif len(x.shape)==2 and x.shape[1]<x.shape[0]:
                warnings.warn('2D hist should be nsamples x nvariables; '
                              'this looks transposed')
        except ValueError:
            # multiple hist with data of different length
            if iterable(x[0]) and not is_string_like(x[0]):
                tx = []
                for i in xrange(len(x)):
                    tx.append( np.array(x[i]) )
                x = tx
            else:
                raise ValueError, 'Can not use providet data to create a histogram'

        # Check whether bins or range are given explicitly. In that
        # case do not autoscale axes.
        binsgiven = (cbook.iterable(bins) or range != None)

        # check the version of the numpy
        if np.__version__ < "1.3": # version 1.1 and 1.2
            hist_kwargs = dict(range=range,
                               normed=bool(normed), new=True)
        else: # version 1.3 and later, drop new=True
            hist_kwargs = dict(range=range,
                               normed=bool(normed))

        n = []
        for i in xrange(len(x)):
            # this will automatically overwrite bins,
            # so that each histogram uses the same bins
            m, bins = np.histogram(x[i], bins, **hist_kwargs)
            n.append(m)

        if cumulative:
            slc = slice(None)
            if cbook.is_numlike(cumulative) and cumulative < 0:
                slc = slice(None,None,-1)

            if normed:
                n = [(m * np.diff(bins))[slc].cumsum()[slc] for m in n]
            else:
                n = [m[slc].cumsum()[slc] for m in n]

        patches = []

        if histtype.startswith('bar'):
            totwidth = np.diff(bins)
            stacked = False

            if rwidth is not None: dr = min(1., max(0., rwidth))
            elif len(n)>1: dr = 0.8
            else: dr = 1.0

            if histtype=='bar':
                width = dr*totwidth/len(n)
                dw = width

                if len(n)>1:
                    boffset = -0.5*dr*totwidth*(1.-1./len(n))
                else:
                    boffset = 0.0
            elif histtype=='barstacked':
                width = dr*totwidth
                boffset, dw = 0.0, 0.0

                stacked = True
            else:
                raise ValueError, 'invalid histtype: %s' % histtype

            if align == 'mid' or align == 'edge':
                boffset += 0.5*totwidth
            elif align == 'right':
                boffset += totwidth
            elif align != 'left' and align != 'center':
                raise ValueError, 'invalid align: %s' % align

            if orientation == 'horizontal':
                for m in n:
                    color = self._get_lines._get_next_cycle_color()
                    patch = self.barh(bins[:-1]+boffset, m, height=width,
                                      left=bottom, align='center', log=log,
                                      color=color)
                    patches.append(patch)
                    if stacked:
                        if bottom is None: bottom = 0.0
                        bottom += m
                    boffset += dw
            elif orientation == 'vertical':
                for m in n:
                    color = self._get_lines._get_next_cycle_color()
                    patch = self.bar(bins[:-1]+boffset, m, width=width,
                                     bottom=bottom, align='center', log=log,
                                     color=color)
                    patches.append(patch)
                    if stacked:
                        if bottom is None: bottom = 0.0
                        bottom += m
                    boffset += dw
            else:
                raise ValueError, 'invalid orientation: %s' % orientation

        elif histtype.startswith('step'):
            x = np.zeros( 2*len(bins), np.float )
            y = np.zeros( 2*len(bins), np.float )

            x[0::2], x[1::2] = bins, bins

            if align == 'left' or align == 'center':
                x -= 0.5*(bins[1]-bins[0])
            elif align == 'right':
                x += 0.5*(bins[1]-bins[0])
            elif align != 'mid' and align != 'edge':
                raise ValueError, 'invalid align: %s' % align

            if log:
                y[0],y[-1] = 1e-100, 1e-100
                if orientation == 'horizontal':
                    self.set_xscale('log')
                elif orientation == 'vertical':
                    self.set_yscale('log')

            fill = False
            if histtype == 'stepfilled':
                fill = True
            elif histtype != 'step':
                raise ValueError, 'invalid histtype: %s' % histtype

            for m in n:
                y[1:-1:2], y[2::2] = m, m
                if orientation == 'horizontal':
                    x,y = y,x
                elif orientation != 'vertical':
                    raise ValueError, 'invalid orientation: %s' % orientation

                color = self._get_lines._get_next_cycle_color()
                if fill:
                    patches.append( self.fill(x, y,
                        closed=False, facecolor=color) )
                else:
                    patches.append( self.fill(x, y,
                        closed=False, edgecolor=color, fill=False) )

            # adopted from adjust_x/ylim part of the bar method
            if orientation == 'horizontal':
                xmin, xmax = 0, self.dataLim.intervalx[1]
                for m in n:
                    xmin = np.amin(m[m!=0]) # filter out the 0 height bins
                xmin = max(xmin*0.9, 1e-100)
                self.dataLim.intervalx = (xmin, xmax)
            elif orientation == 'vertical':
                ymin, ymax = 0, self.dataLim.intervaly[1]
                for m in n:
                    ymin = np.amin(m[m!=0]) # filter out the 0 height bins
                ymin = max(ymin*0.9, 1e-100)
                self.dataLim.intervaly = (ymin, ymax)
            self.autoscale_view()

        else:
            raise ValueError, 'invalid histtype: %s' % histtype

        label = kwargs.pop('label', '')

        for patch in patches:
            for p in patch:
                p.update(kwargs)
                p.set_label(label)
                label = '_nolegend_'

        if binsgiven:
            self.set_autoscale_on(False)
            if orientation == 'vertical':
                self.autoscale_view(scalex=False, scaley=True)
                XL = self.xaxis.get_major_locator().view_limits(bins[0], bins[-1])
                self.set_xbound(XL)
            else:
                self.autoscale_view(scalex=True, scaley=False)
                YL = self.yaxis.get_major_locator().view_limits(bins[0], bins[-1])
                self.set_ybound(YL)

        if len(n)==1:
            return n[0], bins, cbook.silent_list('Patch', patches[0])
        else:
            return n, bins, cbook.silent_list('Lists of Patches', patches)
    hist.__doc__ = cbook.dedent(hist.__doc__) % martist.kwdocd

    def psd(self, x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, pad_to=None,
            sides='default', scale_by_freq=None, **kwargs):
        """
        call signature::

          psd(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
              window=mlab.window_hanning, noverlap=0, pad_to=None,
              sides='default', scale_by_freq=None, **kwargs)

        The power spectral density by Welch's average periodogram
        method.  The vector *x* is divided into *NFFT* length
        segments.  Each segment is detrended by function *detrend* and
        windowed by function *window*.  *noverlap* gives the length of
        the overlap between segments.  The :math:`|\mathrm{fft}(i)|^2`
        of each segment :math:`i` are averaged to compute *Pxx*, with a
        scaling to correct for power loss due to windowing.  *Fs* is the
        sampling frequency.

        %(PSD)s

          *Fc*: integer
            The center frequency of *x* (defaults to 0), which offsets
            the x extents of the plot to reflect the frequency range used
            when a signal is acquired and then filtered and downsampled to
            baseband.

        Returns the tuple (*Pxx*, *freqs*).

        For plotting, the power is plotted as
        :math:`10\log_{10}(P_{xx})` for decibels, though *Pxx* itself
        is returned.

        References:
          Bendat & Piersol -- Random Data: Analysis and Measurement
          Procedures, John Wiley & Sons (1986)

        kwargs control the :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/psd_demo.py
        """
        if not self._hold: self.cla()
        pxx, freqs = mlab.psd(x, NFFT, Fs, detrend, window, noverlap, pad_to,
            sides, scale_by_freq)
        pxx.shape = len(freqs),
        freqs += Fc

        if scale_by_freq in (None, True):
            psd_units = 'dB/Hz'
        else:
            psd_units = 'dB'

        self.plot(freqs, 10*np.log10(pxx), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Power Spectral Density (%s)' % psd_units)
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly
        intv = vmax-vmin
        logi = int(np.log10(intv))
        if logi==0: logi=.1
        step = 10*logi
        #print vmin, vmax, step, intv, math.floor(vmin), math.ceil(vmax)+1
        ticks = np.arange(math.floor(vmin), math.ceil(vmax)+1, step)
        self.set_yticks(ticks)

        return pxx, freqs

    psd_doc_dict = dict()
    psd_doc_dict.update(martist.kwdocd)
    psd_doc_dict.update(mlab.kwdocd)
    psd_doc_dict['PSD'] = cbook.dedent(psd_doc_dict['PSD'])
    psd.__doc__ = cbook.dedent(psd.__doc__) % psd_doc_dict

    def csd(self, x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, pad_to=None,
            sides='default', scale_by_freq=None, **kwargs):
        """
        call signature::

          csd(x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
              window=mlab.window_hanning, noverlap=0, pad_to=None,
              sides='default', scale_by_freq=None, **kwargs)

        The cross spectral density :math:`P_{xy}` by Welch's average
        periodogram method.  The vectors *x* and *y* are divided into
        *NFFT* length segments.  Each segment is detrended by function
        *detrend* and windowed by function *window*.  The product of
        the direct FFTs of *x* and *y* are averaged over each segment
        to compute :math:`P_{xy}`, with a scaling to correct for power
        loss due to windowing.

        Returns the tuple (*Pxy*, *freqs*).  *P* is the cross spectrum
        (complex valued), and :math:`10\log_{10}|P_{xy}|` is
        plotted.

        %(PSD)s

          *Fc*: integer
            The center frequency of *x* (defaults to 0), which offsets
            the x extents of the plot to reflect the frequency range used
            when a signal is acquired and then filtered and downsampled to
            baseband.

        References:
          Bendat & Piersol -- Random Data: Analysis and Measurement
          Procedures, John Wiley & Sons (1986)

        kwargs control the Line2D properties:

        %(Line2D)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/csd_demo.py

        .. seealso:
            :meth:`psd`
                For a description of the optional parameters.
        """
        if not self._hold: self.cla()
        pxy, freqs = mlab.csd(x, y, NFFT, Fs, detrend, window, noverlap,
            pad_to, sides, scale_by_freq)
        pxy.shape = len(freqs),
        # pxy is complex
        freqs += Fc

        self.plot(freqs, 10*np.log10(np.absolute(pxy)), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Cross Spectrum Magnitude (dB)')
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly

        intv = vmax-vmin
        step = 10*int(np.log10(intv))

        ticks = np.arange(math.floor(vmin), math.ceil(vmax)+1, step)
        self.set_yticks(ticks)

        return pxy, freqs
    csd.__doc__ = cbook.dedent(csd.__doc__) % psd_doc_dict

    def cohere(self, x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=0, pad_to=None,
               sides='default', scale_by_freq=None, **kwargs):
        """
        call signature::

          cohere(x, y, NFFT=256, Fs=2, Fc=0, detrend = mlab.detrend_none,
                 window = mlab.window_hanning, noverlap=0, pad_to=None,
                 sides='default', scale_by_freq=None, **kwargs)

        cohere the coherence between *x* and *y*.  Coherence is the normalized
        cross spectral density:

        .. math::

          C_{xy} = \\frac{|P_{xy}|^2}{P_{xx}P_{yy}}

        %(PSD)s

          *Fc*: integer
            The center frequency of *x* (defaults to 0), which offsets
            the x extents of the plot to reflect the frequency range used
            when a signal is acquired and then filtered and downsampled to
            baseband.

        The return value is a tuple (*Cxy*, *f*), where *f* are the
        frequencies of the coherence vector.

        kwargs are applied to the lines.

        References:

          * Bendat & Piersol -- Random Data: Analysis and Measurement
            Procedures, John Wiley & Sons (1986)

        kwargs control the :class:`~matplotlib.lines.Line2D`
        properties of the coherence plot:

        %(Line2D)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/cohere_demo.py
        """
        if not self._hold: self.cla()
        cxy, freqs = mlab.cohere(x, y, NFFT, Fs, detrend, window, noverlap,
            scale_by_freq)
        freqs += Fc

        self.plot(freqs, cxy, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Coherence')
        self.grid(True)

        return cxy, freqs
    cohere.__doc__ = cbook.dedent(cohere.__doc__) % psd_doc_dict

    def specgram(self, x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
                 window=mlab.window_hanning, noverlap=128,
                 cmap=None, xextent=None, pad_to=None, sides='default',
                 scale_by_freq=None):
        """
        call signature::

          specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
                   window=mlab.window_hanning, noverlap=128,
                   cmap=None, xextent=None, pad_to=None, sides='default',
                   scale_by_freq=None)

        Compute a spectrogram of data in *x*.  Data are split into
        *NFFT* length segments and the PSD of each section is
        computed.  The windowing function *window* is applied to each
        segment, and the amount of overlap of each segment is
        specified with *noverlap*.

        %(PSD)s

          *Fc*: integer
            The center frequency of *x* (defaults to 0), which offsets
            the y extents of the plot to reflect the frequency range used
            when a signal is acquired and then filtered and downsampled to
            baseband.

          *cmap*:
            A :class:`matplotlib.cm.Colormap` instance; if *None* use
            default determined by rc

          *xextent*:
            The image extent along the x-axis. xextent = (xmin,xmax)
            The default is (0,max(bins)), where bins is the return
            value from :func:`mlab.specgram`

        Return value is (*Pxx*, *freqs*, *bins*, *im*):

          - *bins* are the time points the spectrogram is calculated over
          - *freqs* is an array of frequencies
          - *Pxx* is a len(times) x len(freqs) array of power
          - *im* is a :class:`matplotlib.image.AxesImage` instance

        Note: If *x* is real (i.e. non-complex), only the positive
        spectrum is shown.  If *x* is complex, both positive and
        negative parts of the spectrum are shown.  This can be
        overridden using the *sides* keyword argument.

        **Example:**

        .. plot:: mpl_examples/pylab_examples/specgram_demo.py
        """
        if not self._hold: self.cla()

        Pxx, freqs, bins = mlab.specgram(x, NFFT, Fs, detrend,
             window, noverlap, pad_to, sides, scale_by_freq)

        Z = 10. * np.log10(Pxx)
        Z = np.flipud(Z)

        if xextent is None: xextent = 0, np.amax(bins)
        xmin, xmax = xextent
        freqs += Fc
        extent = xmin, xmax, freqs[0], freqs[-1]
        im = self.imshow(Z, cmap, extent=extent)
        self.axis('auto')

        return Pxx, freqs, bins, im
    specgram.__doc__ = cbook.dedent(specgram.__doc__) % psd_doc_dict
    del psd_doc_dict #So that this does not become an Axes attribute

    def spy(self, Z, precision=0, marker=None, markersize=None,
            aspect='equal',  **kwargs):
        """
        call signature::

          spy(Z, precision=0, marker=None, markersize=None,
              aspect='equal', **kwargs)

        ``spy(Z)`` plots the sparsity pattern of the 2-D array *Z*.

        If *precision* is 0, any non-zero value will be plotted;
        else, values of :math:`|Z| > precision` will be plotted.

        For :class:`scipy.sparse.spmatrix` instances, there is a
        special case: if *precision* is 'present', any value present in
        the array will be plotted, even if it is identically zero.

        The array will be plotted as it would be printed, with
        the first index (row) increasing down and the second
        index (column) increasing to the right.

        By default aspect is 'equal', so that each array element
        occupies a square space; set the aspect kwarg to 'auto'
        to allow the plot to fill the plot box, or to any scalar
        number to specify the aspect ratio of an array element
        directly.

        Two plotting styles are available: image or marker. Both
        are available for full arrays, but only the marker style
        works for :class:`scipy.sparse.spmatrix` instances.

        If *marker* and *markersize* are *None*, an image will be
        returned and any remaining kwargs are passed to
        :func:`~matplotlib.pyplot.imshow`; else, a
        :class:`~matplotlib.lines.Line2D` object will be returned with
        the value of marker determining the marker type, and any
        remaining kwargs passed to the
        :meth:`~matplotlib.axes.Axes.plot` method.

        If *marker* and *markersize* are *None*, useful kwargs include:

        * *cmap*
        * *alpha*

        .. seealso::
            :func:`~matplotlib.pyplot.imshow`

        For controlling colors, e.g. cyan background and red marks,
        use::

          cmap = mcolors.ListedColormap(['c','r'])

        If *marker* or *markersize* is not *None*, useful kwargs include:

        * *marker*
        * *markersize*
        * *color*

        Useful values for *marker* include:

        * 's'  square (default)
        * 'o'  circle
        * '.'  point
        * ','  pixel

        .. seealso::
            :func:`~matplotlib.pyplot.plot`
        """
        if precision is None:
            precision = 0
            warnings.DeprecationWarning("Use precision=0 instead of None")
            # 2008/10/03
        if marker is None and markersize is None and hasattr(Z, 'tocoo'):
            marker = 's'
        if marker is None and markersize is None:
            Z = np.asarray(Z)
            mask = np.absolute(Z)>precision

            if 'cmap' not in kwargs:
                kwargs['cmap'] = mcolors.ListedColormap(['w', 'k'],
                                                        name='binary')
            nr, nc = Z.shape
            extent = [-0.5, nc-0.5, nr-0.5, -0.5]
            ret = self.imshow(mask, interpolation='nearest', aspect=aspect,
                                extent=extent, origin='upper', **kwargs)
        else:
            if hasattr(Z, 'tocoo'):
                c = Z.tocoo()
                if precision == 'present':
                    y = c.row
                    x = c.col
                else:
                    nonzero = np.absolute(c.data) > precision
                    y = c.row[nonzero]
                    x = c.col[nonzero]
            else:
                Z = np.asarray(Z)
                nonzero = np.absolute(Z)>precision
                y, x = np.nonzero(nonzero)
            if marker is None: marker = 's'
            if markersize is None: markersize = 10
            marks = mlines.Line2D(x, y, linestyle='None',
                         marker=marker, markersize=markersize, **kwargs)
            self.add_line(marks)
            nr, nc = Z.shape
            self.set_xlim(xmin=-0.5, xmax=nc-0.5)
            self.set_ylim(ymin=nr-0.5, ymax=-0.5)
            self.set_aspect(aspect)
            ret = marks
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
        Plot a matrix or array as an image.

        The matrix will be shown the way it would be printed,
        with the first row at the top.  Row and column numbering
        is zero-based.

        Argument:
            *Z*   anything that can be interpreted as a 2-D array

        kwargs all are passed to :meth:`~matplotlib.axes.Axes.imshow`.
        :meth:`matshow` sets defaults for *extent*, *origin*,
        *interpolation*, and *aspect*; use care in overriding the
        *extent* and *origin* kwargs, because they interact.  (Also,
        if you want to change them, you probably should be using
        imshow directly in your own version of matshow.)

        Returns: an :class:`matplotlib.image.AxesImage` instance.
        '''
        Z = np.asarray(Z)
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
    Base class for subplots, which are :class:`Axes` instances with
    additional methods to facilitate generating and manipulating a set
    of :class:`Axes` within a figure.
    """

    def __init__(self, fig, *args, **kwargs):
        """
        *fig* is a :class:`matplotlib.figure.Figure` instance.

        *args* is the tuple (*numRows*, *numCols*, *plotNum*), where
        the array of subplots in the figure has dimensions *numRows*,
        *numCols*, and where *plotNum* is the number of the subplot
        being created.  *plotNum* starts at 1 in the upper left
        corner and increases to the right.

        If *numRows* <= *numCols* <= *plotNum* < 10, *args* can be the
        decimal integer *numRows* * 100 + *numCols* * 10 + *plotNum*.
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
        num -= 1    # convert from matlab to python indexing
                    # ie num in range(0,total)
        if num >= total:
            raise ValueError( 'Subplot number exceeds total subplots')
        self._rows = rows
        self._cols = cols
        self._num = num

        self.update_params()

        # _axes_class is set in the subplot_class_factory
        self._axes_class.__init__(self, fig, self.figbox, **kwargs)

    def get_geometry(self):
        'get the subplot geometry, eg 2,2,3'
        return self._rows, self._cols, self._num+1

    # COVERAGE NOTE: Never used internally or from examples
    def change_geometry(self, numrows, numcols, num):
        'change subplot geometry, eg. from 1,1,1 to 2,2,3'
        self._rows = numrows
        self._cols = numcols
        self._num = num-1
        self.update_params()
        self.set_position(self.figbox)

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

        self.figbox = mtransforms.Bbox.from_bounds(figLeft, figBottom,
                                                   figW, figH)
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

    # COVERAGE NOTE: Never used internally or from examples
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

_subplot_classes = {}
def subplot_class_factory(axes_class=None):
    # This makes a new class that inherits from SubclassBase and the
    # given axes_class (which is assumed to be a subclass of Axes).
    # This is perhaps a little bit roundabout to make a new class on
    # the fly like this, but it means that a new Subplot class does
    # not have to be created for every type of Axes.
    if axes_class is None:
        axes_class = Axes

    new_class = _subplot_classes.get(axes_class)
    if new_class is None:
        new_class = new.classobj("%sSubplot" % (axes_class.__name__),
                                 (SubplotBase, axes_class),
                                 {'_axes_class': axes_class})
        _subplot_classes[axes_class] = new_class

    return new_class

# This is provided for backward compatibility
Subplot = subplot_class_factory()

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
