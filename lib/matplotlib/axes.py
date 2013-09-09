from __future__ import division, print_function
import math
import warnings
from operator import itemgetter
import itertools

import numpy as np
from numpy import ma

import matplotlib
rcParams = matplotlib.rcParams

import matplotlib.artist as martist
from matplotlib.artist import allow_rasterization
import matplotlib.axis as maxis
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates as _  # <-registers a date unit converter
from matplotlib import docstring
import matplotlib.font_manager as font_manager
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.spines as mspines
import matplotlib.quiver as mquiver
import matplotlib.scale as mscale
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer

iterable = cbook.iterable
is_string_like = cbook.is_string_like
is_sequence_of_strings = cbook.is_sequence_of_strings


def _string_to_bool(s):
    if not is_string_like(s):
        return s
    if s == 'on':
        return True
    if s == 'off':
        return False
    raise ValueError("string argument must be either 'on' or 'off'")


def _process_plot_format(fmt):
    """
    Process a MATLAB style color/line style format string.  Return a
    (*linestyle*, *color*) tuple as a result of the processing.  Default
    values are ('-', 'b').  Example format strings include:

    * 'ko': black circles
    * '.b': blue dots
    * 'r--': red dashed lines

    .. seealso::

        :func:`~matplotlib.Line2D.lineStyles` and
        :func:`~matplotlib.pyplot.colors`
            for all possible styles and color format string.
    """

    linestyle = None
    marker = None
    color = None

    # Is fmt just a colorspec?
    try:
        color = mcolors.colorConverter.to_rgb(fmt)

        # We need to differentiate grayscale '1.0' from tri_down marker '1'
        try:
            fmtint = str(int(fmt))
        except ValueError:
            return linestyle, marker, color  # Yes
        else:
            if fmt != fmtint:
                # user definitely doesn't want tri_down marker
                return linestyle, marker, color  # Yes
            else:
                # ignore converted color
                color = None
    except ValueError:
        pass  # No, not just a color.

    # handle the multi char special cases and strip them from the
    # string
    if fmt.find('--') >= 0:
        linestyle = '--'
        fmt = fmt.replace('--', '')
    if fmt.find('-.') >= 0:
        linestyle = '-.'
        fmt = fmt.replace('-.', '')
    if fmt.find(' ') >= 0:
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


class _process_plot_var_args(object):
    """
    Process variable length arguments to the plot command, so that
    plot commands like the following are supported::

      plot(t, s)
      plot(t1, s1, t2, s2)
      plot(t1, s1, 'ko', t2, s2)
      plot(t1, s1, 'ko', t2, s2, 'r--', t3, e3)

    an arbitrary number of *x*, *y*, *fmt* are allowed
    """
    def __init__(self, axes, command='plot'):
        self.axes = axes
        self.command = command
        self.set_color_cycle()

    def __getstate__(self):
        # note: it is not possible to pickle a itertools.cycle instance
        return {'axes': self.axes, 'command': self.command}

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.set_color_cycle()

    def set_color_cycle(self, clist=None):
        if clist is None:
            clist = rcParams['axes.color_cycle']
        self.color_cycle = itertools.cycle(clist)

    def __call__(self, *args, **kwargs):

        if self.axes.xaxis is not None and self.axes.yaxis is not None:
            xunits = kwargs.pop('xunits', self.axes.xaxis.units)

            if self.axes.name == 'polar':
                xunits = kwargs.pop('thetaunits', xunits)

            yunits = kwargs.pop('yunits', self.axes.yaxis.units)

            if self.axes.name == 'polar':
                yunits = kwargs.pop('runits', yunits)

            if xunits != self.axes.xaxis.units:
                self.axes.xaxis.set_units(xunits)

            if yunits != self.axes.yaxis.units:
                self.axes.yaxis.set_units(yunits)

        ret = self._grab_next_args(*args, **kwargs)
        return ret

    def set_lineprops(self, line, **kwargs):
        assert self.command == 'plot', 'set_lineprops only works with "plot"'
        for key, val in kwargs.items():
            funcName = "set_%s" % key
            if not hasattr(line, funcName):
                raise TypeError('There is no line property "%s"' % key)
            func = getattr(line, funcName)
            func(val)

    def set_patchprops(self, fill_poly, **kwargs):
        assert self.command == 'fill', 'set_patchprops only works with "fill"'
        for key, val in kwargs.items():
            funcName = "set_%s" % key
            if not hasattr(fill_poly, funcName):
                raise TypeError('There is no patch property "%s"' % key)
            func = getattr(fill_poly, funcName)
            func(val)

    def _xy_from_xy(self, x, y):
        if self.axes.xaxis is not None and self.axes.yaxis is not None:
            bx = self.axes.xaxis.update_units(x)
            by = self.axes.yaxis.update_units(y)

            if self.command != 'plot':
                # the Line2D class can handle unitized data, with
                # support for post hoc unit changes etc.  Other mpl
                # artists, eg Polygon which _process_plot_var_args
                # also serves on calls to fill, cannot.  So this is a
                # hack to say: if you are not "plot", which is
                # creating Line2D, then convert the data now to
                # floats.  If you are plot, pass the raw data through
                # to Line2D which will handle the conversion.  So
                # polygons will not support post hoc conversions of
                # the unit type since they are not storing the orig
                # data.  Hopefully we can rationalize this at a later
                # date - JDH
                if bx:
                    x = self.axes.convert_xunits(x)
                if by:
                    y = self.axes.convert_yunits(y)

        x = np.atleast_1d(x)  # like asanyarray, but converts scalar to array
        y = np.atleast_1d(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same first dimension")
        if x.ndim > 2 or y.ndim > 2:
            raise ValueError("x and y can be no greater than 2-D")

        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]
        return x, y

    def _makeline(self, x, y, kw, kwargs):
        kw = kw.copy()  # Don't modify the original kw.
        if not 'color' in kw and not 'color' in kwargs.keys():
            kw['color'] = self.color_cycle.next()
            # (can't use setdefault because it always evaluates
            # its second argument)
        seg = mlines.Line2D(x, y,
                            axes=self.axes,
                            **kw
                            )
        self.set_lineprops(seg, **kwargs)
        return seg

    def _makefill(self, x, y, kw, kwargs):
        try:
            facecolor = kw['color']
        except KeyError:
            facecolor = self.color_cycle.next()
        seg = mpatches.Polygon(np.hstack((x[:, np.newaxis],
                                          y[:, np.newaxis])),
                               facecolor=facecolor,
                               fill=True,
                               closed=kw['closed'])
        self.set_patchprops(seg, **kwargs)
        return seg

    def _plot_args(self, tup, kwargs):
        ret = []
        if len(tup) > 1 and is_string_like(tup[-1]):
            linestyle, marker, color = _process_plot_format(tup[-1])
            tup = tup[:-1]
        elif len(tup) == 3:
            raise ValueError('third arg must be a format string')
        else:
            linestyle, marker, color = None, None, None
        kw = {}
        for k, v in zip(('linestyle', 'marker', 'color'),
                        (linestyle, marker, color)):
            if v is not None:
                kw[k] = v

        y = np.atleast_1d(tup[-1])

        if len(tup) == 2:
            x = np.atleast_1d(tup[0])
        else:
            x = np.arange(y.shape[0], dtype=float)

        x, y = self._xy_from_xy(x, y)

        if self.command == 'plot':
            func = self._makeline
        else:
            kw['closed'] = kwargs.get('closed', True)
            func = self._makefill

        ncx, ncy = x.shape[1], y.shape[1]
        for j in xrange(max(ncx, ncy)):
            seg = func(x[:, j % ncx], y[:, j % ncy], kw, kwargs)
            ret.append(seg)
        return ret

    def _grab_next_args(self, *args, **kwargs):

        remaining = args
        while 1:

            if len(remaining) == 0:
                return
            if len(remaining) <= 3:
                for seg in self._plot_args(remaining, kwargs):
                    yield seg
                return

            if is_string_like(remaining[2]):
                isplit = 3
            else:
                isplit = 2

            for seg in self._plot_args(remaining[:isplit], kwargs):
                yield seg
            remaining = remaining[isplit:]


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
                 axisbg=None,  # defaults to rc axes.facecolor
                 frameon=True,
                 sharex=None,  # use Axes instance's xaxis info
                 sharey=None,  # use Axes instance's yaxis info
                 label='',
                 xscale=None,
                 yscale=None,
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
          *adjustable*       [ 'box' | 'datalim' | 'box-forced']
          *alpha*            float: the alpha transparency (can be None)
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
        """ % {'scale': ' | '.join(
            [repr(x) for x in mscale.get_scale_names()])}
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

        self.set_axes_locator(kwargs.get("axes_locator", None))

        self.spines = self._gen_axes_spines()

        # this call may differ for non-sep axes, eg polar
        self._init_axis()

        if axisbg is None:
            axisbg = rcParams['axes.facecolor']
        self._axisbg = axisbg
        self._frameon = frameon
        self._axisbelow = rcParams['axes.axisbelow']

        self._rasterization_zorder = None

        self._hold = rcParams['axes.hold']
        self._connected = {}  # a dict from events to (id, func)
        self.cla()
        # funcs used to format x and y - fall back on major formatters
        self.fmt_xdata = None
        self.fmt_ydata = None

        self.set_cursor_props((1, 'k'))  # set the cursor properties for axes

        self._cachedRenderer = None
        self.set_navigate(True)
        self.set_navigate_mode(None)

        if xscale:
            self.set_xscale(xscale)
        if yscale:
            self.set_yscale(yscale)

        if len(kwargs):
            martist.setp(self, **kwargs)

        if self.xaxis is not None:
            self._xcid = self.xaxis.callbacks.connect('units finalize',
                                                      self.relim)

        if self.yaxis is not None:
            self._ycid = self.yaxis.callbacks.connect('units finalize',
                                                      self.relim)

    def __setstate__(self, state):
        self.__dict__ = state
        # put the _remove_method back on all artists contained within the axes
        for container_name in ['lines', 'collections', 'tables', 'patches',
                               'texts', 'images']:
            container = getattr(self, container_name)
            for artist in container:
                artist._remove_method = container.remove

    def get_window_extent(self, *args, **kwargs):
        """
        get the axes bounding box in display space; *args* and
        *kwargs* are empty
        """
        return self.bbox

    def _init_axis(self):
        "move this out of __init__ because non-separable axes don't use it"
        self.xaxis = maxis.XAxis(self)
        self.spines['bottom'].register_axis(self.xaxis)
        self.spines['top'].register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self.spines['left'].register_axis(self.yaxis)
        self.spines['right'].register_axis(self.yaxis)
        self._update_transScale()

    def set_figure(self, fig):
        """
        Set the class:`~matplotlib.axes.Axes` figure

        accepts a class:`~matplotlib.figure.Figure` instance
        """
        martist.Artist.set_figure(self, fig)

        self.bbox = mtransforms.TransformedBbox(self._position,
                                                fig.transFigure)
        # these will be updated later as data is added
        self.dataLim = mtransforms.Bbox.null()
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

        .. note::

            This method is primarily used by rectilinear projections
            of the :class:`~matplotlib.axes.Axes` class, and is meant
            to be overridden by new kinds of projection axes that need
            different transformations and limits. (See
            :class:`~matplotlib.projections.polar.PolarAxes` for an
            example.

        """
        self.transAxes = mtransforms.BboxTransformTo(self.bbox)

        # Transforms the x and y axis separately by a scale factor.
        # It is assumed that this part will have non-linear components
        # (e.g., for a log scale).
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
            self.transData, self.transAxes)
        self._yaxis_transform = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)

    def get_xaxis_transform(self, which='grid'):
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
        if which == 'grid':
            return self._xaxis_transform
        elif which == 'tick1':
            # for cartesian projection, this is bottom spine
            return self.spines['bottom'].get_spine_transform()
        elif which == 'tick2':
            # for cartesian projection, this is top spine
            return self.spines['top'].get_spine_transform()
        else:
            raise ValueError('unknown value for which')

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
        return (self.get_xaxis_transform(which='tick1') +
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
        return (self.get_xaxis_transform(which='tick2') +
                mtransforms.ScaledTranslation(0, pad_points / 72.0,
                                              self.figure.dpi_scale_trans),
                "bottom", "center")

    def get_yaxis_transform(self, which='grid'):
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
        if which == 'grid':
            return self._yaxis_transform
        elif which == 'tick1':
            # for cartesian projection, this is bottom spine
            return self.spines['left'].get_spine_transform()
        elif which == 'tick2':
            # for cartesian projection, this is top spine
            return self.spines['right'].get_spine_transform()
        else:
            raise ValueError('unknown value for which')

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
        return (self.get_yaxis_transform(which='tick1') +
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
        return (self.get_yaxis_transform(which='tick2') +
                mtransforms.ScaledTranslation(pad_points / 72.0, 0,
                                              self.figure.dpi_scale_trans),
                "center", "left")

    def _update_transScale(self):
        self.transScale.set(
            mtransforms.blended_transform_factory(
                self.xaxis.get_transform(), self.yaxis.get_transform()))
        if hasattr(self, "lines"):
            for line in self.lines:
                try:
                    line._transformed_path.invalidate()
                except AttributeError:
                    pass

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
        """Make the original position the active position"""
        pos = self.get_position(original=True)
        self.set_position(pos, which='active')

    def set_axes_locator(self, locator):
        """
        set axes_locator

        ACCEPT: a callable object which takes an axes instance and renderer and
                 returns a bbox.
        """
        self._axes_locator = locator

    def get_axes_locator(self):
        """
        return axes_locator
        """
        return self._axes_locator

    def _set_artist_props(self, a):
        """set the boilerplate props for artists added to axes"""
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

    def _gen_axes_spines(self, locations=None, offset=0.0, units='inches'):
        """
        Returns a dict whose keys are spine names and values are
        Line2D or Patch instances. Each element is used to draw a
        spine of the axes.

        In the standard axes, this is a single line segment, but in
        other projections it may not be.

        .. note::

            Intended to be overridden by new projection types.

        """
        return {
            'left': mspines.Spine.linear_spine(self, 'left'),
            'right': mspines.Spine.linear_spine(self, 'right'),
            'bottom': mspines.Spine.linear_spine(self, 'bottom'),
            'top': mspines.Spine.linear_spine(self, 'top'), }

    def cla(self):
        """Clear the current axes."""
        # Note: this is called by Axes.__init__()
        self.xaxis.cla()
        self.yaxis.cla()
        for name, spine in self.spines.iteritems():
            spine.cla()

        self.ignore_existing_data_limits = True
        self.callbacks = cbook.CallbackRegistry()

        if self._sharex is not None:
            # major and minor are class instances with
            # locator and formatter attributes
            self.xaxis.major = self._sharex.xaxis.major
            self.xaxis.minor = self._sharex.xaxis.minor
            x0, x1 = self._sharex.get_xlim()
            self.set_xlim(x0, x1, emit=False, auto=None)

            # Save the current formatter/locator so we don't lose it
            majf = self._sharex.xaxis.get_major_formatter()
            minf = self._sharex.xaxis.get_minor_formatter()
            majl = self._sharex.xaxis.get_major_locator()
            minl = self._sharex.xaxis.get_minor_locator()

            # This overwrites the current formatter/locator
            self.xaxis._set_scale(self._sharex.xaxis.get_scale())

            # Reset the formatter/locator
            self.xaxis.set_major_formatter(majf)
            self.xaxis.set_minor_formatter(minf)
            self.xaxis.set_major_locator(majl)
            self.xaxis.set_minor_locator(minl)
        else:
            self.xaxis._set_scale('linear')

        if self._sharey is not None:
            self.yaxis.major = self._sharey.yaxis.major
            self.yaxis.minor = self._sharey.yaxis.minor
            y0, y1 = self._sharey.get_ylim()
            self.set_ylim(y0, y1, emit=False, auto=None)

            # Save the current formatter/locator so we don't lose it
            majf = self._sharey.yaxis.get_major_formatter()
            minf = self._sharey.yaxis.get_minor_formatter()
            majl = self._sharey.yaxis.get_major_locator()
            minl = self._sharey.yaxis.get_minor_locator()

            # This overwrites the current formatter/locator
            self.yaxis._set_scale(self._sharey.yaxis.get_scale())

            # Reset the formatter/locator
            self.yaxis.set_major_formatter(majf)
            self.yaxis.set_minor_formatter(minf)
            self.yaxis.set_major_locator(majl)
            self.yaxis.set_minor_locator(minl)
        else:
            self.yaxis._set_scale('linear')

        self._autoscaleXon = True
        self._autoscaleYon = True
        self._xmargin = rcParams['axes.xmargin']
        self._ymargin = rcParams['axes.ymargin']
        self._tight = False
        self._update_transScale()  # needed?

        self._get_lines = _process_plot_var_args(self)
        self._get_patches_for_fill = _process_plot_var_args(self, 'fill')

        self._gridOn = rcParams['axes.grid']
        self.lines = []
        self.patches = []
        self.texts = []
        self.tables = []
        self.artists = []
        self.images = []
        self._current_image = None  # strictly for pyplot via _sci, _gci
        self.legend_ = None
        self.collections = []  # collection.Collection instances
        self.containers = []

        self.grid(self._gridOn)
        props = font_manager.FontProperties(size=rcParams['axes.titlesize'])

        self.titleOffsetTrans = mtransforms.ScaledTranslation(
            0.0, 5.0 / 72.0, self.figure.dpi_scale_trans)
        self.title = mtext.Text(
            x=0.5, y=1.0, text='',
            fontproperties=props,
            verticalalignment='baseline',
            horizontalalignment='center',
            )
        self._left_title = mtext.Text(
            x=0.0, y=1.0, text='',
            fontproperties=props,
            verticalalignment='baseline',
            horizontalalignment='left', )
        self._right_title = mtext.Text(
            x=1.0, y=1.0, text='',
            fontproperties=props,
            verticalalignment='baseline',
            horizontalalignment='right',
            )

        for _title in (self.title, self._left_title, self._right_title):
            _title.set_transform(self.transAxes + self.titleOffsetTrans)
            _title.set_clip_box(None)
            self._set_artist_props(_title)

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

        self.axison = True

        self.xaxis.set_clip_path(self.patch)
        self.yaxis.set_clip_path(self.patch)

        self._shared_x_axes.clean()
        self._shared_y_axes.clean()

    def clear(self):
        """clear the axes"""
        self.cla()

    def set_color_cycle(self, clist):
        """
        Set the color cycle for any future plot commands on this Axes.

        *clist* is a list of mpl color specifiers.
        """
        self._get_lines.set_color_cycle(clist)
        self._get_patches_for_fill.set_color_cycle(clist)

    def ishold(self):
        """return the HOLD status of the axes"""
        return self._hold

    def hold(self, b=None):
        """
        Call signature::

          hold(b=None)

        Set the hold state.  If *hold* is *None* (default), toggle the
        *hold* state.  Else set the *hold* state to boolean value *b*.

        Examples::

          # toggle hold
          hold()

          # turn hold on
          hold(True)

          # turn hold off
          hold(False)

        When hold is *True*, subsequent plot commands will be added to
        the current axes.  When hold is *False*, the current axes and
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

          ============   =====================================
          value          description
          ============   =====================================
          'box'          change physical size of axes
          'datalim'      change xlim or ylim
          'box-forced'   same as 'box', but axes can be shared
          ============   =====================================

        'box' does not allow axes sharing, as this can cause
        unintended side effect. For cases when sharing axes is
        fine, use 'box-forced'.

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

        .. deprecated:: 1.2
            the option 'normal' for aspect is deprecated. Use 'auto' instead.
        """
        if aspect == 'normal':
            cbook.warn_deprecated(
                '1.2', name='normal', alternative='auto', obj_type='aspect')
            self._aspect = 'auto'

        elif aspect in ('equal', 'auto'):
            self._aspect = aspect
        else:
            self._aspect = float(aspect)  # raise ValueError if necessary

        if adjustable is not None:
            self.set_adjustable(adjustable)
        if anchor is not None:
            self.set_anchor(anchor)

    def get_adjustable(self):
        return self._adjustable

    def set_adjustable(self, adjustable):
        """
        ACCEPTS: [ 'box' | 'datalim' | 'box-forced']
        """
        if adjustable in ('box', 'datalim', 'box-forced'):
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
                             ', '.join(mtransforms.Bbox.coefs.keys()))

    def get_data_ratio(self):
        """
        Returns the aspect ratio of the raw data.

        This method is intended to be overridden by new projection
        types.
        """
        xmin, xmax = self.get_xbound()
        ymin, ymax = self.get_ybound()

        xsize = max(math.fabs(xmax - xmin), 1e-30)
        ysize = max(math.fabs(ymax - ymin), 1e-30)

        return ysize / xsize

    def get_data_ratio_log(self):
        """
        Returns the aspect ratio of the raw data in log scale.
        Will be used when both axis scales are in log.
        """
        xmin, xmax = self.get_xbound()
        ymin, ymax = self.get_ybound()

        xsize = max(math.fabs(math.log10(xmax) - math.log10(xmin)), 1e-30)
        ysize = max(math.fabs(math.log10(ymax) - math.log10(ymin)), 1e-30)

        return ysize / xsize

    def apply_aspect(self, position=None):
        """
        Use :meth:`_aspect` and :meth:`_adjustable` to modify the
        axes box or the view limits.
        """
        if position is None:
            position = self.get_position(original=True)

        aspect = self.get_aspect()

        if self.name != 'polar':
            xscale, yscale = self.get_xscale(), self.get_yscale()
            if xscale == "linear" and yscale == "linear":
                aspect_scale_mode = "linear"
            elif xscale == "log" and yscale == "log":
                aspect_scale_mode = "log"
            elif ((xscale == "linear" and yscale == "log") or
                  (xscale == "log" and yscale == "linear")):
                if aspect is not "auto":
                    warnings.warn(
                        'aspect is not supported for Axes with xscale=%s, '
                        'yscale=%s' % (xscale, yscale))
                    aspect = "auto"
            else:  # some custom projections have their own scales.
                pass
        else:
            aspect_scale_mode = "linear"

        if aspect == 'auto':
            self.set_position(position, which='active')
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

        figW, figH = self.get_figure().get_size_inches()
        fig_aspect = figH / figW
        if self._adjustable in ['box', 'box-forced']:
            if aspect_scale_mode == "log":
                box_aspect = A * self.get_data_ratio_log()
            else:
                box_aspect = A * self.get_data_ratio()
            pb = position.frozen()
            pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
            self.set_position(pb1.anchored(self.get_anchor(), pb), 'active')
            return

        # reset active to original in case it had been changed
        # by prior use of 'box'
        self.set_position(position, which='active')

        xmin, xmax = self.get_xbound()
        ymin, ymax = self.get_ybound()

        if aspect_scale_mode == "log":
            xmin, xmax = math.log10(xmin), math.log10(xmax)
            ymin, ymax = math.log10(ymin), math.log10(ymax)

        xsize = max(math.fabs(xmax - xmin), 1e-30)
        ysize = max(math.fabs(ymax - ymin), 1e-30)

        l, b, w, h = position.bounds
        box_aspect = fig_aspect * (h / w)
        data_ratio = box_aspect / A

        y_expander = (data_ratio * xsize / ysize - 1.0)
        #print 'y_expander', y_expander
        # If y_expander > 0, the dy/dx viewLim ratio needs to increase
        if abs(y_expander) < 0.005:
            #print 'good enough already'
            return

        if aspect_scale_mode == "log":
            dL = self.dataLim
            dL_width = math.log10(dL.x1) - math.log10(dL.x0)
            dL_height = math.log10(dL.y1) - math.log10(dL.y0)
            xr = 1.05 * dL_width
            yr = 1.05 * dL_height
        else:
            dL = self.dataLim
            xr = 1.05 * dL.width
            yr = 1.05 * dL.height

        xmarg = xsize - xr
        ymarg = ysize - yr
        Ysize = data_ratio * xsize
        Xsize = ysize / data_ratio
        Xmarg = Xsize - xr
        Ymarg = Ysize - yr
        xm = 0  # Setting these targets to, e.g., 0.05*xr does not seem to
                # help.
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
            adjust_y = changey or adjy  # (Ymarg > xmarg)
        if adjust_y:
            yc = 0.5 * (ymin + ymax)
            y0 = yc - Ysize / 2.0
            y1 = yc + Ysize / 2.0
            if aspect_scale_mode == "log":
                self.set_ybound((10. ** y0, 10. ** y1))
            else:
                self.set_ybound((y0, y1))
            #print 'New y0, y1:', y0, y1
            #print 'New ysize, ysize/xsize', y1-y0, (y1-y0)/xsize
        else:
            xc = 0.5 * (xmin + xmax)
            x0 = xc - Xsize / 2.0
            x1 = xc + Xsize / 2.0
            if aspect_scale_mode == "log":
                self.set_xbound((10. ** x0, 10. ** x1))
            else:
                self.set_xbound((x0, x1))
            #print 'New x0, x1:', x0, x1
            #print 'New xsize, ysize/xsize', x1-x0, ysize/(x1-x0)

    def axis(self, *v, **kwargs):
        """
        Convenience method for manipulating the x and y view limits
        and the aspect ratio of the plot. For details, see
        :func:`~matplotlib.pyplot.axis`.

        *kwargs* are passed on to :meth:`set_xlim` and
        :meth:`set_ylim`
        """
        if len(v) == 0 and len(kwargs) == 0:
            xmin, xmax = self.get_xlim()
            ymin, ymax = self.get_ylim()
            return xmin, xmax, ymin, ymax

        if len(v) == 1 and is_string_like(v[0]):
            s = v[0].lower()
            if s == 'on':
                self.set_axis_on()
            elif s == 'off':
                self.set_axis_off()
            elif s in ('equal', 'tight', 'scaled', 'normal', 'auto', 'image'):
                self.set_autoscale_on(True)
                self.set_aspect('auto')
                self.autoscale_view(tight=False)
                # self.apply_aspect()
                if s == 'equal':
                    self.set_aspect('equal', adjustable='datalim')
                elif s == 'scaled':
                    self.set_aspect('equal', adjustable='box', anchor='C')
                    self.set_autoscale_on(False)  # Req. by Mark Bakker
                elif s == 'tight':
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

        emit = kwargs.get('emit', True)
        try:
            v[0]
        except IndexError:
            xmin = kwargs.get('xmin', None)
            xmax = kwargs.get('xmax', None)
            auto = False  # turn off autoscaling, unless...
            if xmin is None and xmax is None:
                auto = None  # leave autoscaling state alone
            xmin, xmax = self.set_xlim(xmin, xmax, emit=emit, auto=auto)

            ymin = kwargs.get('ymin', None)
            ymax = kwargs.get('ymax', None)
            auto = False  # turn off autoscaling, unless...
            if ymin is None and ymax is None:
                auto = None  # leave autoscaling state alone
            ymin, ymax = self.set_ylim(ymin, ymax, emit=emit, auto=auto)
            return xmin, xmax, ymin, ymax

        v = v[0]
        if len(v) != 4:
            raise ValueError('v must contain [xmin xmax ymin ymax]')

        self.set_xlim([v[0], v[1]], emit=emit, auto=False)
        self.set_ylim([v[2], v[3]], emit=emit, auto=False)

        return v

    def get_legend(self):
        """
        Return the legend.Legend instance, or None if no legend is defined
        """
        return self.legend_

    def get_images(self):
        """return a list of Axes images contained by the Axes"""
        return cbook.silent_list('AxesImage', self.images)

    def get_lines(self):
        """Return a list of lines contained by the Axes"""
        return cbook.silent_list('Line2D', self.lines)

    def get_xaxis(self):
        """Return the XAxis instance"""
        return self.xaxis

    def get_xgridlines(self):
        """Get the x grid lines as a list of Line2D instances"""
        return cbook.silent_list('Line2D xgridline',
                                 self.xaxis.get_gridlines())

    def get_xticklines(self):
        """Get the xtick lines as a list of Line2D instances"""
        return cbook.silent_list('Text xtickline',
                                 self.xaxis.get_ticklines())

    def get_yaxis(self):
        """Return the YAxis instance"""
        return self.yaxis

    def get_ygridlines(self):
        """Get the y grid lines as a list of Line2D instances"""
        return cbook.silent_list('Line2D ygridline',
                                 self.yaxis.get_gridlines())

    def get_yticklines(self):
        """Get the ytick lines as a list of Line2D instances"""
        return cbook.silent_list('Line2D ytickline',
                                 self.yaxis.get_ticklines())

    #### Adding and tracking artists

    def _sci(self, im):
        """
        helper for :func:`~matplotlib.pyplot.sci`;
        do not use elsewhere.
        """
        if isinstance(im, matplotlib.contour.ContourSet):
            if im.collections[0] not in self.collections:
                raise ValueError(
                    "ContourSet must be in current Axes")
        elif im not in self.images and im not in self.collections:
            raise ValueError(
                "Argument must be an image, collection, or ContourSet in "
                "this Axes")
        self._current_image = im

    def _gci(self):
        """
        Helper for :func:`~matplotlib.pyplot.gci`;
        do not use elsewhere.
        """
        return self._current_image

    def has_data(self):
        """
        Return *True* if any artists have been added to axes.

        This should not be used to determine whether the *dataLim*
        need to be updated, and may not actually be useful for
        anything.
        """
        return (
            len(self.collections) +
            len(self.images) +
            len(self.lines) +
            len(self.patches)) > 0

    def add_artist(self, a):
        """
        Add any :class:`~matplotlib.artist.Artist` to the axes.

        Returns the artist.
        """
        a.set_axes(self)
        self.artists.append(a)
        self._set_artist_props(a)
        a.set_clip_path(self.patch)
        a._remove_method = lambda h: self.artists.remove(h)
        return a

    def add_collection(self, collection, autolim=True):
        """
        Add a :class:`~matplotlib.collections.Collection` instance
        to the axes.

        Returns the collection.
        """
        label = collection.get_label()
        if not label:
            collection.set_label('_collection%d' % len(self.collections))
        self.collections.append(collection)
        self._set_artist_props(collection)

        if collection.get_clip_path() is None:
            collection.set_clip_path(self.patch)

        if (autolim and
            collection._paths is not None and
            len(collection._paths) and
            len(collection._offsets)):
            self.update_datalim(collection.get_datalim(self.transData))

        collection._remove_method = lambda h: self.collections.remove(h)
        return collection

    def add_line(self, line):
        """
        Add a :class:`~matplotlib.lines.Line2D` to the list of plot
        lines

        Returns the line.
        """
        self._set_artist_props(line)
        if line.get_clip_path() is None:
            line.set_clip_path(self.patch)

        self._update_line_limits(line)
        if not line.get_label():
            line.set_label('_line%d' % len(self.lines))
        self.lines.append(line)
        line._remove_method = lambda h: self.lines.remove(h)
        return line

    def _update_line_limits(self, line):
        """
        Figures out the data limit of the given line, updating self.dataLim.
        """
        path = line.get_path()
        if path.vertices.size == 0:
            return

        line_trans = line.get_transform()

        if line_trans == self.transData:
            data_path = path

        elif any(line_trans.contains_branch_seperately(self.transData)):
            # identify the transform to go from line's coordinates
            # to data coordinates
            trans_to_data = line_trans - self.transData

            # if transData is affine we can use the cached non-affine component
            # of line's path. (since the non-affine part of line_trans is
            # entirely encapsulated in trans_to_data).
            if self.transData.is_affine:
                line_trans_path = line._get_transformed_path()
                na_path, _ = line_trans_path.get_transformed_path_and_affine()
                data_path = trans_to_data.transform_path_affine(na_path)
            else:
                data_path = trans_to_data.transform_path(path)
        else:
            # for backwards compatibility we update the dataLim with the
            # coordinate range of the given path, even though the coordinate
            # systems are completely different. This may occur in situations
            # such as when ax.transAxes is passed through for absolute
            # positioning.
            data_path = path

        if data_path.vertices.size > 0:
            updatex, updatey = line_trans.contains_branch_seperately(
                                                               self.transData
                                                                    )
            self.dataLim.update_from_path(data_path,
                                          self.ignore_existing_data_limits,
                                          updatex=updatex,
                                          updatey=updatey)
            self.ignore_existing_data_limits = False

    def add_patch(self, p):
        """
        Add a :class:`~matplotlib.patches.Patch` *p* to the list of
        axes patches; the clipbox will be set to the Axes clipping
        box.  If the transform is not set, it will be set to
        :attr:`transData`.

        Returns the patch.
        """

        self._set_artist_props(p)
        if p.get_clip_path() is None:
            p.set_clip_path(self.patch)
        self._update_patch_limits(p)
        self.patches.append(p)
        p._remove_method = lambda h: self.patches.remove(h)
        return p

    def _update_patch_limits(self, patch):
        """update the data limits for patch *p*"""
        # hist can add zero height Rectangles, which is useful to keep
        # the bins, counts and patches lined up, but it throws off log
        # scaling.  We'll ignore rects with zero height or width in
        # the auto-scaling

        # cannot check for '==0' since unitized data may not compare to zero
        if (isinstance(patch, mpatches.Rectangle) and
                    ((not patch.get_width()) or (not patch.get_height()))):
            return
        vertices = patch.get_path().vertices
        if vertices.size > 0:
            xys = patch.get_patch_transform().transform(vertices)
            if patch.get_data_transform() != self.transData:
                patch_to_data = (patch.get_data_transform() -
                                    self.transData)
                xys = patch_to_data.transform(xys)

            updatex, updatey = patch.get_transform().\
                                    contains_branch_seperately(self.transData)
            self.update_datalim(xys, updatex=updatex,
                                     updatey=updatey)

    def add_table(self, tab):
        """
        Add a :class:`~matplotlib.tables.Table` instance to the
        list of axes tables

        Returns the table.
        """
        self._set_artist_props(tab)
        self.tables.append(tab)
        tab.set_clip_path(self.patch)
        tab._remove_method = lambda h: self.tables.remove(h)
        return tab

    def add_container(self, container):
        """
        Add a :class:`~matplotlib.container.Container` instance
        to the axes.

        Returns the collection.
        """
        label = container.get_label()
        if not label:
            container.set_label('_container%d' % len(self.containers))
        self.containers.append(container)
        container.set_remove_method(lambda h: self.containers.remove(h))
        return container

    def relim(self):
        """
        Recompute the data limits based on current artists.

        At present, :class:`~matplotlib.collections.Collection`
        instances are not supported.
        """
        # Collections are deliberately not supported (yet); see
        # the TODO note in artists.py.
        self.dataLim.ignore(True)
        self.dataLim.set_points(mtransforms.Bbox.null().get_points())
        self.ignore_existing_data_limits = True

        for line in self.lines:
            self._update_line_limits(line)

        for p in self.patches:
            self._update_patch_limits(p)


    def update_datalim(self, xys, updatex=True, updatey=True):
        """
        Update the data lim bbox with seq of xy tups or equiv. 2-D array
        """
        # if no data is set currently, the bbox will ignore its
        # limits and set the bound to be the bounds of the xydata.
        # Otherwise, it will compute the bounds of it's current data
        # and the data in xydata

        if iterable(xys) and not len(xys):
            return
        if not ma.isMaskedArray(xys):
            xys = np.asarray(xys)
        self.dataLim.update_from_data_xy(xys, self.ignore_existing_data_limits,
                                           updatex=updatex, updatey=updatey)
        self.ignore_existing_data_limits = False

    def update_datalim_numerix(self, x, y):
        """
        Update the data lim bbox with seq of xy tups
        """
        # if no data is set currently, the bbox will ignore it's
        # limits and set the bound to be the bounds of the xydata.
        # Otherwise, it will compute the bounds of it's current data
        # and the data in xydata
        if iterable(x) and not len(x):
            return
        self.dataLim.update_from_data(x, y, self.ignore_existing_data_limits)
        self.ignore_existing_data_limits = False

    def update_datalim_bounds(self, bounds):
        """
        Update the datalim to include the given
        :class:`~matplotlib.transforms.Bbox` *bounds*
        """
        self.dataLim.set(mtransforms.Bbox.union([self.dataLim, bounds]))

    def _process_unit_info(self, xdata=None, ydata=None, kwargs=None):
        """Look for unit *kwargs* and update the axis instances as necessary"""

        if self.xaxis is None or self.yaxis is None:
            return

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
            xunits = kwargs.pop('xunits', self.xaxis.units)
            if self.name == 'polar':
                xunits = kwargs.pop('thetaunits', xunits)
            if xunits != self.xaxis.units:
                #print '\tkw setting xunits', xunits
                self.xaxis.set_units(xunits)
                # If the units being set imply a different converter,
                # we need to update.
                if xdata is not None:
                    self.xaxis.update_units(xdata)

            yunits = kwargs.pop('yunits', self.yaxis.units)
            if self.name == 'polar':
                yunits = kwargs.pop('runits', yunits)
            if yunits != self.yaxis.units:
                #print '\tkw setting yunits', yunits
                self.yaxis.set_units(yunits)
                # If the units being set imply a different converter,
                # we need to update.
                if ydata is not None:
                    self.yaxis.update_units(ydata)

    def in_axes(self, mouseevent):
        """
        Return *True* if the given *mouseevent* (in display coords)
        is in the Axes
        """
        return self.patch.contains(mouseevent)[0]

    def get_autoscale_on(self):
        """
        Get whether autoscaling is applied for both axes on plot commands
        """
        return self._autoscaleXon and self._autoscaleYon

    def get_autoscalex_on(self):
        """
        Get whether autoscaling for the x-axis is applied on plot commands
        """
        return self._autoscaleXon

    def get_autoscaley_on(self):
        """
        Get whether autoscaling for the y-axis is applied on plot commands
        """
        return self._autoscaleYon

    def set_autoscale_on(self, b):
        """
        Set whether autoscaling is applied on plot commands

        accepts: [ *True* | *False* ]
        """
        self._autoscaleXon = b
        self._autoscaleYon = b

    def set_autoscalex_on(self, b):
        """
        Set whether autoscaling for the x-axis is applied on plot commands

        accepts: [ *True* | *False* ]
        """
        self._autoscaleXon = b

    def set_autoscaley_on(self, b):
        """
        Set whether autoscaling for the y-axis is applied on plot commands

        accepts: [ *True* | *False* ]
        """
        self._autoscaleYon = b

    def set_xmargin(self, m):
        """
        Set padding of X data limits prior to autoscaling.

        *m* times the data interval will be added to each
        end of that interval before it is used in autoscaling.

        accepts: float in range 0 to 1
        """
        if m < 0 or m > 1:
            raise ValueError("margin must be in range 0 to 1")
        self._xmargin = m

    def set_ymargin(self, m):
        """
        Set padding of Y data limits prior to autoscaling.

        *m* times the data interval will be added to each
        end of that interval before it is used in autoscaling.

        accepts: float in range 0 to 1
        """
        if m < 0 or m > 1:
            raise ValueError("margin must be in range 0 to 1")
        self._ymargin = m

    def margins(self, *args, **kw):
        """
        Set or retrieve autoscaling margins.

        signatures::

            margins()

        returns xmargin, ymargin

        ::

            margins(margin)

            margins(xmargin, ymargin)

            margins(x=xmargin, y=ymargin)

            margins(..., tight=False)

        All three forms above set the xmargin and ymargin parameters.
        All keyword parameters are optional.  A single argument
        specifies both xmargin and ymargin.  The *tight* parameter
        is passed to :meth:`autoscale_view`, which is executed after
        a margin is changed; the default here is *True*, on the
        assumption that when margins are specified, no additional
        padding to match tick marks is usually desired.  Setting
        *tight* to *None* will preserve the previous setting.

        Specifying any margin changes only the autoscaling; for example,
        if *xmargin* is not None, then *xmargin* times the X data
        interval will be added to each end of that interval before
        it is used in autoscaling.

        """
        if not args and not kw:
            return self._xmargin, self._ymargin

        tight = kw.pop('tight', True)
        mx = kw.pop('x', None)
        my = kw.pop('y', None)
        if len(args) == 1:
            mx = my = args[0]
        elif len(args) == 2:
            mx, my = args
        else:
            raise ValueError("more than two arguments were supplied")
        if mx is not None:
            self.set_xmargin(mx)
        if my is not None:
            self.set_ymargin(my)

        scalex = (mx is not None)
        scaley = (my is not None)

        self.autoscale_view(tight=tight, scalex=scalex, scaley=scaley)

    def set_rasterization_zorder(self, z):
        """
        Set zorder value below which artists will be rasterized.  Set
        to `None` to disable rasterizing of artists below a particular
        zorder.
        """
        self._rasterization_zorder = z

    def get_rasterization_zorder(self):
        """
        Get zorder value below which artists will be rasterized
        """
        return self._rasterization_zorder

    def autoscale(self, enable=True, axis='both', tight=None):
        """
        Autoscale the axis view to the data (toggle).

        Convenience method for simple axis view autoscaling.
        It turns autoscaling on or off, and then,
        if autoscaling for either axis is on, it performs
        the autoscaling on the specified axis or axes.

        *enable*: [True | False | None]
            True (default) turns autoscaling on, False turns it off.
            None leaves the autoscaling state unchanged.

        *axis*: ['x' | 'y' | 'both']
            which axis to operate on; default is 'both'

        *tight*: [True | False | None]
            If True, set view limits to data limits;
            if False, let the locator and margins expand the view limits;
            if None, use tight scaling if the only artist is an image,
            otherwise treat *tight* as False.
            The *tight* setting is retained for future autoscaling
            until it is explicitly changed.


        Returns None.
        """
        if enable is None:
            scalex = True
            scaley = True
        else:
            scalex = False
            scaley = False
            if axis in ['x', 'both']:
                self._autoscaleXon = bool(enable)
                scalex = self._autoscaleXon
            if axis in ['y', 'both']:
                self._autoscaleYon = bool(enable)
                scaley = self._autoscaleYon
        self.autoscale_view(tight=tight, scalex=scalex, scaley=scaley)

    def autoscale_view(self, tight=None, scalex=True, scaley=True):
        """
        Autoscale the view limits using the data limits. You can
        selectively autoscale only a single axis, eg, the xaxis by
        setting *scaley* to *False*.  The autoscaling preserves any
        axis direction reversal that has already been done.

        The data limits are not updated automatically when artist data are
        changed after the artist has been added to an Axes instance.  In that
        case, use :meth:`matplotlib.axes.Axes.relim` prior to calling
        autoscale_view.
        """
        if tight is None:
            # if image data only just use the datalim
            _tight = self._tight or (len(self.images) > 0 and
                                     len(self.lines) == 0 and
                                     len(self.patches) == 0)
        else:
            _tight = self._tight = bool(tight)

        if scalex and self._autoscaleXon:
            xshared = self._shared_x_axes.get_siblings(self)
            dl = [ax.dataLim for ax in xshared]
            bb = mtransforms.BboxBase.union(dl)
            x0, x1 = bb.intervalx
            xlocator = self.xaxis.get_major_locator()
            try:
                # e.g., DateLocator has its own nonsingular()
                x0, x1 = xlocator.nonsingular(x0, x1)
            except AttributeError:
                # Default nonsingular for, e.g., MaxNLocator
                x0, x1 = mtransforms.nonsingular(x0, x1, increasing=False,
                                                         expander=0.05)
            if self._xmargin > 0:
                delta = (x1 - x0) * self._xmargin
                x0 -= delta
                x1 += delta
            if not _tight:
                x0, x1 = xlocator.view_limits(x0, x1)
            self.set_xbound(x0, x1)

        if scaley and self._autoscaleYon:
            yshared = self._shared_y_axes.get_siblings(self)
            dl = [ax.dataLim for ax in yshared]
            bb = mtransforms.BboxBase.union(dl)
            y0, y1 = bb.intervaly
            ylocator = self.yaxis.get_major_locator()
            try:
                y0, y1 = ylocator.nonsingular(y0, y1)
            except AttributeError:
                y0, y1 = mtransforms.nonsingular(y0, y1, increasing=False,
                                                         expander=0.05)
            if self._ymargin > 0:
                delta = (y1 - y0) * self._ymargin
                y0 -= delta
                y1 += delta
            if not _tight:
                y0, y1 = ylocator.view_limits(y0, y1)
            self.set_ybound(y0, y1)

    #### Drawing

    @allow_rasterization
    def draw(self, renderer=None, inframe=False):
        """Draw everything (plot lines, axes, labels)"""
        if renderer is None:
            renderer = self._cachedRenderer

        if renderer is None:
            raise RuntimeError('No renderer defined')
        if not self.get_visible():
            return
        renderer.open_group('axes')

        locator = self.get_axes_locator()
        if locator:
            pos = locator(self, renderer)
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        artists = []

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
        if not inframe:
            artists.append(self.title)
            artists.append(self._left_title)
            artists.append(self._right_title)
        artists.extend(self.tables)
        if self.legend_ is not None:
            artists.append(self.legend_)

        # the frame draws the edges around the axes patch -- we
        # decouple these so the patch can be in the background and the
        # frame in the foreground.
        if self.axison and self._frameon:
            artists.extend(self.spines.itervalues())

        if self.figure.canvas.is_saving():
            dsu = [(a.zorder, a) for a in artists]
        else:
            dsu = [(a.zorder, a) for a in artists
                   if not a.get_animated()]

        # add images to dsu if the backend support compositing.
        # otherwise, does the manaul compositing  without adding images to dsu.
        if len(self.images) <= 1 or renderer.option_image_nocomposite():
            dsu.extend([(im.zorder, im) for im in self.images])
            _do_composite = False
        else:
            _do_composite = True

        dsu.sort(key=itemgetter(0))

        # rasterize artists with negative zorder
        # if the minimum zorder is negative, start rasterization
        rasterization_zorder = self._rasterization_zorder
        if (rasterization_zorder is not None and
            len(dsu) > 0 and dsu[0][0] < rasterization_zorder):
            renderer.start_rasterizing()
            dsu_rasterized = [l for l in dsu if l[0] < rasterization_zorder]
            dsu = [l for l in dsu if l[0] >= rasterization_zorder]
        else:
            dsu_rasterized = []

        # the patch draws the background rectangle -- the frame below
        # will draw the edges
        if self.axison and self._frameon:
            self.patch.draw(renderer)

        if _do_composite:
            # make a composite image blending alpha
            # list of (mimage.Image, ox, oy)

            zorder_images = [(im.zorder, im) for im in self.images
                             if im.get_visible()]
            zorder_images.sort(key=lambda x: x[0])

            mag = renderer.get_image_magnification()
            ims = [(im.make_image(mag), 0, 0, im.get_alpha()) for z, im in zorder_images]

            l, b, r, t = self.bbox.extents
            width = mag * ((round(r) + 0.5) - (round(l) - 0.5))
            height = mag * ((round(t) + 0.5) - (round(b) - 0.5))
            im = mimage.from_images(height,
                                    width,
                                    ims)

            im.is_grayscale = False
            l, b, w, h = self.bbox.bounds
            # composite images need special args so they will not
            # respect z-order for now

            gc = renderer.new_gc()
            gc.set_clip_rectangle(self.bbox)
            gc.set_clip_path(mtransforms.TransformedPath(
                    self.patch.get_path(),
                    self.patch.get_transform()))

            renderer.draw_image(gc, round(l), round(b), im)
            gc.restore()

        if dsu_rasterized:
            for zorder, a in dsu_rasterized:
                a.draw(renderer)
            renderer.stop_rasterizing()

        for zorder, a in dsu:
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
        Set whether the axis ticks and gridlines are above or below most
        artists

        ACCEPTS: [ *True* | *False* ]
        """
        self._axisbelow = b

    @docstring.dedent_interpd
    def grid(self, b=None, which='major', axis='both', **kwargs):
        """
        Turn the axes grids on or off.

        Call signature::

           grid(self, b=None, which='major', axis='both', **kwargs)

        Set the axes grids on or off; *b* is a boolean.  (For MATLAB
        compatibility, *b* may also be a string, 'on' or 'off'.)

        If *b* is *None* and ``len(kwargs)==0``, toggle the grid state.  If
        *kwargs* are supplied, it is assumed that you want a grid and *b*
        is thus set to *True*.

        *which* can be 'major' (default), 'minor', or 'both' to control
        whether major tick grids, minor tick grids, or both are affected.

        *axis* can be 'both' (default), 'x', or 'y' to control which
        set of gridlines are drawn.

        *kwargs* are used to set the grid line properties, eg::

           ax.grid(color='r', linestyle='-', linewidth=2)

        Valid :class:`~matplotlib.lines.Line2D` kwargs are

        %(Line2D)s

        """
        if len(kwargs):
            b = True
        b = _string_to_bool(b)

        if axis == 'x' or  axis == 'both':
            self.xaxis.grid(b, which=which, **kwargs)
        if axis == 'y' or  axis == 'both':
            self.yaxis.grid(b, which=which, **kwargs)

    def ticklabel_format(self, **kwargs):
        """
        Change the `~matplotlib.ticker.ScalarFormatter` used by
        default for linear axes.

        Optional keyword arguments:

          ============   =========================================
          Keyword        Description
          ============   =========================================
          *style*        [ 'sci' (or 'scientific') | 'plain' ]
                         plain turns off scientific notation
          *scilimits*    (m, n), pair of integers; if *style*
                         is 'sci', scientific notation will
                         be used for numbers outside the range
                         10`m`:sup: to 10`n`:sup:.
                         Use (0,0) to include all numbers.
          *useOffset*    [True | False | offset]; if True,
                         the offset will be calculated as needed;
                         if False, no offset will be used; if a
                         numeric offset is specified, it will be
                         used.
          *axis*         [ 'x' | 'y' | 'both' ]
          *useLocale*    If True, format the number according to
                         the current locale.  This affects things
                         such as the character used for the
                         decimal separator.  If False, use
                         C-style (English) formatting.  The
                         default setting is controlled by the
                         axes.formatter.use_locale rcparam.
          ============   =========================================

        Only the major ticks are affected.
        If the method is called when the
        :class:`~matplotlib.ticker.ScalarFormatter` is not the
        :class:`~matplotlib.ticker.Formatter` being used, an
        :exc:`AttributeError` will be raised.

        """
        style = kwargs.pop('style', '').lower()
        scilimits = kwargs.pop('scilimits', None)
        useOffset = kwargs.pop('useOffset', None)
        useLocale = kwargs.pop('useLocale', None)
        axis = kwargs.pop('axis', 'both').lower()
        if scilimits is not None:
            try:
                m, n = scilimits
                m + n + 1  # check that both are numbers
            except (ValueError, TypeError):
                raise ValueError("scilimits must be a sequence of 2 integers")
        if style[:3] == 'sci':
            sb = True
        elif style in ['plain', 'comma']:
            sb = False
            if style == 'plain':
                cb = False
            else:
                cb = True
                raise NotImplementedError("comma style remains to be added")
        elif style == '':
            sb = None
        else:
            raise ValueError("%s is not a valid style value")
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
            if useOffset is not None:
                if axis == 'both' or axis == 'x':
                    self.xaxis.major.formatter.set_useOffset(useOffset)
                if axis == 'both' or axis == 'y':
                    self.yaxis.major.formatter.set_useOffset(useOffset)
            if useLocale is not None:
                if axis == 'both' or axis == 'x':
                    self.xaxis.major.formatter.set_useLocale(useLocale)
                if axis == 'both' or axis == 'y':
                    self.yaxis.major.formatter.set_useLocale(useLocale)
        except AttributeError:
            raise AttributeError(
                "This method only works with the ScalarFormatter.")

    def locator_params(self, axis='both', tight=None, **kwargs):
        """
        Control behavior of tick locators.

        Keyword arguments:

        *axis*
            ['x' | 'y' | 'both']  Axis on which to operate;
            default is 'both'.

        *tight*
            [True | False | None] Parameter passed to :meth:`autoscale_view`.
            Default is None, for no change.

        Remaining keyword arguments are passed to directly to the
        :meth:`~matplotlib.ticker.MaxNLocator.set_params` method.

        Typically one might want to reduce the maximum number
        of ticks and use tight bounds when plotting small
        subplots, for example::

            ax.locator_params(tight=True, nbins=4)

        Because the locator is involved in autoscaling,
        :meth:`autoscale_view` is called automatically after
        the parameters are changed.

        This presently works only for the
        :class:`~matplotlib.ticker.MaxNLocator` used
        by default on linear axes, but it may be generalized.
        """
        _x = axis in ['x', 'both']
        _y = axis in ['y', 'both']
        if _x:
            self.xaxis.get_major_locator().set_params(**kwargs)
        if _y:
            self.yaxis.get_major_locator().set_params(**kwargs)
        self.autoscale_view(tight=tight, scalex=_x, scaley=_y)

    def tick_params(self, axis='both', **kwargs):
        """
        Change the appearance of ticks and tick labels.

        Keyword arguments:

        *axis* : ['x' | 'y' | 'both']
            Axis on which to operate; default is 'both'.

        *reset* : [True | False]
            If *True*, set all parameters to defaults
            before processing other keyword arguments.  Default is
            *False*.

        *which* : ['major' | 'minor' | 'both']
            Default is 'major'; apply arguments to *which* ticks.

        *direction* : ['in' | 'out' | 'inout']
            Puts ticks inside the axes, outside the axes, or both.

        *length*
            Tick length in points.

        *width*
            Tick width in points.

        *color*
            Tick color; accepts any mpl color spec.

        *pad*
            Distance in points between tick and label.

        *labelsize*
            Tick label font size in points or as a string (e.g., 'large').

        *labelcolor*
            Tick label color; mpl color spec.

        *colors*
            Changes the tick color and the label color to the same value:
            mpl color spec.

        *zorder*
            Tick and label zorder.

        *bottom*, *top*, *left*, *right* : [bool | 'on' | 'off']
            controls whether to draw the respective ticks.

        *labelbottom*, *labeltop*, *labelleft*, *labelright*
            Boolean or ['on' | 'off'], controls whether to draw the
            respective tick labels.

        Example::

            ax.tick_params(direction='out', length=6, width=2, colors='r')

        This will make all major ticks be red, pointing out of the box,
        and with dimensions 6 points by 2 points.  Tick labels will
        also be red.

        """
        if axis in ['x', 'both']:
            xkw = dict(kwargs)
            xkw.pop('left', None)
            xkw.pop('right', None)
            xkw.pop('labelleft', None)
            xkw.pop('labelright', None)
            self.xaxis.set_tick_params(**xkw)
        if axis in ['y', 'both']:
            ykw = dict(kwargs)
            ykw.pop('top', None)
            ykw.pop('bottom', None)
            ykw.pop('labeltop', None)
            ykw.pop('labelbottom', None)
            self.yaxis.set_tick_params(**ykw)

    def set_axis_off(self):
        """turn off the axis"""
        self.axison = False

    def set_axis_on(self):
        """turn on the axis"""
        self.axison = True

    def get_axis_bgcolor(self):
        """Return the axis background color"""
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
        self.set_xlim(right, left, auto=None)

    def xaxis_inverted(self):
        """Returns *True* if the x-axis is inverted."""
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
        It will not change the _autoscaleXon attribute.
        """
        if upper is None and iterable(lower):
            lower, upper = lower

        old_lower, old_upper = self.get_xbound()

        if lower is None:
            lower = old_lower
        if upper is None:
            upper = old_upper

        if self.xaxis_inverted():
            if lower < upper:
                self.set_xlim(upper, lower, auto=None)
            else:
                self.set_xlim(lower, upper, auto=None)
        else:
            if lower < upper:
                self.set_xlim(lower, upper, auto=None)
            else:
                self.set_xlim(upper, lower, auto=None)

    def get_xlim(self):
        """
        Get the x-axis range [*left*, *right*]
        """
        return tuple(self.viewLim.intervalx)

    def set_xlim(self, left=None, right=None, emit=True, auto=False, **kw):
        """
        Call signature::

          set_xlim(self, *args, **kwargs):

        Set the data limits for the xaxis

        Examples::

          set_xlim((left, right))
          set_xlim(left, right)
          set_xlim(left=1) # right unchanged
          set_xlim(right=1) # left unchanged

        Keyword arguments:

          *left*: scalar
            The left xlim; *xmin*, the previous name, may still be used

          *right*: scalar
            The right xlim; *xmax*, the previous name, may still be used

          *emit*: [ *True* | *False* ]
            Notify observers of limit change

          *auto*: [ *True* | *False* | *None* ]
            Turn *x* autoscaling on (*True*), off (*False*; default),
            or leave unchanged (*None*)

        Note, the *left* (formerly *xmin*) value may be greater than
        the *right* (formerly *xmax*).
        For example, suppose *x* is years before present.
        Then one might use::

          set_ylim(5000, 0)

        so 5000 years ago is on the left of the plot and the
        present is on the right.

        Returns the current xlimits as a length 2 tuple

        ACCEPTS: length 2 sequence of floats
        """
        if 'xmin' in kw:
            left = kw.pop('xmin')
        if 'xmax' in kw:
            right = kw.pop('xmax')
        if kw:
            raise ValueError("unrecognized kwargs: %s" % kw.keys())

        if right is None and iterable(left):
            left, right = left

        self._process_unit_info(xdata=(left, right))
        if left is not None:
            left = self.convert_xunits(left)
        if right is not None:
            right = self.convert_xunits(right)

        old_left, old_right = self.get_xlim()
        if left is None:
            left = old_left
        if right is None:
            right = old_right

        if left == right:
            warnings.warn(('Attempting to set identical left==right results\n'
                   + 'in singular transformations; automatically expanding.\n'
                   + 'left=%s, right=%s') % (left, right))
        left, right = mtransforms.nonsingular(left, right, increasing=False)
        left, right = self.xaxis.limit_range_for_scale(left, right)

        self.viewLim.intervalx = (left, right)
        if auto is not None:
            self._autoscaleXon = bool(auto)

        if emit:
            self.callbacks.process('xlim_changed', self)
            # Call all of the other x-axes that are shared with this one
            for other in self._shared_x_axes.get_siblings(self):
                if other is not self:
                    other.set_xlim(self.viewLim.intervalx,
                                            emit=False, auto=auto)
                    if (other.figure != self.figure and
                        other.figure.canvas is not None):
                        other.figure.canvas.draw_idle()

        return left, right

    def get_xscale(self):
        return self.xaxis.get_scale()
    get_xscale.__doc__ = "Return the xaxis scale string: %s""" % (
                                ", ".join(mscale.get_scale_names()))

    @docstring.dedent_interpd
    def set_xscale(self, value, **kwargs):
        """
        Call signature::

          set_xscale(value)

        Set the scaling of the x-axis: %(scale)s

        ACCEPTS: [%(scale)s]

        Different kwargs are accepted, depending on the scale:
        %(scale_docs)s
        """
        self.xaxis._set_scale(value, **kwargs)
        self.autoscale_view(scaley=False)
        self._update_transScale()

    def get_xticks(self, minor=False):
        """Return the x ticks as a list of locations"""
        return self.xaxis.get_ticklocs(minor=minor)

    def set_xticks(self, ticks, minor=False):
        """
        Set the x ticks with list of *ticks*

        ACCEPTS: sequence of floats
        """
        return self.xaxis.set_ticks(ticks, minor=minor)

    def get_xmajorticklabels(self):
        """
        Get the xtick labels as a list of :class:`~matplotlib.text.Text`
        instances.
        """
        return cbook.silent_list('Text xticklabel',
                                 self.xaxis.get_majorticklabels())

    def get_xminorticklabels(self):
        """
        Get the x minor tick labels as a list of
        :class:`matplotlib.text.Text` instances.
        """
        return cbook.silent_list('Text xticklabel',
                                 self.xaxis.get_minorticklabels())

    def get_xticklabels(self, minor=False):
        """
        Get the x tick labels as a list of :class:`~matplotlib.text.Text`
        instances.
        """
        return cbook.silent_list('Text xticklabel',
                                 self.xaxis.get_ticklabels(minor=minor))

    @docstring.dedent_interpd
    def set_xticklabels(self, labels, fontdict=None, minor=False, **kwargs):
        """
        Call signature::

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

    def invert_yaxis(self):
        """
        Invert the y-axis.
        """
        bottom, top = self.get_ylim()
        self.set_ylim(top, bottom, auto=None)

    def yaxis_inverted(self):
        """Returns *True* if the y-axis is inverted."""
        bottom, top = self.get_ylim()
        return top < bottom

    def get_ybound(self):
        """
        Return y-axis numerical bounds in the form of
        ``lowerBound < upperBound``
        """
        bottom, top = self.get_ylim()
        if bottom < top:
            return bottom, top
        else:
            return top, bottom

    def set_ybound(self, lower=None, upper=None):
        """
        Set the lower and upper numerical bounds of the y-axis.
        This method will honor axes inversion regardless of parameter order.
        It will not change the _autoscaleYon attribute.
        """
        if upper is None and iterable(lower):
            lower, upper = lower

        old_lower, old_upper = self.get_ybound()

        if lower is None:
            lower = old_lower
        if upper is None:
            upper = old_upper

        if self.yaxis_inverted():
            if lower < upper:
                self.set_ylim(upper, lower, auto=None)
            else:
                self.set_ylim(lower, upper, auto=None)
        else:
            if lower < upper:
                self.set_ylim(lower, upper, auto=None)
            else:
                self.set_ylim(upper, lower, auto=None)

    def get_ylim(self):
        """
        Get the y-axis range [*bottom*, *top*]
        """
        return tuple(self.viewLim.intervaly)

    def set_ylim(self, bottom=None, top=None, emit=True, auto=False, **kw):
        """
        Call signature::

          set_ylim(self, *args, **kwargs):

        Set the data limits for the yaxis

        Examples::

          set_ylim((bottom, top))
          set_ylim(bottom, top)
          set_ylim(bottom=1) # top unchanged
          set_ylim(top=1) # bottom unchanged

        Keyword arguments:

          *bottom*: scalar
            The bottom ylim; the previous name, *ymin*, may still be used

          *top*: scalar
            The top ylim; the previous name, *ymax*, may still be used

          *emit*: [ *True* | *False* ]
            Notify observers of limit change

          *auto*: [ *True* | *False* | *None* ]
            Turn *y* autoscaling on (*True*), off (*False*; default),
            or leave unchanged (*None*)

        Note, the *bottom* (formerly *ymin*) value may be greater than
        the *top* (formerly *ymax*).
        For example, suppose *y* is depth in the ocean.
        Then one might use::

          set_ylim(5000, 0)

        so 5000 m depth is at the bottom of the plot and the
        surface, 0 m, is at the top.

        Returns the current ylimits as a length 2 tuple

        ACCEPTS: length 2 sequence of floats
        """
        if 'ymin' in kw:
            bottom = kw.pop('ymin')
        if 'ymax' in kw:
            top = kw.pop('ymax')
        if kw:
            raise ValueError("unrecognized kwargs: %s" % kw.keys())

        if top is None and iterable(bottom):
            bottom, top = bottom

        if bottom is not None:
            bottom = self.convert_yunits(bottom)
        if top is not None:
            top = self.convert_yunits(top)

        old_bottom, old_top = self.get_ylim()

        if bottom is None:
            bottom = old_bottom
        if top is None:
            top = old_top

        if bottom == top:
            warnings.warn(('Attempting to set identical bottom==top results\n'
                   + 'in singular transformations; automatically expanding.\n'
                   + 'bottom=%s, top=%s') % (bottom, top))

        bottom, top = mtransforms.nonsingular(bottom, top, increasing=False)
        bottom, top = self.yaxis.limit_range_for_scale(bottom, top)

        self.viewLim.intervaly = (bottom, top)
        if auto is not None:
            self._autoscaleYon = bool(auto)

        if emit:
            self.callbacks.process('ylim_changed', self)
            # Call all of the other y-axes that are shared with this one
            for other in self._shared_y_axes.get_siblings(self):
                if other is not self:
                    other.set_ylim(self.viewLim.intervaly,
                                            emit=False, auto=auto)
                    if (other.figure != self.figure and
                        other.figure.canvas is not None):
                        other.figure.canvas.draw_idle()

        return bottom, top

    def get_yscale(self):
        return self.yaxis.get_scale()
    get_yscale.__doc__ = "Return the yaxis scale string: %s""" % (
                                ", ".join(mscale.get_scale_names()))

    @docstring.dedent_interpd
    def set_yscale(self, value, **kwargs):
        """
        Call signature::

          set_yscale(value)

        Set the scaling of the y-axis: %(scale)s

        ACCEPTS: [%(scale)s]

        Different kwargs are accepted, depending on the scale:
        %(scale_docs)s
        """
        self.yaxis._set_scale(value, **kwargs)
        self.autoscale_view(scalex=False)
        self._update_transScale()

    def get_yticks(self, minor=False):
        """Return the y ticks as a list of locations"""
        return self.yaxis.get_ticklocs(minor=minor)

    def set_yticks(self, ticks, minor=False):
        """
        Set the y ticks with list of *ticks*

        ACCEPTS: sequence of floats

        Keyword arguments:

          *minor*: [ *False* | *True* ]
            Sets the minor ticks if *True*
        """
        return self.yaxis.set_ticks(ticks, minor=minor)

    def get_ymajorticklabels(self):
        """
        Get the major y tick labels as a list of
        :class:`~matplotlib.text.Text` instances.
        """
        return cbook.silent_list('Text yticklabel',
                                 self.yaxis.get_majorticklabels())

    def get_yminorticklabels(self):
        """
        Get the minor y tick labels as a list of
        :class:`~matplotlib.text.Text` instances.
        """
        return cbook.silent_list('Text yticklabel',
                                 self.yaxis.get_minorticklabels())

    def get_yticklabels(self, minor=False):
        """
        Get the y tick labels as a list of :class:`~matplotlib.text.Text`
        instances
        """
        return cbook.silent_list('Text yticklabel',
                                 self.yaxis.get_ticklabels(minor=minor))

    @docstring.dedent_interpd
    def set_yticklabels(self, labels, fontdict=None, minor=False, **kwargs):
        """
        Call signature::

          set_yticklabels(labels, fontdict=None, minor=False, **kwargs)

        Set the y tick labels with list of strings *labels*.  Return a list of
        :class:`~matplotlib.text.Text` instances.

        *kwargs* set :class:`~matplotlib.text.Text` properties for the labels.
        Valid properties are
        %(Text)s

        ACCEPTS: sequence of strings
        """
        return self.yaxis.set_ticklabels(labels, fontdict,
                                         minor=minor, **kwargs)

    def xaxis_date(self, tz=None):
        """
        Sets up x-axis ticks and labels that treat the x data as dates.

        *tz* is a timezone string or :class:`tzinfo` instance.
        Defaults to rc value.
        """
        # should be enough to inform the unit conversion interface
        # dates are coming in
        self.xaxis.axis_date(tz)

    def yaxis_date(self, tz=None):
        """
        Sets up y-axis ticks and labels that treat the y data as dates.

        *tz* is a timezone string or :class:`tzinfo` instance.
        Defaults to rc value.
        """
        self.yaxis.axis_date(tz)

    def format_xdata(self, x):
        """
        Return *x* string formatted.  This function will use the attribute
        self.fmt_xdata if it is callable, else will fall back on the xaxis
        major formatter
        """
        try:
            return self.fmt_xdata(x)
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
        try:
            return self.fmt_ydata(y)
        except TypeError:
            func = self.yaxis.get_major_formatter().format_data_short
            val = func(y)
            return val

    def format_coord(self, x, y):
        """Return a format string formatting the *x*, *y* coord"""
        if x is None:
            xs = '???'
        else:
            xs = self.format_xdata(x)
        if y is None:
            ys = '???'
        else:
            ys = self.format_ydata(y)
        return 'x=%s y=%s' % (xs, ys)

    #### Interactive manipulation

    def can_zoom(self):
        """
        Return *True* if this axes supports the zoom box button functionality.
        """
        return True

    def can_pan(self):
        """
        Return *True* if this axes supports any pan/zoom button functionality.
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

        ACCEPTS: [ *True* | *False* ]
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
            lim=self.viewLim.frozen(),
            trans=self.transData.frozen(),
            trans_inverse=self.transData.inverted().frozen(),
            bbox=self.bbox.frozen(),
            x=x,
            y=y
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
            if key == 'control':
                if abs(dx) > abs(dy):
                    dy = dx
                else:
                    dx = dy
            elif key == 'x':
                dy = 0
            elif key == 'y':
                dx = 0
            elif key == 'shift':
                if 2 * abs(dx) < abs(dy):
                    dx = 0
                elif 2 * abs(dy) < abs(dx):
                    dy = 0
                elif abs(dx) > abs(dy):
                    dy = dy / abs(dy) * abs(dx)
                else:
                    dx = dx / abs(dx) * abs(dy)
            return (dx, dy)

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
                start = np.array([p.x, p.y])
                oldpoints = p.lim.transformed(p.trans)
                newpoints = start + alpha * (oldpoints - start)
                result = mtransforms.Bbox(newpoints) \
                    .transformed(p.trans_inverse)
            except OverflowError:
                warnings.warn('Overflow while panning')
                return

        self.set_xlim(*result.intervalx)
        self.set_ylim(*result.intervaly)

    def get_cursor_props(self):
        """
        Return the cursor propertiess as a (*linewidth*, *color*)
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
        if len(args) == 1:
            lw, c = args[0]
        elif len(args) == 2:
            lw, c = args
        else:
            raise ValueError('args must be a (linewidth, color) tuple')
        c = mcolors.colorConverter.to_rgba(c)
        self._cursorProps = lw, c

    def get_children(self):
        """return a list of child artists"""
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
        children.append(self._left_title)
        children.append(self._right_title)
        children.append(self.patch)
        children.extend(self.spines.itervalues())
        return children

    def contains(self, mouseevent):
        """
        Test whether the mouse event occured in the axes.

        Returns *True* / *False*, {}
        """
        if callable(self._contains):
            return self._contains(self, mouseevent)

        return self.patch.contains(mouseevent)

    def contains_point(self, point):
        """
        Returns *True* if the point (tuple of x,y) is inside the axes
        (the area defined by the its patch). A pixel coordinate is
        required.

        """
        return self.patch.contains_point(point, radius=1.0)

    def pick(self, *args):
        """
        Call signature::

            pick(mouseevent)

        each child artist will fire a pick event if mouseevent is over
        the artist and the artist has picker set
        """
        martist.Artist.pick(self, args[0])

    ### Labelling

    def get_title(self, loc="center"):
        """Get an axes title.

        Get one of the three available axes titles. The available titles
        are positioned above the axes in the center, flush with the left
        edge, and flush with the right edge.

        Parameters
        ----------
        loc : {'center', 'left', 'right'}, str, optional
            Which title to get, defaults to 'center'

        Returns
        -------
        title: str
            The title text string.

        """
        try:
            title = {'left': self._left_title,
                     'center': self.title,
                     'right': self._right_title}[loc.lower()]
        except KeyError:
            raise ValueError("'%s' is not a valid location" % loc)
        return title.get_text()

    @docstring.dedent_interpd
    def set_title(self, label, fontdict=None, loc="center", **kwargs):
        """
        Set a title for the axes.

        Set one of the three available axes titles. The available titles
        are positioned above the axes in the center, flush with the left
        edge, and flush with the right edge.

        Parameters
        ----------
        label : str
            Text to use for the title

        fontdict : dict
            A dictionary controlling the appearance of the title text,
            the default `fontdict` is::

               {'fontsize': rcParams['axes.titlesize'],
                'verticalalignment': 'baseline',
                'horizontalalignment': loc}

        loc : {'center', 'left', 'right'}, str, optional
            Which title to set, defaults to 'center'

        Returns
        -------
        text : :class:`~matplotlib.text.Text`
            The matplotlib text instance representing the title

        Other parameters
        ----------------
        Other keyword arguments are text properties, see
        :class:`~matplotlib.text.Text` for a list of valid text
        properties.
        """
        try:
            title = {'left': self._left_title,
                     'center': self.title,
                     'right': self._right_title}[loc.lower()]
        except KeyError:
            raise ValueError("'%s' is not a valid location" % loc)
        default = {
            'fontsize': rcParams['axes.titlesize'],
            'verticalalignment': 'baseline',
            'horizontalalignment': loc.lower()
            }
        title.set_text(label)
        title.update(default)
        if fontdict is not None:
            title.update(fontdict)
        title.update(kwargs)
        return title

    def get_xlabel(self):
        """
        Get the xlabel text string.
        """
        label = self.xaxis.get_label()
        return label.get_text()

    @docstring.dedent_interpd
    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set the label for the xaxis.

        Parameters
        ----------
        xlabel : string
            x label

        labelpad : scalar, optional, default: None
            spacing in points between the label and the x-axis

        Other parameters
        ----------------
        kwargs : `~matplotlib.text.Text` properties

        See also
        --------
        text : for information on how override and the optional args work
        """
        if labelpad is not None:
            self.xaxis.labelpad = labelpad
        return self.xaxis.set_label_text(xlabel, fontdict, **kwargs)

    def get_ylabel(self):
        """
        Get the ylabel text string.
        """
        label = self.yaxis.get_label()
        return label.get_text()

    @docstring.dedent_interpd
    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, **kwargs):
        """
        Set the label for the yaxis

        Parameters
        ----------
        ylabel : string
            y label

        labelpad : scalar, optional, default: None
            spacing in points between the label and the x-axis

        Other parameters
        ----------------
        kwargs : `~matplotlib.text.Text` properties

        See also
        --------
        text : for information on how override and the optional args work

        """
        if labelpad is not None:
            self.yaxis.labelpad = labelpad
        return self.yaxis.set_label_text(ylabel, fontdict, **kwargs)

    @docstring.dedent_interpd
    def text(self, x, y, s, fontdict=None,
             withdash=False, **kwargs):
        """
        Add text to the axes.

        Add text in string *s* to axis at location *x*, *y*, data
        coordinates.

        Parameters
        ----------
        s : string
            text

        x, y : scalars
            data coordinates

        fontdict : dictionary, optional, default: None
            A dictionary to override the default text properties. If fontdict
            is None, the defaults are determined by your rc parameters.

        withdash : boolean, optional, default: False
            Creates a `~matplotlib.text.TextWithDash` instance instead of a
            `~matplotlib.text.Text` instance.

        Other parameters
        ----------------
        kwargs : `~matplotlib.text.Text` properties.
            Other miscellaneous text parameters.

        Examples
        --------
        Individual keyword arguments can be used to override any given
        parameter::

            >>> text(x, y, s, fontsize=12)

        The default transform specifies that text is in data coords,
        alternatively, you can specify text in axis coords (0,0 is
        lower-left and 1,1 is upper-right).  The example below places
        text in the center of the axes::

            >>> text(0.5, 0.5,'matplotlib', horizontalalignment='center',
            ...      verticalalignment='center',
            ...      transform=ax.transAxes)

        You can put a rectangular box around the text instance (e.g., to
        set a background color) by using the keyword *bbox*.  *bbox* is
        a dictionary of `~matplotlib.patches.Rectangle`
        properties.  For example::

            >>> text(x, y, s, bbox=dict(facecolor='red', alpha=0.5))
        """
        default = {
            'verticalalignment': 'baseline',
            'horizontalalignment': 'left',
            'transform': self.transData,
            'clip_on': False
            }

        # At some point if we feel confident that TextWithDash
        # is robust as a drop-in replacement for Text and that
        # the performance impact of the heavier-weight class
        # isn't too significant, it may make sense to eliminate
        # the withdash kwarg and simply delegate whether there's
        # a dash to TextWithDash and dashlength.
        if withdash:
            t = mtext.TextWithDash(
                x=x, y=y, text=s)
        else:
            t = mtext.Text(
                x=x, y=y, text=s)
        self._set_artist_props(t)

        t.update(default)
        if fontdict is not None:
            t.update(fontdict)
        t.update(kwargs)
        self.texts.append(t)
        t._remove_method = lambda h: self.texts.remove(h)

        t.set_clip_path(self.patch)
        return t

    @docstring.dedent_interpd
    def annotate(self, *args, **kwargs):
        """
        Create an annotation: a piece of text referring to a data
        point.

        Call signature::

          annotate(s, xy, xytext=None, xycoords='data',
                   textcoords='data', arrowprops=None, **kwargs)

        Keyword arguments:

        %(Annotation)s

        .. plot:: mpl_examples/pylab_examples/annotation_demo2.py
        """
        a = mtext.Annotation(*args, **kwargs)
        a.set_transform(mtransforms.IdentityTransform())
        self._set_artist_props(a)
        if kwargs.has_key('clip_on'):
            a.set_clip_path(self.patch)
        self.texts.append(a)
        a._remove_method = lambda h: self.texts.remove(h)
        return a

    #### Lines and spans

    @docstring.dedent_interpd
    def axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        """
        Add a horizontal line across the axis.

        Call signature::

          axhline(y=0, xmin=0, xmax=1, **kwargs)

        Draw a horizontal line at *y* from *xmin* to *xmax*.  With the
        default values of *xmin* = 0 and *xmax* = 1, this line will
        always span the horizontal extent of the axes, regardless of
        the xlim settings, even if you change them, e.g., with the
        :meth:`set_xlim` command.  That is, the horizontal extent is
        in axes coords: 0=left, 0.5=middle, 1.0=right but the *y*
        location is in data coordinates.

        Return value is the :class:`~matplotlib.lines.Line2D`
        instance.  kwargs are the same as kwargs to plot, and can be
        used to control the line properties.  e.g.,

        * draw a thick red hline at *y* = 0 that spans the xrange::

            >>> axhline(linewidth=4, color='r')

        * draw a default hline at *y* = 1 that spans the xrange::

            >>> axhline(y=1)

        * draw a default hline at *y* = .5 that spans the the middle half of
          the xrange::

            >>> axhline(y=.5, xmin=0.25, xmax=0.75)

        Valid kwargs are :class:`~matplotlib.lines.Line2D` properties,
        with the exception of 'transform':

        %(Line2D)s

        .. seealso::

            :meth:`axhspan`
                for example plot and source code
        """

        if "transform" in kwargs:
            raise ValueError(
                "'transform' is not allowed as a kwarg;"
                + "axhline generates its own transform.")
        ymin, ymax = self.get_ybound()

        # We need to strip away the units for comparison with
        # non-unitized bounds
        self._process_unit_info(ydata=y, kwargs=kwargs)
        yy = self.convert_yunits(y)
        scaley = (yy < ymin) or (yy > ymax)

        trans = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)
        l = mlines.Line2D([xmin, xmax], [y, y], transform=trans, **kwargs)
        self.add_line(l)
        self.autoscale_view(scalex=False, scaley=scaley)
        return l

    @docstring.dedent_interpd
    def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        """
        Add a vertical line across the axes.

        Call signature::

          axvline(x=0, ymin=0, ymax=1, **kwargs)

        Draw a vertical line at *x* from *ymin* to *ymax*.  With the
        default values of *ymin* = 0 and *ymax* = 1, this line will
        always span the vertical extent of the axes, regardless of the
        ylim settings, even if you change them, e.g., with the
        :meth:`set_ylim` command.  That is, the vertical extent is in
        axes coords: 0=bottom, 0.5=middle, 1.0=top but the *x* location
        is in data coordinates.

        Return value is the :class:`~matplotlib.lines.Line2D`
        instance.  kwargs are the same as kwargs to plot, and can be
        used to control the line properties.  e.g.,

        * draw a thick red vline at *x* = 0 that spans the yrange::

            >>> axvline(linewidth=4, color='r')

        * draw a default vline at *x* = 1 that spans the yrange::

            >>> axvline(x=1)

        * draw a default vline at *x* = .5 that spans the the middle half of
          the yrange::

            >>> axvline(x=.5, ymin=0.25, ymax=0.75)

        Valid kwargs are :class:`~matplotlib.lines.Line2D` properties,
        with the exception of 'transform':

        %(Line2D)s

        .. seealso::

            :meth:`axhspan`
                for example plot and source code
        """

        if "transform" in kwargs:
            raise ValueError(
                "'transform' is not allowed as a kwarg;"
                + "axvline generates its own transform.")
        xmin, xmax = self.get_xbound()

        # We need to strip away the units for comparison with
        # non-unitized bounds
        self._process_unit_info(xdata=x, kwargs=kwargs)
        xx = self.convert_xunits(x)
        scalex = (xx < xmin) or (xx > xmax)

        trans = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)
        l = mlines.Line2D([x, x], [ymin, ymax], transform=trans, **kwargs)
        self.add_line(l)
        self.autoscale_view(scalex=scalex, scaley=False)
        return l

    @docstring.dedent_interpd
    def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kwargs):
        """
        Add a horizontal span (rectangle) across the axis.

        Call signature::

          axhspan(ymin, ymax, xmin=0, xmax=1, **kwargs)

        *y* coords are in data units and *x* coords are in axes (relative
        0-1) units.

        Draw a horizontal span (rectangle) from *ymin* to *ymax*.
        With the default values of *xmin* = 0 and *xmax* = 1, this
        always spans the xrange, regardless of the xlim settings, even
        if you change them, e.g., with the :meth:`set_xlim` command.
        That is, the horizontal extent is in axes coords: 0=left,
        0.5=middle, 1.0=right but the *y* location is in data
        coordinates.

        Return value is a :class:`matplotlib.patches.Polygon`
        instance.

        Examples:

        * draw a gray rectangle from *y* = 0.25-0.75 that spans the
          horizontal extent of the axes::

            >>> axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)

        Valid kwargs are :class:`~matplotlib.patches.Polygon` properties:

        %(Polygon)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/axhspan_demo.py

        """
        trans = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)

        # process the unit information
        self._process_unit_info([xmin, xmax], [ymin, ymax], kwargs=kwargs)

        # first we need to strip away the units
        xmin, xmax = self.convert_xunits([xmin, xmax])
        ymin, ymax = self.convert_yunits([ymin, ymax])

        verts = (xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        self.add_patch(p)
        self.autoscale_view(scalex=False)
        return p

    @docstring.dedent_interpd
    def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        """
        Add a vertical span (rectangle) across the axes.

        Call signature::

          axvspan(xmin, xmax, ymin=0, ymax=1, **kwargs)

        *x* coords are in data units and *y* coords are in axes (relative
        0-1) units.

        Draw a vertical span (rectangle) from *xmin* to *xmax*.  With
        the default values of *ymin* = 0 and *ymax* = 1, this always
        spans the yrange, regardless of the ylim settings, even if you
        change them, e.g., with the :meth:`set_ylim` command.  That is,
        the vertical extent is in axes coords: 0=bottom, 0.5=middle,
        1.0=top but the *y* location is in data coordinates.

        Return value is the :class:`matplotlib.patches.Polygon`
        instance.

        Examples:

        * draw a vertical green translucent rectangle from x=1.25 to 1.55 that
          spans the yrange of the axes::

            >>> axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

        Valid kwargs are :class:`~matplotlib.patches.Polygon`
        properties:

        %(Polygon)s

        .. seealso::

            :meth:`axhspan`
                for example plot and source code
        """
        trans = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)

        # process the unit information
        self._process_unit_info([xmin, xmax], [ymin, ymax], kwargs=kwargs)

        # first we need to strip away the units
        xmin, xmax = self.convert_xunits([xmin, xmax])
        ymin, ymax = self.convert_yunits([ymin, ymax])

        verts = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
        p = mpatches.Polygon(verts, **kwargs)
        p.set_transform(trans)
        self.add_patch(p)
        self.autoscale_view(scaley=False)
        return p

    @docstring.dedent
    def hlines(self, y, xmin, xmax, colors='k', linestyles='solid',
                     label='', **kwargs):
        """
        Plot horizontal lines.

        Plot horizontal lines at each `y` from `xmin` to `xmax`.

        Parameters
        ----------
        y : scalar or 1D array_like
            y-indexes where to plot the lines.

        xmin, xmax : scalar or 1D array_like
            Respective beginning and end of each line. If scalars are
            provided, all lines will have same length.

        colors : array_like of colors, optional, default: 'k'

        linestyles : ['solid' | 'dashed' | 'dashdot' | 'dotted'], optional

        label : string, optional, default: ''

        Returns
        -------
        lines : `~matplotlib.collections.LineCollection`

        Other parameters
        ----------------
        kwargs :  `~matplotlib.collections.LineCollection` properties.

        See also
        --------
        vlines : vertical lines

        Examples
        --------
        .. plot:: mpl_examples/pylab_examples/vline_hline_demo.py

        """

        # We do the conversion first since not all unitized data is uniform
        # process the unit information
        self._process_unit_info([xmin, xmax], y, kwargs=kwargs)
        y = self.convert_yunits(y)
        xmin = self.convert_xunits(xmin)
        xmax = self.convert_xunits(xmax)

        if not iterable(y):
            y = [y]
        if not iterable(xmin):
            xmin = [xmin]
        if not iterable(xmax):
            xmax = [xmax]

        y = np.asarray(y)
        xmin = np.asarray(xmin)
        xmax = np.asarray(xmax)

        if len(xmin) == 1:
            xmin = np.resize(xmin, y.shape)
        if len(xmax) == 1:
            xmax = np.resize(xmax, y.shape)

        if len(xmin) != len(y):
            raise ValueError('xmin and y are unequal sized sequences')
        if len(xmax) != len(y):
            raise ValueError('xmax and y are unequal sized sequences')

        verts = [((thisxmin, thisy), (thisxmax, thisy))
                 for thisxmin, thisxmax, thisy in zip(xmin, xmax, y)]
        coll = mcoll.LineCollection(verts, colors=colors,
                                    linestyles=linestyles, label=label)
        self.add_collection(coll)
        coll.update(kwargs)

        if len(y) > 0:
            minx = min(xmin.min(), xmax.min())
            maxx = max(xmin.max(), xmax.max())
            miny = y.min()
            maxy = y.max()

            corners = (minx, miny), (maxx, maxy)

            self.update_datalim(corners)
            self.autoscale_view()

        return coll

    @docstring.dedent_interpd
    def vlines(self, x, ymin, ymax, colors='k', linestyles='solid',
                     label='', **kwargs):
        """
        Plot vertical lines.

        Plot vertical lines at each `x` from `ymin` to `ymax`.

        Parameters
        ----------
        x : scalar or 1D array_like
            x-indexes where to plot the lines.

        xmin, xmax : scalar or 1D array_like
            Respective beginning and end of each line. If scalars are
            provided, all lines will have same length.

        colors : array_like of colors, optional, default: 'k'

        linestyles : ['solid' | 'dashed' | 'dashdot' | 'dotted'], optional

        label : string, optional, default: ''

        Returns
        -------
        lines : `~matplotlib.collections.LineCollection`

        Other parameters
        ----------------
        kwargs : `~matplotlib.collections.LineCollection` properties.

        See also
        --------
        hlines : horizontal lines

        Examples
        ---------
        .. plot:: mpl_examples/pylab_examples/vline_hline_demo.py

        """

        self._process_unit_info(xdata=x, ydata=[ymin, ymax], kwargs=kwargs)

        # We do the conversion first since not all unitized data is uniform
        x = self.convert_xunits(x)
        ymin = self.convert_yunits(ymin)
        ymax = self.convert_yunits(ymax)

        if not iterable(x):
            x = [x]
        if not iterable(ymin):
            ymin = [ymin]
        if not iterable(ymax):
            ymax = [ymax]

        x = np.asarray(x)
        ymin = np.asarray(ymin)
        ymax = np.asarray(ymax)
        if len(ymin) == 1:
            ymin = np.resize(ymin, x.shape)
        if len(ymax) == 1:
            ymax = np.resize(ymax, x.shape)

        if len(ymin) != len(x):
            raise ValueError('ymin and x are unequal sized sequences')
        if len(ymax) != len(x):
            raise ValueError('ymax and x are unequal sized sequences')

        Y = np.array([ymin, ymax]).T

        verts = [((thisx, thisymin), (thisx, thisymax))
                 for thisx, (thisymin, thisymax) in zip(x, Y)]
        #print 'creating line collection'
        coll = mcoll.LineCollection(verts, colors=colors,
                                    linestyles=linestyles, label=label)
        self.add_collection(coll)
        coll.update(kwargs)

        if len(x) > 0:
            minx = min(x)
            maxx = max(x)

            miny = min(min(ymin), min(ymax))
            maxy = max(max(ymin), max(ymax))

            corners = (minx, miny), (maxx, maxy)
            self.update_datalim(corners)
            self.autoscale_view()

        return coll

    @docstring.dedent_interpd
    def eventplot(self, positions, orientation='horizontal', lineoffsets=1,
                  linelengths=1, linewidths=None, colors=None,
                  linestyles='solid', **kwargs):
        """
        Plot identical parallel lines at specific positions.

        Call signature::

          eventplot(positions, orientation='horizontal', lineoffsets=0,
                    linelengths=1, linewidths=None, color =None,
                    linestyles='solid'

        Plot parallel lines at the given positions.  positions should be a 1D
        or 2D array-like object, with each row corresponding to a row or column
        of lines.

        This type of plot is commonly used in neuroscience for representing
        neural events, where it is commonly called a spike raster, dot raster,
        or raster plot.

        However, it is useful in any situation where you wish to show the
        timing or position of multiple sets of discrete events, such as the
        arrival times of people to a business on each day of the month or the
        date of hurricanes each year of the last century.

        *orientation* : [ 'horizonal' | 'vertical' ]
          'horizonal' : the lines will be vertical and arranged in rows
          "vertical' : lines will be horizontal and arranged in columns

        *lineoffsets* :
          A float or array-like containing floats.

        *linelengths* :
          A float or array-like containing floats.

        *linewidths* :
          A float or array-like containing floats.

        *colors*
          must be a sequence of RGBA tuples (eg arbitrary color
          strings, etc, not allowed) or a list of such sequences

        *linestyles* :
          [ 'solid' | 'dashed' | 'dashdot' | 'dotted' ] or an array of these
          values

        For linelengths, linewidths, colors, and linestyles, if only a single
        value is given, that value is applied to all lines.  If an array-like
        is given, it must have the same length as positions, and each value
        will be applied to the corresponding row or column in positions.

        Returns a list of :class:`matplotlib.collections.EventCollection`
        objects that were added.

        kwargs are :class:`~matplotlib.collections.LineCollection` properties:

        %(LineCollection)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/eventplot_demo.py
        """
        self._process_unit_info(xdata=positions,
                                ydata=[lineoffsets, linelengths],
                                kwargs=kwargs)

        # We do the conversion first since not all unitized data is uniform
        positions = self.convert_xunits(positions)
        lineoffsets = self.convert_yunits(lineoffsets)
        linelengths = self.convert_yunits(linelengths)

        if not iterable(positions):
            positions = [positions]
        elif any(iterable(position) for position in positions):
            positions = [np.asanyarray(position) for position in positions]
        else:
            positions = [np.asanyarray(positions)]

        if len(positions) == 0:
            return []

        if not iterable(lineoffsets):
            lineoffsets = [lineoffsets]
        if not iterable(linelengths):
            linelengths = [linelengths]
        if not iterable(linewidths):
            linewidths = [linewidths]
        if not iterable(colors):
            colors = [colors]
        if hasattr(linestyles, 'lower') or not iterable(linestyles):
            linestyles = [linestyles]

        lineoffsets = np.asarray(lineoffsets)
        linelengths = np.asarray(linelengths)
        linewidths = np.asarray(linewidths)

        if len(lineoffsets) == 0:
            lineoffsets = [None]
        if len(linelengths) == 0:
            linelengths = [None]
        if len(linewidths) == 0:
            lineoffsets = [None]
        if len(linewidths) == 0:
            lineoffsets = [None]
        if len(colors) == 0:
            colors = [None]

        if len(lineoffsets) == 1 and len(positions) != 1:
            lineoffsets = np.tile(lineoffsets, len(positions))
            lineoffsets[0] = 0
            lineoffsets = np.cumsum(lineoffsets)
        if len(linelengths) == 1:
            linelengths = np.tile(linelengths, len(positions))
        if len(linewidths) == 1:
            linewidths = np.tile(linewidths, len(positions))
        if len(colors) == 1:
            colors = np.asanyarray(colors)
            colors = np.tile(colors, [len(positions), 1])
        if len(linestyles) == 1:
            linestyles = [linestyles] * len(positions)

        if len(lineoffsets) != len(positions):
            raise ValueError('lineoffsets and positions are unequal sized '
                             'sequences')
        if len(linelengths) != len(positions):
            raise ValueError('linelengths and positions are unequal sized '
                             'sequences')
        if len(linewidths) != len(positions):
            raise ValueError('linewidths and positions are unequal sized '
                             'sequences')
        if len(colors) != len(positions):
            raise ValueError('colors and positions are unequal sized '
                             'sequences')
        if len(linestyles) != len(positions):
            raise ValueError('linestyles and positions are unequal sized '
                             'sequences')

        colls = []
        for position, lineoffset, linelength, linewidth, color, linestyle in \
            itertools.izip(positions, lineoffsets, linelengths, linewidths,
                           colors, linestyles):
            coll = mcoll.EventCollection(position,
                                         orientation=orientation,
                                         lineoffset=lineoffset,
                                         linelength=linelength,
                                         linewidth=linewidth,
                                         color=color,
                                         linestyle=linestyle)
            self.add_collection(coll)
            coll.update(kwargs)
            colls.append(coll)

        if len(positions) > 0:
            minpos = min(position.min() for position in positions)
            maxpos = max(position.max() for position in positions)

            minline = (lineoffsets - linelengths).min()
            maxline = (lineoffsets + linelengths).max()

            if colls[0].is_horizontal():
                corners = (minpos, minline), (maxpos, maxline)
            else:
                corners = (minline, minpos), (maxline, maxpos)
            self.update_datalim(corners)
            self.autoscale_view()

        return colls

    #### Basic plotting
    @docstring.dedent_interpd
    def plot(self, *args, **kwargs):
        """
        Plot lines and/or markers to the
        :class:`~matplotlib.axes.Axes`.  *args* is a variable length
        argument, allowing for multiple *x*, *y* pairs with an
        optional format string.  For example, each of the following is
        legal::

            plot(x, y)        # plot x and y using default line style and color
            plot(x, y, 'bo')  # plot x and y using blue circle markers
            plot(y)           # plot y using x as index array 0..N-1
            plot(y, 'r+')     # ditto, but with red plusses

        If *x* and/or *y* is 2-dimensional, then the corresponding columns
        will be plotted.

        An arbitrary number of *x*, *y*, *fmt* groups can be
        specified, as in::

            a.plot(x1, y1, 'g^', x2, y2, 'g-')

        Return value is a list of lines that were added.

        By default, each line is assigned a different color specified by a
        'color cycle'.  To change this behavior, you can edit the
        axes.color_cycle rcParam. Alternatively, you can use
        :meth:`~matplotlib.axes.Axes.set_default_color_cycle`.

        The following format string characters are accepted to control
        the line style or marker:

        ================    ===============================
        character           description
        ================    ===============================
        ``'-'``             solid line style
        ``'--'``            dashed line style
        ``'-.'``            dash-dot line style
        ``':'``             dotted line style
        ``'.'``             point marker
        ``','``             pixel marker
        ``'o'``             circle marker
        ``'v'``             triangle_down marker
        ``'^'``             triangle_up marker
        ``'<'``             triangle_left marker
        ``'>'``             triangle_right marker
        ``'1'``             tri_down marker
        ``'2'``             tri_up marker
        ``'3'``             tri_left marker
        ``'4'``             tri_right marker
        ``'s'``             square marker
        ``'p'``             pentagon marker
        ``'*'``             star marker
        ``'h'``             hexagon1 marker
        ``'H'``             hexagon2 marker
        ``'+'``             plus marker
        ``'x'``             x marker
        ``'D'``             diamond marker
        ``'d'``             thin_diamond marker
        ``'|'``             vline marker
        ``'_'``             hline marker
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
                 markerfacecolor='blue', markersize=12).

        See :class:`~matplotlib.lines.Line2D` for details.

        The kwargs are :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        kwargs *scalex* and *scaley*, if defined, are passed on to
        :meth:`~matplotlib.axes.Axes.autoscale_view` to determine
        whether the *x* and *y* axes are autoscaled; the default is
        *True*.
        """
        scalex = kwargs.pop('scalex', True)
        scaley = kwargs.pop('scaley', True)

        if not self._hold:
            self.cla()
        lines = []

        for line in self._get_lines(*args, **kwargs):
            self.add_line(line)
            lines.append(line)

        self.autoscale_view(scalex=scalex, scaley=scaley)
        return lines

    @docstring.dedent_interpd
    def plot_date(self, x, y, fmt='bo', tz=None, xdate=True, ydate=False,
                  **kwargs):
        """
        Plot with data with dates.

        Call signature::

           plot_date(x, y, fmt='bo', tz=None, xdate=True,
                     ydate=False, **kwargs)

        Similar to the :func:`~matplotlib.pyplot.plot` command, except
        the *x* or *y* (or both) data is considered to be dates, and the
        axis is labeled accordingly.

        *x* and/or *y* can be a sequence of dates represented as float
        days since 0001-01-01 UTC.

        Keyword arguments:

          *fmt*: string
            The plot format string.

          *tz*: [ *None* | timezone string | :class:`tzinfo` instance]
            The time zone to use in labeling dates. If *None*, defaults to rc
            value.

          *xdate*: [ *True* | *False* ]
            If *True*, the *x*-axis will be labeled with dates.

          *ydate*: [ *False* | *True* ]
            If *True*, the *y*-axis will be labeled with dates.

        Note if you are using custom date tickers and formatters, it
        may be necessary to set the formatters/locators after the call
        to :meth:`plot_date` since :meth:`plot_date` will set the
        default tick locator to
        :class:`matplotlib.dates.AutoDateLocator` (if the tick
        locator is not already set to a
        :class:`matplotlib.dates.DateLocator` instance) and the
        default tick formatter to
        :class:`matplotlib.dates.AutoDateFormatter` (if the tick
        formatter is not already set to a
        :class:`matplotlib.dates.DateFormatter` instance).

        Valid kwargs are :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        .. seealso::

           :mod:`~matplotlib.dates` for helper functions

           :func:`~matplotlib.dates.date2num`,
           :func:`~matplotlib.dates.num2date` and
           :func:`~matplotlib.dates.drange` for help on creating the required
           floating point dates.
        """

        if not self._hold:
            self.cla()

        ret = self.plot(x, y, fmt, **kwargs)

        if xdate:
            self.xaxis_date(tz)
        if ydate:
            self.yaxis_date(tz)

        self.autoscale_view()

        return ret

    @docstring.dedent_interpd
    def loglog(self, *args, **kwargs):
        """
        Make a plot with log scaling on both the *x* and *y* axis.

        Call signature::

          loglog(*args, **kwargs)

        :func:`~matplotlib.pyplot.loglog` supports all the keyword
        arguments of :func:`~matplotlib.pyplot.plot` and
        :meth:`matplotlib.axes.Axes.set_xscale` /
        :meth:`matplotlib.axes.Axes.set_yscale`.

        Notable keyword arguments:

          *basex*/*basey*: scalar > 1
            Base of the *x*/*y* logarithm

          *subsx*/*subsy*: [ *None* | sequence ]
            The location of the minor *x*/*y* ticks; *None* defaults
            to autosubs, which depend on the number of decades in the
            plot; see :meth:`matplotlib.axes.Axes.set_xscale` /
            :meth:`matplotlib.axes.Axes.set_yscale` for details

          *nonposx*/*nonposy*: ['mask' | 'clip' ]
            Non-positive values in *x* or *y* can be masked as
            invalid, or clipped to a very small positive number

        The remaining valid kwargs are
        :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/log_demo.py

        """
        if not self._hold:
            self.cla()

        dx = {'basex': kwargs.pop('basex', 10),
              'subsx': kwargs.pop('subsx', None),
              'nonposx': kwargs.pop('nonposx', 'mask'),
              }
        dy = {'basey': kwargs.pop('basey', 10),
              'subsy': kwargs.pop('subsy', None),
              'nonposy': kwargs.pop('nonposy', 'mask'),
              }

        self.set_xscale('log', **dx)
        self.set_yscale('log', **dy)

        b = self._hold
        self._hold = True  # we've already processed the hold
        l = self.plot(*args, **kwargs)
        self._hold = b  # restore the hold

        return l

    @docstring.dedent_interpd
    def semilogx(self, *args, **kwargs):
        """
        Make a plot with log scaling on the *x* axis.

        Call signature::

          semilogx(*args, **kwargs)

        :func:`semilogx` supports all the keyword arguments of
        :func:`~matplotlib.pyplot.plot` and
        :meth:`matplotlib.axes.Axes.set_xscale`.

        Notable keyword arguments:

          *basex*: scalar > 1
            Base of the *x* logarithm

          *subsx*: [ *None* | sequence ]
            The location of the minor xticks; *None* defaults to
            autosubs, which depend on the number of decades in the
            plot; see :meth:`~matplotlib.axes.Axes.set_xscale` for
            details.

          *nonposx*: [ 'mask' | 'clip' ]
            Non-positive values in *x* can be masked as
            invalid, or clipped to a very small positive number

        The remaining valid kwargs are
        :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        .. seealso::

            :meth:`loglog`
                For example code and figure
        """
        if not self._hold:
            self.cla()
        d = {'basex': kwargs.pop('basex', 10),
             'subsx': kwargs.pop('subsx', None),
             'nonposx': kwargs.pop('nonposx', 'mask'),
             }

        self.set_xscale('log', **d)
        b = self._hold
        self._hold = True  # we've already processed the hold
        l = self.plot(*args, **kwargs)
        self._hold = b  # restore the hold
        return l

    @docstring.dedent_interpd
    def semilogy(self, *args, **kwargs):
        """
        Make a plot with log scaling on the *y* axis.

        call signature::

          semilogy(*args, **kwargs)

        :func:`semilogy` supports all the keyword arguments of
        :func:`~matplotlib.pylab.plot` and
        :meth:`matplotlib.axes.Axes.set_yscale`.

        Notable keyword arguments:

          *basey*: scalar > 1
            Base of the *y* logarithm

          *subsy*: [ *None* | sequence ]
            The location of the minor yticks; *None* defaults to
            autosubs, which depend on the number of decades in the
            plot; see :meth:`~matplotlib.axes.Axes.set_yscale` for
            details.

          *nonposy*: [ 'mask' | 'clip' ]
            Non-positive values in *y* can be masked as
            invalid, or clipped to a very small positive number

        The remaining valid kwargs are
        :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        .. seealso::

            :meth:`loglog`
                For example code and figure
        """
        if not self._hold:
            self.cla()
        d = {'basey': kwargs.pop('basey', 10),
             'subsy': kwargs.pop('subsy', None),
             'nonposy': kwargs.pop('nonposy', 'mask'),
             }
        self.set_yscale('log', **d)
        b = self._hold
        self._hold = True  # we've already processed the hold
        l = self.plot(*args, **kwargs)
        self._hold = b  # restore the hold

        return l

    @docstring.dedent_interpd
    def acorr(self, x, **kwargs):
        """
        Plot the autocorrelation of *x*.

        Call signature::

            acorr(x, normed=True, detrend=mlab.detrend_none, usevlines=True,
                  maxlags=10, **kwargs)

        If *normed* = *True*, normalize the data by the autocorrelation at
        0-th lag.  *x* is detrended by the *detrend* callable (default no
        normalization).

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
        ``(2*len(x)-1)`` lags.

        The return value is a tuple (*lags*, *c*, *linecol*, *b*)
        where

          - *linecol* is the
            :class:`~matplotlib.collections.LineCollection`

          - *b* is the *x*-axis.

        .. seealso::

            :meth:`~matplotlib.axes.Axes.plot` or
            :meth:`~matplotlib.axes.Axes.vlines`
            For documentation on valid kwargs.

        **Example:**

        :func:`~matplotlib.pyplot.xcorr` is top graph, and
        :func:`~matplotlib.pyplot.acorr` is bottom graph.

        .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
        """
        return self.xcorr(x, x, **kwargs)

    @docstring.dedent_interpd
    def xcorr(self, x, y, normed=True, detrend=mlab.detrend_none,
              usevlines=True, maxlags=10, **kwargs):
        """
        Plot the cross correlation between *x* and *y*.

        Call signature::

            xcorr(self, x, y, normed=True, detrend=mlab.detrend_none,
              usevlines=True, maxlags=10, **kwargs)

        If *normed* = *True*, normalize the data by the cross
        correlation at 0-th lag.  *x* and y are detrended by the
        *detrend* callable (default no normalization).  *x* and *y*
        must be equal length.

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

        :func:`~matplotlib.pyplot.xcorr` is top graph, and
        :func:`~matplotlib.pyplot.acorr` is bottom graph.

        .. plot:: mpl_examples/pylab_examples/xcorr_demo.py
        """

        Nx = len(x)
        if Nx != len(y):
            raise ValueError('x and y must be equal length')

        x = detrend(np.asarray(x))
        y = detrend(np.asarray(y))

        c = np.correlate(x, y, mode=2)

        if normed:
            c /= np.sqrt(np.dot(x, x) * np.dot(y, y))

        if maxlags is None:
            maxlags = Nx - 1

        if maxlags >= Nx or maxlags < 1:
            raise ValueError('maglags must be None or strictly '
                             'positive < %d' % Nx)

        lags = np.arange(-maxlags, maxlags + 1)
        c = c[Nx - 1 - maxlags:Nx + maxlags]

        if usevlines:
            a = self.vlines(lags, [0], c, **kwargs)
            b = self.axhline(**kwargs)
        else:

            kwargs.setdefault('marker', 'o')
            kwargs.setdefault('linestyle', 'None')
            a, = self.plot(lags, c, **kwargs)
            b = None
        return lags, c, a, b

    def _get_legend_handles(self, legend_handler_map=None):
        "return artists that will be used as handles for legend"
        handles_original = self.lines + self.patches + \
                           self.collections + self.containers

        # collections
        handler_map = mlegend.Legend.get_default_handler_map()

        if legend_handler_map is not None:
            handler_map = handler_map.copy()
            handler_map.update(legend_handler_map)

        handles = []
        for h in handles_original:
            if h.get_label() == "_nolegend_":  # .startswith('_'):
                continue
            if mlegend.Legend.get_legend_handler(handler_map, h):
                handles.append(h)

        return handles

    def get_legend_handles_labels(self, legend_handler_map=None):
        """
        Return handles and labels for legend

        ``ax.legend()`` is equivalent to ::

          h, l = ax.get_legend_handles_labels()
          ax.legend(h, l)

        """

        handles = []
        labels = []
        for handle in self._get_legend_handles(legend_handler_map):
            label = handle.get_label()
            if label and not label.startswith('_'):
                handles.append(handle)
                labels.append(label)

        return handles, labels

    def legend(self, *args, **kwargs):
        """
        Place a legend on the current axes.

        Call signature::

           legend(*args, **kwargs)

        Places legend at location *loc*.  Labels are a sequence of
        strings and *loc* can be a string or an integer specifying the
        legend location.

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

           legend((line1, line2, line3), ('label1', 'label2', 'label3'), loc=2)

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


        Users can specify any arbitrary location for the legend using the
        *bbox_to_anchor* keyword argument. bbox_to_anchor can be an instance
        of BboxBase(or its derivatives) or a tuple of 2 or 4 floats.
        For example::

           loc = 'upper right', bbox_to_anchor = (0.5, 0.5)

        will place the legend so that the upper right corner of the legend at
        the center of the axes.

        The legend location can be specified in other coordinate, by using the
        *bbox_transform* keyword.

        The loc itslef can be a 2-tuple giving x,y of the lower-left corner of
        the legend in axes coords (*bbox_to_anchor* is ignored).

        Keyword arguments:

          *prop*: [ *None* | FontProperties | dict ]
            A :class:`matplotlib.font_manager.FontProperties`
            instance. If *prop* is a dictionary, a new instance will be
            created with *prop*. If *None*, use rc settings.

          *fontsize*: [size in points | 'xx-small' | 'x-small' | 'small' |
                      'medium' | 'large' | 'x-large' | 'xx-large']
            Set the font size.  May be either a size string, relative to
            the default font size, or an absolute font size in points. This
            argument is only used if prop is not specified.

          *numpoints*: integer
            The number of points in the legend for line

          *scatterpoints*: integer
            The number of points in the legend for scatter plot

          *scatteryoffsets*: list of floats
            a list of yoffsets for scatter symbols in legend

          *markerscale*: [ *None* | scalar ]
            The relative size of legend markers vs. original. If *None*,
            use rc settings.

          *frameon*: [ *True* | *False* ]
            if *True*, draw a frame around the legend.
            The default is set by the rcParam 'legend.frameon'

          *fancybox*: [ *None* | *False* | *True* ]
            if *True*, draw a frame with a round fancybox.  If *None*,
            use rc settings

          *shadow*: [ *None* | *False* | *True* ]
            If *True*, draw a shadow behind legend. If *None*,
            use rc settings.

          *framealpha*: [*None* | float]
            If not None, alpha channel for legend frame. Default *None*.

          *ncol* : integer
            number of columns. default is 1

          *mode* : [ "expand" | *None* ]
            if mode is "expand", the legend will be horizontally expanded
            to fill the axes area (or *bbox_to_anchor*)

          *bbox_to_anchor*: an instance of BboxBase or a tuple of 2 or 4 floats
            the bbox that the legend will be anchored.

          *bbox_transform* : [ an instance of Transform | *None* ]
            the transform for the bbox. transAxes if *None*.

          *title* : string
            the legend title

        Padding and spacing between various elements use following
        keywords parameters. These values are measure in font-size
        units. e.g., a fontsize of 10 points and a handlelength=5
        implies a handlelength of 50 points.  Values from rcParams
        will be used if None.

        ================   ====================================================
        Keyword            Description
        ================   ====================================================
        borderpad          the fractional whitespace inside the legend border
        labelspacing       the vertical space between the legend entries
        handlelength       the length of the legend handles
        handletextpad      the pad between the legend handle and text
        borderaxespad      the pad between the axes and legend border
        columnspacing      the spacing between columns
        ================   ====================================================

        .. note::

           Not all kinds of artist are supported by the legend command.
           See :ref:`plotting-guide-legend` for details.

        **Example:**

        .. plot:: mpl_examples/api/legend_demo.py

        .. seealso::
            :ref:`plotting-guide-legend`.

        """

        if len(args) == 0:
            handles, labels = self.get_legend_handles_labels()
            if len(handles) == 0:
                warnings.warn("No labeled objects found. "
                              "Use label='...' kwarg on individual plots.")
                return None

        elif len(args) == 1:
            # LABELS
            labels = args[0]
            handles = [h for h, label in zip(self._get_legend_handles(),
                                             labels)]

        elif len(args) == 2:
            if is_string_like(args[1]) or isinstance(args[1], int):
                # LABELS, LOC
                labels, loc = args
                handles = [h for h, label in zip(self._get_legend_handles(),
                                                 labels)]
                kwargs['loc'] = loc
            else:
                # LINES, LABELS
                handles, labels = args

        elif len(args) == 3:
            # LINES, LABELS, LOC
            handles, labels, loc = args
            kwargs['loc'] = loc
        else:
            raise TypeError('Invalid arguments to legend')

        # Why do we need to call "flatten" here? -JJL
        # handles = cbook.flatten(handles)

        self.legend_ = mlegend.Legend(self, handles, labels, **kwargs)
        return self.legend_

    #### Specialized plotting

    def step(self, x, y, *args, **kwargs):
        """
        Make a step plot.

        Call signature::

          step(x, y, *args, **kwargs)

        Additional keyword args to :func:`step` are the same as those
        for :func:`~matplotlib.pyplot.plot`.

        *x* and *y* must be 1-D sequences, and it is assumed, but not checked,
        that *x* is uniformly increasing.

        Keyword arguments:

        *where*: [ 'pre' | 'post' | 'mid'  ]
          If 'pre', the interval from x[i] to x[i+1] has level y[i+1]

          If 'post', that interval has level y[i]

          If 'mid', the jumps in *y* occur half-way between the
          *x*-values.
        """

        where = kwargs.pop('where', 'pre')
        if where not in ('pre', 'post', 'mid'):
            raise ValueError("'where' argument to step must be "
                             "'pre', 'post' or 'mid'")
        usr_linestyle = kwargs.pop('linestyle', '')
        kwargs['linestyle'] = 'steps-' + where + usr_linestyle

        return self.plot(x, y, *args, **kwargs)

    @docstring.dedent_interpd
    def bar(self, left, height, width=0.8, bottom=None, **kwargs):
        """
        Make a bar plot.

        Make a bar plot with rectangles bounded by:

          `left`, `left` + `width`, `bottom`, `bottom` + `height`
                (left, right, bottom and top edges)

        Parameters
        ----------
        left : sequence of scalars
            the x coordinates of the left sides of the bars

        height : sequence of scalars
            the heights of the bars

        width : scalar or array-like, optional, default: 0.8
            the width(s) of the bars

        bottom : scalar or array-like, optional, default: None
            the y coordinate(s) of the bars

        color : scalar or array-like, optional
            the colors of the bar faces

        edgecolor : scalar or array-like, optional
            the colors of the bar edges

        linewidth : scalar or array-like, optional, default: None
            width of bar edge(s). If None, use default
            linewidth; If 0, don't draw edges.

        xerr : scalar or array-like, optional, default: None
            if not None, will be used to generate errorbar(s) on the bar chart

        yerr :scalar or array-like, optional, default: None
            if not None, will be used to generate errorbar(s) on the bar chart

        ecolor : scalar or array-like, optional, default: None
            specifies the color of errorbar(s)

        capsize : integer, optional, default: 3
           determines the length in points of the error bar caps

        error_kw :
            dictionary of kwargs to be passed to errorbar method. *ecolor* and
            *capsize* may be specified here rather than as independent kwargs.

        align : ['edge' | 'center'], optional, default: 'edge'
            If `edge`, aligns bars by their left edges (for vertical bars) and
            by their bottom edges (for horizontal bars). If `center`, interpret
            the `left` argument as the coordinates of the centers of the bars.

        orientation : 'vertical' | 'horizontal', optional, default: 'vertical'
            The orientation of the bars.

        log : boolean, optional, default: False
            If true, sets the axis to be log scale

        Returns
        -------
        :class:`matplotlib.patches.Rectangle` instances.

        Notes
        -----
        The optional arguments `color`, `edgecolor`, `linewidth`,
        `xerr`, and `yerr` can be either scalars or sequences of
        length equal to the number of bars.  This enables you to use
        bar as the basis for stacked bar charts, or candlestick plots.
        Detail: `xerr` and `yerr` are passed directly to
        :meth:`errorbar`, so they can also have shape 2xN for
        independent specification of lower and upper errors.

        Other optional kwargs:

        %(Rectangle)s

        **Example:** A stacked bar chart.

        .. plot:: mpl_examples/pylab_examples/bar_stacked.py
        """
        if not self._hold:
            self.cla()
        color = kwargs.pop('color', None)
        edgecolor = kwargs.pop('edgecolor', None)
        linewidth = kwargs.pop('linewidth', None)

        # Because xerr and yerr will be passed to errorbar,
        # most dimension checking and processing will be left
        # to the errorbar method.
        xerr = kwargs.pop('xerr', None)
        yerr = kwargs.pop('yerr', None)
        error_kw = kwargs.pop('error_kw', dict())
        ecolor = kwargs.pop('ecolor', None)
        capsize = kwargs.pop('capsize', 3)
        error_kw.setdefault('ecolor', ecolor)
        error_kw.setdefault('capsize', capsize)

        align = kwargs.pop('align', 'edge')
        orientation = kwargs.pop('orientation', 'vertical')
        log = kwargs.pop('log', False)
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
                self.set_yscale('log', nonposy='clip')
            # size width and bottom according to length of left
            if _bottom is None:
                if self.get_yscale() == 'log':
                    adjust_ylim = True
                bottom = [0]

            nbars = len(left)
            if len(width) == 1:
                width *= nbars
            if len(bottom) == 1:
                bottom *= nbars
        elif orientation == 'horizontal':
            self._process_unit_info(xdata=width, ydata=bottom, kwargs=kwargs)
            if log:
                self.set_xscale('log', nonposx='clip')
            # size left and height according to length of bottom
            if _left is None:
                if self.get_xscale() == 'log':
                    adjust_xlim = True
                left = [0]

            nbars = len(bottom)
            if len(left) == 1:
                left *= nbars
            if len(height) == 1:
                height *= nbars
        else:
            raise ValueError('invalid orientation: %s' % orientation)

        if len(linewidth) < nbars:
            linewidth *= nbars

        if color is None:
            color = [None] * nbars
        else:
            color = list(mcolors.colorConverter.to_rgba_array(color))
            if len(color) == 0:  # until to_rgba_array is changed
                color = [[0, 0, 0, 0]]
            if len(color) < nbars:
                color *= nbars

        if edgecolor is None:
            edgecolor = [None] * nbars
        else:
            edgecolor = list(mcolors.colorConverter.to_rgba_array(edgecolor))
            if len(edgecolor) == 0:     # until to_rgba_array is changed
                edgecolor = [[0, 0, 0, 0]]
            if len(edgecolor) < nbars:
                edgecolor *= nbars

        # FIXME: convert the following to proper input validation
        # raising ValueError; don't use assert for this.
        assert len(left) == nbars, ("incompatible sizes: argument 'left' must "
                                    "be length %d or scalar" % nbars)
        assert len(height) == nbars, ("incompatible sizes: argument 'height' "
                                      "must be length %d or scalar" %
                                      nbars)
        assert len(width) == nbars, ("incompatible sizes: argument 'width' "
                                     "must be length %d or scalar" %
                                     nbars)
        assert len(bottom) == nbars, ("incompatible sizes: argument 'bottom' "
                                      "must be length %d or scalar" %
                                      nbars)

        patches = []

        # lets do some conversions now since some types cannot be
        # subtracted uniformly
        if self.xaxis is not None:
            left = self.convert_xunits(left)
            width = self.convert_xunits(width)
            if xerr is not None:
                xerr = self.convert_xunits(xerr)

        if self.yaxis is not None:
            bottom = self.convert_yunits(bottom)
            height = self.convert_yunits(height)
            if yerr is not None:
                yerr = self.convert_yunits(yerr)

        if align == 'edge':
            pass
        elif align == 'center':
            if orientation == 'vertical':
                left = [left[i] - width[i] / 2. for i in xrange(len(left))]
            elif orientation == 'horizontal':
                bottom = [bottom[i] - height[i] / 2.
                          for i in xrange(len(bottom))]

        else:
            raise ValueError('invalid alignment: %s' % align)

        args = zip(left, bottom, width, height, color, edgecolor, linewidth)
        for l, b, w, h, c, e, lw in args:
            if h < 0:
                b += h
                h = abs(h)
            if w < 0:
                l += w
                w = abs(w)
            r = mpatches.Rectangle(
                xy=(l, b), width=w, height=h,
                facecolor=c,
                edgecolor=e,
                linewidth=lw,
                label='_nolegend_'
                )
            r.update(kwargs)
            r.get_path()._interpolation_steps = 100
            #print r.get_label(), label, 'label' in kwargs
            self.add_patch(r)
            patches.append(r)

        holdstate = self._hold
        self.hold(True)  # ensure hold is on before plotting errorbars

        if xerr is not None or yerr is not None:
            if orientation == 'vertical':
                # using list comps rather than arrays to preserve unit info
                x = [l + 0.5 * w for l, w in zip(left, width)]
                y = [b + h for b, h in zip(bottom, height)]

            elif orientation == 'horizontal':
                # using list comps rather than arrays to preserve unit info
                x = [l + w for l, w in zip(left, width)]
                y = [b + 0.5 * h for b, h in zip(bottom, height)]

            if "label" not in error_kw:
                error_kw["label"] = '_nolegend_'

            errorbar = self.errorbar(x, y,
                                     yerr=yerr, xerr=xerr,
                                     fmt=None, **error_kw)
        else:
            errorbar = None

        self.hold(holdstate)  # restore previous hold state

        if adjust_xlim:
            xmin, xmax = self.dataLim.intervalx
            xmin = np.amin([w for w in width if w > 0])
            if xerr is not None:
                xmin = xmin - np.amax(xerr)
            xmin = max(xmin * 0.9, 1e-100)
            self.dataLim.intervalx = (xmin, xmax)

        if adjust_ylim:
            ymin, ymax = self.dataLim.intervaly
            ymin = np.amin([h for h in height if h > 0])
            if yerr is not None:
                ymin = ymin - np.amax(yerr)
            ymin = max(ymin * 0.9, 1e-100)
            self.dataLim.intervaly = (ymin, ymax)
        self.autoscale_view()

        bar_container = BarContainer(patches, errorbar, label=label)
        self.add_container(bar_container)

        return bar_container

    @docstring.dedent_interpd
    def barh(self, bottom, width, height=0.8, left=None, **kwargs):
        """
        Make a horizontal bar plot.

        Call signature::

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

        patches = self.bar(left=left, height=height, width=width,
                           bottom=bottom, orientation='horizontal', **kwargs)
        return patches

    @docstring.dedent_interpd
    def broken_barh(self, xranges, yrange, **kwargs):
        """
        Plot horizontal bars.

        Call signature::

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

    def stem(self, *args, **kwargs):
        """
        Create a stem plot.

        Call signatures::

          stem(y, linefmt='b-', markerfmt='bo', basefmt='r-')
          stem(x, y, linefmt='b-', markerfmt='bo', basefmt='r-')

        A stem plot plots vertical lines (using *linefmt*) at each *x*
        location from the baseline to *y*, and places a marker there
        using *markerfmt*.  A horizontal line at 0 is is plotted using
        *basefmt*.

        If no *x* values are provided, the default is (0, 1, ..., len(y) - 1)

        Return value is a tuple (*markerline*, *stemlines*,
        *baseline*).

        .. seealso::
            This
            `document <http://www.mathworks.com/help/techdoc/ref/stem.html>`_
            for details.


        **Example:**

        .. plot:: mpl_examples/pylab_examples/stem_plot.py
        """
        remember_hold = self._hold
        if not self._hold:
            self.cla()
        self.hold(True)

        # Assume there's at least one data array
        y = np.asarray(args[0], dtype=np.float)
        args = args[1:]

        # Try a second one
        try:
            second = np.asarray(args[0], dtype=np.float)
            x, y = y, second
            args = args[1:]
        except (IndexError, ValueError):
            # The second array doesn't make sense, or it doesn't exist
            second = np.arange(len(y))
            x = second

        # Popping some defaults
        try:
            linefmt = kwargs.pop('linefmt', args[0])
        except IndexError:
            linefmt = kwargs.pop('linefmt', 'b-')
        try:
            markerfmt = kwargs.pop('markerfmt', args[1])
        except IndexError:
            markerfmt = kwargs.pop('markerfmt', 'bo')
        try:
            basefmt = kwargs.pop('basefmt', args[2])
        except IndexError:
            basefmt = kwargs.pop('basefmt', 'r-')

        bottom = kwargs.pop('bottom', None)
        label = kwargs.pop('label', None)

        markerline, = self.plot(x, y, markerfmt, label="_nolegend_")

        if bottom is None:
            bottom = 0

        stemlines = []
        for thisx, thisy in zip(x, y):
            l, = self.plot([thisx, thisx], [bottom, thisy], linefmt,
                           label="_nolegend_")
            stemlines.append(l)

        baseline, = self.plot([np.amin(x), np.amax(x)], [bottom, bottom],
                              basefmt, label="_nolegend_")

        self.hold(remember_hold)

        stem_container = StemContainer((markerline, stemlines, baseline),
                                       label=label)
        self.add_container(stem_container)

        return stem_container

    def pie(self, x, explode=None, labels=None, colors=None,
            autopct=None, pctdistance=0.6, shadow=False,
            labeldistance=1.1, startangle=None, radius=None):
        r"""
        Plot a pie chart.

        Call signature::

          pie(x, explode=None, labels=None,
              colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'),
              autopct=None, pctdistance=0.6, shadow=False,
              labeldistance=1.1, startangle=None, radius=None)

        Make a pie chart of array *x*.  The fractional area of each
        wedge is given by x/sum(x).  If sum(x) <= 1, then the values
        of x give the fractional area directly and the array will not
        be normalized.  The wedges are plotted counterclockwise,
        by default starting from the x-axis.

        Keyword arguments:

          *explode*: [ *None* | len(x) sequence ]
            If not *None*, is a ``len(x)`` array which specifies the
            fraction of the radius with which to offset each wedge.

          *colors*: [ *None* | color sequence ]
            A sequence of matplotlib color args through which the pie chart
            will cycle.

          *labels*: [ *None* | len(x) sequence of strings ]
            A sequence of strings providing the labels for each wedge

          *autopct*: [ *None* | format string | format function ]
            If not *None*, is a string or function used to label the wedges
            with their numeric value.  The label will be placed inside the
            wedge.  If it is a format string, the label will be ``fmt%pct``.
            If it is a function, it will be called.

          *pctdistance*: scalar
            The ratio between the center of each pie slice and the
            start of the text generated by *autopct*.  Ignored if
            *autopct* is *None*; default is 0.6.

          *labeldistance*: scalar
            The radial distance at which the pie labels are drawn

          *shadow*: [ *False* | *True* ]
            Draw a shadow beneath the pie.

          *startangle*: [ *None* | Offset angle ]
            If not *None*, rotates the start of the pie chart by *angle*
            degrees counterclockwise from the x-axis.

          *radius*: [ *None* | scalar ]
          The radius of the pie, if *radius* is *None* it will be set to 1.

        The pie chart will probably look best if the figure and axes are
        square, or the Axes aspect is equal.  e.g.::

          figure(figsize=(8,8))
          ax = axes([0.1, 0.1, 0.8, 0.8])

        or::

          axes(aspect=1)

        Return value:
          If *autopct* is *None*, return the tuple (*patches*, *texts*):

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
        if sx > 1:
            x = np.divide(x, sx)

        if labels is None:
            labels = [''] * len(x)
        if explode is None:
            explode = [0] * len(x)
        assert(len(x) == len(labels))
        assert(len(x) == len(explode))
        if colors is None:
            colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')

        center = 0, 0
        if radius is None:
            radius = 1

        # Starting theta1 is the start fraction of the circle
        if startangle is None:
            theta1 = 0
        else:
            theta1 = startangle / 360.0

        texts = []
        slices = []
        autotexts = []

        i = 0
        for frac, label, expl in cbook.safezip(x, labels, explode):
            x, y = center
            theta2 = theta1 + frac
            thetam = 2 * math.pi * 0.5 * (theta1 + theta2)
            x += expl * math.cos(thetam)
            y += expl * math.sin(thetam)

            w = mpatches.Wedge((x, y), radius, 360. * theta1, 360. * theta2,
                      facecolor=colors[i % len(colors)])
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
                shad.set_zorder(0.9 * w.get_zorder())
                shad.set_label('_nolegend_')
                self.add_patch(shad)

            xt = x + labeldistance * radius * math.cos(thetam)
            yt = y + labeldistance * radius * math.sin(thetam)
            label_alignment = xt > 0 and 'left' or 'right'

            t = self.text(xt, yt, label,
                          size=rcParams['xtick.labelsize'],
                          horizontalalignment=label_alignment,
                          verticalalignment='center')

            texts.append(t)

            if autopct is not None:
                xt = x + pctdistance * radius * math.cos(thetam)
                yt = y + pctdistance * radius * math.sin(thetam)
                if is_string_like(autopct):
                    s = autopct % (100. * frac)
                elif callable(autopct):
                    s = autopct(100. * frac)
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

        if autopct is None:
            return slices, texts
        else:
            return slices, texts, autotexts

    @docstring.dedent_interpd
    def errorbar(self, x, y, yerr=None, xerr=None,
                 fmt='-', ecolor=None, elinewidth=None, capsize=3,
                 barsabove=False, lolims=False, uplims=False,
                 xlolims=False, xuplims=False, errorevery=1, capthick=None,
                 **kwargs):
        """
        Plot an errorbar graph.

        Call signature::

          errorbar(x, y, yerr=None, xerr=None,
                   fmt='-', ecolor=None, elinewidth=None, capsize=3,
                   barsabove=False, lolims=False, uplims=False,
                   xlolims=False, xuplims=False, errorevery=1,
                   capthick=None)

        Plot *x* versus *y* with error deltas in *yerr* and *xerr*.
        Vertical errorbars are plotted if *yerr* is not *None*.
        Horizontal errorbars are plotted if *xerr* is not *None*.

        *x*, *y*, *xerr*, and *yerr* can all be scalars, which plots a
        single error bar at *x*, *y*.

        Optional keyword arguments:

          *xerr*/*yerr*: [ scalar | N, Nx1, or 2xN array-like ]
            If a scalar number, len(N) array-like object, or an Nx1
            array-like object, errorbars are drawn at +/-value relative
            to the data.

            If a sequence of shape 2xN, errorbars are drawn at -row1
            and +row2 relative to the data.

          *fmt*: '-'
            The plot format symbol. If *fmt* is *None*, only the
            errorbars are plotted.  This is used for adding
            errorbars to a bar plot, for example.

          *ecolor*: [ *None* | mpl color ]
            A matplotlib color arg which gives the color the errorbar lines;
            if *None*, use the marker color.

          *elinewidth*: scalar
            The linewidth of the errorbar lines. If *None*, use the linewidth.

          *capsize*: scalar
            The length of the error bar caps in points

          *capthick*: scalar
            An alias kwarg to *markeredgewidth* (a.k.a. - *mew*). This
            setting is a more sensible name for the property that
            controls the thickness of the error bar cap in points. For
            backwards compatibility, if *mew* or *markeredgewidth* are given,
            then they will over-ride *capthick*.  This may change in future
            releases.

          *barsabove*: [ *True* | *False* ]
            if *True*, will plot the errorbars above the plot
            symbols. Default is below.

          *lolims* / *uplims* / *xlolims* / *xuplims*: [ *False* | *True* ]
            These arguments can be used to indicate that a value gives
            only upper/lower limits. In that case a caret symbol is
            used to indicate this. lims-arguments may be of the same
            type as *xerr* and *yerr*.

          *errorevery*: positive integer
            subsamples the errorbars. e.g., if everyerror=5, errorbars for
            every 5-th datapoint will be plotted. The data plot itself still
            shows all data points.

        All other keyword arguments are passed on to the plot command for the
        markers. For example, this code makes big red squares with
        thick green edges::

          x,y,yerr = rand(3,10)
          errorbar(x, y, yerr, marker='s',
                   mfc='red', mec='green', ms=20, mew=4)

        where *mfc*, *mec*, *ms* and *mew* are aliases for the longer
        property names, *markerfacecolor*, *markeredgecolor*, *markersize*
        and *markeredgewith*.

        valid kwargs for the marker properties are

        %(Line2D)s

        Returns (*plotline*, *caplines*, *barlinecols*):

            *plotline*: :class:`~matplotlib.lines.Line2D` instance
                *x*, *y* plot markers and/or line

            *caplines*: list of error bar cap
                :class:`~matplotlib.lines.Line2D` instances
            *barlinecols*: list of
                :class:`~matplotlib.collections.LineCollection` instances for
                the horizontal and vertical error ranges.

        **Example:**

        .. plot:: mpl_examples/statistics/errorbar_demo.py

        """

        if errorevery < 1:
            raise ValueError(
                'errorevery has to be a strictly positive integer')

        self._process_unit_info(xdata=x, ydata=y, kwargs=kwargs)
        if not self._hold:
            self.cla()
        holdstate = self._hold
        self._hold = True

        label = kwargs.pop("label", None)

        # make sure all the args are iterable; use lists not arrays to
        # preserve units
        if not iterable(x):
            x = [x]

        if not iterable(y):
            y = [y]

        if xerr is not None:
            if not iterable(xerr):
                xerr = [xerr] * len(x)

        if yerr is not None:
            if not iterable(yerr):
                yerr = [yerr] * len(y)

        l0 = None

        if barsabove and fmt is not None:
            l0, = self.plot(x, y, fmt, label="_nolegend_", **kwargs)

        barcols = []
        caplines = []

        lines_kw = {'label': '_nolegend_'}
        if elinewidth:
            lines_kw['linewidth'] = elinewidth
        else:
            if 'linewidth' in kwargs:
                lines_kw['linewidth'] = kwargs['linewidth']
            if 'lw' in kwargs:
                lines_kw['lw'] = kwargs['lw']
        if 'transform' in kwargs:
            lines_kw['transform'] = kwargs['transform']
        if 'alpha' in kwargs:
            lines_kw['alpha'] = kwargs['alpha']
        if 'zorder' in kwargs:
            lines_kw['zorder'] = kwargs['zorder']

        # arrays fine here, they are booleans and hence not units
        if not iterable(lolims):
            lolims = np.asarray([lolims] * len(x), bool)
        else:
            lolims = np.asarray(lolims, bool)

        if not iterable(uplims):
            uplims = np.array([uplims] * len(x), bool)
        else:
            uplims = np.asarray(uplims, bool)

        if not iterable(xlolims):
            xlolims = np.array([xlolims] * len(x), bool)
        else:
            xlolims = np.asarray(xlolims, bool)

        if not iterable(xuplims):
            xuplims = np.array([xuplims] * len(x), bool)
        else:
            xuplims = np.asarray(xuplims, bool)

        everymask = np.arange(len(x)) % errorevery == 0

        def xywhere(xs, ys, mask):
            """
            return xs[mask], ys[mask] where mask is True but xs and
            ys are not arrays
            """
            assert len(xs) == len(ys)
            assert len(xs) == len(mask)
            xs = [thisx for thisx, b in zip(xs, mask) if b]
            ys = [thisy for thisy, b in zip(ys, mask) if b]
            return xs, ys

        if capsize > 0:
            plot_kw = {
                'ms': 2 * capsize,
                'label': '_nolegend_'}
            if capthick is not None:
                # 'mew' has higher priority, I believe,
                # if both 'mew' and 'markeredgewidth' exists.
                # So, save capthick to markeredgewidth so that
                # explicitly setting mew or markeredgewidth will
                # over-write capthick.
                plot_kw['markeredgewidth'] = capthick
            # For backwards-compat, allow explicit setting of
            # 'mew' or 'markeredgewidth' to over-ride capthick.
            if 'markeredgewidth' in kwargs:
                plot_kw['markeredgewidth'] = kwargs['markeredgewidth']
            if 'mew' in kwargs:
                plot_kw['mew'] = kwargs['mew']
            if 'transform' in kwargs:
                plot_kw['transform'] = kwargs['transform']
            if 'alpha' in kwargs:
                plot_kw['alpha'] = kwargs['alpha']
            if 'zorder' in kwargs:
                plot_kw['zorder'] = kwargs['zorder']

        if xerr is not None:
            if (iterable(xerr) and len(xerr) == 2 and
                iterable(xerr[0]) and iterable(xerr[1])):
                # using list comps rather than arrays to preserve units
                left = [thisx - thiserr for (thisx, thiserr)
                        in cbook.safezip(x, xerr[0])]
                right = [thisx + thiserr for (thisx, thiserr)
                         in cbook.safezip(x, xerr[1])]
            else:
                # using list comps rather than arrays to preserve units
                left = [thisx - thiserr for (thisx, thiserr)
                        in cbook.safezip(x, xerr)]
                right = [thisx + thiserr for (thisx, thiserr)
                         in cbook.safezip(x, xerr)]

            yo, _ = xywhere(y, right, everymask)
            lo, ro = xywhere(left, right, everymask)
            barcols.append(self.hlines(yo, lo, ro, **lines_kw))
            if capsize > 0:
                if xlolims.any():
                    # can't use numpy logical indexing since left and
                    # y are lists
                    leftlo, ylo = xywhere(left, y, xlolims & everymask)

                    caplines.extend(
                        self.plot(leftlo, ylo, ls='None',
                                  marker=mlines.CARETLEFT, **plot_kw))
                    xlolims = ~xlolims
                    leftlo, ylo = xywhere(left, y, xlolims & everymask)
                    caplines.extend(self.plot(leftlo, ylo, 'k|', **plot_kw))
                else:

                    leftlo, ylo = xywhere(left, y, everymask)
                    caplines.extend(self.plot(leftlo, ylo, 'k|', **plot_kw))

                if xuplims.any():

                    rightup, yup = xywhere(right, y, xuplims & everymask)
                    caplines.extend(
                        self.plot(rightup,  yup, ls='None',
                                  marker=mlines.CARETRIGHT, **plot_kw))
                    xuplims = ~xuplims
                    rightup, yup = xywhere(right, y, xuplims & everymask)
                    caplines.extend(self.plot(rightup, yup, 'k|', **plot_kw))
                else:
                    rightup, yup = xywhere(right, y, everymask)
                    caplines.extend(self.plot(rightup, yup, 'k|', **plot_kw))

        if yerr is not None:
            if (iterable(yerr) and len(yerr) == 2 and
                iterable(yerr[0]) and iterable(yerr[1])):
                # using list comps rather than arrays to preserve units
                lower = [thisy - thiserr for (thisy, thiserr)
                         in cbook.safezip(y, yerr[0])]
                upper = [thisy + thiserr for (thisy, thiserr)
                         in cbook.safezip(y, yerr[1])]
            else:
                # using list comps rather than arrays to preserve units
                lower = [thisy - thiserr for (thisy, thiserr)
                         in cbook.safezip(y, yerr)]
                upper = [thisy + thiserr for (thisy, thiserr)
                         in cbook.safezip(y, yerr)]

            xo, _ = xywhere(x, lower, everymask)
            lo, uo = xywhere(lower, upper, everymask)
            barcols.append(self.vlines(xo, lo, uo, **lines_kw))
            if capsize > 0:

                if lolims.any():
                    xlo, lowerlo = xywhere(x, lower, lolims & everymask)
                    caplines.extend(
                        self.plot(xlo, lowerlo, ls='None',
                                  marker=mlines.CARETDOWN, **plot_kw))
                    lolims = ~lolims
                    xlo, lowerlo = xywhere(x, lower, lolims & everymask)
                    caplines.extend(self.plot(xlo, lowerlo, 'k_', **plot_kw))
                else:
                    xlo, lowerlo = xywhere(x, lower, everymask)
                    caplines.extend(self.plot(xlo, lowerlo, 'k_', **plot_kw))

                if uplims.any():
                    xup, upperup = xywhere(x, upper, uplims & everymask)

                    caplines.extend(
                        self.plot(xup, upperup, ls='None',
                                  marker=mlines.CARETUP, **plot_kw))
                    uplims = ~uplims
                    xup, upperup = xywhere(x, upper, uplims & everymask)
                    caplines.extend(self.plot(xup, upperup, 'k_', **plot_kw))
                else:
                    xup, upperup = xywhere(x, upper, everymask)
                    caplines.extend(self.plot(xup, upperup, 'k_', **plot_kw))

        if not barsabove and fmt is not None:
            l0, = self.plot(x, y, fmt, **kwargs)

        if ecolor is None:
            if l0 is None:
                ecolor = self._get_lines.color_cycle.next()
            else:
                ecolor = l0.get_color()

        for l in barcols:
            l.set_color(ecolor)
        for l in caplines:
            l.set_color(ecolor)

        self.autoscale_view()
        self._hold = holdstate

        errorbar_container = ErrorbarContainer((l0, tuple(caplines),
                                                tuple(barcols)),
                                               has_xerr=(xerr is not None),
                                               has_yerr=(yerr is not None),
                                               label=label)
        self.containers.append(errorbar_container)

        return errorbar_container  # (l0, caplines, barcols)

    def boxplot(self, x, notch=False, sym='b+', vert=True, whis=1.5,
                positions=None, widths=None, patch_artist=False,
                bootstrap=None, usermedians=None, conf_intervals=None):
        """
        Make a box and whisker plot.

        Call signature::

          boxplot(x, notch=False, sym='+', vert=True, whis=1.5,
                  positions=None, widths=None, patch_artist=False,
                  bootstrap=None, usermedians=None, conf_intervals=None)

        Make a box and whisker plot for each column of *x* or each
        vector in sequence *x*.  The box extends from the lower to
        upper quartile values of the data, with a line at the median.
        The whiskers extend from the box to show the range of the
        data.  Flier points are those past the end of the whiskers.

        Function Arguments:

          *x* :
            Array or a sequence of vectors.

          *notch* : [ False (default) | True ]
            If False (default), produces a rectangular box plot.
            If True, will produce a notched box plot

          *sym* : [ default 'b+' ]
            The default symbol for flier points.
            Enter an empty string ('') if you don't want to show fliers.

          *vert* : [ False | True (default) ]
            If True (default), makes the boxes vertical.
            If False, makes horizontal boxes.

          *whis* : [ default 1.5 ]
            Defines the length of the whiskers as a function of the inner
            quartile range.  They extend to the most extreme data point
            within ( ``whis*(75%-25%)`` ) data range.

          *bootstrap* : [ *None* (default) | integer ]
            Specifies whether to bootstrap the confidence intervals
            around the median for notched boxplots. If bootstrap==None,
            no bootstrapping is performed, and notches are calculated
            using a Gaussian-based asymptotic approximation  (see McGill, R.,
            Tukey, J.W., and Larsen, W.A., 1978, and Kendall and Stuart,
            1967). Otherwise, bootstrap specifies the number of times to
            bootstrap the median to determine it's 95% confidence intervals.
            Values between 1000 and 10000 are recommended.

          *usermedians* : [ default None ]
            An array or sequence whose first dimension (or length) is
            compatible with *x*. This overrides the medians computed by
            matplotlib for each element of *usermedians* that is not None.
            When an element of *usermedians* == None, the median will be
            computed directly as normal.

          *conf_intervals* : [ default None ]
            Array or sequence whose first dimension (or length) is compatible
            with *x* and whose second dimension is 2. When the current element
            of *conf_intervals* is not None, the notch locations computed by
            matplotlib are overridden (assuming notch is True). When an
            element of *conf_intervals* is None, boxplot compute notches the
            method specified by the other kwargs (e.g., *bootstrap*).

          *positions* : [ default 1,2,...,n ]
            Sets the horizontal positions of the boxes. The ticks and limits
            are automatically set to match the positions.

          *widths* : [ default 0.5 ]
            Either a scalar or a vector and sets the width of each box. The
            default is 0.5, or ``0.15*(distance between extreme positions)``
            if that is smaller.

          *patch_artist* : [ False (default) | True ]
            If False produces boxes with the Line2D artist
            If True produces boxes with the Patch artist

        Returns a dictionary mapping each component of the boxplot
        to a list of the :class:`matplotlib.lines.Line2D`
        instances created. That dictionary has the following keys
        (assuming vertical boxplots):

            - boxes: the main body of the boxplot showing the quartiles
              and the median's confidence intervals if enabled.
            - medians: horizonal lines at the median of each box.
            - whiskers: the vertical lines extending to the most extreme,
              n-outlier data points.
            - caps: the horizontal lines at the ends of the whiskers.
            - fliers: points representing data that extend beyone the
              whiskers (outliers).

        **Example:**

        .. plot:: pyplots/boxplot_demo.py
        """
        def bootstrapMedian(data, N=5000):
            # determine 95% confidence intervals of the median
            M = len(data)
            percentile = [2.5, 97.5]
            estimate = np.zeros(N)
            for n in range(N):
                bsIndex = np.random.random_integers(0, M - 1, M)
                bsData = data[bsIndex]
                estimate[n] = mlab.prctile(bsData, 50)
            CI = mlab.prctile(estimate, percentile)
            return CI

        def computeConfInterval(data, med, iq, bootstrap):
            if bootstrap is not None:
                # Do a bootstrap estimate of notch locations.
                # get conf. intervals around median
                CI = bootstrapMedian(data, N=bootstrap)
                notch_min = CI[0]
                notch_max = CI[1]
            else:
                # Estimate notch locations using Gaussian-based
                # asymptotic approximation.
                #
                # For discussion: McGill, R., Tukey, J.W.,
                # and Larsen, W.A. (1978) "Variations of
                # Boxplots", The American Statistician, 32:12-16.
                N = len(data)
                notch_min = med - 1.57 * iq / np.sqrt(N)
                notch_max = med + 1.57 * iq / np.sqrt(N)
            return notch_min, notch_max

        if not self._hold:
            self.cla()
        holdStatus = self._hold
        whiskers, caps, boxes, medians, fliers = [], [], [], [], []

        # convert x to a list of vectors
        if hasattr(x, 'shape'):
            if len(x.shape) == 1:
                if hasattr(x[0], 'shape'):
                    x = list(x)
                else:
                    x = [x, ]
            elif len(x.shape) == 2:
                nr, nc = x.shape
                if nr == 1:
                    x = [x]
                elif nc == 1:
                    x = [x.ravel()]
                else:
                    x = [x[:, i] for i in xrange(nc)]
            else:
                raise ValueError("input x can have no more than 2 dimensions")
        if not hasattr(x[0], '__len__'):
            x = [x]
        col = len(x)

        # sanitize user-input medians
        msg1 = "usermedians must either be a list/tuple or a 1d array"
        msg2 = "usermedians' length must be compatible with x"
        if usermedians is not None:
            if hasattr(usermedians, 'shape'):
                if len(usermedians.shape) != 1:
                    raise ValueError(msg1)
                elif usermedians.shape[0] != col:
                    raise ValueError(msg2)
            elif len(usermedians) != col:
                raise ValueError(msg2)

        #sanitize user-input confidence intervals
        msg1 = "conf_intervals must either be a list of tuples or a 2d array"
        msg2 = "conf_intervals' length must be compatible with x"
        msg3 = "each conf_interval, if specificied, must have two values"
        if conf_intervals is not None:
            if hasattr(conf_intervals, 'shape'):
                if len(conf_intervals.shape) != 2:
                    raise ValueError(msg1)
                elif conf_intervals.shape[0] != col:
                    raise ValueError(msg2)
                elif conf_intervals.shape[1] == 2:
                    raise ValueError(msg3)
            else:
                if len(conf_intervals) != col:
                    raise ValueError(msg2)
                for ci in conf_intervals:
                    if ci is not None and len(ci) != 2:
                        raise ValueError(msg3)

        # get some plot info
        if positions is None:
            positions = range(1, col + 1)
        if widths is None:
            distance = max(positions) - min(positions)
            widths = min(0.15 * max(distance, 1.0), 0.5)
        if isinstance(widths, float) or isinstance(widths, int):
            widths = np.ones((col,), float) * widths

        # loop through columns, adding each to plot
        self.hold(True)
        for i, pos in enumerate(positions):
            d = np.ravel(x[i])
            row = len(d)
            if row == 0:
                # no data, skip this position
                continue

            # get median and quartiles
            q1, med, q3 = mlab.prctile(d, [25, 50, 75])

            # replace with input medians if available
            if usermedians is not None:
                if usermedians[i] is not None:
                    med = usermedians[i]

            # get high extreme
            iq = q3 - q1
            hi_val = q3 + whis * iq
            wisk_hi = np.compress(d <= hi_val, d)
            if len(wisk_hi) == 0 or np.max(wisk_hi) < q3:
                wisk_hi = q3
            else:
                wisk_hi = max(wisk_hi)

            # get low extreme
            lo_val = q1 - whis * iq
            wisk_lo = np.compress(d >= lo_val, d)
            if len(wisk_lo) == 0 or np.min(wisk_lo) > q1:
                wisk_lo = q1
            else:
                wisk_lo = min(wisk_lo)

            # get fliers - if we are showing them
            flier_hi = []
            flier_lo = []
            flier_hi_x = []
            flier_lo_x = []
            if len(sym) != 0:
                flier_hi = np.compress(d > wisk_hi, d)
                flier_lo = np.compress(d < wisk_lo, d)
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

            # calculate 'notch' plot
            if notch:
                # conf. intervals from user, if available
                if (conf_intervals is not None and
                    conf_intervals[i] is not None):
                    notch_max = np.max(conf_intervals[i])
                    notch_min = np.min(conf_intervals[i])
                else:
                    notch_min, notch_max = computeConfInterval(d, med, iq,
                                                               bootstrap)

                # make our notched box vectors
                box_x = [box_x_min, box_x_max, box_x_max, cap_x_max, box_x_max,
                         box_x_max, box_x_min, box_x_min, cap_x_min, box_x_min,
                         box_x_min]
                box_y = [q1, q1, notch_min, med, notch_max, q3, q3, notch_max,
                         med, notch_min, q1]
                # make our median line vectors
                med_x = [cap_x_min, cap_x_max]
                med_y = [med, med]
            # calculate 'regular' plot
            else:
                # make our box vectors
                box_x = [box_x_min, box_x_max, box_x_max, box_x_min, box_x_min]
                box_y = [q1, q1, q3, q3, q1]
                # make our median line vectors
                med_x = [box_x_min, box_x_max]

            def to_vc(xs, ys):
                # convert arguments to verts and codes
                verts = []
                #codes = []
                for xi, yi in zip(xs, ys):
                    verts.append((xi, yi))
                verts.append((0, 0))  # ignored
                codes = [mpath.Path.MOVETO] + \
                        [mpath.Path.LINETO] * (len(verts) - 2) + \
                        [mpath.Path.CLOSEPOLY]
                return verts, codes

            def patch_list(xs, ys):
                verts, codes = to_vc(xs, ys)
                path = mpath.Path(verts, codes)
                patch = mpatches.PathPatch(path)
                self.add_artist(patch)
                return [patch]

            # vertical or horizontal plot?
            if vert:

                def doplot(*args):
                    return self.plot(*args)

                def dopatch(xs, ys):
                    return patch_list(xs, ys)
            else:

                def doplot(*args):
                    shuffled = []
                    for i in xrange(0, len(args), 3):
                        shuffled.extend([args[i + 1], args[i], args[i + 2]])
                    return self.plot(*shuffled)

                def dopatch(xs, ys):
                    xs, ys = ys, xs  # flip X, Y
                    return patch_list(xs, ys)

            if patch_artist:
                median_color = 'k'
            else:
                median_color = 'r'

            whiskers.extend(doplot(wisk_x, [q1, wisk_lo], 'b--',
                                   wisk_x, [q3, wisk_hi], 'b--'))
            caps.extend(doplot(cap_x, [wisk_hi, wisk_hi], 'k-',
                               cap_x, [wisk_lo, wisk_lo], 'k-'))
            if patch_artist:
                boxes.extend(dopatch(box_x, box_y))
            else:
                boxes.extend(doplot(box_x, box_y, 'b-'))

            medians.extend(doplot(med_x, med_y, median_color + '-'))
            fliers.extend(doplot(flier_hi_x, flier_hi, sym,
                                 flier_lo_x, flier_lo, sym))

        # fix our axes/ticks up a little
        if vert:
            setticks, setlim = self.set_xticks, self.set_xlim
        else:
            setticks, setlim = self.set_yticks, self.set_ylim

        newlimits = min(positions) - 0.5, max(positions) + 0.5
        setlim(newlimits)
        setticks(positions)

        # reset hold status
        self.hold(holdStatus)

        return dict(whiskers=whiskers, caps=caps, boxes=boxes,
                    medians=medians, fliers=fliers)

    @docstring.dedent_interpd
    def scatter(self, x, y, s=20, c='b', marker='o', cmap=None, norm=None,
                vmin=None, vmax=None, alpha=None, linewidths=None,
                verts=None, **kwargs):
        """
        Make a scatter plot of x vs y, where x and y are sequence like objects
        of the same lengths.

        Parameters
        ----------
        x, y : array_like, shape (n, )
            Input data

        s : scalar or array_like, shape (n, ), optional, default: 20
            size in points^2.

        c : color or sequence of color, optional, default : 'b'
            `c` can be a single color format string, or a sequence of color
            specifications of length `N`, or a sequence of `N` numbers to be
            mapped to colors using the `cmap` and `norm` specified via kwargs
            (see below). Note that `c` should not be a single numeric RGB or
            RGBA sequence because that is indistinguishable from an array of
            values to be colormapped.  `c` can be a 2-D array in which the
            rows are RGB or RGBA, however.

        marker : `~matplotlib.markers.MarkerStyle`, optional, default: 'o'
            See `~matplotlib.markers` for more information on the different
            styles of markers scatter supports.

        cmap : `~matplotlib.colors.Colormap`, optional, default: None
            A `~matplotlib.colors.Colormap` instance or registered name.
            `cmap` is only used if `c` is an array of floats. If None,
            defaults to rc `image.cmap`.

        norm : `~matplotlib.colors.Normalize`, optional, default: None
            A `~matplotlib.colors.Normalize` instance is used to scale
            luminance data to 0, 1. `norm` is only used if `c` is an array of
            floats. If `None`, use the default :func:`normalize`.

        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with `norm` to normalize
            luminance data.  If either are `None`, the min and max of the
            color array is used.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.

        alpha : scalar, optional, default: None
            The alpha blending value, between 0 (transparent) and 1 (opaque)

        linewidths : scalar or array_like, optional, default: None
            If None, defaults to (lines.linewidth,).  Note that this is a
            tuple, and if you set the linewidths argument you must set it as a
            sequence of floats, as required by
            `~matplotlib.collections.RegularPolyCollection`.

        Returns
        -------
        paths : `~matplotlib.collections.PathCollection`

        Other parameters
        ----------------
        kwargs : `~matplotlib.collections.Collection` properties

        Notes
        ------
        Any or all of `x`, `y`, `s`, and `c` may be masked arrays, in
        which case all masks will be combined and only unmasked points
        will be plotted.

        Examples
        --------
        .. plot:: mpl_examples/shapes_and_collections/scatter_demo.py

        """

        if not self._hold:
            self.cla()

        self._process_unit_info(xdata=x, ydata=y, kwargs=kwargs)
        x = self.convert_xunits(x)
        y = self.convert_yunits(y)

        # np.ma.ravel yields an ndarray, not a masked array,
        # unless its argument is a masked array.
        x = np.ma.ravel(x)
        y = np.ma.ravel(y)
        if x.size != y.size:
            raise ValueError("x and y must be the same size")

        s = np.ma.ravel(s)  # This doesn't have to match x, y in size.

        c_is_stringy = is_string_like(c) or is_sequence_of_strings(c)
        if not c_is_stringy:
            c = np.asanyarray(c)
            if c.size == x.size:
                c = np.ma.ravel(c)

        x, y, s, c = cbook.delete_masked_points(x, y, s, c)

        scales = s   # Renamed for readability below.

        if c_is_stringy:
            colors = mcolors.colorConverter.to_rgba_array(c, alpha)
        else:
            # The inherent ambiguity is resolved in favor of color
            # mapping, not interpretation as rgb or rgba:
            if c.size == x.size:
                colors = None  # use cmap, norm after collection is created
            else:
                colors = mcolors.colorConverter.to_rgba_array(c, alpha)

        faceted = kwargs.pop('faceted', None)
        edgecolors = kwargs.get('edgecolors', None)
        if faceted is not None:
            cbook.warn_deprecated(
                '1.2', name='faceted', alternative='edgecolor',
                obj_type='option')
            if faceted:
                edgecolors = None
            else:
                edgecolors = 'none'

        # to be API compatible
        if marker is None and not (verts is None):
            marker = (verts, 0)
            verts = None

        marker_obj = mmarkers.MarkerStyle(marker)
        path = marker_obj.get_path().transformed(
            marker_obj.get_transform())
        if not marker_obj.is_filled():
            edgecolors = 'face'

        collection = mcoll.PathCollection(
                (path,), scales,
                facecolors=colors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                offsets=zip(x, y),
                transOffset=kwargs.pop('transform', self.transData),
                )
        collection.set_transform(mtransforms.IdentityTransform())
        collection.set_alpha(alpha)
        collection.update(kwargs)

        if colors is None:
            if norm is not None:
                assert(isinstance(norm, mcolors.Normalize))
            collection.set_array(np.asarray(c))
            collection.set_cmap(cmap)
            collection.set_norm(norm)

            if vmin is not None or vmax is not None:
                collection.set_clim(vmin, vmax)
            else:
                collection.autoscale_None()

        # The margin adjustment is a hack to deal with the fact that we don't
        # want to transform all the symbols whose scales are in points
        # to data coords to get the exact bounding box for efficiency
        # reasons.  It can be done right if this is deemed important.
        # Also, only bother with this padding if there is anything to draw.
        if self._xmargin < 0.05 and x.size > 0:
            self.set_xmargin(0.05)

        if self._ymargin < 0.05 and x.size > 0:
            self.set_ymargin(0.05)

        self.add_collection(collection)
        self.autoscale_view()

        return collection

    @docstring.dedent_interpd
    def hexbin(self, x, y, C=None, gridsize=100, bins=None,
               xscale='linear', yscale='linear', extent=None,
               cmap=None, norm=None, vmin=None, vmax=None,
               alpha=None, linewidths=None, edgecolors='none',
               reduce_C_function=np.mean, mincnt=None, marginals=False,
               **kwargs):
        """
        Make a hexagonal binning plot.

        Call signature::

           hexbin(x, y, C = None, gridsize = 100, bins = None,
                  xscale = 'linear', yscale = 'linear',
                  cmap=None, norm=None, vmin=None, vmax=None,
                  alpha=None, linewidths=None, edgecolors='none'
                  reduce_C_function = np.mean, mincnt=None, marginals=True
                  **kwargs)

        Make a hexagonal binning plot of *x* versus *y*, where *x*,
        *y* are 1-D sequences of the same length, *N*. If *C* is *None*
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

        *bins*: [ *None* | 'log' | integer | sequence ]
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

        *mincnt*: [ *None* | a positive integer ]
           If not *None*, only display cells with more than *mincnt*
           number of points in the cell

        *marginals*: [ *True* | *False* ]
           if marginals is *True*, plot the marginal density as
           colormapped rectagles along the bottom of the x-axis and
           left of the y-axis

        *extent*: [ *None* | scalars (left, right, bottom, top) ]
           The limits of the bins. The default assigns the limits
           based on gridsize, x, y, xscale and yscale.

        Other keyword arguments controlling color mapping and normalization
        arguments:

        *cmap*: [ *None* | Colormap ]
           a :class:`matplotlib.colors.Colormap` instance. If *None*,
           defaults to rc ``image.cmap``.

        *norm*: [ *None* | Normalize ]
           :class:`matplotlib.colors.Normalize` instance is used to
           scale luminance data to 0,1.

        *vmin* / *vmax*: scalar
           *vmin* and *vmax* are used in conjunction with *norm* to normalize
           luminance data.  If either are *None*, the min and max of the color
           array *C* is used.  Note if you pass a norm instance, your settings
           for *vmin* and *vmax* will be ignored.

        *alpha*: scalar between 0 and 1, or *None*
           the alpha value for the patches

        *linewidths*: [ *None* | scalar ]
           If *None*, defaults to rc lines.linewidth. Note that this
           is a tuple, and if you set the linewidths argument you
           must set it as a sequence of floats, as required by
           :class:`~matplotlib.collections.RegularPolyCollection`.

        Other keyword arguments controlling the Collection properties:

        *edgecolors*: [ *None* | ``'none'`` | mpl color | color sequence ]
           If ``'none'``, draws the edges in the same color as the fill color.
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
        :meth:`~matplotlib.collections.PolyCollection.get_array` on
        this :class:`~matplotlib.collections.PolyCollection` to get
        the counts in each hexagon. If *marginals* is *True*, horizontal
        bar and vertical bar (both PolyCollections) will be attached
        to the return collection as attributes *hbar* and *vbar*.


        **Example:**

        .. plot:: mpl_examples/pylab_examples/hexbin_demo.py

        """

        if not self._hold:
            self.cla()

        self._process_unit_info(xdata=x, ydata=y, kwargs=kwargs)

        x, y, C = cbook.delete_masked_points(x, y, C)

        # Set the size of the hexagon grid
        if iterable(gridsize):
            nx, ny = gridsize
        else:
            nx = gridsize
            ny = int(nx / math.sqrt(3))
        # Count the number of data in each hexagon
        x = np.array(x, float)
        y = np.array(y, float)
        if xscale == 'log':
            if np.any(x <= 0.0):
                raise ValueError("x contains non-positive values, so can not"
                                 " be log-scaled")
            x = np.log10(x)
        if yscale == 'log':
            if np.any(y <= 0.0):
                raise ValueError("y contains non-positive values, so can not"
                                 " be log-scaled")
            y = np.log10(y)
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
        else:
            xmin = np.amin(x)
            xmax = np.amax(x)
            ymin = np.amin(y)
            ymax = np.amax(y)
        # In the x-direction, the hexagons exactly cover the region from
        # xmin to xmax. Need some padding to avoid roundoff errors.
        padding = 1.e-9 * (xmax - xmin)
        xmin -= padding
        xmax += padding
        sx = (xmax - xmin) / nx
        sy = (ymax - ymin) / ny

        if marginals:
            xorig = x.copy()
            yorig = y.copy()

        x = (x - xmin) / sx
        y = (y - ymin) / sy
        ix1 = np.round(x).astype(int)
        iy1 = np.round(y).astype(int)
        ix2 = np.floor(x).astype(int)
        iy2 = np.floor(y).astype(int)

        nx1 = nx + 1
        ny1 = ny + 1
        nx2 = nx
        ny2 = ny
        n = nx1 * ny1 + nx2 * ny2

        d1 = (x - ix1) ** 2 + 3.0 * (y - iy1) ** 2
        d2 = (x - ix2 - 0.5) ** 2 + 3.0 * (y - iy2 - 0.5) ** 2
        bdist = (d1 < d2)
        if C is None:
            accum = np.zeros(n)
            # Create appropriate views into "accum" array.
            lattice1 = accum[:nx1 * ny1]
            lattice2 = accum[nx1 * ny1:]
            lattice1.shape = (nx1, ny1)
            lattice2.shape = (nx2, ny2)

            for i in xrange(len(x)):
                if bdist[i]:
                    if ((ix1[i] >= 0) and (ix1[i] < nx1) and
                        (iy1[i] >= 0) and (iy1[i] < ny1)):
                        lattice1[ix1[i], iy1[i]] += 1
                else:
                    if ((ix2[i] >= 0) and (ix2[i] < nx2) and
                        (iy2[i] >= 0) and (iy2[i] < ny2)):
                        lattice2[ix2[i], iy2[i]] += 1

            # threshold
            if mincnt is not None:
                for i in xrange(nx1):
                    for j in xrange(ny1):
                        if lattice1[i, j] < mincnt:
                            lattice1[i, j] = np.nan
                for i in xrange(nx2):
                    for j in xrange(ny2):
                        if lattice2[i, j] < mincnt:
                            lattice2[i, j] = np.nan
            accum = np.hstack((lattice1.astype(float).ravel(),
                               lattice2.astype(float).ravel()))
            good_idxs = ~np.isnan(accum)

        else:
            if mincnt is None:
                mincnt = 0

            # create accumulation arrays
            lattice1 = np.empty((nx1, ny1), dtype=object)
            for i in xrange(nx1):
                for j in xrange(ny1):
                    lattice1[i, j] = []
            lattice2 = np.empty((nx2, ny2), dtype=object)
            for i in xrange(nx2):
                for j in xrange(ny2):
                    lattice2[i, j] = []

            for i in xrange(len(x)):
                if bdist[i]:
                    if ((ix1[i] >= 0) and (ix1[i] < nx1) and
                        (iy1[i] >= 0) and (iy1[i] < ny1)):
                        lattice1[ix1[i], iy1[i]].append(C[i])
                else:
                    if ((ix2[i] >= 0) and (ix2[i] < nx2) and
                        (iy2[i] >= 0) and (iy2[i] < ny2)):
                        lattice2[ix2[i], iy2[i]].append(C[i])

            for i in xrange(nx1):
                for j in xrange(ny1):
                    vals = lattice1[i, j]
                    if len(vals) > mincnt:
                        lattice1[i, j] = reduce_C_function(vals)
                    else:
                        lattice1[i, j] = np.nan
            for i in xrange(nx2):
                for j in xrange(ny2):
                    vals = lattice2[i, j]
                    if len(vals) > mincnt:
                        lattice2[i, j] = reduce_C_function(vals)
                    else:
                        lattice2[i, j] = np.nan

            accum = np.hstack((lattice1.astype(float).ravel(),
                               lattice2.astype(float).ravel()))
            good_idxs = ~np.isnan(accum)

        offsets = np.zeros((n, 2), float)
        offsets[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
        offsets[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
        offsets[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
        offsets[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
        offsets[:, 0] *= sx
        offsets[:, 1] *= sy
        offsets[:, 0] += xmin
        offsets[:, 1] += ymin
        # remove accumulation bins with no data
        offsets = offsets[good_idxs, :]
        accum = accum[good_idxs]

        polygon = np.zeros((6, 2), float)
        polygon[:, 0] = sx * np.array([0.5, 0.5, 0.0, -0.5, -0.5, 0.0])
        polygon[:, 1] = sy * np.array([-0.5, 0.5, 1.0,  0.5, -0.5, -1.0]) / 3.0

        if edgecolors == 'none':
            edgecolors = 'face'

        if xscale == 'log' or yscale == 'log':
            polygons = np.expand_dims(polygon, 0) + np.expand_dims(offsets, 1)
            if xscale == 'log':
                polygons[:, :, 0] = 10.0 ** polygons[:, :, 0]
                xmin = 10.0 ** xmin
                xmax = 10.0 ** xmax
                self.set_xscale(xscale)
            if yscale == 'log':
                polygons[:, :, 1] = 10.0 ** polygons[:, :, 1]
                ymin = 10.0 ** ymin
                ymax = 10.0 ** ymax
                self.set_yscale(yscale)
            collection = mcoll.PolyCollection(
                polygons,
                edgecolors=edgecolors,
                linewidths=linewidths,
                )
        else:
            collection = mcoll.PolyCollection(
                [polygon],
                edgecolors=edgecolors,
                linewidths=linewidths,
                offsets=offsets,
                transOffset=mtransforms.IdentityTransform(),
                offset_position="data"
                )

        if isinstance(norm, mcolors.LogNorm):
            if (accum == 0).any():
                # make sure we have not zeros
                accum += 1

        # autoscale the norm with curren accum values if it hasn't
        # been set
        if norm is not None:
            if norm.vmin is None and norm.vmax is None:
                norm.autoscale(accum)

        # Transform accum if needed
        if bins == 'log':
            accum = np.log10(accum + 1)
        elif bins != None:
            if not iterable(bins):
                minimum, maximum = min(accum), max(accum)
                bins -= 1  # one less edge than bins
                bins = minimum + (maximum - minimum) * np.arange(bins) / bins
            bins = np.sort(bins)
            accum = bins.searchsorted(accum)

        if norm is not None:
            assert(isinstance(norm, mcolors.Normalize))
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
        self.update_datalim(corners)
        self.autoscale_view(tight=True)

        # add the collection last
        self.add_collection(collection)
        if not marginals:
            return collection

        if C is None:
            C = np.ones(len(x))

        def coarse_bin(x, y, coarse):
            ind = coarse.searchsorted(x).clip(0, len(coarse) - 1)
            mus = np.zeros(len(coarse))
            for i in range(len(coarse)):
                mu = reduce_C_function(y[ind == i])
                mus[i] = mu
            return mus

        coarse = np.linspace(xmin, xmax, gridsize)

        xcoarse = coarse_bin(xorig, C, coarse)
        valid = ~np.isnan(xcoarse)
        verts, values = [], []
        for i, val in enumerate(xcoarse):
            thismin = coarse[i]
            if i < len(coarse) - 1:
                thismax = coarse[i + 1]
            else:
                thismax = thismin + np.diff(coarse)[-1]

            if not valid[i]:
                continue

            verts.append([(thismin, 0),
                          (thismin, 0.05),
                          (thismax, 0.05),
                          (thismax, 0)])
            values.append(val)

        values = np.array(values)
        trans = mtransforms.blended_transform_factory(
            self.transData, self.transAxes)

        hbar = mcoll.PolyCollection(verts, transform=trans, edgecolors='face')

        hbar.set_array(values)
        hbar.set_cmap(cmap)
        hbar.set_norm(norm)
        hbar.set_alpha(alpha)
        hbar.update(kwargs)
        self.add_collection(hbar)

        coarse = np.linspace(ymin, ymax, gridsize)
        ycoarse = coarse_bin(yorig, C, coarse)
        valid = ~np.isnan(ycoarse)
        verts, values = [], []
        for i, val in enumerate(ycoarse):
            thismin = coarse[i]
            if i < len(coarse) - 1:
                thismax = coarse[i + 1]
            else:
                thismax = thismin + np.diff(coarse)[-1]
            if not valid[i]:
                continue
            verts.append([(0, thismin), (0.0, thismax),
                          (0.05, thismax), (0.05, thismin)])
            values.append(val)

        values = np.array(values)

        trans = mtransforms.blended_transform_factory(
            self.transAxes, self.transData)

        vbar = mcoll.PolyCollection(verts, transform=trans, edgecolors='face')
        vbar.set_array(values)
        vbar.set_cmap(cmap)
        vbar.set_norm(norm)
        vbar.set_alpha(alpha)
        vbar.update(kwargs)
        self.add_collection(vbar)

        collection.hbar = hbar
        collection.vbar = vbar

        def on_changed(collection):
            hbar.set_cmap(collection.get_cmap())
            hbar.set_clim(collection.get_clim())
            vbar.set_cmap(collection.get_cmap())
            vbar.set_clim(collection.get_clim())

        collection.callbacksSM.connect('changed', on_changed)

        return collection

    @docstring.dedent_interpd
    def arrow(self, x, y, dx, dy, **kwargs):
        """
        Add an arrow to the axes.

        Call signature::

           arrow(x, y, dx, dy, **kwargs)

        Draws arrow on specified axis from (*x*, *y*) to (*x* + *dx*,
        *y* + *dy*). Uses FancyArrow patch to construct the arrow.

        The resulting arrow is affected by the axes aspect ratio and limits.
        This may produce an arrow whose head is not square with its stem. To
        create an arrow whose head is square with its stem, use
        :meth:`annotate`.

        Optional kwargs control the arrow construction and properties:

        %(FancyArrow)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/arrow_demo.py
        """
        # Strip away units for the underlying patch since units
        # do not make sense to most patch-like code
        x = self.convert_xunits(x)
        y = self.convert_yunits(y)
        dx = self.convert_xunits(dx)
        dy = self.convert_yunits(dy)

        a = mpatches.FancyArrow(x, y, dx, dy, **kwargs)
        self.add_artist(a)
        return a

    def quiverkey(self, *args, **kw):
        qk = mquiver.QuiverKey(*args, **kw)
        self.add_artist(qk)
        return qk
    quiverkey.__doc__ = mquiver.QuiverKey.quiverkey_doc

    def quiver(self, *args, **kw):
        if not self._hold:
            self.cla()
        q = mquiver.Quiver(self, *args, **kw)
        self.add_collection(q, False)
        self.update_datalim(q.XY)
        self.autoscale_view()
        return q
    quiver.__doc__ = mquiver.Quiver.quiver_doc

    def stackplot(self, x, *args, **kwargs):
        return mstack.stackplot(self, x, *args, **kwargs)
    stackplot.__doc__ = mstack.stackplot.__doc__

    def streamplot(self, x, y, u, v, density=1, linewidth=None, color=None,
                   cmap=None, norm=None, arrowsize=1, arrowstyle='-|>',
                   minlength=0.1, transform=None):
        if not self._hold:
            self.cla()
        stream_container = mstream.streamplot(self, x, y, u, v,
                                              density=density,
                                              linewidth=linewidth,
                                              color=color,
                                              cmap=cmap,
                                              norm=norm,
                                              arrowsize=arrowsize,
                                              arrowstyle=arrowstyle,
                                              minlength=minlength,
                                              transform=transform)
        return stream_container
    streamplot.__doc__ = mstream.streamplot.__doc__

    @docstring.dedent_interpd
    def barbs(self, *args, **kw):
        """
        %(barbs_doc)s

        **Example:**

        .. plot:: mpl_examples/pylab_examples/barb_demo.py
        """
        if not self._hold:
            self.cla()
        b = mquiver.Barbs(self, *args, **kw)
        self.add_collection(b)
        self.update_datalim(b.get_offsets())
        self.autoscale_view()
        return b

    @docstring.dedent_interpd
    def fill(self, *args, **kwargs):
        """
        Plot filled polygons.

        Call signature::

          fill(*args, **kwargs)

        *args* is a variable length argument, allowing for multiple
        *x*, *y* pairs with an optional color format string; see
        :func:`~matplotlib.pyplot.plot` for details on the argument
        parsing.  For example, to plot a polygon with vertices at *x*,
        *y* in blue.::

          ax.fill(x,y, 'b' )

        An arbitrary number of *x*, *y*, *color* groups can be specified::

          ax.fill(x1, y1, 'g', x2, y2, 'r')

        Return value is a list of :class:`~matplotlib.patches.Patch`
        instances that were added.

        The same color strings that :func:`~matplotlib.pyplot.plot`
        supports are supported by the fill format string.

        If you would like to fill below a curve, e.g., shade a region
        between 0 and *y* along *x*, use :meth:`fill_between`

        The *closed* kwarg will close the polygon when *True* (default).

        kwargs control the :class:`~matplotlib.patches.Polygon` properties:

        %(Polygon)s

        **Example:**

        .. plot:: mpl_examples/lines_bars_and_markers/fill_demo.py

        """
        if not self._hold:
            self.cla()

        patches = []
        for poly in self._get_patches_for_fill(*args, **kwargs):
            self.add_patch(poly)
            patches.append(poly)
        self.autoscale_view()
        return patches

    @docstring.dedent_interpd
    def fill_between(self, x, y1, y2=0, where=None, interpolate=False,
                     **kwargs):
        """
        Make filled polygons between two curves.

        Call signature::

          fill_between(x, y1, y2=0, where=None, **kwargs)

        Create a :class:`~matplotlib.collections.PolyCollection`
        filling the regions between *y1* and *y2* where
        ``where==True``

          *x* :
            An N-length array of the x data

          *y1* :
            An N-length array (or scalar) of the y data

          *y2* :
            An N-length array (or scalar) of the y data

          *where* :
            If *None*, default to fill between everywhere.  If not *None*,
            it is an N-length numpy boolean array and the fill will
            only happen over the regions where ``where==True``.

          *interpolate* :
            If *True*, interpolate between the two lines to find the
            precise point of intersection.  Otherwise, the start and
            end points of the filled region will only occur on explicit
            values in the *x* array.

          *kwargs* :
            Keyword args passed on to the
            :class:`~matplotlib.collections.PolyCollection`.

        kwargs control the :class:`~matplotlib.patches.Polygon` properties:

        %(PolyCollection)s

        .. plot:: mpl_examples/pylab_examples/fill_between_demo.py

        .. seealso::

            :meth:`fill_betweenx`
                for filling between two sets of x-values

        """
        # Handle united data, such as dates
        self._process_unit_info(xdata=x, ydata=y1, kwargs=kwargs)
        self._process_unit_info(ydata=y2)

        # Convert the arrays so we can work with them
        x = ma.masked_invalid(self.convert_xunits(x))
        y1 = ma.masked_invalid(self.convert_yunits(y1))
        y2 = ma.masked_invalid(self.convert_yunits(y2))

        if y1.ndim == 0:
            y1 = np.ones_like(x) * y1
        if y2.ndim == 0:
            y2 = np.ones_like(x) * y2

        if where is None:
            where = np.ones(len(x), np.bool)
        else:
            where = np.asarray(where, np.bool)

        if not (x.shape == y1.shape == y2.shape == where.shape):
            raise ValueError("Argument dimensions are incompatible")

        mask = reduce(ma.mask_or, [ma.getmask(a) for a in (x, y1, y2)])
        if mask is not ma.nomask:
            where &= ~mask

        polys = []
        for ind0, ind1 in mlab.contiguous_regions(where):
            xslice = x[ind0:ind1]
            y1slice = y1[ind0:ind1]
            y2slice = y2[ind0:ind1]

            if not len(xslice):
                continue

            N = len(xslice)
            X = np.zeros((2 * N + 2, 2), np.float)

            if interpolate:
                def get_interp_point(ind):
                    im1 = max(ind - 1, 0)
                    x_values = x[im1:ind + 1]
                    diff_values = y1[im1:ind + 1] - y2[im1:ind + 1]
                    y1_values = y1[im1:ind + 1]

                    if len(diff_values) == 2:
                        if np.ma.is_masked(diff_values[1]):
                            return x[im1], y1[im1]
                        elif np.ma.is_masked(diff_values[0]):
                            return x[ind], y1[ind]

                    diff_order = diff_values.argsort()
                    diff_root_x = np.interp(
                        0, diff_values[diff_order], x_values[diff_order])
                    diff_root_y = np.interp(diff_root_x, x_values, y1_values)
                    return diff_root_x, diff_root_y

                start = get_interp_point(ind0)
                end = get_interp_point(ind1)
            else:
                # the purpose of the next two lines is for when y2 is a
                # scalar like 0 and we want the fill to go all the way
                # down to 0 even if none of the y1 sample points do
                start = xslice[0], y2slice[0]
                end = xslice[-1], y2slice[-1]

            X[0] = start
            X[N + 1] = end

            X[1:N + 1, 0] = xslice
            X[1:N + 1, 1] = y1slice
            X[N + 2:, 0] = xslice[::-1]
            X[N + 2:, 1] = y2slice[::-1]

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

    @docstring.dedent_interpd
    def fill_betweenx(self, y, x1, x2=0, where=None, **kwargs):
        """
        Make filled polygons between two horizontal curves.

        Call signature::

          fill_betweenx(y, x1, x2=0, where=None, **kwargs)

        Create a :class:`~matplotlib.collections.PolyCollection`
        filling the regions between *x1* and *x2* where
        ``where==True``

          *y* :
            An N-length array of the y data

          *x1* :
            An N-length array (or scalar) of the x data

          *x2* :
            An N-length array (or scalar) of the x data

          *where* :
             If *None*, default to fill between everywhere.  If not *None*,
             it is a N length numpy boolean array and the fill will
             only happen over the regions where ``where==True``

          *kwargs* :
            keyword args passed on to the
            :class:`~matplotlib.collections.PolyCollection`

        kwargs control the :class:`~matplotlib.patches.Polygon` properties:

        %(PolyCollection)s

        .. plot:: mpl_examples/pylab_examples/fill_betweenx_demo.py

        .. seealso::

            :meth:`fill_between`
                for filling between two sets of y-values

        """
        # Handle united data, such as dates
        self._process_unit_info(ydata=y, xdata=x1, kwargs=kwargs)
        self._process_unit_info(xdata=x2)

        # Convert the arrays so we can work with them
        y = ma.masked_invalid(self.convert_yunits(y))
        x1 = ma.masked_invalid(self.convert_xunits(x1))
        x2 = ma.masked_invalid(self.convert_xunits(x2))

        if x1.ndim == 0:
            x1 = np.ones_like(y) * x1
        if x2.ndim == 0:
            x2 = np.ones_like(y) * x2

        if where is None:
            where = np.ones(len(y), np.bool)
        else:
            where = np.asarray(where, np.bool)

        if not (y.shape == x1.shape == x2.shape == where.shape):
            raise ValueError("Argument dimensions are incompatible")

        mask = reduce(ma.mask_or, [ma.getmask(a) for a in (y, x1, x2)])
        if mask is not ma.nomask:
            where &= ~mask

        polys = []
        for ind0, ind1 in mlab.contiguous_regions(where):
            yslice = y[ind0:ind1]
            x1slice = x1[ind0:ind1]
            x2slice = x2[ind0:ind1]

            if not len(yslice):
                continue

            N = len(yslice)
            Y = np.zeros((2 * N + 2, 2), np.float)

            # the purpose of the next two lines is for when x2 is a
            # scalar like 0 and we want the fill to go all the way
            # down to 0 even if none of the x1 sample points do
            Y[0] = x2slice[0], yslice[0]
            Y[N + 1] = x2slice[-1], yslice[-1]

            Y[1:N + 1, 0] = x1slice
            Y[1:N + 1, 1] = yslice
            Y[N + 2:, 0] = x2slice[::-1]
            Y[N + 2:, 1] = yslice[::-1]

            polys.append(Y)

        collection = mcoll.PolyCollection(polys, **kwargs)

        # now update the datalim and autoscale
        X1Y = np.array([x1[where], y[where]]).T
        X2Y = np.array([x2[where], y[where]]).T
        self.dataLim.update_from_data_xy(X1Y, self.ignore_existing_data_limits,
                                         updatex=True, updatey=True)

        self.dataLim.update_from_data_xy(X2Y, self.ignore_existing_data_limits,
                                         updatex=False, updatey=True)
        self.add_collection(collection)
        self.autoscale_view()
        return collection

    #### plotting z(x,y): imshow, pcolor and relatives, contour

    @docstring.dedent_interpd
    def imshow(self, X, cmap=None, norm=None, aspect=None,
               interpolation=None, alpha=None, vmin=None, vmax=None,
               origin=None, extent=None, shape=None, filternorm=1,
               filterrad=4.0, imlim=None, resample=None, url=None, **kwargs):
        """
        Display an image on the axes.

        Parameters
        -----------
        X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4)
            Display the image in `X` to current axes.  `X` may be a float
            array, a uint8 array or a PIL image. If `X` is an array, it
            can have the following shapes:

            - MxN -- luminance (grayscale, float array only)
            - MxNx3 -- RGB (float or uint8 array)
            - MxNx4 -- RGBA (float or uint8 array)

            The value for each component of MxNx3 and MxNx4 float arrays
            should be in the range 0.0 to 1.0; MxN float arrays may be
            normalised.

        cmap : `~matplotlib.colors.Colormap`, optional, default: None
            If None, default to rc `image.cmap` value. `cmap` is ignored when
            `X` has RGB(A) information

        aspect : ['auto' | 'equal' | scalar], optional, default: None
            If 'auto', changes the image aspect ratio to match that of the
            axes.

            If 'equal', and `extent` is None, changes the axes aspect ratio to
            match that of the image. If `extent` is not `None`, the axes
            aspect ratio is changed to match that of the extent.

            If None, default to rc ``image.aspect`` value.

        interpolation : string, optional, default: None
            Acceptable values are 'none', 'nearest', 'bilinear', 'bicubic',
            'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
            'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
            'lanczos'

            If `interpolation` is None, default to rc `image.interpolation`.
            See also the `filternorm` and `filterrad` parameters.
            If `interpolation` is 'none', then no interpolation is performed
            on the Agg, ps and pdf backends. Other backends will fall back to
            'nearest'.

        norm : `~matplotlib.colors.Normalize`, optional, default: None
            A `~matplotlib.colors.Normalize` instance is used to scale
            luminance data to 0, 1. If `None`, use the default
            func:`normalize`. `norm` is only used if `X` is an array of
            floats.

        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with norm to normalize
            luminance data.  Note if you pass a `norm` instance, your
            settings for `vmin` and `vmax` will be ignored.

        alpha : scalar, optional, default: None
            The alpha blending value, between 0 (transparent) and 1 (opaque)

        origin : ['upper' | 'lower'], optional, default: None
            Place the [0,0] index of the array in the upper left or lower left
            corner of the axes. If None, default to rc `image.origin`.

        extent : scalars (left, right, bottom, top), optional, default: None
            Data limits for the axes.  The default assigns zero-based row,
            column indices to the `x`, `y` centers of the pixels.

        shape : scalars (columns, rows), optional, default: None
            For raw buffer images

        filternorm : scalar, optional, default: 1
            A parameter for the antigrain image resize filter.  From the
            antigrain documentation, if `filternorm` = 1, the filter
            normalizes integer values and corrects the rounding errors. It
            doesn't do anything with the source floating point values, it
            corrects only integers according to the rule of 1.0 which means
            that any sum of pixel weights must be equal to 1.0.  So, the
            filter function must produce a graph of the proper shape.

        filterrad : scalar, optional, default: 4.0
            The filter radius for filters that have a radius parameter, i.e.
            when interpolation is one of: 'sinc', 'lanczos' or 'blackman'

        Returns
        --------
        image : `~matplotlib.image.AxesImage`

        Other parameters
        ----------------
        kwargs : `~matplotlib.artist.Artist` properties.

        See also
        --------
        matshow : Plot a matrix or an array as an image.

        Examples
        --------

        .. plot:: mpl_examples/pylab_examples/image_demo.py

        """

        if not self._hold:
            self.cla()

        if norm is not None:
            assert(isinstance(norm, mcolors.Normalize))
        if aspect is None:
            aspect = rcParams['image.aspect']
        self.set_aspect(aspect)
        im = mimage.AxesImage(self, cmap, norm, interpolation, origin, extent,
                       filternorm=filternorm,
                       filterrad=filterrad, resample=resample, **kwargs)

        im.set_data(X)
        im.set_alpha(alpha)
        self._set_artist_props(im)
        if im.get_clip_path() is None:
            # image does not already have clipping set, clip to axes patch
            im.set_clip_path(self.patch)
        #if norm is None and shape is None:
        #    im.set_clim(vmin, vmax)
        if vmin is not None or vmax is not None:
            im.set_clim(vmin, vmax)
        else:
            im.autoscale_None()
        im.set_url(url)

        # update ax.dataLim, and, if autoscaling, set viewLim
        # to tightly fit the image, regardless of dataLim.
        im.set_extent(im.get_extent())

        self.images.append(im)
        im._remove_method = lambda h: self.images.remove(h)

        return im

    @staticmethod
    def _pcolorargs(funcname, *args, **kw):
        # This takes one kwarg, allmatch.
        # If allmatch is True, then the incoming X, Y, C must
        # have matching dimensions, taking into account that
        # X and Y can be 1-D rather than 2-D.  This perfect
        # match is required for Gouroud shading.  For flat
        # shading, X and Y specify boundaries, so we need
        # one more boundary than color in each direction.
        # For convenience, and consistent with Matlab, we
        # discard the last row and/or column of C if necessary
        # to meet this condition.  This is done if allmatch
        # is False.

        allmatch = kw.pop("allmatch", False)

        if len(args) == 1:
            C = args[0]
            numRows, numCols = C.shape
            if allmatch:
                X, Y = np.meshgrid(np.arange(numCols), np.arange(numRows))
            else:
                X, Y = np.meshgrid(np.arange(numCols + 1),
                                   np.arange(numRows + 1))
            return X, Y, C

        if len(args) == 3:
            X, Y, C = args
            numRows, numCols = C.shape
        else:
            raise TypeError(
                'Illegal arguments to %s; see help(%s)' % (funcname, funcname))

        Nx = X.shape[-1]
        Ny = Y.shape[0]
        if len(X.shape) != 2 or X.shape[0] == 1:
            x = X.reshape(1, Nx)
            X = x.repeat(Ny, axis=0)
        if len(Y.shape) != 2 or Y.shape[1] == 1:
            y = Y.reshape(Ny, 1)
            Y = y.repeat(Nx, axis=1)
        if X.shape != Y.shape:
            raise TypeError(
                'Incompatible X, Y inputs to %s; see help(%s)' % (
                funcname, funcname))
        if allmatch:
            if not (Nx == numCols and Ny == numRows):
                raise TypeError('Dimensions of C %s are incompatible with'
                                ' X (%d) and/or Y (%d); see help(%s)' % (
                                    C.shape, Nx, Ny, funcname))
        else:
            if not (numCols in (Nx, Nx-1) and numRows in (Ny, Ny-1)):
                raise TypeError('Dimensions of C %s are incompatible with'
                                ' X (%d) and/or Y (%d); see help(%s)' % (
                                    C.shape, Nx, Ny, funcname))
            C = C[:Ny-1, :Nx-1]
        return X, Y, C

    @docstring.dedent_interpd
    def pcolor(self, *args, **kwargs):
        """
        Create a pseudocolor plot of a 2-D array.

        .. note::

            pcolor can be very slow for large arrays; consider
            using the similar but much faster
            :func:`~matplotlib.pyplot.pcolormesh` instead.

        Call signatures::

          pcolor(C, **kwargs)
          pcolor(X, Y, C, **kwargs)

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

          *cmap*: [ *None* | Colormap ]
            A :class:`matplotlib.colors.Colormap` instance. If *None*, use
            rc settings.

          *norm*: [ *None* | Normalize ]
            An :class:`matplotlib.colors.Normalize` instance is used
            to scale luminance data to 0,1. If *None*, defaults to
            :func:`normalize`.

          *vmin*/*vmax*: [ *None* | scalar ]
            *vmin* and *vmax* are used in conjunction with *norm* to
            normalize luminance data.  If either is *None*, it
            is autoscaled to the respective min or max
            of the color array *C*.  If not *None*, *vmin* or
            *vmax* passed in here override any pre-existing values
            supplied in the *norm* instance.

          *shading*: [ 'flat' | 'faceted' ]
            If 'faceted', a black grid is drawn around each rectangle; if
            'flat', edges are not drawn. Default is 'flat', contrary to
            MATLAB.

            This kwarg is deprecated; please use 'edgecolors' instead:
              * shading='flat' -- edgecolors='none'
              * shading='faceted  -- edgecolors='k'

          *edgecolors*: [ *None* | ``'none'`` | color | color sequence]
            If *None*, the rc setting is used by default.

            If ``'none'``, edges will not be visible.

            An mpl color or sequence of colors will set the edge color

          *alpha*: ``0 <= scalar <= 1``   or *None*
            the alpha blending value

        Return value is a :class:`matplotlib.collections.Collection`
        instance.

        .. _axes-pcolor-grid-orientation:

        The grid orientation follows the MATLAB convention: an
        array *C* with shape (*nrows*, *ncolumns*) is plotted with
        the column number as *X* and the row number as *Y*, increasing
        up; hence it is plotted the way the array would be printed,
        except that the *Y* axis is reversed.  That is, *C* is taken
        as *C*(*y*, *x*).

        Similarly for :func:`meshgrid`::

          x = np.arange(5)
          y = np.arange(3)
          X, Y = np.meshgrid(x, y)

        is equivalent to::

          X = array([[0, 1, 2, 3, 4],
                     [0, 1, 2, 3, 4],
                     [0, 1, 2, 3, 4]])

          Y = array([[0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2]])

        so if you have::

          C = rand(len(x), len(y))

        then you need to transpose C::

          pcolor(X, Y, C.T)

        or::

          pcolor(C.T)

        MATLAB :func:`pcolor` always discards the last row and column
        of *C*, but matplotlib displays the last row and column if *X* and
        *Y* are not specified, or if *X* and *Y* have one more row and
        column than *C*.

        kwargs can be used to control the
        :class:`~matplotlib.collections.PolyCollection` properties:

        %(PolyCollection)s

        .. note::

            The default *antialiaseds* is False if the default
            *edgecolors*="none" is used.  This eliminates artificial lines
            at patch boundaries, and works regardless of the value of
            alpha.  If *edgecolors* is not "none", then the default
            *antialiaseds* is taken from
            rcParams['patch.antialiased'], which defaults to *True*.
            Stroking the edges may be preferred if *alpha* is 1, but
            will cause artifacts otherwise.

        .. seealso::

            :func:`~matplotlib.pyplot.pcolormesh`
                For an explanation of the differences between
                pcolor and pcolormesh.
        """

        if not self._hold:
            self.cla()

        alpha = kwargs.pop('alpha', None)
        norm = kwargs.pop('norm', None)
        cmap = kwargs.pop('cmap', None)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        if 'shading' in kwargs:
            cbook.warn_deprecated(
                '1.2', name='shading', alternative='edgecolors',
                obj_type='option')
        shading = kwargs.pop('shading', 'flat')

        X, Y, C = self._pcolorargs('pcolor', *args, allmatch=False)
        Ny, Nx = X.shape

        # convert to MA, if necessary.
        C = ma.asarray(C)
        X = ma.asarray(X)
        Y = ma.asarray(Y)
        mask = ma.getmaskarray(X) + ma.getmaskarray(Y)
        xymask = (mask[0:-1, 0:-1] + mask[1:, 1:] +
                  mask[0:-1, 1:] + mask[1:, 0:-1])
        # don't plot if C or any of the surrounding vertices are masked.
        mask = ma.getmaskarray(C) + xymask

        newaxis = np.newaxis
        compress = np.compress

        ravelmask = (mask == 0).ravel()
        X1 = compress(ravelmask, ma.filled(X[0:-1, 0:-1]).ravel())
        Y1 = compress(ravelmask, ma.filled(Y[0:-1, 0:-1]).ravel())
        X2 = compress(ravelmask, ma.filled(X[1:, 0:-1]).ravel())
        Y2 = compress(ravelmask, ma.filled(Y[1:, 0:-1]).ravel())
        X3 = compress(ravelmask, ma.filled(X[1:, 1:]).ravel())
        Y3 = compress(ravelmask, ma.filled(Y[1:, 1:]).ravel())
        X4 = compress(ravelmask, ma.filled(X[0:-1, 1:]).ravel())
        Y4 = compress(ravelmask, ma.filled(Y[0:-1, 1:]).ravel())
        npoly = len(X1)

        xy = np.concatenate((X1[:, newaxis], Y1[:, newaxis],
                             X2[:, newaxis], Y2[:, newaxis],
                             X3[:, newaxis], Y3[:, newaxis],
                             X4[:, newaxis], Y4[:, newaxis],
                             X1[:, newaxis], Y1[:, newaxis]),
                             axis=1)
        verts = xy.reshape((npoly, 5, 2))

        C = compress(ravelmask, ma.filled(C[0:Ny - 1, 0:Nx - 1]).ravel())

        linewidths = (0.25,)
        if 'linewidth' in kwargs:
            kwargs['linewidths'] = kwargs.pop('linewidth')
        kwargs.setdefault('linewidths', linewidths)

        if shading == 'faceted':
            edgecolors = 'k',
        else:
            edgecolors = 'none'

        if 'edgecolor' in kwargs:
            kwargs['edgecolors'] = kwargs.pop('edgecolor')
        ec = kwargs.setdefault('edgecolors', edgecolors)

        # aa setting will default via collections to patch.antialiased
        # unless the boundary is not stroked, in which case the
        # default will be False; with unstroked boundaries, aa
        # makes artifacts that are often disturbing.
        if 'antialiased' in kwargs:
            kwargs['antialiaseds'] = kwargs.pop('antialiased')
        if 'antialiaseds' not in kwargs and (is_string_like(ec) and
                ec.lower() == "none"):
            kwargs['antialiaseds'] = False

        collection = mcoll.PolyCollection(verts, **kwargs)

        collection.set_alpha(alpha)
        collection.set_array(C)
        if norm is not None:
            assert(isinstance(norm, mcolors.Normalize))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection.set_clim(vmin, vmax)
        collection.autoscale_None()
        self.grid(False)

        x = X.compressed()
        y = Y.compressed()

        # Transform from native to data coordinates?
        t = collection._transform
        if (not isinstance(t, mtransforms.Transform)
            and hasattr(t, '_as_mpl_transform')):
            t = t._as_mpl_transform(self.axes)

        if t and any(t.contains_branch_seperately(self.transData)):
            trans_to_data = t - self.transData
            pts = np.vstack([x, y]).T.astype(np.float)
            transformed_pts = trans_to_data.transform(pts)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]

        minx = np.amin(x)
        maxx = np.amax(x)
        miny = np.amin(y)
        maxy = np.amax(y)

        corners = (minx, miny), (maxx, maxy)
        self.update_datalim(corners)
        self.autoscale_view()
        self.add_collection(collection)
        return collection

    @docstring.dedent_interpd
    def pcolormesh(self, *args, **kwargs):
        """
        Plot a quadrilateral mesh.

        Call signatures::

          pcolormesh(C)
          pcolormesh(X, Y, C)
          pcolormesh(C, **kwargs)

        Create a pseudocolor plot of a 2-D array.

        pcolormesh is similar to :func:`~matplotlib.pyplot.pcolor`,
        but uses a different mechanism and returns a different
        object; pcolor returns a
        :class:`~matplotlib.collections.PolyCollection` but pcolormesh
        returns a
        :class:`~matplotlib.collections.QuadMesh`.  It is much faster,
        so it is almost always preferred for large arrays.

        *C* may be a masked array, but *X* and *Y* may not.  Masked
        array support is implemented via *cmap* and *norm*; in
        contrast, :func:`~matplotlib.pyplot.pcolor` simply does not
        draw quadrilaterals with masked colors or vertices.

        Keyword arguments:

          *cmap*: [ *None* | Colormap ]
            A :class:`matplotlib.colors.Colormap` instance. If *None*, use
            rc settings.

          *norm*: [ *None* | Normalize ]
            A :class:`matplotlib.colors.Normalize` instance is used to
            scale luminance data to 0,1. If *None*, defaults to
            :func:`normalize`.

          *vmin*/*vmax*: [ *None* | scalar ]
            *vmin* and *vmax* are used in conjunction with *norm* to
            normalize luminance data.  If either is *None*, it
            is autoscaled to the respective min or max
            of the color array *C*.  If not *None*, *vmin* or
            *vmax* passed in here override any pre-existing values
            supplied in the *norm* instance.

          *shading*: [ 'flat' | 'gouraud' ]
            'flat' indicates a solid color for each quad.  When
            'gouraud', each quad will be Gouraud shaded.  When gouraud
            shading, edgecolors is ignored.

          *edgecolors*: [*None* | ``'None'`` | ``'face'`` | color |
                         color sequence]
            If *None*, the rc setting is used by default.

            If ``'None'``, edges will not be visible.

            If ``'face'``, edges will have the same color as the faces.

            An mpl color or sequence of colors will set the edge color

          *alpha*: ``0 <= scalar <= 1``  or *None*
            the alpha blending value

        Return value is a :class:`matplotlib.collections.QuadMesh`
        object.

        kwargs can be used to control the
        :class:`matplotlib.collections.QuadMesh` properties:

        %(QuadMesh)s

        .. seealso::

            :func:`~matplotlib.pyplot.pcolor`
                For an explanation of the grid orientation and the
                expansion of 1-D *X* and/or *Y* to 2-D arrays.
        """
        if not self._hold:
            self.cla()

        alpha = kwargs.pop('alpha', None)
        norm = kwargs.pop('norm', None)
        cmap = kwargs.pop('cmap', None)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        shading = kwargs.pop('shading', 'flat').lower()
        antialiased = kwargs.pop('antialiased', False)
        kwargs.setdefault('edgecolors', 'None')

        allmatch = (shading == 'gouraud')

        X, Y, C = self._pcolorargs('pcolormesh', *args, allmatch=allmatch)
        Ny, Nx = X.shape

        # convert to one dimensional arrays
        C = C.ravel()
        X = X.ravel()
        Y = Y.ravel()

        coords = np.zeros(((Nx * Ny), 2), dtype=float)
        coords[:, 0] = X
        coords[:, 1] = Y

        collection = mcoll.QuadMesh(
            Nx - 1, Ny - 1, coords,
            antialiased=antialiased, shading=shading, **kwargs)
        collection.set_alpha(alpha)
        collection.set_array(C)
        if norm is not None:
            assert(isinstance(norm, mcolors.Normalize))
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection.set_clim(vmin, vmax)
        collection.autoscale_None()

        self.grid(False)

        # Transform from native to data coordinates?
        t = collection._transform
        if (not isinstance(t, mtransforms.Transform)
            and hasattr(t, '_as_mpl_transform')):
            t = t._as_mpl_transform(self.axes)

        if t and any(t.contains_branch_seperately(self.transData)):
            trans_to_data = t - self.transData
            pts = np.vstack([X, Y]).T.astype(np.float)
            transformed_pts = trans_to_data.transform(pts)
            X = transformed_pts[..., 0]
            Y = transformed_pts[..., 1]

        minx = np.amin(X)
        maxx = np.amax(X)
        miny = np.amin(Y)
        maxy = np.amax(Y)

        corners = (minx, miny), (maxx, maxy)
        self.update_datalim(corners)
        self.autoscale_view()
        self.add_collection(collection)
        return collection

    @docstring.dedent_interpd
    def pcolorfast(self, *args, **kwargs):
        """
        pseudocolor plot of a 2-D array

        Experimental; this is a pcolor-type method that
        provides the fastest possible rendering with the Agg
        backend, and that can handle any quadrilateral grid.
        It supports only flat shading (no outlines), it lacks
        support for log scaling of the axes, and it does not
        have a pyplot wrapper.

        Call signatures::

          ax.pcolorfast(C, **kwargs)
          ax.pcolorfast(xr, yr, C, **kwargs)
          ax.pcolorfast(x, y, C, **kwargs)
          ax.pcolorfast(X, Y, C, **kwargs)

        C is the 2D array of color values corresponding to quadrilateral
        cells. Let (nr, nc) be its shape.  C may be a masked array.

        ``ax.pcolorfast(C, **kwargs)`` is equivalent to
        ``ax.pcolorfast([0,nc], [0,nr], C, **kwargs)``

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

          *cmap*: [ *None* | Colormap ]
            A :class:`matplotlib.colors.Colormap` instance from cm. If *None*,
            use rc settings.

          *norm*: [ *None* | Normalize ]
            A :class:`matplotlib.colors.Normalize` instance is used to scale
            luminance data to 0,1. If *None*, defaults to normalize()

          *vmin*/*vmax*: [ *None* | scalar ]
            *vmin* and *vmax* are used in conjunction with norm to normalize
            luminance data.  If either are *None*, the min and max
            of the color array *C* is used.  If you pass a norm instance,
            *vmin* and *vmax* will be *None*.

          *alpha*: ``0 <= scalar <= 1``  or *None*
            the alpha blending value

        Return value is an image if a regular or rectangular grid
        is specified, and a :class:`~matplotlib.collections.QuadMesh`
        collection in the general quadrilateral case.

        """

        if not self._hold:
            self.cla()

        alpha = kwargs.pop('alpha', None)
        norm = kwargs.pop('norm', None)
        cmap = kwargs.pop('cmap', None)
        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        if norm is not None:
            assert(isinstance(norm, mcolors.Normalize))

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
                    if (np.ptp(dx) < 0.01 * np.abs(dx.mean()) and
                        np.ptp(dy) < 0.01 * np.abs(dy.mean())):
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
            C = ma.ravel(C)  # data point in each cell is value
                             # at lower left corner
            X = x.ravel()
            Y = y.ravel()
            Nx = nc + 1
            Ny = nr + 1

            # The following needs to be cleaned up; the renderer
            # requires separate contiguous arrays for X and Y,
            # but the QuadMesh class requires the 2D array.
            coords = np.empty(((Nx * Ny), 2), np.float64)
            coords[:, 0] = X
            coords[:, 1] = Y

            # The QuadMesh class can also be changed to
            # handle relevant superclass kwargs; the initializer
            # should do much more than it does now.
            collection = mcoll.QuadMesh(nc, nr, coords, 0, edgecolors="None")
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
        if not self._hold:
            self.cla()
        kwargs['filled'] = False
        return mcontour.QuadContourSet(self, *args, **kwargs)
    contour.__doc__ = mcontour.QuadContourSet.contour_doc

    def contourf(self, *args, **kwargs):
        if not self._hold:
            self.cla()
        kwargs['filled'] = True
        return mcontour.QuadContourSet(self, *args, **kwargs)
    contourf.__doc__ = mcontour.QuadContourSet.contour_doc

    def clabel(self, CS, *args, **kwargs):
        return CS.clabel(*args, **kwargs)
    clabel.__doc__ = mcontour.ContourSet.clabel.__doc__

    @docstring.dedent_interpd
    def table(self, **kwargs):
        """
        Add a table to the current axes.

        Call signature::

          table(cellText=None, cellColours=None,
                cellLoc='right', colWidths=None,
                rowLabels=None, rowColours=None, rowLoc='left',
                colLabels=None, colColours=None, colLoc='center',
                loc='bottom', bbox=None):

        Returns a :class:`matplotlib.table.Table` instance.  For finer
        grained control over tables, use the
        :class:`~matplotlib.table.Table` class and add it to the axes
        with :meth:`~matplotlib.axes.Axes.add_table`.

        Thanks to John Gill for providing the class and table.

        kwargs control the :class:`~matplotlib.table.Table`
        properties:

        %(Table)s
        """
        return mtable.table(self, **kwargs)

    def _make_twin_axes(self, *kl, **kwargs):
        """
        make a twinx axes of self. This is used for twinx and twiny.
        """
        ax2 = self.figure.add_axes(self.get_position(True), *kl, **kwargs)
        return ax2

    def twinx(self):
        """
        Call signature::

          ax = twinx()

        create a twin of Axes for generating a plot with a sharex
        x-axis but independent y axis.  The y-axis of self will have
        ticks on left and the returned axes will have ticks on the
        right.

        .. note::
            For those who are 'picking' artists while using twinx, pick
            events are only called for the artists in the top-most axes.
        """

        ax2 = self._make_twin_axes(sharex=self, frameon=False)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_offset_position('right')
        self.yaxis.tick_left()
        ax2.xaxis.set_visible(False)
        return ax2

    def twiny(self):
        """
        Call signature::

          ax = twiny()

        create a twin of Axes for generating a plot with a shared
        y-axis but independent x axis.  The x-axis of self will have
        ticks on bottom and the returned axes will have ticks on the
        top.

        .. note::
            For those who are 'picking' artists while using twiny, pick
            events are only called for the artists in the top-most axes.
        """

        ax2 = self._make_twin_axes(sharey=self, frameon=False)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        self.xaxis.tick_bottom()
        ax2.yaxis.set_visible(False)
        return ax2

    def get_shared_x_axes(self):
        'Return a copy of the shared axes Grouper object for x axes'
        return self._shared_x_axes

    def get_shared_y_axes(self):
        'Return a copy of the shared axes Grouper object for y axes'
        return self._shared_y_axes

    #### Data analysis

    @docstring.dedent_interpd
    def hist(self, x, bins=10, range=None, normed=False, weights=None,
             cumulative=False, bottom=None, histtype='bar', align='mid',
             orientation='vertical', rwidth=None, log=False,
             color=None, label=None, stacked=False,
             **kwargs):
        """
        Plot a histogram.

        Compute and draw the histogram of *x*. The return value is a
        tuple (*n*, *bins*, *patches*) or ([*n0*, *n1*, ...], *bins*,
        [*patches0*, *patches1*,...]) if the input contains multiple
        data.

        Multiple data can be provided via *x* as a list of datasets
        of potentially different length ([*x0*, *x1*, ...]), or as
        a 2-D ndarray in which each column is a dataset.  Note that
        the ndarray form is transposed relative to the list form.

        Masked arrays are not supported at present.

        Parameters
        ----------
        x : array_like, shape (n, )
            Input values.

        bins : integer or array_like, optional, default: 10
            If an integer is given, `bins + 1` bin edges are returned,
            consistently with :func:`numpy.histogram` for numpy version >=
            1.3.

            Unequally spaced bins are supported if `bins` is a sequence.

        range : tuple, optional, default: None
            The lower and upper range of the bins. Lower and upper outliers
            are ignored. If not provided, `range` is (x.min(), x.max()). Range
            has no effect if `bins` is a sequence.

            If `bins` is a sequence or `range` is specified, autoscaling
            is based on the specified bin range instead of the
            range of x.

        normed : boolean, optional, default: False
            If `True`, the first element of the return tuple will
            be the counts normalized to form a probability density, i.e.,
            ``n/(len(x)`dbin)``, ie the integral of the histogram will sum to
            1. If *stacked* is also *True*, the sum of the histograms is
            normalized to 1.

        weights : array_like, shape (n, ), optional, default: None
            An array of weights, of the same shape as `x`.  Each value in `x`
            only contributes its associated weight towards the bin count
            (instead of 1).  If `normed` is True, the weights are normalized,
            so that the integral of the density over the range remains 1.

        cumulative : boolean, optional, default : True
            If `True`, then a histogram is computed where each bin gives the
            counts in that bin plus all bins for smaller values. The last bin
            gives the total number of datapoints.  If `normed` is also `True`
            then the histogram is normalized such that the last bin equals 1.
            If `cumulative` evaluates to less than 0 (e.g., -1), the direction
            of accumulation is reversed.  In this case, if `normed` is also
            `True`, then the histogram is normalized such that the first bin
            equals 1.

        histtype : ['bar' | 'barstacked' | 'step' | 'stepfilled'], optional
            The type of histogram to draw.

            - 'bar' is a traditional bar-type histogram.  If multiple data
              are given the bars are aranged side by side.

            - 'barstacked' is a bar-type histogram where multiple
              data are stacked on top of each other.

            - 'step' generates a lineplot that is by default
              unfilled.

            - 'stepfilled' generates a lineplot that is by default
              filled.

        align : ['left' | 'mid' | 'right'], optional, default: 'mid'
            Controls how the histogram is plotted.

                - 'left': bars are centered on the left bin edges.

                - 'mid': bars are centered between the bin edges.

                - 'right': bars are centered on the right bin edges.

        orientation : ['horizontal' | 'vertical'], optional
            If 'horizontal', `~matplotlib.pyplot.barh` will be used for
            bar-type histograms and the *bottom* kwarg will be the left edges.

        rwidth : scalar, optional, default: None
            The relative width of the bars as a fraction of the bin width.  If
            `None`, automatically compute the width. Ignored if `histtype` =
            'step' or 'stepfilled'.

        log : boolean, optional, default : False
            If `True`, the histogram axis will be set to a log scale. If `log`
            is `True` and `x` is a 1D array, empty bins will be filtered out
            and only the non-empty (`n`, `bins`, `patches`) will be returned.

        color : color or array_like of colors, optional, default: None
            Color spec or sequence of color specs, one per dataset.  Default
            (`None`) uses the standard line color sequence.

        label : string, optional, default: ''
            String, or sequence of strings to match multiple datasets.  Bar
            charts yield multiple patches per dataset, but only the first gets
            the label, so that the legend command will work as expected.

        stacked : boolean, optional, default : False
            If `True`, multiple data are stacked on top of each other If
            `False` multiple data are aranged side by side if histtype is
            'bar' or on top of each other if histtype is 'step'

        Returns
        -------
        tuple : (n, bins, patches) or ([n0, n1, ...], bins, [patches0, patches1,...])

        Other Parameters
        ----------------
        kwargs : `~matplotlib.patches.Patch` properties

        See also
        --------
        hist2d : 2D histograms

        Notes
        -----
        Until numpy release 1.5, the underlying numpy histogram function was
        incorrect with `normed`=`True` if bin sizes were unequal.  MPL
        inherited that error.  It is now corrected within MPL when using
        earlier numpy versions.

        Examples
        --------
        .. plot:: mpl_examples/statistics/histogram_demo_features.py

        """
        if not self._hold:
            self.cla()

        # xrange becomes range after 2to3
        bin_range = range
        range = __builtins__["range"]

        # NOTE: the range keyword overwrites the built-in func range !!!
        #       needs to be fixed in numpy                           !!!

        # Validate string inputs here so we don't have to clutter
        # subsequent code.
        if histtype not in ['bar', 'barstacked', 'step', 'stepfilled']:
            raise ValueError("histtype %s is not recognized" % histtype)

        if align not in ['left', 'mid', 'right']:
            raise ValueError("align kwarg %s is not recognized" % align)

        if orientation not in ['horizontal', 'vertical']:
            raise ValueError(
                "orientation kwarg %s is not recognized" % orientation)

        if histtype == 'barstacked' and not stacked:
            stacked = True

        # Massage 'x' for processing.
        # NOTE: Be sure any changes here is also done below to 'weights'
        if isinstance(x, np.ndarray) or not iterable(x[0]):
            # TODO: support masked arrays;
            x = np.asarray(x)
            if x.ndim == 2:
                x = x.T  # 2-D input with columns as datasets; switch to rows
            elif x.ndim == 1:
                x = x.reshape(1, x.shape[0])  # new view, single row
            else:
                raise ValueError("x must be 1D or 2D")
            if x.shape[1] < x.shape[0]:
                warnings.warn(
                    '2D hist input should be nsamples x nvariables;\n '
                    'this looks transposed (shape is %d x %d)' % x.shape[::-1])
        else:
            # multiple hist with data of different length
            x = [np.asarray(xi) for xi in x]

        nx = len(x)  # number of datasets

        if color is None:
            color = [self._get_lines.color_cycle.next()
                     for i in xrange(nx)]
        else:
            color = mcolors.colorConverter.to_rgba_array(color)
            if len(color) != nx:
                raise ValueError("color kwarg must have one color per dataset")

        # We need to do to 'weights' what was done to 'x'
        if weights is not None:
            if isinstance(weights, np.ndarray) or not iterable(weights[0]):
                w = np.array(weights)
                if w.ndim == 2:
                    w = w.T
                elif w.ndim == 1:
                    w.shape = (1, w.shape[0])
                else:
                    raise ValueError("weights must be 1D or 2D")
            else:
                w = [np.asarray(wi) for wi in weights]

            if len(w) != nx:
                raise ValueError('weights should have the same shape as x')
            for i in xrange(nx):
                if len(w[i]) != len(x[i]):
                    raise ValueError(
                        'weights should have the same shape as x')
        else:
            w = [None]*nx

        # Save the datalimits for the same reason:
        _saved_bounds = self.dataLim.bounds

        # Check whether bins or range are given explicitly. In that
        # case use those values for autoscaling.
        binsgiven = (cbook.iterable(bins) or bin_range is not None)

        # If bins are not specified either explicitly or via range,
        # we need to figure out the range required for all datasets,
        # and supply that to np.histogram.
        if not binsgiven:
            xmin = np.inf
            xmax = -np.inf
            for xi in x:
                xmin = min(xmin, xi.min())
                xmax = max(xmax, xi.max())
            bin_range = (xmin, xmax)

        #hist_kwargs = dict(range=range, normed=bool(normed))
        # We will handle the normed kwarg within mpl until we
        # get to the point of requiring numpy >= 1.5.
        hist_kwargs = dict(range=bin_range)

        n = []
        mlast = bottom
        for i in xrange(nx):
            # this will automatically overwrite bins,
            # so that each histogram uses the same bins
            m, bins = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
            m = m.astype(float) # causes problems later if it's an int
            if mlast is None:
                mlast = np.zeros(len(bins)-1, m.dtype)
            if normed and not stacked:
                db = np.diff(bins)
                m = (m.astype(float) / db) / m.sum()
            if stacked:
                m += mlast
                mlast[:] = m
            n.append(m)

        if stacked and normed:
            db = np.diff(bins)
            for m in n:
                m[:] = (m.astype(float) / db) / n[-1].sum()
        if cumulative:
            slc = slice(None)
            if cbook.is_numlike(cumulative) and cumulative < 0:
                slc = slice(None, None, -1)

            if normed:
                n = [(m * np.diff(bins))[slc].cumsum()[slc] for m in n]
            else:
                n = [m[slc].cumsum()[slc] for m in n]

        patches = []

        if histtype.startswith('bar'):
            # Save autoscale state for later restoration; turn autoscaling
            # off so we can do it all a single time at the end, instead
            # of having it done by bar or fill and then having to be redone.
            _saved_autoscalex = self.get_autoscalex_on()
            _saved_autoscaley = self.get_autoscaley_on()
            self.set_autoscalex_on(False)
            self.set_autoscaley_on(False)

            totwidth = np.diff(bins)

            if rwidth is not None:
                dr = min(1.0, max(0.0, rwidth))
            elif len(n) > 1:
                dr = 0.8
            else:
                dr = 1.0

            if histtype == 'bar' and not stacked:
                width = dr*totwidth/nx
                dw = width

                if nx > 1:
                    boffset = -0.5*dr*totwidth*(1.0-1.0/nx)
                else:
                    boffset = 0.0
                stacked = False
            elif histtype == 'barstacked' or stacked:
                width = dr*totwidth
                boffset, dw = 0.0, 0.0

            if align == 'mid' or align == 'edge':
                boffset += 0.5*totwidth
            elif align == 'right':
                boffset += totwidth

            if orientation == 'horizontal':
                _barfunc = self.barh
                bottom_kwarg = 'left'
            else:  # orientation == 'vertical'
                _barfunc = self.bar
                bottom_kwarg = 'bottom'

            for m, c in zip(n, color):
                if bottom is None:
                    bottom = np.zeros(len(m), np.float)
                if stacked:
                    height = m - bottom
                else:
                    height = m
                patch = _barfunc(bins[:-1]+boffset, height, width,
                                 align='center', log=log,
                                 color=c, **{bottom_kwarg: bottom})
                patches.append(patch)
                if stacked:
                    bottom[:] = m
                boffset += dw

            self.set_autoscalex_on(_saved_autoscalex)
            self.set_autoscaley_on(_saved_autoscaley)
            self.autoscale_view()

        elif histtype.startswith('step'):
            # these define the perimeter of the polygon
            x = np.zeros(4 * len(bins) - 3, np.float)
            y = np.zeros(4 * len(bins) - 3, np.float)

            x[0:2*len(bins)-1:2], x[1:2*len(bins)-1:2] = bins, bins[:-1]
            x[2*len(bins)-1:] = x[1:2*len(bins)-1][::-1]

            if log:
                if orientation == 'horizontal':
                    self.set_xscale('log', nonposx='clip')
                    logbase = self.xaxis._scale.base
                else:  # orientation == 'vertical'
                    self.set_yscale('log', nonposy='clip')
                    logbase = self.yaxis._scale.base

                # Setting a minimum of 0 results in problems for log plots
                if normed:
                    # For normed data, set to log base * minimum data value
                    # (gives 1 full tick-label unit for the lowest filled bin)
                    ndata = np.array(n)
                    minimum = (np.min(ndata[ndata > 0])) / logbase
                else:
                    # For non-normed data, set the min to log base,
                    # again so that there is 1 full tick-label unit
                    # for the lowest bin
                    minimum = 1.0 / logbase

                y[0], y[-1] = minimum, minimum
            else:
                minimum = np.min(bins)

            if align == 'left' or align == 'center':
                x -= 0.5*(bins[1]-bins[0])
            elif align == 'right':
                x += 0.5*(bins[1]-bins[0])

            # If fill kwarg is set, it will be passed to the patch collection,
            # overriding this
            fill = (histtype == 'stepfilled')

            xvals, yvals = [], []
            for m in n:
                # starting point for drawing polygon
                y[0] = y[1]
                # top of the previous polygon becomes the bottom
                y[2*len(bins)-1:] = y[1:2*len(bins)-1][::-1]
                # set the top of this polygon
                y[1:2*len(bins)-1:2], y[2:2*len(bins)-1:2] = m, m
                if log:
                    y[y < minimum] = minimum
                if orientation == 'horizontal':
                    x, y = y, x

                xvals.append(x.copy())
                yvals.append(y.copy())

            if fill:
                # add patches in reverse order so that when stacking,
                # items lower in the stack are plottted on top of
                # items higher in the stack
                for x, y, c in reversed(zip(xvals, yvals, color)):
                    patches.append(self.fill(
                        x, y,
                        closed=True,
                        facecolor=c))
            else:
                for x, y, c in reversed(zip(xvals, yvals, color)):
                    split = 2 * len(bins)
                    patches.append(self.fill(
                        x[:split], y[:split],
                        closed=False, edgecolor=c,
                        fill=False))

            # we return patches, so put it back in the expected order
            patches.reverse()

            # adopted from adjust_x/ylim part of the bar method
            if orientation == 'horizontal':
                xmin0 = max(_saved_bounds[0]*0.9, minimum)
                xmax = self.dataLim.intervalx[1]
                for m in n:
                    xmin = np.amin(m[m != 0]) # filter out the 0 height bins
                xmin = max(xmin*0.9, minimum)
                xmin = min(xmin0, xmin)
                self.dataLim.intervalx = (xmin, xmax)
            elif orientation == 'vertical':
                ymin0 = max(_saved_bounds[1]*0.9, minimum)
                ymax = self.dataLim.intervaly[1]
                for m in n:
                    ymin = np.amin(m[m != 0]) # filter out the 0 height bins
                ymin = max(ymin*0.9, minimum)
                ymin = min(ymin0, ymin)
                self.dataLim.intervaly = (ymin, ymax)

        if label is None:
            labels = [None]
        elif is_string_like(label):
            labels = [label]
        elif is_sequence_of_strings(label):
            labels = list(label)
        else:
            raise ValueError(
                'invalid label: must be string or sequence of strings')

        if len(labels) < nx:
            labels += [None] * (nx - len(labels))

        for (patch, lbl) in zip(patches, labels):
            if patch:
                p = patch[0]
                p.update(kwargs)
                if lbl is not None:
                    p.set_label(lbl)

                p.set_snap(False)

                for p in patch[1:]:
                    p.update(kwargs)
                    p.set_label('_nolegend_')

        if binsgiven:
            if orientation == 'vertical':
                self.update_datalim(
                    [(bins[0], 0), (bins[-1], 0)], updatey=False)
            else:
                self.update_datalim(
                    [(0, bins[0]), (0, bins[-1])], updatex=False)

        if nx == 1:
            return n[0], bins, cbook.silent_list('Patch', patches[0])
        else:
            return n, bins, cbook.silent_list('Lists of Patches', patches)

    @docstring.dedent_interpd
    def hist2d(self, x, y, bins=10, range=None, normed=False, weights=None,
               cmin=None, cmax=None, **kwargs):
        """
        Make a 2D histogram plot.

        Parameters
        ----------
        x, y: array_like, shape (n, )
            Input values

        bins: [None | int | [int, int] | array_like | [array, array]]

            The bin specification:

                - If int, the number of bins for the two dimensions
                  (nx=ny=bins).

                - If [int, int], the number of bins in each dimension
                  (nx, ny = bins).

                - If array_like, the bin edges for the two dimensions
                  (x_edges=y_edges=bins).

                - If [array, array], the bin edges in each dimension
                  (x_edges, y_edges = bins).

            The default value is 10.

        range : array_like shape(2, 2), optional, default: None
             The leftmost and rightmost edges of the bins along each dimension
             (if not specified explicitly in the bins parameters): [[xmin,
             xmax], [ymin, ymax]]. All values outside of this range will be
             considered outliers and not tallied in the histogram.

        normed : boolean, optional, default: False
             Normalize histogram.

        weights : array_like, shape (n, ), optional, default: None
            An array of values w_i weighing each sample (x_i, y_i).

        cmin : scalar, optional, default: None
             All bins that has count less than cmin will not be displayed and
             these count values in the return value count histogram will also
             be set to nan upon return

        cmax : scalar, optional, default: None
             All bins that has count more than cmax will not be displayed (set
             to none before passing to imshow) and these count values in the
             return value count histogram will also be set to nan upon return

        Returns
        -------
        The return value is ``(counts, xedges, yedges, Image)``.

        Other parameters
        -----------------
        kwargs : :meth:`pcolorfast` properties.

        See also
        --------
        hist : 1D histogram

        Notes
        -----
        Rendering the histogram with a logarithmic color scale is
        accomplished by passing a :class:`colors.LogNorm` instance to
        the *norm* keyword argument.

        Examples
        --------
        .. plot:: mpl_examples/pylab_examples/hist2d_demo.py
        """

        # xrange becomes range after 2to3
        bin_range = range
        range = __builtins__["range"]
        h, xedges, yedges = np.histogram2d(x, y, bins=bins, range=bin_range,
                                           normed=normed, weights=weights)

        if cmin is not None:
            h[h < cmin] = None
        if cmax is not None:
            h[h > cmax] = None

        pc = self.pcolorfast(xedges, yedges, h.T, **kwargs)
        self.set_xlim(xedges[0], xedges[-1])
        self.set_ylim(yedges[0], yedges[-1])

        return h, xedges, yedges, pc

    @docstring.dedent_interpd
    def psd(self, x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, pad_to=None,
            sides='default', scale_by_freq=None, **kwargs):
        """
        Plot the power spectral density.

        Call signature::

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

          *noverlap*: integer
            The number of points of overlap between blocks.  The default value
            is 0 (no overlap).

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
        if not self._hold:
            self.cla()
        pxx, freqs = mlab.psd(x, NFFT, Fs, detrend, window, noverlap, pad_to,
                              sides, scale_by_freq)
        pxx.shape = len(freqs),
        freqs += Fc

        if scale_by_freq in (None, True):
            psd_units = 'dB/Hz'
        else:
            psd_units = 'dB'

        self.plot(freqs, 10 * np.log10(pxx), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Power Spectral Density (%s)' % psd_units)
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly
        intv = vmax - vmin
        logi = int(np.log10(intv))
        if logi == 0:
            logi = .1
        step = 10 * logi
        #print vmin, vmax, step, intv, math.floor(vmin), math.ceil(vmax)+1
        ticks = np.arange(math.floor(vmin), math.ceil(vmax) + 1, step)
        self.set_yticks(ticks)

        return pxx, freqs

    @docstring.dedent_interpd
    def csd(self, x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
            window=mlab.window_hanning, noverlap=0, pad_to=None,
            sides='default', scale_by_freq=None, **kwargs):
        """
        Plot cross-spectral density.

        Call signature::

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

          *noverlap*: integer
            The number of points of overlap between blocks.  The
            default value is 0 (no overlap).

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
        if not self._hold:
            self.cla()
        pxy, freqs = mlab.csd(x, y, NFFT, Fs, detrend, window, noverlap,
            pad_to, sides, scale_by_freq)
        pxy.shape = len(freqs),
        # pxy is complex
        freqs += Fc

        self.plot(freqs, 10 * np.log10(np.absolute(pxy)), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Cross Spectrum Magnitude (dB)')
        self.grid(True)
        vmin, vmax = self.viewLim.intervaly

        intv = vmax - vmin
        step = 10 * int(np.log10(intv))

        ticks = np.arange(math.floor(vmin), math.ceil(vmax) + 1, step)
        self.set_yticks(ticks)

        return pxy, freqs

    @docstring.dedent_interpd
    def cohere(self, x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=0, pad_to=None,
               sides='default', scale_by_freq=None, **kwargs):
        """
        Plot the coherence between *x* and *y*.

        Call signature::

          cohere(x, y, NFFT=256, Fs=2, Fc=0, detrend = mlab.detrend_none,
                 window = mlab.window_hanning, noverlap=0, pad_to=None,
                 sides='default', scale_by_freq=None, **kwargs)

        Plot the coherence between *x* and *y*.  Coherence is the
        normalized cross spectral density:

        .. math::

          C_{xy} = \\frac{|P_{xy}|^2}{P_{xx}P_{yy}}

        %(PSD)s

          *noverlap*: integer
            The number of points of overlap between blocks.  The
            default value is 0 (no overlap).

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
        if not self._hold:
            self.cla()
        cxy, freqs = mlab.cohere(x, y, NFFT, Fs, detrend, window, noverlap,
            scale_by_freq)
        freqs += Fc

        self.plot(freqs, cxy, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Coherence')
        self.grid(True)

        return cxy, freqs

    @docstring.dedent_interpd
    def specgram(self, x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
                 window=mlab.window_hanning, noverlap=128,
                 cmap=None, xextent=None, pad_to=None, sides='default',
                 scale_by_freq=None, **kwargs):
        """
        Plot a spectrogram.

        Call signature::

          specgram(x, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
                   window=mlab.window_hanning, noverlap=128,
                   cmap=None, xextent=None, pad_to=None, sides='default',
                   scale_by_freq=None, **kwargs)

        Compute and plot a spectrogram of data in *x*.  Data are split into
        *NFFT* length segments and the PSD of each section is
        computed.  The windowing function *window* is applied to each
        segment, and the amount of overlap of each segment is
        specified with *noverlap*. The spectrogram is plotted in decibels
        as a colormap (using imshow).

        %(PSD)s

          *noverlap*: integer
            The number of points of overlap between blocks.  The
            default value is 128.

          *Fc*: integer
            The center frequency of *x* (defaults to 0), which offsets
            the y extents of the plot to reflect the frequency range used
            when a signal is acquired and then filtered and downsampled to
            baseband.

          *cmap*:
            A :class:`matplotlib.colors.Colormap` instance; if *None*, use
            default determined by rc

          *xextent*:
            The image extent along the x-axis. xextent = (xmin,xmax)
            The default is (0,max(bins)), where bins is the return
            value from :func:`~matplotlib.mlab.specgram`

          *kwargs*:

            Additional kwargs are passed on to imshow which makes the
            specgram image

          Return value is (*Pxx*, *freqs*, *bins*, *im*):

          - *bins* are the time points the spectrogram is calculated over
          - *freqs* is an array of frequencies
          - *Pxx* is an array of shape `(len(times), len(freqs))` of power
          - *im* is a :class:`~matplotlib.image.AxesImage` instance

        .. note::

            If *x* is real (i.e. non-complex), only the positive
            spectrum is shown.  If *x* is complex, both positive and
            negative parts of the spectrum are shown.  This can be
            overridden using the *sides* keyword argument.

        Also note that while the plot is in dB, the *Pxx* array returned is
        linear in power.

        **Example:**

        .. plot:: mpl_examples/pylab_examples/specgram_demo.py
        """
        if not self._hold:
            self.cla()

        Pxx, freqs, bins = mlab.specgram(x, NFFT, Fs, detrend,
             window, noverlap, pad_to, sides, scale_by_freq)

        Z = 10. * np.log10(Pxx)
        Z = np.flipud(Z)

        if xextent is None:
            xextent = 0, np.amax(bins)
        xmin, xmax = xextent
        freqs += Fc
        extent = xmin, xmax, freqs[0], freqs[-1]
        im = self.imshow(Z, cmap, extent=extent, **kwargs)
        self.axis('auto')

        return Pxx, freqs, bins, im

    def spy(self, Z, precision=0, marker=None, markersize=None,
            aspect='equal', **kwargs):
        """
        Plot the sparsity pattern on a 2-D array.

        Call signature::

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
               For image options.

        For controlling colors, e.g., cyan background and red marks,
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
               For plotting options
        """
        if marker is None and markersize is None and hasattr(Z, 'tocoo'):
            marker = 's'
        if marker is None and markersize is None:
            Z = np.asarray(Z)
            mask = np.absolute(Z) > precision

            if 'cmap' not in kwargs:
                kwargs['cmap'] = mcolors.ListedColormap(['w', 'k'],
                                                        name='binary')
            nr, nc = Z.shape
            extent = [-0.5, nc - 0.5, nr - 0.5, -0.5]
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
                nonzero = np.absolute(Z) > precision
                y, x = np.nonzero(nonzero)
            if marker is None:
                marker = 's'
            if markersize is None:
                markersize = 10
            marks = mlines.Line2D(x, y, linestyle='None',
                         marker=marker, markersize=markersize, **kwargs)
            self.add_line(marks)
            nr, nc = Z.shape
            self.set_xlim(xmin=-0.5, xmax=nc - 0.5)
            self.set_ylim(ymin=nr - 0.5, ymax=-0.5)
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
        """
        Plot a matrix or array as an image.

        The matrix will be shown the way it would be printed, with the first
        row at the top.  Row and column numbering is zero-based.

        Parameters
        ----------
        Z : array_like shape (n, m)
            The matrix to be displayed.

        Returns
        -------
        image : `~matplotlib.image.AxesImage`

        Other parameters
        ----------------
        kwargs : `~matplotlib.axes.Axes.imshow` arguments
            Sets `origin` to 'upper', 'interpolation' to 'nearest' and
            'aspect' to equal.

        See also
        --------
        imshow : plot an image

        Examples
        --------
        .. plot:: mpl_examples/pylab_examples/matshow.py

        """
        Z = np.asanyarray(Z)
        nr, nc = Z.shape
        kw = {'origin': 'upper',
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

    def get_default_bbox_extra_artists(self):
        return [artist for artist in self.get_children()
                if artist.get_visible()]

    def get_tightbbox(self, renderer, call_axes_locator=True):
        """
        Return the tight bounding box of the axes.
        The dimension of the Bbox in canvas coordinate.

        If *call_axes_locator* is *False*, it does not call the
        _axes_locator attribute, which is necessary to get the correct
        bounding box. ``call_axes_locator==False`` can be used if the
        caller is only intereted in the relative size of the tightbbox
        compared to the axes bbox.
        """

        bb = []

        if not self.get_visible():
            return None

        locator = self.get_axes_locator()
        if locator and call_axes_locator:
            pos = locator(self, renderer)
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        bb.append(self.get_window_extent(renderer))

        if self.title.get_visible():
            bb.append(self.title.get_window_extent(renderer))
        if self._left_title.get_visible():
            bb.append(self._left_title.get_window_extent(renderer))
        if self._right_title.get_visible():
            bb.append(self._right_title.get_window_extent(renderer))

        bb_xaxis = self.xaxis.get_tightbbox(renderer)
        if bb_xaxis:
            bb.append(bb_xaxis)

        bb_yaxis = self.yaxis.get_tightbbox(renderer)
        if bb_yaxis:
            bb.append(bb_yaxis)

        _bbox = mtransforms.Bbox.union(
            [b for b in bb if b.width != 0 or b.height != 0])

        return _bbox

    def minorticks_on(self):
        'Add autoscaling minor ticks to the axes.'
        for ax in (self.xaxis, self.yaxis):
            if ax.get_scale() == 'log':
                s = ax._scale
                ax.set_minor_locator(mticker.LogLocator(s.base, s.subs))
            else:
                ax.set_minor_locator(mticker.AutoMinorLocator())

    def minorticks_off(self):
        """Remove minor ticks from the axes."""
        self.xaxis.set_minor_locator(mticker.NullLocator())
        self.yaxis.set_minor_locator(mticker.NullLocator())

    def tricontour(self, *args, **kwargs):
        return mtri.tricontour(self, *args, **kwargs)
    tricontour.__doc__ = mtri.TriContourSet.tricontour_doc

    def tricontourf(self, *args, **kwargs):
        return mtri.tricontourf(self, *args, **kwargs)
    tricontourf.__doc__ = mtri.TriContourSet.tricontour_doc

    def tripcolor(self, *args, **kwargs):
        return mtri.tripcolor(self, *args, **kwargs)
    tripcolor.__doc__ = mtri.tripcolor.__doc__

    def triplot(self, *args, **kwargs):
        mtri.triplot(self, *args, **kwargs)
    triplot.__doc__ = mtri.triplot.__doc__


from matplotlib.gridspec import GridSpec, SubplotSpec


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

        if len(args) == 1:
            if isinstance(args[0], SubplotSpec):
                self._subplotspec = args[0]
            else:
                try:
                    s = str(int(args[0]))
                    rows, cols, num = map(int, s)
                except ValueError:
                    raise ValueError(
                         'Single argument to subplot must be a 3-digit '
                         'integer')
                self._subplotspec = GridSpec(rows, cols)[num - 1]
                # num - 1 for converting from MATLAB to python indexing
        elif len(args) == 3:
            rows, cols, num = args
            rows = int(rows)
            cols = int(cols)
            if isinstance(num, tuple) and len(num) == 2:
                num = [int(n) for n in num]
                self._subplotspec = GridSpec(rows, cols)[num[0] - 1:num[1]]
            else:
                self._subplotspec = GridSpec(rows, cols)[int(num) - 1]
                # num - 1 for converting from MATLAB to python indexing
        else:
            raise ValueError('Illegal argument(s) to subplot: %s' % (args,))

        self.update_params()

        # _axes_class is set in the subplot_class_factory
        self._axes_class.__init__(self, fig, self.figbox, **kwargs)

    def __reduce__(self):
        # get the first axes class which does not inherit from a subplotbase
        not_subplotbase = lambda c: issubclass(c, Axes) and \
                                    not issubclass(c, SubplotBase)
        axes_class = [c for c in self.__class__.mro() if not_subplotbase(c)][0]
        r = [_PicklableSubplotClassConstructor(),
             (axes_class,),
             self.__getstate__()]
        return tuple(r)

    def get_geometry(self):
        """get the subplot geometry, eg 2,2,3"""
        rows, cols, num1, num2 = self.get_subplotspec().get_geometry()
        return rows, cols, num1 + 1  # for compatibility

    # COVERAGE NOTE: Never used internally or from examples
    def change_geometry(self, numrows, numcols, num):
        """change subplot geometry, e.g., from 1,1,1 to 2,2,3"""
        self._subplotspec = GridSpec(numrows, numcols)[num - 1]
        self.update_params()
        self.set_position(self.figbox)

    def get_subplotspec(self):
        """get the SubplotSpec instance associated with the subplot"""
        return self._subplotspec

    def set_subplotspec(self, subplotspec):
        """set the SubplotSpec instance associated with the subplot"""
        self._subplotspec = subplotspec

    def update_params(self):
        """update the subplot position from fig.subplotpars"""

        self.figbox, self.rowNum, self.colNum, self.numRows, self.numCols = \
                     self.get_subplotspec().get_position(self.figure,
                                                         return_all=True)

    def is_first_col(self):
        return self.colNum == 0

    def is_first_row(self):
        return self.rowNum == 0

    def is_last_row(self):
        return self.rowNum == self.numRows - 1

    def is_last_col(self):
        return self.colNum == self.numCols - 1

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

    def _make_twin_axes(self, *kl, **kwargs):
        """
        make a twinx axes of self. This is used for twinx and twiny.
        """
        from matplotlib.projections import process_projection_requirements
        kl = (self.get_subplotspec(),) + kl
        projection_class, kwargs, key = process_projection_requirements(
            self.figure, *kl, **kwargs)

        ax2 = subplot_class_factory(projection_class)(self.figure,
                                                      *kl, **kwargs)
        self.figure.add_subplot(ax2)
        return ax2

_subplot_classes = {}


def subplot_class_factory(axes_class=None):
    # This makes a new class that inherits from SubplotBase and the
    # given axes_class (which is assumed to be a subclass of Axes).
    # This is perhaps a little bit roundabout to make a new class on
    # the fly like this, but it means that a new Subplot class does
    # not have to be created for every type of Axes.
    if axes_class is None:
        axes_class = Axes

    new_class = _subplot_classes.get(axes_class)
    if new_class is None:
        new_class = type("%sSubplot" % (axes_class.__name__),
                         (SubplotBase, axes_class),
                         {'_axes_class': axes_class})
        _subplot_classes[axes_class] = new_class

    return new_class

# This is provided for backward compatibility
Subplot = subplot_class_factory()


class _PicklableSubplotClassConstructor(object):
    """
    This stub class exists to return the appropriate subplot
    class when __call__-ed with an axes class. This is purely to
    allow Pickling of Axes and Subplots.
    """
    def __call__(self, axes_class):
        # create a dummy object instance
        subplot_instance = _PicklableSubplotClassConstructor()
        subplot_class = subplot_class_factory(axes_class)
        # update the class to the desired subplot class
        subplot_instance.__class__ = subplot_class
        return subplot_instance


docstring.interpd.update(Axes=martist.kwdoc(Axes))
docstring.interpd.update(Subplot=martist.kwdoc(Axes))

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
