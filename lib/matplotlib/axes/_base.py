from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six.moves import xrange

import itertools
import warnings
import math
from operator import itemgetter

import numpy as np
from numpy import ma

import matplotlib

from matplotlib import cbook
from matplotlib.cbook import (_check_1d, _string_to_bool, iterable,
                              index_of, get_label)
from matplotlib import docstring
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker
import matplotlib.axis as maxis
import matplotlib.scale as mscale
import matplotlib.spines as mspines
import matplotlib.font_manager as font_manager
import matplotlib.text as mtext
import matplotlib.image as mimage
from matplotlib.offsetbox import OffsetBox
from matplotlib.artist import allow_rasterization

from matplotlib.rcsetup import cycler

rcParams = matplotlib.rcParams

is_string_like = cbook.is_string_like
is_sequence_of_strings = cbook.is_sequence_of_strings


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
        self.set_prop_cycle()

    def __getstate__(self):
        # note: it is not possible to pickle a itertools.cycle instance
        return {'axes': self.axes, 'command': self.command}

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        self.set_prop_cycle()

    def set_prop_cycle(self, *args, **kwargs):
        if not (args or kwargs) or (len(args) == 1 and args[0] is None):
            prop_cycler = rcParams['axes.prop_cycle']
            if prop_cycler is None and 'axes.color_cycle' in rcParams:
                clist = rcParams['axes.color_cycle']
                prop_cycler = cycler('color', clist)
        else:
            prop_cycler = cycler(*args, **kwargs)

        self.prop_cycler = itertools.cycle(prop_cycler)
        # This should make a copy
        self._prop_keys = prop_cycler.keys

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

    def set_patchprops(self, fill_poly, **kwargs):
        assert self.command == 'fill', 'set_patchprops only works with "fill"'
        fill_poly.set(**kwargs)

    def _xy_from_xy(self, x, y):
        if self.axes.xaxis is not None and self.axes.yaxis is not None:
            bx = self.axes.xaxis.update_units(x)
            by = self.axes.yaxis.update_units(y)

            if self.command != 'plot':
                # the Line2D class can handle unitized data, with
                # support for post hoc unit changes etc.  Other mpl
                # artists, e.g., Polygon which _process_plot_var_args
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

        # like asanyarray, but converts scalar to array, and doesn't change
        # existing compatible sequences
        x = _check_1d(x)
        y = _check_1d(y)
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same first dimension")
        if x.ndim > 2 or y.ndim > 2:
            raise ValueError("x and y can be no greater than 2-D")

        if x.ndim == 1:
            x = x[:, np.newaxis]
        if y.ndim == 1:
            y = y[:, np.newaxis]
        return x, y

    def _getdefaults(self, ignore, *kwargs):
        """
        Only advance the cycler if the cycler has information that
        is not specified in any of the supplied tuple of dicts.
        Ignore any keys specified in the `ignore` set.

        Returns a copy of defaults dictionary if there are any
        keys that are not found in any of the supplied dictionaries.
        If the supplied dictionaries have non-None values for
        everything the property cycler has, then just return
        an empty dictionary. Ignored keys are excluded from the
        returned dictionary.

        """
        prop_keys = self._prop_keys
        if ignore is None:
            ignore = set([])
        prop_keys = prop_keys - ignore

        if any(all(kw.get(k, None) is None for kw in kwargs)
               for k in prop_keys):
            # Need to copy this dictionary or else the next time around
            # in the cycle, the dictionary could be missing entries.
            default_dict = six.next(self.prop_cycler).copy()
            for p in ignore:
                default_dict.pop(p, None)
        else:
            default_dict = {}
        return default_dict

    def _setdefaults(self, defaults, *kwargs):
        """
        Given a defaults dictionary, and any other dictionaries,
        update those other dictionaries with information in defaults if
        none of the other dictionaries contains that information.

        """
        for k in defaults:
            if all(kw.get(k, None) is None for kw in kwargs):
                for kw in kwargs:
                    kw[k] = defaults[k]

    def _makeline(self, x, y, kw, kwargs):
        kw = kw.copy()  # Don't modify the original kw.
        kw.update(kwargs)
        default_dict = self._getdefaults(None, kw)
        self._setdefaults(default_dict, kw)
        seg = mlines.Line2D(x, y, **kw)
        return seg

    def _makefill(self, x, y, kw, kwargs):
        kw = kw.copy()  # Don't modify the original kw.
        kwargs = kwargs.copy()

        # Ignore 'marker'-related properties as they aren't Polygon
        # properties, but they are Line2D properties, and so they are
        # likely to appear in the default cycler construction.
        # This is done here to the defaults dictionary as opposed to the
        # other two dictionaries because we do want to capture when a
        # *user* explicitly specifies a marker which should be an error.
        # We also want to prevent advancing the cycler if there are no
        # defaults needed after ignoring the given properties.
        ignores = set(['marker', 'markersize', 'markeredgecolor',
                       'markerfacecolor', 'markeredgewidth'])
        # Also ignore anything provided by *kwargs*.
        for k, v in six.iteritems(kwargs):
            if v is not None:
                ignores.add(k)

        # Only using the first dictionary to use as basis
        # for getting defaults for back-compat reasons.
        # Doing it with both seems to mess things up in
        # various places (probably due to logic bugs elsewhere).
        default_dict = self._getdefaults(ignores, kw)
        self._setdefaults(default_dict, kw)

        # Looks like we don't want "color" to be interpreted to
        # mean both facecolor and edgecolor for some reason.
        # So the "kw" dictionary is thrown out, and only its
        # 'color' value is kept and translated as a 'facecolor'.
        # This design should probably be revisited as it increases
        # complexity.
        facecolor = kw.get('color', None)

        # Throw out 'color' as it is now handled as a facecolor
        default_dict.pop('color', None)

        # To get other properties set from the cycler
        # modify the kwargs dictionary.
        self._setdefaults(default_dict, kwargs)

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

        # Don't allow any None value; These will be up-converted
        # to one element array of None which causes problems
        # downstream.
        if any(v is None for v in tup):
            raise ValueError("x and y must not be None")

        kw = {}
        for k, v in zip(('linestyle', 'marker', 'color'),
                        (linestyle, marker, color)):
            if v is not None:
                kw[k] = v

        if 'label' not in kwargs or kwargs['label'] is None:
            kwargs['label'] = get_label(tup[-1], None)

        if len(tup) == 2:
            x = _check_1d(tup[0])
            y = _check_1d(tup[-1])
        else:
            x, y = index_of(tup[-1])

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


class _AxesBase(martist.Artist):
    """
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
        # self.set_axes(self)
        self.axes = self
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

        # this call may differ for non-sep axes, e.g., polar
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
            self.update(kwargs)

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
        self._stale = True

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
        self.stale = True

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
        self.stale = True

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

        a.axes = self
        if a.mouseover:
            self.mouseover_set.add(a)

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

        # stash the current visibility state
        if hasattr(self, 'patch'):
            patch_visible = self.patch.get_visible()
        else:
            patch_visible = True

        xaxis_visible = self.xaxis.get_visible()
        yaxis_visible = self.yaxis.get_visible()

        self.xaxis.cla()
        self.yaxis.cla()
        for name, spine in six.iteritems(self.spines):
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

        # update the minor locator for x and y axis based on rcParams
        if (rcParams['xtick.minor.visible']):
            self.xaxis.set_minor_locator(mticker.AutoMinorLocator())

        if (rcParams['ytick.minor.visible']):
            self.yaxis.set_minor_locator(mticker.AutoMinorLocator())

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
        self.mouseover_set = set()
        self._current_image = None  # strictly for pyplot via _sci, _gci
        self.legend_ = None
        self.collections = []  # collection.Collection instances
        self.containers = []

        self.grid(False)  # Disable grid on init to use rcParameter
        self.grid(self._gridOn, which=rcParams['axes.grid.which'],
                    axis=rcParams['axes.grid.axis'])
        props = font_manager.FontProperties(
                    size=rcParams['axes.titlesize'],
                    weight=rcParams['axes.titleweight']
                )

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
            fontproperties=props.copy(),
            verticalalignment='baseline',
            horizontalalignment='left', )
        self._right_title = mtext.Text(
            x=1.0, y=1.0, text='',
            fontproperties=props.copy(),
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

        self.set_axis_on()

        self.xaxis.set_clip_path(self.patch)
        self.yaxis.set_clip_path(self.patch)

        self._shared_x_axes.clean()
        self._shared_y_axes.clean()
        if self._sharex:
            self.xaxis.set_visible(xaxis_visible)
            self.patch.set_visible(patch_visible)

        if self._sharey:
            self.yaxis.set_visible(yaxis_visible)
            self.patch.set_visible(patch_visible)
        self.stale = True

    def clear(self):
        """clear the axes"""
        self.cla()

    def set_prop_cycle(self, *args, **kwargs):
        """
        Set the property cycle for any future plot commands on this Axes.

        set_prop_cycle(arg)
        set_prop_cycle(label, itr)
        set_prop_cycle(label1=itr1[, label2=itr2[, ...]])

        Form 1 simply sets given `Cycler` object.

        Form 2 creates and sets  a `Cycler` from a label and an iterable.

        Form 3 composes and sets  a `Cycler` as an inner product of the
        pairs of keyword arguments. In other words, all of the
        iterables are cycled simultaneously, as if through zip().

        Parameters
        ----------
        arg : Cycler
            Set the given Cycler.
            Can also be `None` to reset to the cycle defined by the
            current style.

        label : str
            The property key. Must be a valid `Artist` property.
            For example, 'color' or 'linestyle'. Aliases are allowed,
            such as 'c' for 'color' and 'lw' for 'linewidth'.

        itr : iterable
            Finite-length iterable of the property values. These values
            are validated and will raise a ValueError if invalid.

        See Also
        --------
            :func:`cycler`      Convenience function for creating your
                                own cyclers.

        """
        if args and kwargs:
            raise TypeError("Cannot supply both positional and keyword "
                            "arguments to this method.")
        if len(args) == 1 and args[0] is None:
            prop_cycle = None
        else:
            prop_cycle = cycler(*args, **kwargs)
        self._get_lines.set_prop_cycle(prop_cycle)
        self._get_patches_for_fill.set_prop_cycle(prop_cycle)

    def set_color_cycle(self, clist):
        """
        Set the color cycle for any future plot commands on this Axes.

        *clist* is a list of mpl color specifiers.

        .. deprecated:: 1.5
        """
        cbook.warn_deprecated(
                '1.5', name='set_color_cycle', alternative='set_prop_cycle')
        if clist is None:
            # Calling set_color_cycle() or set_prop_cycle() with None
            # effectively resets the cycle, but you can't do
            # set_prop_cycle('color', None). So we are special-casing this.
            self.set_prop_cycle(None)
        else:
            self.set_prop_cycle('color', clist)

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
        self.stale = True

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
        self.stale = True

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
        if (anchor in list(six.iterkeys(mtransforms.Bbox.coefs)) or
                len(anchor) == 2):
            self._anchor = anchor
        else:
            raise ValueError('argument must be among %s' %
                             ', '.join(six.iterkeys(mtransforms.Bbox.coefs)))
        self.stale = True

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
                if aspect != "auto":
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

        # Ensure at drawing time that any Axes involved in axis-sharing
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
        # If y_expander > 0, the dy/dx viewLim ratio needs to increase
        if abs(y_expander) < 0.005:
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
        # Setting these targets to, e.g., 0.05*xr does not seem to
        # help.
        xm = 0
        ym = 0

        changex = (self in self._shared_y_axes and
                   self not in self._shared_x_axes)
        changey = (self in self._shared_x_axes and
                   self not in self._shared_y_axes)
        if changex and changey:
            warnings.warn("adjustable='datalim' cannot work with shared "
                          "x and y axes")
            return
        if changex:
            adjust_y = False
        else:
            if xmarg > xm and ymarg > ym:
                adjy = ((Ymarg > 0 and y_expander < 0) or
                        (Xmarg < 0 and y_expander > 0))
            else:
                adjy = y_expander > 0
            adjust_y = changey or adjy  # (Ymarg > xmarg)
        if adjust_y:
            yc = 0.5 * (ymin + ymax)
            y0 = yc - Ysize / 2.0
            y1 = yc + Ysize / 2.0
            if aspect_scale_mode == "log":
                self.set_ybound((10. ** y0, 10. ** y1))
            else:
                self.set_ybound((y0, y1))
        else:
            xc = 0.5 * (xmin + xmax)
            x0 = xc - Xsize / 2.0
            x1 = xc + Xsize / 2.0
            if aspect_scale_mode == "log":
                self.set_xbound((10. ** x0, 10. ** x1))
            else:
                self.set_xbound((x0, x1))

    def axis(self, *v, **kwargs):
        """Set axis properties.

        Valid signatures::

          xmin, xmax, ymin, ymax = axis()
          xmin, xmax, ymin, ymax = axis(list_arg)
          xmin, xmax, ymin, ymax = axis(string_arg)
          xmin, xmax, ymin, ymax = axis(**kwargs)

        Parameters
        ----------
        v : list of float or {'on', 'off', 'equal', 'tight', 'scaled',\
            'normal', 'auto', 'image', 'square'}
            Optional positional argument

            Axis data limits set from a list; or a command relating to axes:

                ========== ================================================
                Value      Description
                ========== ================================================
                'on'       Toggle axis lines and labels on
                'off'      Toggle axis lines and labels off
                'equal'    Equal scaling by changing limits
                'scaled'   Equal scaling by changing box dimensions
                'tight'    Limits set such that all data is shown
                'auto'     Automatic scaling, fill rectangle with data
                'normal'   Same as 'auto'; deprecated
                'image'    'scaled' with axis limits equal to data limits
                'square'   Square plot; similar to 'scaled', but initially\
                           forcing xmax-xmin = ymax-ymin
                ========== ================================================

        emit : bool, optional
            Passed to set_{x,y}lim functions, if observers
            are notified of axis limit change

        xmin, ymin, xmax, ymax : float, optional
            The axis limits to be set

        Returns
        -------
        xmin, xmax, ymin, ymax : float
            The axis limits

        """

        if len(v) == 0 and len(kwargs) == 0:
            xmin, xmax = self.get_xlim()
            ymin, ymax = self.get_ylim()
            return xmin, xmax, ymin, ymax

        emit = kwargs.get('emit', True)

        if len(v) == 1 and is_string_like(v[0]):
            s = v[0].lower()
            if s == 'on':
                self.set_axis_on()
            elif s == 'off':
                self.set_axis_off()
            elif s in ('equal', 'tight', 'scaled', 'normal',
                       'auto', 'image', 'square'):
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
                elif s == 'square':
                    self.set_aspect('equal', adjustable='box', anchor='C')
                    self.set_autoscale_on(False)
                    xlim = self.get_xlim()
                    ylim = self.get_ylim()
                    edge_size = max(np.diff(xlim), np.diff(ylim))
                    self.set_xlim([xlim[0], xlim[0] + edge_size],
                                  emit=emit, auto=False)
                    self.set_ylim([ylim[0], ylim[0] + edge_size],
                                  emit=emit, auto=False)
            else:
                raise ValueError('Unrecognized string %s to axis; '
                                 'try on or off' % s)
            xmin, xmax = self.get_xlim()
            ymin, ymax = self.get_ylim()
            return xmin, xmax, ymin, ymax

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

    # Adding and tracking artists

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
        """Add any :class:`~matplotlib.artist.Artist` to the axes.

        Use `add_artist` only for artists for which there is no dedicated
        "add" method; and if necessary, use a method such as
        `update_datalim` or `update_datalim_numerix` to manually update the
        dataLim if the artist is to be included in autoscaling.

        Returns the artist.
        """
        a.axes = self
        self.artists.append(a)
        self._set_artist_props(a)
        a.set_clip_path(self.patch)
        a._remove_method = lambda h: self.artists.remove(h)
        self.stale = True
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

        if autolim:
            self.update_datalim(collection.get_datalim(self.transData))

        collection._remove_method = lambda h: self.collections.remove(h)
        self.stale = True
        return collection

    def add_image(self, image):
        """
        Add a :class:`~matplotlib.image.AxesImage` to the axes.

        Returns the image.
        """
        self._set_artist_props(image)
        self.images.append(image)
        image._remove_method = lambda h: self.images.remove(h)
        self.stale = True
        return image

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
        self.stale = True
        return line

    def _add_text(self, txt):
        """

        """
        self._set_artist_props(txt)
        self.texts.append(txt)
        txt._remove_method = lambda h: self.texts.remove(h)
        self.stale = True
        return txt

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
                self.transData)
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
        # issue #2150 - we update the limits if patch has non zero width
        # or height.
        if (isinstance(patch, mpatches.Rectangle) and
                ((not patch.get_width()) and (not patch.get_height()))):
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

    def relim(self, visible_only=False):
        """
        Recompute the data limits based on current artists. If you want to
        exclude invisible artists from the calculation, set
        ``visible_only=True``

        At present, :class:`~matplotlib.collections.Collection`
        instances are not supported.
        """
        # Collections are deliberately not supported (yet); see
        # the TODO note in artists.py.
        self.dataLim.ignore(True)
        self.dataLim.set_points(mtransforms.Bbox.null().get_points())
        self.ignore_existing_data_limits = True

        for line in self.lines:
            if not visible_only or line.get_visible():
                self._update_line_limits(line)

        for p in self.patches:
            if not visible_only or p.get_visible():
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

        if xdata is not None:
            # we only need to update if there is nothing set yet.
            if not self.xaxis.have_units():
                self.xaxis.update_units(xdata)

        if ydata is not None:
            # we only need to update if there is nothing set yet.
            if not self.yaxis.have_units():
                self.yaxis.update_units(ydata)

        # process kwargs 2nd since these will override default units
        if kwargs is not None:
            xunits = kwargs.pop('xunits', self.xaxis.units)
            if self.name == 'polar':
                xunits = kwargs.pop('thetaunits', xunits)
            if xunits != self.xaxis.units:
                self.xaxis.set_units(xunits)
                # If the units being set imply a different converter,
                # we need to update.
                if xdata is not None:
                    self.xaxis.update_units(xdata)

            yunits = kwargs.pop('yunits', self.yaxis.units)
            if self.name == 'polar':
                yunits = kwargs.pop('runits', yunits)
            if yunits != self.yaxis.units:
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
        self.stale = True

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
        self.stale = True

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
        elif len(args) > 2:
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
        self.stale = True

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
        selectively autoscale only a single axis, e.g., the xaxis by
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
            # ignore non-finite data limits if good limits exist
            finite_dl = [d for d in dl if np.isfinite(d).all()]
            if len(finite_dl):
                dl = finite_dl

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
            # ignore non-finite data limits if good limits exist
            finite_dl = [d for d in dl if np.isfinite(d).all()]
            if len(finite_dl):
                dl = finite_dl

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

    def _get_axis_list(self):
        return (self.xaxis, self.yaxis)

    # Drawing

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
        # prevent triggering call backs during the draw process
        self._stale = True
        locator = self.get_axes_locator()
        if locator:
            pos = locator(self, renderer)
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        artists = self.get_children()
        artists.remove(self.patch)

        # the frame draws the edges around the axes patch -- we
        # decouple these so the patch can be in the background and the
        # frame in the foreground. Do this before drawing the axis
        # objects so that the spine has the opportunity to update them.
        if not (self.axison and self._frameon):
            for spine in six.itervalues(self.spines):
                artists.remove(spine)

        if self.axison and not inframe:
            if self._axisbelow:
                self.xaxis.set_zorder(0.5)
                self.yaxis.set_zorder(0.5)
            else:
                self.xaxis.set_zorder(2.5)
                self.yaxis.set_zorder(2.5)
        else:
            for _axis in self._get_axis_list():
                artists.remove(_axis)

        if inframe:
            artists.remove(self.title)
            artists.remove(self._left_title)
            artists.remove(self._right_title)

        # add images to dsu if the backend supports compositing.
        # otherwise, does the manual compositing  without adding images to dsu.
        if len(self.images) <= 1 or renderer.option_image_nocomposite():
            _do_composite = False
        else:
            _do_composite = True
            for im in self.images:
                artists.remove(im)

        if self.figure.canvas.is_saving():
            dsu = [(a.zorder, a) for a in artists]
        else:
            dsu = [(a.zorder, a) for a in artists
                   if (not a.get_animated() or a in self.images)]

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
            # make a composite image, blending alpha
            # list of (mimage.Image, ox, oy)

            zorder_images = [(im.zorder, im) for im in self.images
                             if im.get_visible()]
            zorder_images.sort(key=lambda x: x[0])

            mag = renderer.get_image_magnification()
            ims = [(im.make_image(mag), 0, 0, im.get_alpha())
                   for z, im in zorder_images]

            l, b, r, t = self.bbox.extents
            width = int(mag * ((round(r) + 0.5) - (round(l) - 0.5)))
            height = int(mag * ((round(t) + 0.5) - (round(b) - 0.5)))
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
        self.stale = False

    def draw_artist(self, a):
        """
        This method can only be used after an initial draw which
        caches the renderer.  It is used to efficiently update Axes
        data (axis ticks, labels, etc are not updated)
        """
        if self._cachedRenderer is None:
            msg = ('draw_artist can only be used after an initial draw which'
                   ' caches the render')
            raise AttributeError(msg)
        a.draw(self._cachedRenderer)

    def redraw_in_frame(self):
        """
        This method can only be used after an initial draw which
        caches the renderer.  It is used to efficiently update Axes
        data (axis ticks, labels, etc are not updated)
        """
        if self._cachedRenderer is None:
            msg = ('redraw_in_frame can only be used after an initial draw'
                   ' which caches the render')
            raise AttributeError(msg)
        self.draw(self._cachedRenderer, inframe=True)

    def get_renderer_cache(self):
        return self._cachedRenderer

    # Axes rectangle characteristics

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
        self.stale = True

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
        self.stale = True

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

        *kwargs* are used to set the grid line properties, e.g.,::

           ax.grid(color='r', linestyle='-', linewidth=2)

        Valid :class:`~matplotlib.lines.Line2D` kwargs are

        %(Line2D)s

        """
        if len(kwargs):
            b = True
        b = _string_to_bool(b)

        if axis == 'x' or axis == 'both':
            self.xaxis.grid(b, which=which, **kwargs)
        if axis == 'y' or axis == 'both':
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
        self.stale = True

    def set_axis_on(self):
        """turn on the axis"""
        self.axison = True
        self.stale = True

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
        self.stale = True
    # data limits, ticks, tick labels, and formatting

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
            raise ValueError("unrecognized kwargs: %s" %
                             list(six.iterkeys(kw)))

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
            warnings.warn(
                ('Attempting to set identical left==right results\n'
                 'in singular transformations; automatically expanding.\n'
                 'left=%s, right=%s') % (left, right))
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
        self.stale = True
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
        # If the scale is being set to log, clip nonposx to prevent headaches
        # around zero
        if value.lower() == 'log' and 'nonposx' not in kwargs.keys():
            kwargs['nonposx'] = 'clip'

        g = self.get_shared_x_axes()
        for ax in g.get_siblings(self):
            ax.xaxis._set_scale(value, **kwargs)
            ax._update_transScale()
            ax.stale = True

        self.autoscale_view(scaley=False)

    def get_xticks(self, minor=False):
        """Return the x ticks as a list of locations"""
        return self.xaxis.get_ticklocs(minor=minor)

    def set_xticks(self, ticks, minor=False):
        """
        Set the x ticks with list of *ticks*

        ACCEPTS: sequence of floats
        """
        ret = self.xaxis.set_ticks(ticks, minor=minor)
        self.stale = True
        return ret

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

    def get_xticklabels(self, minor=False, which=None):
        """
        Get the x tick labels as a list of :class:`~matplotlib.text.Text`
        instances.

        Parameters
        ----------
        minor : bool
           If True return the minor ticklabels,
           else return the major ticklabels

        which : None, ('minor', 'major', 'both')
           Overrides `minor`.

           Selects which ticklabels to return

        Returns
        -------
        ret : list
           List of :class:`~matplotlib.text.Text` instances.
        """
        return cbook.silent_list('Text xticklabel',
                                 self.xaxis.get_ticklabels(minor=minor,
                                                           which=which))

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
        if fontdict is not None:
            kwargs.update(fontdict)
        ret = self.xaxis.set_ticklabels(labels,
                                        minor=minor, **kwargs)
        self.stale = True
        return ret

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
            raise ValueError("unrecognized kwargs: %s" %
                             list(six.iterkeys(kw)))

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
            warnings.warn(
                ('Attempting to set identical bottom==top results\n'
                 'in singular transformations; automatically expanding.\n'
                 'bottom=%s, top=%s') % (bottom, top))

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
        self.stale = True
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
        # If the scale is being set to log, clip nonposy to prevent headaches
        # around zero
        if value.lower() == 'log' and 'nonposy' not in kwargs.keys():
            kwargs['nonposy'] = 'clip'

        g = self.get_shared_y_axes()
        for ax in g.get_siblings(self):
            ax.yaxis._set_scale(value, **kwargs)
            ax._update_transScale()
            ax.stale = True
        self.autoscale_view(scalex=False)

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
        ret = self.yaxis.set_ticks(ticks, minor=minor)
        return ret

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

    def get_yticklabels(self, minor=False, which=None):
        """
        Get the x tick labels as a list of :class:`~matplotlib.text.Text`
        instances.

        Parameters
        ----------
        minor : bool
           If True return the minor ticklabels,
           else return the major ticklabels

        which : None, ('minor', 'major', 'both')
           Overrides `minor`.

           Selects which ticklabels to return

        Returns
        -------
        ret : list
           List of :class:`~matplotlib.text.Text` instances.
        """
        return cbook.silent_list('Text yticklabel',
                                 self.yaxis.get_ticklabels(minor=minor,
                                                           which=which))

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
        if fontdict is not None:
            kwargs.update(fontdict)
        return self.yaxis.set_ticklabels(labels,
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

    def minorticks_on(self):
        'Add autoscaling minor ticks to the axes.'
        for ax in (self.xaxis, self.yaxis):
            scale = ax.get_scale()
            if scale == 'log':
                s = ax._scale
                ax.set_minor_locator(mticker.LogLocator(s.base, s.subs))
            elif scale == 'symlog':
                s = ax._scale
                ax.set_minor_locator(
                    mticker.SymmetricalLogLocator(s.base, s.subs))
            else:
                ax.set_minor_locator(mticker.AutoMinorLocator())

    def minorticks_off(self):
        """Remove minor ticks from the axes."""
        self.xaxis.set_minor_locator(mticker.NullLocator())
        self.yaxis.set_minor_locator(mticker.NullLocator())

    # Interactive manipulation

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

    def _get_view(self):
        """
        Save information required to reproduce the current view.

        Called before a view is changed, such as during a pan or zoom
        initiated by the user. You may return any information you deem
        necessary to describe the view.

        .. note::

            Intended to be overridden by new projection types, but if not, the
            default implementation saves the view limits. You *must* implement
            :meth:`_set_view` if you implement this method.
        """
        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()
        return (xmin, xmax, ymin, ymax)

    def _set_view(self, view):
        """
        Apply a previously saved view.

        Called when restoring a view, such as with the navigation buttons.

        .. note::

            Intended to be overridden by new projection types, but if not, the
            default implementation restores the view limits. You *must*
            implement :meth:`_get_view` if you implement this method.
        """
        xmin, xmax, ymin, ymax = view
        self.set_xlim((xmin, xmax))
        self.set_ylim((ymin, ymax))

    def _set_view_from_bbox(self, bbox, direction='in',
                            mode=None, twinx=False, twiny=False):
        """
        Update view from a selection bbox.

        .. note::

            Intended to be overridden by new projection types, but if not, the
            default implementation sets the view limits to the bbox directly.

        Parameters
        ----------

        bbox : tuple
            The selected bounding box limits, in *display* coordinates.

        direction : str
            The direction to apply the bounding box.
                * `'in'` - The bounding box describes the view directly, i.e.,
                           it zooms in.
                * `'out'` - The bounding box describes the size to make the
                            existing view, i.e., it zooms out.

        mode : str or None
            The selection mode, whether to apply the bounding box in only the
            `'x'` direction, `'y'` direction or both (`None`).

        twinx : bool
            Whether this axis is twinned in the *x*-direction.

        twiny : bool
            Whether this axis is twinned in the *y*-direction.
        """

        lastx, lasty, x, y = bbox

        # zoom to rect
        inverse = self.transData.inverted()
        lastx, lasty = inverse.transform_point((lastx, lasty))
        x, y = inverse.transform_point((x, y))
        Xmin, Xmax = self.get_xlim()
        Ymin, Ymax = self.get_ylim()

        if twinx:
            x0, x1 = Xmin, Xmax
        else:
            if Xmin < Xmax:
                if x < lastx:
                    x0, x1 = x, lastx
                else:
                    x0, x1 = lastx, x
                if x0 < Xmin:
                    x0 = Xmin
                if x1 > Xmax:
                    x1 = Xmax
            else:
                if x > lastx:
                    x0, x1 = x, lastx
                else:
                    x0, x1 = lastx, x
                if x0 > Xmin:
                    x0 = Xmin
                if x1 < Xmax:
                    x1 = Xmax

        if twiny:
            y0, y1 = Ymin, Ymax
        else:
            if Ymin < Ymax:
                if y < lasty:
                    y0, y1 = y, lasty
                else:
                    y0, y1 = lasty, y
                if y0 < Ymin:
                    y0 = Ymin
                if y1 > Ymax:
                    y1 = Ymax
            else:
                if y > lasty:
                    y0, y1 = y, lasty
                else:
                    y0, y1 = lasty, y
                if y0 > Ymin:
                    y0 = Ymin
                if y1 < Ymax:
                    y1 = Ymax

        if direction == 'in':
            if mode == 'x':
                self.set_xlim((x0, x1))
            elif mode == 'y':
                self.set_ylim((y0, y1))
            else:
                self.set_xlim((x0, x1))
                self.set_ylim((y0, y1))
        elif direction == 'out':
            if self.get_xscale() == 'log':
                alpha = np.log(Xmax / Xmin) / np.log(x1 / x0)
                rx1 = pow(Xmin / x0, alpha) * Xmin
                rx2 = pow(Xmax / x0, alpha) * Xmin
            else:
                alpha = (Xmax - Xmin) / (x1 - x0)
                rx1 = alpha * (Xmin - x0) + Xmin
                rx2 = alpha * (Xmax - x0) + Xmin
            if self.get_yscale() == 'log':
                alpha = np.log(Ymax / Ymin) / np.log(y1 / y0)
                ry1 = pow(Ymin / y0, alpha) * Ymin
                ry2 = pow(Ymax / y0, alpha) * Ymin
            else:
                alpha = (Ymax - Ymin) / (y1 - y0)
                ry1 = alpha * (Ymin - y0) + Ymin
                ry2 = alpha * (Ymax - y0) + Ymin

            if mode == 'x':
                self.set_xlim((rx1, rx2))
            elif mode == 'y':
                self.set_ylim((ry1, ry2))
            else:
                self.set_xlim((rx1, rx2))
                self.set_ylim((ry1, ry2))

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
            y=y)

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
        children.extend(self.collections)
        children.extend(self.patches)
        children.extend(self.lines)
        children.extend(self.texts)
        children.extend(self.artists)
        children.extend(six.itervalues(self.spines))
        children.append(self.xaxis)
        children.append(self.yaxis)
        children.append(self.title)
        children.append(self._left_title)
        children.append(self._right_title)
        children.extend(self.tables)
        children.extend(self.images)
        if self.legend_ is not None:
            children.append(self.legend_)
        children.append(self.patch)
        return children

    def contains(self, mouseevent):
        """
        Test whether the mouse event occured in the axes.

        Returns *True* / *False*, {}
        """
        if six.callable(self._contains):
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

        for child in self.get_children():
            if isinstance(child, OffsetBox) and child.get_visible():
                bb.append(child.get_window_extent(renderer))

        _bbox = mtransforms.Bbox.union(
            [b for b in bb if b.width != 0 or b.height != 0])

        return _bbox

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
        ax2 = self._make_twin_axes(sharex=self)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_offset_position('right')
        self.yaxis.tick_left()
        ax2.xaxis.set_visible(False)
        ax2.patch.set_visible(False)
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

        ax2 = self._make_twin_axes(sharey=self)
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        self.xaxis.tick_bottom()
        ax2.yaxis.set_visible(False)
        ax2.patch.set_visible(False)
        return ax2

    def get_shared_x_axes(self):
        'Return a copy of the shared axes Grouper object for x axes'
        return self._shared_x_axes

    def get_shared_y_axes(self):
        'Return a copy of the shared axes Grouper object for y axes'
        return self._shared_y_axes
