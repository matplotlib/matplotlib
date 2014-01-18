"""
This module contains all the 2D line class which can draw with a
variety of line styles, markers and colors.
"""

# TODO: expose cap and join style attrs
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import warnings

import numpy as np
from numpy import ma
from matplotlib import verbose
from . import artist
from .artist import Artist
from .cbook import iterable, is_string_like, is_numlike, ls_mapper
from .colors import colorConverter
from .path import Path
from .transforms import Bbox, TransformedPath, IdentityTransform

from matplotlib import rcParams
from .artist import allow_rasterization
from matplotlib import docstring
from matplotlib.markers import MarkerStyle
# Imported here for backward compatibility, even though they don't
# really belong.
from matplotlib.markers import TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN
from matplotlib.markers import CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN


def segment_hits(cx, cy, x, y, radius):
    """
    Determine if any line segments are within radius of a
    point. Returns the list of line segments that are within that
    radius.
    """
    # Process single points specially
    if len(x) < 2:
        res, = np.nonzero((cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2)
        return res

    # We need to lop the last element off a lot.
    xr, yr = x[:-1], y[:-1]

    # Only look at line segments whose nearest point to C on the line
    # lies within the segment.
    dx, dy = x[1:] - xr, y[1:] - yr
    Lnorm_sq = dx ** 2 + dy ** 2  # Possibly want to eliminate Lnorm==0
    u = ((cx - xr) * dx + (cy - yr) * dy) / Lnorm_sq
    candidates = (u >= 0) & (u <= 1)
    #if any(candidates): print "candidates",xr[candidates]

    # Note that there is a little area near one side of each point
    # which will be near neither segment, and another which will
    # be near both, depending on the angle of the lines.  The
    # following radius test eliminates these ambiguities.
    point_hits = (cx - x) ** 2 + (cy - y) ** 2 <= radius ** 2
    #if any(point_hits): print "points",xr[candidates]
    candidates = candidates & ~(point_hits[:-1] | point_hits[1:])

    # For those candidates which remain, determine how far they lie away
    # from the line.
    px, py = xr + u * dx, yr + u * dy
    line_hits = (cx - px) ** 2 + (cy - py) ** 2 <= radius ** 2
    #if any(line_hits): print "lines",xr[candidates]
    line_hits = line_hits & candidates
    points, = point_hits.ravel().nonzero()
    lines, = line_hits.ravel().nonzero()
    #print points,lines
    return np.concatenate((points, lines))


def _mark_every_path(markevery, tpath, affine, ax_transform):
    """
    Helper function that sorts out how to deal the input
    `markevery` and returns the points where markers should be drawn.

    Takes in the `markevery` value and the line path and returns the
    sub-sampled path.
    """
    # pull out the two bits of data we want from the path
    codes, verts = tpath.codes, tpath.vertices

    def _slice_or_none(in_v, slc):
        '''
        Helper function to cope with `codes` being an
        ndarray or `None`
        '''
        if in_v is None:
            return None
        return in_v[slc]

    # if just a float, assume starting at 0.0 and make a tuple
    if isinstance(markevery, float):
        markevery = (0.0, markevery)
    # if just an int, assume starting at 0 and make a tuple
    elif isinstance(markevery, int):
        markevery = (0, markevery)

    if isinstance(markevery, tuple):
        if len(markevery) != 2:
            raise ValueError('`markevery` is a tuple but its '
                'len is not 2; '
                'markevery=%s' % (markevery,))
        start, step = markevery
        # if step is an int, old behavior
        if isinstance(step, int):
            #tuple of 2 int is for backwards compatibility,
            if not(isinstance(start, int)):
                raise ValueError('`markevery` is a tuple with '
                    'len 2 and second element is an int, but '
                    'the first element is not an int; '
                    'markevery=%s' % (markevery,))
            # just return, we are done here

            return Path(verts[slice(start, None, step)],
                        _slice_or_none(codes, slice(start, None, step)))

        elif isinstance(step, float):
            if not (isinstance(start, int) or
                    isinstance(start, float)):
                raise ValueError('`markevery` is a tuple with '
                    'len 2 and second element is a float, but '
                    'the first element is not a float or an '
                    'int; '
                    'markevery=%s' % (markevery,))
            #calc cumulative distance along path (in display
            # coords):
            disp_coords = affine.transform(tpath.vertices)
            delta = np.empty((len(disp_coords), 2),
                             dtype=float)
            delta[0, :] = 0.0
            delta[1:, :] = (disp_coords[1:, :] -
                                disp_coords[:-1, :])
            delta = np.sum(delta**2, axis=1)
            delta = np.sqrt(delta)
            delta = np.cumsum(delta)
            #calc distance between markers along path based on
            # the axes bounding box diagonal being a distance
            # of unity:
            scale = ax_transform.transform(
                np.array([[0, 0], [1, 1]]))
            scale = np.diff(scale, axis=0)
            scale = np.sum(scale**2)
            scale = np.sqrt(scale)
            marker_delta = np.arange(start * scale,
                                     delta[-1],
                                     step * scale)
            #find closest actual data point that is closest to
            # the theoretical distance along the path:
            inds = np.abs(delta[np.newaxis, :] -
                            marker_delta[:, np.newaxis])
            inds = inds.argmin(axis=1)
            inds = np.unique(inds)
            # return, we are done here
            return Path(verts[inds],
                        _slice_or_none(codes, inds))
        else:
            raise ValueError('`markevery` is a tuple with '
                'len 2, but its second element is not an int '
                'or a float; '
                'markevery=%s' % (markevery,))

    elif isinstance(markevery, slice):
        # mazol tov, it's already a slice, just return
        return Path(verts[markevery],
                    _slice_or_none(codes, markevery))

    elif iterable(markevery):
        #fancy indexing
        try:
            return Path(verts[markevery],
                    _slice_or_none(codes, markevery))

        except (ValueError, IndexError):
            raise ValueError('`markevery` is iterable but '
                'not a valid form of numpy fancy indexing; '
                'markevery=%s' % (markevery,))
    else:
        raise ValueError('Value of `markevery` is not '
            'recognized; '
            'markevery=%s' % (markevery,))


class Line2D(Artist):
    """
    A line - the line can have both a solid linestyle connecting all
    the vertices, and a marker at each vertex.  Additionally, the
    drawing of the solid line is influenced by the drawstyle, eg one
    can create "stepped" lines in various styles.


    """
    lineStyles = _lineStyles = {  # hidden names deprecated
        '-':    '_draw_solid',
        '--':   '_draw_dashed',
        '-.':   '_draw_dash_dot',
        ':':    '_draw_dotted',
        'None': '_draw_nothing',
        ' ':    '_draw_nothing',
        '':     '_draw_nothing',
    }

    _drawStyles_l = {
        'default':    '_draw_lines',
        'steps-mid':  '_draw_steps_mid',
        'steps-pre':  '_draw_steps_pre',
        'steps-post': '_draw_steps_post',
    }

    _drawStyles_s = {
        'steps': '_draw_steps_pre',
    }

    drawStyles = {}
    drawStyles.update(_drawStyles_l)
    drawStyles.update(_drawStyles_s)
    # Need a list ordered with long names first:
    drawStyleKeys = (list(six.iterkeys(_drawStyles_l)) +
                     list(six.iterkeys(_drawStyles_s)))

    # Referenced here to maintain API.  These are defined in
    # MarkerStyle
    markers = MarkerStyle.markers
    filled_markers = MarkerStyle.filled_markers
    fillStyles = MarkerStyle.fillstyles

    zorder = 2
    validCap = ('butt', 'round', 'projecting')
    validJoin = ('miter', 'round', 'bevel')

    def __str__(self):
        if self._label != "":
            return "Line2D(%s)" % (self._label)
        elif hasattr(self, '_x') and len(self._x) > 3:
            return "Line2D((%g,%g),(%g,%g),...,(%g,%g))"\
                % (self._x[0], self._y[0], self._x[0],
                   self._y[0], self._x[-1], self._y[-1])
        elif hasattr(self, '_x'):
            return "Line2D(%s)"\
                % (",".join(["(%g,%g)" % (x, y) for x, y
                             in zip(self._x, self._y)]))
        else:
            return "Line2D()"

    def __init__(self, xdata, ydata,
                 linewidth=None,  # all Nones default to rc
                 linestyle=None,
                 color=None,
                 marker=None,
                 markersize=None,
                 markeredgewidth=None,
                 markeredgecolor=None,
                 markerfacecolor=None,
                 markerfacecoloralt='none',
                 fillstyle='full',
                 antialiased=None,
                 dash_capstyle=None,
                 solid_capstyle=None,
                 dash_joinstyle=None,
                 solid_joinstyle=None,
                 pickradius=5,
                 drawstyle=None,
                 markevery=None,
                 **kwargs
                 ):
        """
        Create a :class:`~matplotlib.lines.Line2D` instance with *x*
        and *y* data in sequences *xdata*, *ydata*.

        The kwargs are :class:`~matplotlib.lines.Line2D` properties:

        %(Line2D)s

        See :meth:`set_linestyle` for a decription of the line styles,
        :meth:`set_marker` for a description of the markers, and
        :meth:`set_drawstyle` for a description of the draw styles.

        """
        Artist.__init__(self)

        #convert sequences to numpy arrays
        if not iterable(xdata):
            raise RuntimeError('xdata must be a sequence')
        if not iterable(ydata):
            raise RuntimeError('ydata must be a sequence')

        if linewidth is None:
            linewidth = rcParams['lines.linewidth']

        if linestyle is None:
            linestyle = rcParams['lines.linestyle']
        if marker is None:
            marker = rcParams['lines.marker']
        if color is None:
            color = rcParams['lines.color']

        if markersize is None:
            markersize = rcParams['lines.markersize']
        if antialiased is None:
            antialiased = rcParams['lines.antialiased']
        if dash_capstyle is None:
            dash_capstyle = rcParams['lines.dash_capstyle']
        if dash_joinstyle is None:
            dash_joinstyle = rcParams['lines.dash_joinstyle']
        if solid_capstyle is None:
            solid_capstyle = rcParams['lines.solid_capstyle']
        if solid_joinstyle is None:
            solid_joinstyle = rcParams['lines.solid_joinstyle']

        if drawstyle is None:
            drawstyle = 'default'

        self.set_dash_capstyle(dash_capstyle)
        self.set_dash_joinstyle(dash_joinstyle)
        self.set_solid_capstyle(solid_capstyle)
        self.set_solid_joinstyle(solid_joinstyle)

        self.set_linestyle(linestyle)
        self.set_drawstyle(drawstyle)
        self.set_linewidth(linewidth)
        self.set_color(color)
        self._marker = MarkerStyle()
        self.set_marker(marker)
        self.set_markevery(markevery)
        self.set_antialiased(antialiased)
        self.set_markersize(markersize)
        self._dashSeq = None

        self.set_markerfacecolor(markerfacecolor)
        self.set_markerfacecoloralt(markerfacecoloralt)
        self.set_markeredgecolor(markeredgecolor)
        self.set_markeredgewidth(markeredgewidth)
        self.set_fillstyle(fillstyle)

        self.verticalOffset = None

        # update kwargs before updating data to give the caller a
        # chance to init axes (and hence unit support)
        self.update(kwargs)
        self.pickradius = pickradius
        self.ind_offset = 0
        if is_numlike(self._picker):
            self.pickradius = self._picker

        self._xorig = np.asarray([])
        self._yorig = np.asarray([])
        self._invalidx = True
        self._invalidy = True
        self.set_data(xdata, ydata)

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred on the line.  The pick
        radius determines the precision of the location test (usually
        within five points of the value).  Use
        :meth:`~matplotlib.lines.Line2D.get_pickradius` or
        :meth:`~matplotlib.lines.Line2D.set_pickradius` to view or
        modify it.

        Returns *True* if any values are within the radius along with
        ``{'ind': pointlist}``, where *pointlist* is the set of points
        within the radius.

        TODO: sort returned indices by distance
        """
        if six.callable(self._contains):
            return self._contains(self, mouseevent)

        if not is_numlike(self.pickradius):
            raise ValueError("pick radius should be a distance")

        # Make sure we have data to plot
        if self._invalidy or self._invalidx:
            self.recache()
        if len(self._xy) == 0:
            return False, {}

        # Convert points to pixels
        transformed_path = self._get_transformed_path()
        path, affine = transformed_path.get_transformed_path_and_affine()
        path = affine.transform_path(path)
        xy = path.vertices
        xt = xy[:, 0]
        yt = xy[:, 1]

        # Convert pick radius from points to pixels
        if self.figure is None:
            warnings.warn('no figure set when check if mouse is on line')
            pixels = self.pickradius
        else:
            pixels = self.figure.dpi / 72. * self.pickradius

        # the math involved in checking for containment (here and inside of
        # segment_hits) assumes that it is OK to overflow.  In case the
        # application has set the error flags such that an exception is raised
        # on overflow, we temporarily set the appropriate error flags here and
        # set them back when we are finished.
        olderrflags = np.seterr(all='ignore')
        try:
            # Check for collision
            if self._linestyle in ['None', None]:
                # If no line, return the nearby point(s)
                d = (xt - mouseevent.x) ** 2 + (yt - mouseevent.y) ** 2
                ind, = np.nonzero(np.less_equal(d, pixels ** 2))
            else:
                # If line, return the nearby segment(s)
                ind = segment_hits(mouseevent.x, mouseevent.y, xt, yt, pixels)
        finally:
            np.seterr(**olderrflags)

        ind += self.ind_offset

        # Debugging message
        if False and self._label != '':
            print("Checking line", self._label,
                  "at", mouseevent.x, mouseevent.y)
            print('xt', xt)
            print('yt', yt)
            #print 'dx,dy', (xt-mouseevent.x)**2., (yt-mouseevent.y)**2.
            print('ind', ind)

        # Return the point(s) within radius
        return len(ind) > 0, dict(ind=ind)

    def get_pickradius(self):
        """return the pick radius used for containment tests"""
        return self.pickradius

    def set_pickradius(self, d):
        """Sets the pick radius used for containment tests

        ACCEPTS: float distance in points
        """
        self.pickradius = d

    def get_fillstyle(self):
        """
        return the marker fillstyle
        """
        return self._marker.get_fillstyle()

    def set_fillstyle(self, fs):
        """
        Set the marker fill style; 'full' means fill the whole marker.
        'none' means no filling; other options are for half-filled markers.

        ACCEPTS: ['full' | 'left' | 'right' | 'bottom' | 'top' | 'none']
        """
        self._marker.set_fillstyle(fs)

    def set_markevery(self, every):
        """Set the markevery property to subsample the plot when using markers.

        e.g., if `every=5`, every 5-th marker will be plotted.

        Parameters
        ----------
        every: None | int | length-2 tuple of int | slice | list/array of int |
        float | length-2 tuple of float
            Which markers to plot.

            - every=None, every point will be plotted.
            - every=N, every N-th marker will be plotted starting with
              marker 0.
            - every=(start, N), every N-th marker, starting at point
              start, will be plotted.
            - every=slice(start, end, N), every N-th marker, starting at
              point start, upto but not including point end, will be plotted.
            - every=[i, j, m, n], only markers at points i, j, m, and n
              will be plotted.
            - every=0.1, (i.e. a float) then markers will be spaced at
              approximately equal distances along the line; the distance
              along the line between markers is determined by multiplying the
              display-coordinate distance of the axes bounding-box diagonal
              by the value of every.
            - every=(0.5, 0.1) (i.e. a length-2 tuple of float), the
              same functionality as every=0.1 is exhibited but the first
              marker will be 0.5 multiplied by the
              display-cordinate-diagonal-distance along the line.

        Notes
        -----
        Setting the markevery property will only show markers at actual data
        points.  When using float arguments to set the markevery property
        on irregularly spaced data, the markers will likely not appear evenly
        spaced because the actual data points do not coincide with the
        theoretical spacing between markers.

        When using a start offset to specify the first marker, the offset will
        be from the first data point which may be different from the first
        the visible data point if the plot is zoomed in.

        If zooming in on a plot when using float arguments then the actual
        data points that have markers will change because the distance between
        markers is always determined from the display-coordinates
        axes-bounding-box-diagonal regardless of the actual axes data limits.

        """

        self._markevery = every

    def get_markevery(self):
        """return the markevery setting"""
        return self._markevery

    def set_picker(self, p):
        """Sets the event picker details for the line.

        ACCEPTS: float distance in points or callable pick function
        ``fn(artist, event)``
        """
        if six.callable(p):
            self._contains = p
        else:
            self.pickradius = p
        self._picker = p

    def get_window_extent(self, renderer):
        bbox = Bbox([[0, 0], [0, 0]])
        trans_data_to_xy = self.get_transform().transform
        bbox.update_from_data_xy(trans_data_to_xy(self.get_xydata()),
                                 ignore=True)
        # correct for marker size, if any
        if self._marker:
            ms = (self._markersize / 72.0 * self.figure.dpi) * 0.5
            bbox = bbox.padded(ms)
        return bbox

    def set_axes(self, ax):
        Artist.set_axes(self, ax)
        if ax.xaxis is not None:
            self._xcid = ax.xaxis.callbacks.connect('units',
                                                    self.recache_always)
        if ax.yaxis is not None:
            self._ycid = ax.yaxis.callbacks.connect('units',
                                                    self.recache_always)
    set_axes.__doc__ = Artist.set_axes.__doc__

    def set_data(self, *args):
        """
        Set the x and y data

        ACCEPTS: 2D array (rows are x, y) or two 1D arrays
        """
        if len(args) == 1:
            x, y = args[0]
        else:
            x, y = args

        self.set_xdata(x)
        self.set_ydata(y)

    def recache_always(self):
        self.recache(always=True)

    def recache(self, always=False):
        if always or self._invalidx:
            xconv = self.convert_xunits(self._xorig)
            if ma.isMaskedArray(self._xorig):
                x = ma.asarray(xconv, np.float_)
            else:
                x = np.asarray(xconv, np.float_)
            x = x.ravel()
        else:
            x = self._x
        if always or self._invalidy:
            yconv = self.convert_yunits(self._yorig)
            if ma.isMaskedArray(self._yorig):
                y = ma.asarray(yconv, np.float_)
            else:
                y = np.asarray(yconv, np.float_)
            y = y.ravel()
        else:
            y = self._y

        if len(x) == 1 and len(y) > 1:
            x = x * np.ones(y.shape, np.float_)
        if len(y) == 1 and len(x) > 1:
            y = y * np.ones(x.shape, np.float_)

        if len(x) != len(y):
            raise RuntimeError('xdata and ydata must be the same length')

        x = x.reshape((len(x), 1))
        y = y.reshape((len(y), 1))

        if ma.isMaskedArray(x) or ma.isMaskedArray(y):
            self._xy = ma.concatenate((x, y), 1)
        else:
            self._xy = np.concatenate((x, y), 1)
        self._x = self._xy[:, 0]  # just a view
        self._y = self._xy[:, 1]  # just a view

        self._subslice = False
        if (self.axes and len(x) > 100 and self._is_sorted(x) and
                self.axes.name == 'rectilinear' and
                self.axes.get_xscale() == 'linear' and
                self._markevery is None and
                self.get_clip_on() is True):
            self._subslice = True
        if hasattr(self, '_path'):
            interpolation_steps = self._path._interpolation_steps
        else:
            interpolation_steps = 1
        self._path = Path(self._xy, None, interpolation_steps)
        self._transformed_path = None
        self._invalidx = False
        self._invalidy = False

    def _transform_path(self, subslice=None):
        """
        Puts a TransformedPath instance at self._transformed_path,
        all invalidation of the transform is then handled by the
        TransformedPath instance.
        """
        # Masked arrays are now handled by the Path class itself
        if subslice is not None:
            _path = Path(self._xy[subslice, :])
        else:
            _path = self._path
        self._transformed_path = TransformedPath(_path, self.get_transform())

    def _get_transformed_path(self):
        """
        Return the :class:`~matplotlib.transforms.TransformedPath` instance
        of this line.
        """
        if self._transformed_path is None:
            self._transform_path()
        return self._transformed_path

    def set_transform(self, t):
        """
        set the Transformation instance used by this artist

        ACCEPTS: a :class:`matplotlib.transforms.Transform` instance
        """
        Artist.set_transform(self, t)
        self._invalidx = True
        self._invalidy = True

    def _is_sorted(self, x):
        """return true if x is sorted"""
        if len(x) < 2:
            return 1
        return np.amin(x[1:] - x[0:-1]) >= 0

    @allow_rasterization
    def draw(self, renderer):
        """draw the Line with `renderer` unless visibility is False"""
        if not self.get_visible():
            return

        if self._invalidy or self._invalidx:
            self.recache()
        self.ind_offset = 0  # Needed for contains() method.
        if self._subslice and self.axes:
            # Need to handle monotonically decreasing case also...
            x0, x1 = self.axes.get_xbound()
            i0, = self._x.searchsorted([x0], 'left')
            i1, = self._x.searchsorted([x1], 'right')
            subslice = slice(max(i0 - 1, 0), i1 + 1)
            self.ind_offset = subslice.start
            self._transform_path(subslice)

        transf_path = self._get_transformed_path()

        if self.get_path_effects():
            from matplotlib.patheffects import PathEffectRenderer
            renderer = PathEffectRenderer(self.get_path_effects(), renderer)

        renderer.open_group('line2d', self.get_gid())
        gc = renderer.new_gc()
        self._set_gc_clip(gc)

        ln_color_rgba = self._get_rgba_ln_color()
        gc.set_foreground(ln_color_rgba, isRGBA=True)
        gc.set_alpha(ln_color_rgba[3])

        gc.set_antialiased(self._antialiased)
        gc.set_linewidth(self._linewidth)

        if self.is_dashed():
            cap = self._dashcapstyle
            join = self._dashjoinstyle
        else:
            cap = self._solidcapstyle
            join = self._solidjoinstyle
        gc.set_joinstyle(join)
        gc.set_capstyle(cap)
        gc.set_snap(self.get_snap())
        if self.get_sketch_params() is not None:
            gc.set_sketch_params(*self.get_sketch_params())

        funcname = self._lineStyles.get(self._linestyle, '_draw_nothing')
        if funcname != '_draw_nothing':
            tpath, affine = transf_path.get_transformed_path_and_affine()
            if len(tpath.vertices):
                self._lineFunc = getattr(self, funcname)
                funcname = self.drawStyles.get(self._drawstyle, '_draw_lines')
                drawFunc = getattr(self, funcname)
                drawFunc(renderer, gc, tpath, affine.frozen())

        if self._marker:
            gc = renderer.new_gc()
            self._set_gc_clip(gc)
            rgbaFace = self._get_rgba_face()
            rgbaFaceAlt = self._get_rgba_face(alt=True)
            edgecolor = self.get_markeredgecolor()
            if is_string_like(edgecolor) and edgecolor.lower() == 'none':
                gc.set_linewidth(0)
                gc.set_foreground(rgbaFace, isRGBA=True)
            else:
                gc.set_foreground(edgecolor)
                gc.set_linewidth(self._markeredgewidth)

            marker = self._marker
            tpath, affine = transf_path.get_transformed_points_and_affine()
            if len(tpath.vertices):
                # subsample the markers if markevery is not None
                markevery = self.get_markevery()
                if markevery is not None:
                    subsampled = _mark_every_path(markevery, tpath,
                                                  affine, self.axes.transAxes)
                else:
                    subsampled = tpath

                snap = marker.get_snap_threshold()
                if type(snap) == float:
                    snap = renderer.points_to_pixels(self._markersize) >= snap
                gc.set_snap(snap)
                gc.set_joinstyle(marker.get_joinstyle())
                gc.set_capstyle(marker.get_capstyle())
                marker_path = marker.get_path()
                marker_trans = marker.get_transform()
                w = renderer.points_to_pixels(self._markersize)
                if marker.get_marker() != ',':
                    # Don't scale for pixels, and don't stroke them
                    marker_trans = marker_trans.scale(w)
                else:
                    gc.set_linewidth(0)
                if rgbaFace is not None:
                    gc.set_alpha(rgbaFace[3])

                renderer.draw_markers(gc, marker_path, marker_trans,
                                      subsampled, affine.frozen(),
                                      rgbaFace)

                alt_marker_path = marker.get_alt_path()
                if alt_marker_path:
                    if rgbaFaceAlt is not None:
                        gc.set_alpha(rgbaFaceAlt[3])
                    alt_marker_trans = marker.get_alt_transform()
                    alt_marker_trans = alt_marker_trans.scale(w)

                    renderer.draw_markers(
                            gc, alt_marker_path, alt_marker_trans, subsampled,
                            affine.frozen(), rgbaFaceAlt)

            gc.restore()

        gc.restore()
        renderer.close_group('line2d')

    def get_antialiased(self):
        return self._antialiased

    def get_color(self):
        return self._color

    def get_drawstyle(self):
        return self._drawstyle

    def get_linestyle(self):
        return self._linestyle

    def get_linewidth(self):
        return self._linewidth

    def get_marker(self):
        return self._marker.get_marker()

    def get_markeredgecolor(self):
        mec = self._markeredgecolor
        if (is_string_like(mec) and mec == 'auto'):
            if self._marker.get_marker() in ('.', ','):
                return self._color
            if self._marker.is_filled() and self.get_fillstyle() != 'none':
                return 'k'  # Bad hard-wired default...
            else:
                return self._color
        else:
            return mec

    def get_markeredgewidth(self):
        return self._markeredgewidth

    def _get_markerfacecolor(self, alt=False):
        if alt:
            fc = self._markerfacecoloralt
        else:
            fc = self._markerfacecolor

        if (is_string_like(fc) and fc.lower() == 'auto'):
            if self.get_fillstyle() == 'none':
                return 'none'
            else:
                return self._color
        else:
            return fc

    def get_markerfacecolor(self):
        return self._get_markerfacecolor(alt=False)

    def get_markerfacecoloralt(self):
        return self._get_markerfacecolor(alt=True)

    def get_markersize(self):
        return self._markersize

    def get_data(self, orig=True):
        """
        Return the xdata, ydata.

        If *orig* is *True*, return the original data.
        """
        return self.get_xdata(orig=orig), self.get_ydata(orig=orig)

    def get_xdata(self, orig=True):
        """
        Return the xdata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
        if orig:
            return self._xorig
        if self._invalidx:
            self.recache()
        return self._x

    def get_ydata(self, orig=True):
        """
        Return the ydata.

        If *orig* is *True*, return the original data, else the
        processed data.
        """
        if orig:
            return self._yorig
        if self._invalidy:
            self.recache()
        return self._y

    def get_path(self):
        """
        Return the :class:`~matplotlib.path.Path` object associated
        with this line.
        """
        if self._invalidy or self._invalidx:
            self.recache()
        return self._path

    def get_xydata(self):
        """
        Return the *xy* data as a Nx2 numpy array.
        """
        if self._invalidy or self._invalidx:
            self.recache()
        return self._xy

    def set_antialiased(self, b):
        """
        True if line should be drawin with antialiased rendering

        ACCEPTS: [True | False]
        """
        self._antialiased = b

    def set_color(self, color):
        """
        Set the color of the line

        ACCEPTS: any matplotlib color
        """
        self._color = color

    def set_drawstyle(self, drawstyle):
        """
        Set the drawstyle of the plot

        'default' connects the points with lines. The steps variants
        produce step-plots. 'steps' is equivalent to 'steps-pre' and
        is maintained for backward-compatibility.

        ACCEPTS: ['default' | 'steps' | 'steps-pre' | 'steps-mid' |
                  'steps-post']
        """
        self._drawstyle = drawstyle

    def set_linewidth(self, w):
        """
        Set the line width in points

        ACCEPTS: float value in points
        """
        self._linewidth = w

    def set_linestyle(self, linestyle):
        """
        Set the linestyle of the line (also accepts drawstyles)


        ================    =================
        linestyle           description
        ================    =================
        ``'-'``             solid
        ``'--'``            dashed
        ``'-.'``            dash_dot
        ``':'``             dotted
        ``'None'``          draw nothing
        ``' '``             draw nothing
        ``''``              draw nothing
        ================    =================

        'steps' is equivalent to 'steps-pre' and is maintained for
        backward-compatibility.

        .. seealso::

            :meth:`set_drawstyle`
               To set the drawing style (stepping) of the plot.

        ACCEPTS: [``'-'`` | ``'--'`` | ``'-.'`` | ``':'`` | ``'None'`` |
                  ``' '`` | ``''``]
        and any drawstyle in combination with a linestyle, e.g., ``'steps--'``.
        """

        for ds in self.drawStyleKeys:  # long names are first in the list
            if linestyle.startswith(ds):
                self.set_drawstyle(ds)
                if len(linestyle) > len(ds):
                    linestyle = linestyle[len(ds):]
                else:
                    linestyle = '-'
                break

        if linestyle not in self._lineStyles:
            if linestyle in ls_mapper:
                linestyle = ls_mapper[linestyle]
            else:
                verbose.report('Unrecognized line style %s, %s' %
                               (linestyle, type(linestyle)))
        if linestyle in [' ', '']:
            linestyle = 'None'
        self._linestyle = linestyle

    @docstring.dedent_interpd
    def set_marker(self, marker):
        """
        Set the line marker

        Parameters
        -----------

        marker: marker style
            See `~matplotlib.markers` for full description of possible
            argument

        """
        self._marker.set_marker(marker)

    def set_markeredgecolor(self, ec):
        """
        Set the marker edge color

        ACCEPTS: any matplotlib color
        """
        if ec is None:
            ec = 'auto'
        self._markeredgecolor = ec

    def set_markeredgewidth(self, ew):
        """
        Set the marker edge width in points

        ACCEPTS: float value in points
        """
        if ew is None:
            ew = rcParams['lines.markeredgewidth']
        self._markeredgewidth = ew

    def set_markerfacecolor(self, fc):
        """
        Set the marker face color.

        ACCEPTS: any matplotlib color
        """
        if fc is None:
            fc = 'auto'

        self._markerfacecolor = fc

    def set_markerfacecoloralt(self, fc):
        """
        Set the alternate marker face color.

        ACCEPTS: any matplotlib color
        """
        if fc is None:
            fc = 'auto'

        self._markerfacecoloralt = fc

    def set_markersize(self, sz):
        """
        Set the marker size in points

        ACCEPTS: float
        """
        self._markersize = sz

    def set_xdata(self, x):
        """
        Set the data np.array for x

        ACCEPTS: 1D array
        """
        self._xorig = x
        self._invalidx = True

    def set_ydata(self, y):
        """
        Set the data np.array for y

        ACCEPTS: 1D array
        """
        self._yorig = y
        self._invalidy = True

    def set_dashes(self, seq):
        """
        Set the dash sequence, sequence of dashes with on off ink in
        points.  If seq is empty or if seq = (None, None), the
        linestyle will be set to solid.

        ACCEPTS: sequence of on/off ink in points
        """
        if seq == (None, None) or len(seq) == 0:
            self.set_linestyle('-')
        else:
            self.set_linestyle('--')
        self._dashSeq = seq  # TODO: offset ignored for now

    def _draw_lines(self, renderer, gc, path, trans):
        self._lineFunc(renderer, gc, path, trans)

    def _draw_steps_pre(self, renderer, gc, path, trans):
        vertices = self._xy
        steps = ma.zeros((2 * len(vertices) - 1, 2), np.float_)

        steps[0::2, 0], steps[1::2, 0] = vertices[:, 0], vertices[:-1, 0]
        steps[0::2, 1], steps[1:-1:2, 1] = vertices[:, 1], vertices[1:, 1]

        path = Path(steps)
        path = path.transformed(self.get_transform())
        self._lineFunc(renderer, gc, path, IdentityTransform())

    def _draw_steps_post(self, renderer, gc, path, trans):
        vertices = self._xy
        steps = ma.zeros((2 * len(vertices) - 1, 2), np.float_)

        steps[::2, 0], steps[1:-1:2, 0] = vertices[:, 0], vertices[1:, 0]
        steps[0::2, 1], steps[1::2, 1] = vertices[:, 1], vertices[:-1, 1]

        path = Path(steps)
        path = path.transformed(self.get_transform())
        self._lineFunc(renderer, gc, path, IdentityTransform())

    def _draw_steps_mid(self, renderer, gc, path, trans):
        vertices = self._xy
        steps = ma.zeros((2 * len(vertices), 2), np.float_)

        steps[1:-1:2, 0] = 0.5 * (vertices[:-1, 0] + vertices[1:, 0])
        steps[2::2, 0] = 0.5 * (vertices[:-1, 0] + vertices[1:, 0])
        steps[0, 0] = vertices[0, 0]
        steps[-1, 0] = vertices[-1, 0]
        steps[0::2, 1], steps[1::2, 1] = vertices[:, 1], vertices[:, 1]

        path = Path(steps)
        path = path.transformed(self.get_transform())
        self._lineFunc(renderer, gc, path, IdentityTransform())

    def _draw_solid(self, renderer, gc, path, trans):
        gc.set_linestyle('solid')
        renderer.draw_path(gc, path, trans)

    def _draw_dashed(self, renderer, gc, path, trans):
        gc.set_linestyle('dashed')
        if self._dashSeq is not None:
            gc.set_dashes(0, self._dashSeq)

        renderer.draw_path(gc, path, trans)

    def _draw_dash_dot(self, renderer, gc, path, trans):
        gc.set_linestyle('dashdot')
        renderer.draw_path(gc, path, trans)

    def _draw_dotted(self, renderer, gc, path, trans):
        gc.set_linestyle('dotted')
        renderer.draw_path(gc, path, trans)

    def update_from(self, other):
        """copy properties from other to self"""
        Artist.update_from(self, other)
        self._linestyle = other._linestyle
        self._linewidth = other._linewidth
        self._color = other._color
        self._markersize = other._markersize
        self._markerfacecolor = other._markerfacecolor
        self._markerfacecoloralt = other._markerfacecoloralt
        self._markeredgecolor = other._markeredgecolor
        self._markeredgewidth = other._markeredgewidth
        self._dashSeq = other._dashSeq
        self._dashcapstyle = other._dashcapstyle
        self._dashjoinstyle = other._dashjoinstyle
        self._solidcapstyle = other._solidcapstyle
        self._solidjoinstyle = other._solidjoinstyle

        self._linestyle = other._linestyle
        self._marker = MarkerStyle(other._marker.get_marker(),
                                   other._marker.get_fillstyle())
        self._drawstyle = other._drawstyle

    def _get_rgb_face(self, alt=False):
        facecolor = self._get_markerfacecolor(alt=alt)
        if is_string_like(facecolor) and facecolor.lower() == 'none':
            rgbFace = None
        else:
            rgbFace = colorConverter.to_rgb(facecolor)
        return rgbFace

    def _get_rgba_face(self, alt=False):
        facecolor = self._get_markerfacecolor(alt=alt)
        if is_string_like(facecolor) and facecolor.lower() == 'none':
            rgbaFace = None
        else:
            rgbaFace = colorConverter.to_rgba(facecolor, self._alpha)
        return rgbaFace

    def _get_rgba_ln_color(self, alt=False):
        return colorConverter.to_rgba(self._color, self._alpha)

    # some aliases....
    def set_aa(self, val):
        'alias for set_antialiased'
        self.set_antialiased(val)

    def set_c(self, val):
        'alias for set_color'
        self.set_color(val)

    def set_ls(self, val):
        """alias for set_linestyle"""
        self.set_linestyle(val)

    def set_lw(self, val):
        """alias for set_linewidth"""
        self.set_linewidth(val)

    def set_mec(self, val):
        """alias for set_markeredgecolor"""
        self.set_markeredgecolor(val)

    def set_mew(self, val):
        """alias for set_markeredgewidth"""
        self.set_markeredgewidth(val)

    def set_mfc(self, val):
        """alias for set_markerfacecolor"""
        self.set_markerfacecolor(val)

    def set_mfcalt(self, val):
        """alias for set_markerfacecoloralt"""
        self.set_markerfacecoloralt(val)

    def set_ms(self, val):
        """alias for set_markersize"""
        self.set_markersize(val)

    def get_aa(self):
        """alias for get_antialiased"""
        return self.get_antialiased()

    def get_c(self):
        """alias for get_color"""
        return self.get_color()

    def get_ls(self):
        """alias for get_linestyle"""
        return self.get_linestyle()

    def get_lw(self):
        """alias for get_linewidth"""
        return self.get_linewidth()

    def get_mec(self):
        """alias for get_markeredgecolor"""
        return self.get_markeredgecolor()

    def get_mew(self):
        """alias for get_markeredgewidth"""
        return self.get_markeredgewidth()

    def get_mfc(self):
        """alias for get_markerfacecolor"""
        return self.get_markerfacecolor()

    def get_mfcalt(self, alt=False):
        """alias for get_markerfacecoloralt"""
        return self.get_markerfacecoloralt()

    def get_ms(self):
        """alias for get_markersize"""
        return self.get_markersize()

    def set_dash_joinstyle(self, s):
        """
        Set the join style for dashed linestyles
        ACCEPTS: ['miter' | 'round' | 'bevel']
        """
        s = s.lower()
        if s not in self.validJoin:
            raise ValueError('set_dash_joinstyle passed "%s";\n' % (s,)
                             + 'valid joinstyles are %s' % (self.validJoin,))
        self._dashjoinstyle = s

    def set_solid_joinstyle(self, s):
        """
        Set the join style for solid linestyles
        ACCEPTS: ['miter' | 'round' | 'bevel']
        """
        s = s.lower()
        if s not in self.validJoin:
            raise ValueError('set_solid_joinstyle passed "%s";\n' % (s,)
                             + 'valid joinstyles are %s' % (self.validJoin,))
        self._solidjoinstyle = s

    def get_dash_joinstyle(self):
        """
        Get the join style for dashed linestyles
        """
        return self._dashjoinstyle

    def get_solid_joinstyle(self):
        """
        Get the join style for solid linestyles
        """
        return self._solidjoinstyle

    def set_dash_capstyle(self, s):
        """
        Set the cap style for dashed linestyles

        ACCEPTS: ['butt' | 'round' | 'projecting']
        """
        s = s.lower()
        if s not in self.validCap:
            raise ValueError('set_dash_capstyle passed "%s";\n' % (s,)
                             + 'valid capstyles are %s' % (self.validCap,))

        self._dashcapstyle = s

    def set_solid_capstyle(self, s):
        """
        Set the cap style for solid linestyles

        ACCEPTS: ['butt' | 'round' |  'projecting']
        """
        s = s.lower()
        if s not in self.validCap:
            raise ValueError('set_solid_capstyle passed "%s";\n' % (s,)
                             + 'valid capstyles are %s' % (self.validCap,))

        self._solidcapstyle = s

    def get_dash_capstyle(self):
        """
        Get the cap style for dashed linestyles
        """
        return self._dashcapstyle

    def get_solid_capstyle(self):
        """
        Get the cap style for solid linestyles
        """
        return self._solidcapstyle

    def is_dashed(self):
        'return True if line is dashstyle'
        return self._linestyle in ('--', '-.', ':')


class VertexSelector:
    """
    Manage the callbacks to maintain a list of selected vertices for
    :class:`matplotlib.lines.Line2D`. Derived classes should override
    :meth:`~matplotlib.lines.VertexSelector.process_selected` to do
    something with the picks.

    Here is an example which highlights the selected verts with red
    circles::

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.lines as lines

        class HighlightSelected(lines.VertexSelector):
            def __init__(self, line, fmt='ro', **kwargs):
                lines.VertexSelector.__init__(self, line)
                self.markers, = self.axes.plot([], [], fmt, **kwargs)

            def process_selected(self, ind, xs, ys):
                self.markers.set_data(xs, ys)
                self.canvas.draw()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x, y = np.random.rand(2, 30)
        line, = ax.plot(x, y, 'bs-', picker=5)

        selector = HighlightSelected(line)
        plt.show()

    """
    def __init__(self, line):
        """
        Initialize the class with a :class:`matplotlib.lines.Line2D`
        instance.  The line should already be added to some
        :class:`matplotlib.axes.Axes` instance and should have the
        picker property set.
        """
        if not hasattr(line, 'axes'):
            raise RuntimeError('You must first add the line to the Axes')

        if line.get_picker() is None:
            raise RuntimeError('You must first set the picker property '
                               'of the line')

        self.axes = line.axes
        self.line = line
        self.canvas = self.axes.figure.canvas
        self.cid = self.canvas.mpl_connect('pick_event', self.onpick)

        self.ind = set()

    def process_selected(self, ind, xs, ys):
        """
        Default "do nothing" implementation of the
        :meth:`process_selected` method.

        *ind* are the indices of the selected vertices.  *xs* and *ys*
        are the coordinates of the selected vertices.
        """
        pass

    def onpick(self, event):
        """When the line is picked, update the set of selected indicies."""
        if event.artist is not self.line:
            return

        for i in event.ind:
            if i in self.ind:
                self.ind.remove(i)
            else:
                self.ind.add(i)

        ind = list(self.ind)
        ind.sort()
        xdata, ydata = self.line.get_data()
        self.process_selected(ind, xdata[ind], ydata[ind])


lineStyles = Line2D._lineStyles
lineMarkers = MarkerStyle.markers
drawStyles = Line2D.drawStyles
fillStyles = MarkerStyle.fillstyles

docstring.interpd.update(Line2D=artist.kwdoc(Line2D))

# You can not set the docstring of an instancemethod,
# but you can on the underlying function.  Go figure.
docstring.dedent_interpd(Line2D.__init__)
