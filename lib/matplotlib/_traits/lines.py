"""
matplotlib.lines.Line2D refactored in traitlets
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import warnings

import numpy as np

from . import artist, colors as mcolors, docstring, rcParams
from .artist import Artist, allow_rasterization

# import matplotlib._traits.artist as artist
# from matplotlib._traits.artist import Artist, allow_rasterization

from .cbook import (
    iterable, is_numlike, ls_mapper, ls_mapper_r, STEP_LOOKUP_MAP)
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, TransformedPath, IdentityTransform

# Imported here for backward compatibility, even though they don't
# really belong.
from numpy import ma
from . import _path
from .markers import (
    CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN,
    CARETLEFTBASE, CARETRIGHTBASE, CARETUPBASE, CARETDOWNBASE,
    TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN)

import matplotlib.lines.Line2D as b_Line2D
from .artist import Artist


class Line2D(Artist):
        """
        A line - the line can have both a solid linestyle connecting all
        the vertices, and a marker at each vertex.  Additionally, the
        drawing of the solid line is influenced by the drawstyle, e.g., one
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

        # drawStyles should now be deprecated.
        drawStyles = {}
        drawStyles.update(_drawStyles_l)
        drawStyles.update(_drawStyles_s)
        # Need a list ordered with long names first:
        drawStyleKeys = list(_drawStyles_l) + list(_drawStyles_s)

        # Referenced here to maintain API.  These are defined in
        # MarkerStyle
        markers = MarkerStyle.markers
        filled_markers = MarkerStyle.filled_markers
        fillStyles = MarkerStyle.fillstyles

        zorder = 2
        validCap = ('butt', 'round', 'projecting')
        validJoin = ('miter', 'round', 'bevel')

        # not sure how much this will have to be refactored
        def __str__(self):
            if self._label != "":
                return "Line2D(%s)" % (self._label)
            elif self._x is None:
                return "Line2D()"
            elif len(self._x) > 3:
                return "Line2D((%g,%g),(%g,%g),...,(%g,%g))"\
                    % (self._x[0], self._y[0], self._x[0],
                       self._y[0], self._x[-1], self._y[-1])
            else:
                return "Line2D(%s)"\
                    % (",".join(["(%g,%g)" % (x, y) for x, y
                                 in zip(self._x, self._y)]))
                                 
        #this will have to be edited according to the traits
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
                     fillstyle=None,
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

            if isinstance(linestyle, six.string_types):
                ds, ls = self._split_drawstyle_linestyle(linestyle)
                if ds is not None and drawstyle is not None and ds != drawstyle:
                    raise ValueError("Inconsistent drawstyle ({0!r}) and "
                                     "linestyle ({1!r})".format(drawstyle,
                                                                linestyle)
                                     )
                linestyle = ls

                if ds is not None:
                    drawstyle = ds

            if drawstyle is None:
                drawstyle = 'default'

            self._dashcapstyle = None
            self._dashjoinstyle = None
            self._solidjoinstyle = None
            self._solidcapstyle = None
            self.set_dash_capstyle(dash_capstyle)
            self.set_dash_joinstyle(dash_joinstyle)
            self.set_solid_capstyle(solid_capstyle)
            self.set_solid_joinstyle(solid_joinstyle)

            self._linestyles = None
            self._drawstyle = None
            self._linewidth = linewidth

            # scaled dash + offset
            self._dashSeq = None
            self._dashOffset = 0
            # unscaled dash + offset
            # this is needed scaling the dash pattern by linewidth
            self._us_dashSeq = None
            self._us_dashOffset = 0

            self.set_linestyle(linestyle)
            self.set_drawstyle(drawstyle)
            self.set_linewidth(linewidth)

            self._color = None
            self.set_color(color)
            self._marker = MarkerStyle(marker, fillstyle)

            self._markevery = None
            self._markersize = None
            self._antialiased = None

            self.set_markevery(markevery)
            self.set_antialiased(antialiased)
            self.set_markersize(markersize)

            self._markeredgecolor = None
            self._markeredgewidth = None
            self._markerfacecolor = None
            self._markerfacecoloralt = None

            self.set_markerfacecolor(markerfacecolor)
            self.set_markerfacecoloralt(markerfacecoloralt)
            self.set_markeredgecolor(markeredgecolor)
            self.set_markeredgewidth(markeredgewidth)

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
            self._x = None
            self._y = None
            self._xy = None
            self._path = None
            self._transformed_path = None
            self._subslice = False
            self._x_filled = None  # used in subslicing; only x is needed

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
            if callable(self._contains):
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
            with np.errstate(all='ignore'):
                # Check for collision
                if self._linestyle in ['None', None]:
                    # If no line, return the nearby point(s)
                    d = (xt - mouseevent.x) ** 2 + (yt - mouseevent.y) ** 2
                    ind, = np.nonzero(np.less_equal(d, pixels ** 2))
                else:
                    # If line, return the nearby segment(s)
                    ind = segment_hits(mouseevent.x, mouseevent.y, xt, yt, pixels)
                    if self._drawstyle.startswith("steps"):
                        ind //= 2

            ind += self.ind_offset

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
            self.stale = True

        def set_markevery(self, every):
            """Set the markevery property to subsample the plot when using markers.

            e.g., if `every=5`, every 5-th marker will be plotted.

            ACCEPTS: [None | int | length-2 tuple of int | slice |
            list/array of int | float | length-2 tuple of float]

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
            if self._markevery != every:
                self.stale = True
            self._markevery = every

        def get_markevery(self):
            """return the markevery setting"""
            return self._markevery

        def set_picker(self, p):
            """Sets the event picker details for the line.

            ACCEPTS: float distance in points or callable pick function
            ``fn(artist, event)``
            """
            if callable(p):
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

        # @Artist.axes.setter
        def axes(self, ax):
            # call the set method from the base-class property
            Artist.axes.fset(self, ax)
            if ax is not None:
                # connect unit-related callbacks
                if ax.xaxis is not None:
                    self._xcid = ax.xaxis.callbacks.connect('units',
                                                            self.recache_always)
                if ax.yaxis is not None:
                    self._ycid = ax.yaxis.callbacks.connect('units',
                                                            self.recache_always)

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
                if isinstance(self._xorig, np.ma.MaskedArray):
                    x = np.ma.asarray(xconv, float).filled(np.nan)
                else:
                    x = np.asarray(xconv, float)
                x = x.ravel()
            else:
                x = self._x
            if always or self._invalidy:
                yconv = self.convert_yunits(self._yorig)
                if isinstance(self._yorig, np.ma.MaskedArray):
                    y = np.ma.asarray(yconv, float).filled(np.nan)
                else:
                    y = np.asarray(yconv, float)
                y = y.ravel()
            else:
                y = self._y

            if len(x) == 1 and len(y) > 1:
                x = x * np.ones(y.shape, float)
            if len(y) == 1 and len(x) > 1:
                y = y * np.ones(x.shape, float)

            if len(x) != len(y):
                raise RuntimeError('xdata and ydata must be the same length')

            self._xy = np.empty((len(x), 2), dtype=float)
            self._xy[:, 0] = x
            self._xy[:, 1] = y

            self._x = self._xy[:, 0]  # just a view
            self._y = self._xy[:, 1]  # just a view

            self._subslice = False
            if (self.axes and len(x) > 1000 and self._is_sorted(x) and
                    self.axes.name == 'rectilinear' and
                    self.axes.get_xscale() == 'linear' and
                    self._markevery is None and
                    self.get_clip_on() is True):
                self._subslice = True
                nanmask = np.isnan(x)
                if nanmask.any():
                    self._x_filled = self._x.copy()
                    indices = np.arange(len(x))
                    self._x_filled[nanmask] = np.interp(indices[nanmask],
                            indices[~nanmask], self._x[~nanmask])
                else:
                    self._x_filled = self._x

            if self._path is not None:
                interpolation_steps = self._path._interpolation_steps
            else:
                interpolation_steps = 1
            xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy.T)
            self._path = Path(np.asarray(xy).T,
                              _interpolation_steps=interpolation_steps)
            self._transformed_path = None
            self._invalidx = False
            self._invalidy = False

        def _transform_path(self, subslice=None):
            """
            Puts a TransformedPath instance at self._transformed_path;
            all invalidation of the transform is then handled by the
            TransformedPath instance.
            """
            # Masked arrays are now handled by the Path class itself
            if subslice is not None:
                xy = STEP_LOOKUP_MAP[self._drawstyle](*self._xy[subslice, :].T)
                _path = Path(np.asarray(xy).T,
                             _interpolation_steps=self._path._interpolation_steps)
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
            self.stale = True

        def _is_sorted(self, x):
            """return True if x is sorted in ascending order"""
            # We don't handle the monotonically decreasing case.
            return _path.is_sorted(x)

        @allow_rasterization
        def draw(self, renderer):
            """draw the Line with `renderer` unless visibility is False"""
            if not self.get_visible():
                return

            if self._invalidy or self._invalidx:
                self.recache()
            self.ind_offset = 0  # Needed for contains() method.
            if self._subslice and self.axes:
                x0, x1 = self.axes.get_xbound()
                i0, = self._x_filled.searchsorted([x0], 'left')
                i1, = self._x_filled.searchsorted([x1], 'right')
                subslice = slice(max(i0 - 1, 0), i1 + 1)
                self.ind_offset = subslice.start
                self._transform_path(subslice)

            transf_path = self._get_transformed_path()

            if self.get_path_effects():
                from matplotlib.patheffects import PathEffectRenderer
                renderer = PathEffectRenderer(self.get_path_effects(), renderer)

            renderer.open_group('line2d', self.get_gid())
            if self._lineStyles[self._linestyle] != '_draw_nothing':
                tpath, affine = transf_path.get_transformed_path_and_affine()
                if len(tpath.vertices):
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

                    gc.set_dashes(self._dashOffset, self._dashSeq)
                    renderer.draw_path(gc, tpath, affine.frozen())
                    gc.restore()

            if self._marker and self._markersize > 0:
                gc = renderer.new_gc()
                self._set_gc_clip(gc)
                rgbaFace = self._get_rgba_face()
                rgbaFaceAlt = self._get_rgba_face(alt=True)
                edgecolor = self.get_markeredgecolor()
                if (isinstance(edgecolor, six.string_types)
                        and edgecolor.lower() == 'none'):
                    gc.set_linewidth(0)
                    gc.set_foreground(rgbaFace, isRGBA=True)
                else:
                    gc.set_foreground(edgecolor)
                    gc.set_linewidth(self._markeredgewidth)
                    mec = self._markeredgecolor
                    if (isinstance(mec, six.string_types) and mec == 'auto' and
                            rgbaFace is not None):
                        gc.set_alpha(rgbaFace[3])
                    else:
                        gc.set_alpha(self.get_alpha())

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

                    if (isinstance(marker.get_marker(), six.string_types) and
                            marker.get_marker() == ','):
                        gc.set_linewidth(0)
                    else:
                        # Don't scale for pixels, and don't stroke them
                        marker_trans = marker_trans.scale(w)

                    renderer.draw_markers(gc, marker_path, marker_trans,
                                          subsampled, affine.frozen(),
                                          rgbaFace)

                    alt_marker_path = marker.get_alt_path()
                    if alt_marker_path:
                        alt_marker_trans = marker.get_alt_transform()
                        alt_marker_trans = alt_marker_trans.scale(w)
                        if (isinstance(mec, six.string_types) and mec == 'auto' and
                                rgbaFaceAlt is not None):
                            gc.set_alpha(rgbaFaceAlt[3])
                        else:
                            gc.set_alpha(self.get_alpha())

                        renderer.draw_markers(
                                gc, alt_marker_path, alt_marker_trans, subsampled,
                                affine.frozen(), rgbaFaceAlt)

                gc.restore()

            renderer.close_group('line2d')
            self.stale = False

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
            if isinstance(mec, six.string_types) and mec == 'auto':
                if rcParams['_internal.classic_mode']:
                    if self._marker.get_marker() in ('.', ','):
                        return self._color
                    if self._marker.is_filled() and self.get_fillstyle() != 'none':
                         return 'k'  # Bad hard-wired default...
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

            if (isinstance(fc, six.string_types) and fc.lower() == 'auto'):
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
            if self._antialiased != b:
                self.stale = True
            self._antialiased = b

        def set_color(self, color):
            """
            Set the color of the line

            ACCEPTS: any matplotlib color
            """
            self._color = color
            self.stale = True

        def set_drawstyle(self, drawstyle):
            """
            Set the drawstyle of the plot

            'default' connects the points with lines. The steps variants
            produce step-plots. 'steps' is equivalent to 'steps-pre' and
            is maintained for backward-compatibility.

            ACCEPTS: ['default' | 'steps' | 'steps-pre' | 'steps-mid' |
                      'steps-post']
            """
            if drawstyle is None:
                drawstyle = 'default'
            if drawstyle not in self.drawStyles:
                raise ValueError('Unrecognized drawstyle {!r}'.format(drawstyle))
            if self._drawstyle != drawstyle:
                self.stale = True
            self._drawstyle = drawstyle

        def set_linewidth(self, w):
            """
            Set the line width in points

            ACCEPTS: float value in points
            """
            w = float(w)

            if self._linewidth != w:
                self.stale = True
            self._linewidth = w
            # rescale the dashes + offset
            self._dashOffset, self._dashSeq = _scale_dashes(
                self._us_dashOffset, self._us_dashSeq, self._linewidth)

        def _split_drawstyle_linestyle(self, ls):
            '''Split drawstyle from linestyle string

            If `ls` is only a drawstyle default to returning a linestyle
            of '-'.

            Parameters
            ----------
            ls : str
                The linestyle to be processed

            Returns
            -------
            ret_ds : str or None
                If the linestyle string does not contain a drawstyle prefix
                return None, otherwise return it.

            ls : str
                The linestyle with the drawstyle (if any) stripped.
            '''
            ret_ds = None
            for ds in self.drawStyleKeys:  # long names are first in the list
                if ls.startswith(ds):
                    ret_ds = ds
                    if len(ls) > len(ds):
                        ls = ls[len(ds):]
                    else:
                        ls = '-'
                    break

            return ret_ds, ls

        def set_linestyle(self, ls):
            """
            Set the linestyle of the line (also accepts drawstyles,
            e.g., ``'steps--'``)


            ===========================   =================
            linestyle                     description
            ===========================   =================
            ``'-'`` or ``'solid'``        solid line
            ``'--'`` or  ``'dashed'``     dashed line
            ``'-.'`` or  ``'dashdot'``    dash-dotted line
            ``':'`` or ``'dotted'``       dotted line
            ``'None'``                    draw nothing
            ``' '``                       draw nothing
            ``''``                        draw nothing
            ===========================   =================

            'steps' is equivalent to 'steps-pre' and is maintained for
            backward-compatibility.

            Alternatively a dash tuple of the following form can be provided::

                (offset, onoffseq),

            where ``onoffseq`` is an even length tuple of on and off ink
            in points.


            ACCEPTS: ['solid' | 'dashed', 'dashdot', 'dotted' |
                       (offset, on-off-dash-seq) |
                       ``'-'`` | ``'--'`` | ``'-.'`` | ``':'`` | ``'None'`` |
                       ``' '`` | ``''``]

            .. seealso::

                :meth:`set_drawstyle`
                   To set the drawing style (stepping) of the plot.

            Parameters
            ----------
            ls : { ``'-'``,  ``'--'``, ``'-.'``, ``':'``} and more see description
                The line style.
            """
            if isinstance(ls, six.string_types):
                ds, ls = self._split_drawstyle_linestyle(ls)
                if ds is not None:
                    self.set_drawstyle(ds)

                if ls in [' ', '', 'none']:
                    ls = 'None'

                if ls not in self._lineStyles:
                    try:
                        ls = ls_mapper_r[ls]
                    except KeyError:
                        raise ValueError(("You passed in an invalid linestyle, "
                                          "`{0}`.  See "
                                          "docs of Line2D.set_linestyle for "
                                          "valid values.").format(ls))
                self._linestyle = ls
            else:
                self._linestyle = '--'

            # get the unscaled dashes
            self._us_dashOffset, self._us_dashSeq = _get_dash_pattern(ls)
            # compute the linewidth scaled dashes
            self._dashOffset, self._dashSeq = _scale_dashes(
                self._us_dashOffset, self._us_dashSeq, self._linewidth)

        @docstring.dedent_interpd
        def set_marker(self, marker):
            """
            Set the line marker

            ACCEPTS: :mod:`A valid marker style <matplotlib.markers>`

            Parameters
            ----------

            marker: marker style
                See `~matplotlib.markers` for full description of possible
                argument

            """
            self._marker.set_marker(marker)
            self.stale = True

        def set_markeredgecolor(self, ec):
            """
            Set the marker edge color

            ACCEPTS: any matplotlib color
            """
            if ec is None:
                ec = 'auto'
            if self._markeredgecolor is None or \
               np.any(self._markeredgecolor != ec):
                self.stale = True
            self._markeredgecolor = ec

        def set_markeredgewidth(self, ew):
            """
            Set the marker edge width in points

            ACCEPTS: float value in points
            """
            if ew is None:
                ew = rcParams['lines.markeredgewidth']
            if self._markeredgewidth != ew:
                self.stale = True
            self._markeredgewidth = ew

        def set_markerfacecolor(self, fc):
            """
            Set the marker face color.

            ACCEPTS: any matplotlib color
            """
            if fc is None:
                fc = 'auto'
            if np.any(self._markerfacecolor != fc):
                self.stale = True
            self._markerfacecolor = fc

        def set_markerfacecoloralt(self, fc):
            """
            Set the alternate marker face color.

            ACCEPTS: any matplotlib color
            """
            if fc is None:
                fc = 'auto'
            if np.any(self._markerfacecoloralt != fc):
                self.stale = True
            self._markerfacecoloralt = fc

        def set_markersize(self, sz):
            """
            Set the marker size in points

            ACCEPTS: float
            """
            sz = float(sz)
            if self._markersize != sz:
                self.stale = True
            self._markersize = sz

        def set_xdata(self, x):
            """
            Set the data np.array for x

            ACCEPTS: 1D array
            """
            self._xorig = x
            self._invalidx = True
            self.stale = True

        def set_ydata(self, y):
            """
            Set the data np.array for y

            ACCEPTS: 1D array
            """
            self._yorig = y
            self._invalidy = True
            self.stale = True

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
                self.set_linestyle((0, seq))

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
            self._us_dashSeq = other._us_dashSeq
            self._dashOffset = other._dashOffset
            self._us_dashOffset = other._us_dashOffset
            self._dashcapstyle = other._dashcapstyle
            self._dashjoinstyle = other._dashjoinstyle
            self._solidcapstyle = other._solidcapstyle
            self._solidjoinstyle = other._solidjoinstyle

            self._linestyle = other._linestyle
            self._marker = MarkerStyle(other._marker.get_marker(),
                                       other._marker.get_fillstyle())
            self._drawstyle = other._drawstyle

        def _get_rgba_face(self, alt=False):
            facecolor = self._get_markerfacecolor(alt=alt)
            if (isinstance(facecolor, six.string_types)
                    and facecolor.lower() == 'none'):
                rgbaFace = None
            else:
                rgbaFace = mcolors.to_rgba(facecolor, self._alpha)
            return rgbaFace

        def _get_rgba_ln_color(self, alt=False):
            return mcolors.to_rgba(self._color, self._alpha)

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
            if self._dashjoinstyle != s:
                self.stale = True
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

            if self._solidjoinstyle != s:
                self.stale = True
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
            if self._dashcapstyle != s:
                self.stale = True
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
            if self._solidcapstyle != s:
                self.stale = True
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


#for monkey patching
# b_Line2D = Line2D()
