"""
This module contains all the 2D line class which can draw with a
variety of line styles, markers and colors.
"""

# TODO: expose cap and join style attrs
from __future__ import division

import numpy as np
from numpy import ma
from matplotlib import verbose
import artist
from artist import Artist
from cbook import iterable, is_string_like, is_numlike, ls_mapper, dedent,\
flatten
from colors import colorConverter
from path import Path
from transforms import Affine2D, Bbox, TransformedPath, IdentityTransform

from matplotlib import rcParams
# special-purpose marker identifiers:
(TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN,
    CARETLEFT, CARETRIGHT, CARETUP, CARETDOWN) = range(8)


# COVERAGE NOTE: Never called internally or from examples
def unmasked_index_ranges(mask, compressed = True):
    warnings.warn("Import this directly from matplotlib.cbook",
                   DeprecationWarning)
    # Warning added 2008/07/22
    from matplotlib.cbook import unmasked_index_ranges as _unmasked_index_ranges
    return _unmasked_index_ranges(mask, compressed=compressed)


def segment_hits(cx, cy, x, y, radius):
    """
    Determine if any line segments are within radius of a
    point. Returns the list of line segments that are within that
    radius.
    """
    # Process single points specially
    if len(x) < 2:
        res, = np.nonzero( (cx - x)**2 + (cy - y)**2 <= radius**2 )
        return res

    # We need to lop the last element off a lot.
    xr,yr = x[:-1],y[:-1]

    # Only look at line segments whose nearest point to C on the line
    # lies within the segment.
    dx,dy = x[1:]-xr, y[1:]-yr
    Lnorm_sq = dx**2+dy**2    # Possibly want to eliminate Lnorm==0
    u = ( (cx-xr)*dx + (cy-yr)*dy )/Lnorm_sq
    candidates = (u>=0) & (u<=1)
    #if any(candidates): print "candidates",xr[candidates]

    # Note that there is a little area near one side of each point
    # which will be near neither segment, and another which will
    # be near both, depending on the angle of the lines.  The
    # following radius test eliminates these ambiguities.
    point_hits = (cx - x)**2 + (cy - y)**2 <= radius**2
    #if any(point_hits): print "points",xr[candidates]
    candidates = candidates & ~(point_hits[:-1] | point_hits[1:])

    # For those candidates which remain, determine how far they lie away
    # from the line.
    px,py = xr+u*dx,yr+u*dy
    line_hits = (cx-px)**2 + (cy-py)**2 <= radius**2
    #if any(line_hits): print "lines",xr[candidates]
    line_hits = line_hits & candidates
    points, = point_hits.ravel().nonzero()
    lines, = line_hits.ravel().nonzero()
    #print points,lines
    return np.concatenate((points,lines))

class Line2D(Artist):
    """
    A line - the line can have both a solid linestyle connecting all
    the vertices, and a marker at each vertex.  Additionally, the
    drawing of the solid line is influenced by the drawstyle, eg one
    can create "stepped" lines in various styles.


    """
    lineStyles = _lineStyles =  { # hidden names deprecated
        '-'          : '_draw_solid',
        '--'         : '_draw_dashed',
        '-.'         : '_draw_dash_dot',
        ':'          : '_draw_dotted',
        'None'       : '_draw_nothing',
        ' '          : '_draw_nothing',
        ''           : '_draw_nothing',
    }

    _drawStyles_l = {
        'default'    : '_draw_lines',
        'steps-mid'  : '_draw_steps_mid',
        'steps-pre'  : '_draw_steps_pre',
        'steps-post' : '_draw_steps_post',
    }

    _drawStyles_s = {
        'steps'      : '_draw_steps_pre',
    }
    drawStyles = {}
    drawStyles.update(_drawStyles_l)
    drawStyles.update(_drawStyles_s)

    markers = _markers =  {  # hidden names deprecated
        '.'  : '_draw_point',
        ','  : '_draw_pixel',
        'o'  : '_draw_circle',
        'v'  : '_draw_triangle_down',
        '^'  : '_draw_triangle_up',
        '<'  : '_draw_triangle_left',
        '>'  : '_draw_triangle_right',
        '1'  : '_draw_tri_down',
        '2'  : '_draw_tri_up',
        '3'  : '_draw_tri_left',
        '4'  : '_draw_tri_right',
        's'  : '_draw_square',
        'p'  : '_draw_pentagon',
        '*'  : '_draw_star',
        'h'  : '_draw_hexagon1',
        'H'  : '_draw_hexagon2',
        '+'  : '_draw_plus',
        'x'  : '_draw_x',
        'D'  : '_draw_diamond',
        'd'  : '_draw_thin_diamond',
        '|'  : '_draw_vline',
        '_'  : '_draw_hline',
        TICKLEFT    : '_draw_tickleft',
        TICKRIGHT   : '_draw_tickright',
        TICKUP      : '_draw_tickup',
        TICKDOWN    : '_draw_tickdown',
        CARETLEFT   : '_draw_caretleft',
        CARETRIGHT  : '_draw_caretright',
        CARETUP     : '_draw_caretup',
        CARETDOWN   : '_draw_caretdown',
        'None' : '_draw_nothing',
        ' ' : '_draw_nothing',
        '' : '_draw_nothing',
    }

    filled_markers = ('o', '^', 'v', '<', '>',
                        's', 'd', 'D', 'h', 'H', 'p', '*')

    zorder = 2
    validCap = ('butt', 'round', 'projecting')
    validJoin =   ('miter', 'round', 'bevel')

    def __str__(self):
        if self._label != "":
            return "Line2D(%s)"%(self._label)
        elif hasattr(self, '_x') and len(self._x) > 3:
            return "Line2D((%g,%g),(%g,%g),...,(%g,%g))"\
                %(self._x[0],self._y[0],self._x[0],self._y[0],self._x[-1],self._y[-1])
        elif hasattr(self, '_x'):
            return "Line2D(%s)"\
                %(",".join(["(%g,%g)"%(x,y) for x,y in zip(self._x,self._y)]))
        else:
            return "Line2D()"

    def __init__(self, xdata, ydata,
                 linewidth       = None, # all Nones default to rc
                 linestyle       = None,
                 color           = None,
                 marker          = None,
                 markersize      = None,
                 markeredgewidth = None,
                 markeredgecolor = None,
                 markerfacecolor = None,
                 antialiased     = None,
                 dash_capstyle   = None,
                 solid_capstyle  = None,
                 dash_joinstyle  = None,
                 solid_joinstyle = None,
                 pickradius      = 5,
                 drawstyle       = None,
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

        if linewidth is None   : linewidth=rcParams['lines.linewidth']

        if linestyle is None   : linestyle=rcParams['lines.linestyle']
        if marker is None      : marker=rcParams['lines.marker']
        if color is None       : color=rcParams['lines.color']

        if markersize is None  : markersize=rcParams['lines.markersize']
        if antialiased is None : antialiased=rcParams['lines.antialiased']
        if dash_capstyle is None : dash_capstyle=rcParams['lines.dash_capstyle']
        if dash_joinstyle is None : dash_joinstyle=rcParams['lines.dash_joinstyle']
        if solid_capstyle is None : solid_capstyle=rcParams['lines.solid_capstyle']
        if solid_joinstyle is None : solid_joinstyle=rcParams['lines.solid_joinstyle']

        if drawstyle is None : drawstyle='default'

        self.set_dash_capstyle(dash_capstyle)
        self.set_dash_joinstyle(dash_joinstyle)
        self.set_solid_capstyle(solid_capstyle)
        self.set_solid_joinstyle(solid_joinstyle)


        self.set_linestyle(linestyle)
        self.set_drawstyle(drawstyle)
        self.set_linewidth(linewidth)
        self.set_color(color)
        self.set_marker(marker)
        self.set_antialiased(antialiased)
        self.set_markersize(markersize)
        self._dashSeq = None


        self.set_markerfacecolor(markerfacecolor)
        self.set_markeredgecolor(markeredgecolor)
        self.set_markeredgewidth(markeredgewidth)
        self._point_size_reduction = 0.5

        self.verticalOffset = None

        # update kwargs before updating data to give the caller a
        # chance to init axes (and hence unit support)
        self.update(kwargs)
        self.pickradius = pickradius
        if is_numlike(self._picker):
            self.pickradius = self._picker

        self._xorig = np.asarray([])
        self._yorig = np.asarray([])
        self._invalid = True
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
        if callable(self._contains): return self._contains(self,mouseevent)

        if not is_numlike(self.pickradius):
            raise ValueError,"pick radius should be a distance"

        # Make sure we have data to plot
        if self._invalid:
            self.recache()
        if len(self._xy)==0: return False,{}

        # Convert points to pixels
        path, affine = self._transformed_path.get_transformed_path_and_affine()
        path = affine.transform_path(path)
        xy = path.vertices
        xt = xy[:, 0]
        yt = xy[:, 1]

        # Convert pick radius from points to pixels
        if self.figure == None:
            warning.warn('no figure set when check if mouse is on line')
            pixels = self.pickradius
        else:
            pixels = self.figure.dpi/72. * self.pickradius

        # Check for collision
        if self._linestyle in ['None',None]:
            # If no line, return the nearby point(s)
            d = (xt-mouseevent.x)**2 + (yt-mouseevent.y)**2
            ind, = np.nonzero(np.less_equal(d, pixels**2))
        else:
            # If line, return the nearby segment(s)
            ind = segment_hits(mouseevent.x,mouseevent.y,xt,yt,pixels)

        # Debugging message
        if False and self._label != u'':
            print "Checking line",self._label,"at",mouseevent.x,mouseevent.y
            print 'xt', xt
            print 'yt', yt
            #print 'dx,dy', (xt-mouseevent.x)**2., (yt-mouseevent.y)**2.
            print 'ind',ind

        # Return the point(s) within radius
        return len(ind)>0,dict(ind=ind)

    def get_pickradius(self):
        'return the pick radius used for containment tests'
        return self.pickradius

    def setpickradius(self,d):
        """Sets the pick radius used for containment tests

        ACCEPTS: float distance in points
        """
        self.pickradius = d

    def set_picker(self,p):
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
        bbox = Bbox.unit()
        bbox.update_from_data_xy(self.get_transform().transform(self.get_xydata()),
                                 ignore=True)
        # correct for marker size, if any
        if self._marker is not None:
            ms = (self._markersize / 72.0 * self.figure.dpi) * 0.5
            bbox = bbox.padded(ms)
        return bbox

    def set_axes(self, ax):
        Artist.set_axes(self, ax)
        if ax.xaxis is not None:
            self._xcid = ax.xaxis.callbacks.connect('units', self.recache)
        if ax.yaxis is not None:
            self._ycid = ax.yaxis.callbacks.connect('units', self.recache)
    set_axes.__doc__ = Artist.set_axes.__doc__

    def set_data(self, *args):
        """
        Set the x and y data

        ACCEPTS: 2D array
        """
        if len(args)==1:
            x, y = args[0]
        else:
            x, y = args

        not_masked = 0
        if not ma.isMaskedArray(x):
            x = np.asarray(x)
            not_masked += 1
        if not ma.isMaskedArray(y):
            y = np.asarray(y)
            not_masked += 1

        if (not_masked < 2 or
            (x is not self._xorig and
             (x.shape != self._xorig.shape or np.any(x != self._xorig))) or
            (y is not self._yorig and
              (y.shape != self._yorig.shape or np.any(y != self._yorig)))):
            self._xorig = x
            self._yorig = y
            self._invalid = True

    def recache(self):
        #if self.axes is None: print 'recache no axes'
        #else: print 'recache units', self.axes.xaxis.units, self.axes.yaxis.units
        if ma.isMaskedArray(self._xorig) or ma.isMaskedArray(self._yorig):
            x = ma.asarray(self.convert_xunits(self._xorig), float)
            y = ma.asarray(self.convert_yunits(self._yorig), float)
            x = ma.ravel(x)
            y = ma.ravel(y)
        else:
            x = np.asarray(self.convert_xunits(self._xorig), float)
            y = np.asarray(self.convert_yunits(self._yorig), float)
            x = np.ravel(x)
            y = np.ravel(y)

        if len(x)==1 and len(y)>1:
            x = x * np.ones(y.shape, float)
        if len(y)==1 and len(x)>1:
            y = y * np.ones(x.shape, float)

        if len(x) != len(y):
            raise RuntimeError('xdata and ydata must be the same length')

        x = x.reshape((len(x), 1))
        y = y.reshape((len(y), 1))

        if ma.isMaskedArray(x) or ma.isMaskedArray(y):
            self._xy = ma.concatenate((x, y), 1)
        else:
            self._xy = np.concatenate((x, y), 1)
        self._x = self._xy[:, 0] # just a view
        self._y = self._xy[:, 1] # just a view

        # Masked arrays are now handled by the Path class itself
        self._path = Path(self._xy)
        self._transformed_path = TransformedPath(self._path, self.get_transform())

        self._invalid = False

    def set_transform(self, t):
        """
        set the Transformation instance used by this artist

        ACCEPTS: a :class:`matplotlib.transforms.Transform` instance
        """
        Artist.set_transform(self, t)
        self._invalid = True
        # self._transformed_path = TransformedPath(self._path, self.get_transform())

    def _is_sorted(self, x):
        "return true if x is sorted"
        if len(x)<2: return 1
        return np.alltrue(x[1:]-x[0:-1]>=0)

    def draw(self, renderer):
        if self._invalid:
            self.recache()

        renderer.open_group('line2d')

        if not self._visible: return
        gc = renderer.new_gc()
        self._set_gc_clip(gc)

        gc.set_foreground(self._color)
        gc.set_antialiased(self._antialiased)
        gc.set_linewidth(self._linewidth)
        gc.set_alpha(self._alpha)
        if self.is_dashed():
            cap = self._dashcapstyle
            join = self._dashjoinstyle
        else:
            cap = self._solidcapstyle
            join = self._solidjoinstyle
        gc.set_joinstyle(join)
        gc.set_capstyle(cap)
        gc.set_snap(self.get_snap())

        funcname = self._lineStyles.get(self._linestyle, '_draw_nothing')
        if funcname != '_draw_nothing':
            tpath, affine = self._transformed_path.get_transformed_path_and_affine()
            self._lineFunc = getattr(self, funcname)
            funcname = self.drawStyles.get(self._drawstyle, '_draw_lines')
            drawFunc = getattr(self, funcname)
            drawFunc(renderer, gc, tpath, affine.frozen())

        if self._marker is not None:
            gc = renderer.new_gc()
            self._set_gc_clip(gc)
            gc.set_foreground(self.get_markeredgecolor())
            gc.set_linewidth(self._markeredgewidth)
            gc.set_alpha(self._alpha)
            funcname = self._markers.get(self._marker, '_draw_nothing')
            if funcname != '_draw_nothing':
                tpath, affine = self._transformed_path.get_transformed_points_and_affine()
                markerFunc = getattr(self, funcname)
                markerFunc(renderer, gc, tpath, affine.frozen())

        renderer.close_group('line2d')

    def get_antialiased(self): return self._antialiased
    def get_color(self): return self._color
    def get_drawstyle(self): return self._drawstyle
    def get_linestyle(self): return self._linestyle

    def get_linewidth(self): return self._linewidth
    def get_marker(self): return self._marker

    def get_markeredgecolor(self):
        if (is_string_like(self._markeredgecolor) and
            self._markeredgecolor == 'auto'):
            if self._marker in self.filled_markers:
                return 'k'
            else:
                return self._color
        else:
            return self._markeredgecolor


        return self._markeredgecolor
    def get_markeredgewidth(self): return self._markeredgewidth

    def get_markerfacecolor(self):
        if (self._markerfacecolor is None or
            (is_string_like(self._markerfacecolor) and
             self._markerfacecolor.lower()=='none') ):
            return self._markerfacecolor
        elif (is_string_like(self._markerfacecolor) and
              self._markerfacecolor.lower() == 'auto'):
            return self._color
        else:
            return self._markerfacecolor


    def get_markersize(self): return self._markersize

    def get_data(self, orig=True):
        """
        Return the xdata, ydata.

        If *orig* is *True*, return the original data
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
        if self._invalid:
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
        if self._invalid:
            self.recache()
        return self._y

    def get_path(self):
        """
        Return the :class:`~matplotlib.path.Path` object associated
        with this line.
        """
        if self._invalid:
            self.recache()
        return self._path

    def get_xydata(self):
        """
        Return the *xy* data as a Nx2 numpy array.
        """
        if self._invalid:
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

        ACCEPTS: [ 'default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post' ]
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
        '-'                 solid
        '--'                dashed
        '-.'                dash_dot
        ':'                 dotted
        'None'              draw nothing
        ' '                 draw nothing
        ''                  draw nothing
        ================    =================

        'steps' is equivalent to 'steps-pre' and is maintained for
        backward-compatibility.

        .. seealso::
            :meth:`set_drawstyle`

        ACCEPTS: [ '-' | '--' | '-.' | ':' | 'None' | ' ' | '' ] and
        any drawstyle in combination with a linestyle, e.g. 'steps--'.
        """

        # handle long drawstyle names before short ones !
        for ds in flatten([k.keys() for k in (self._drawStyles_l,
                self._drawStyles_s)], is_string_like):
            if linestyle.startswith(ds):
                self.set_drawstyle(ds)
                if len(linestyle) > len(ds):
                    linestyle = linestyle[len(ds):]
                else:
                    linestyle = '-'

        if linestyle not in self._lineStyles:
            if linestyle in ls_mapper:
                linestyle = ls_mapper[linestyle]
            else:
                verbose.report('Unrecognized line style %s, %s' %
                                            (linestyle, type(linestyle)))
        if linestyle in [' ','']:
            linestyle = 'None'
        self._linestyle = linestyle

    def set_marker(self, marker):
        """
        Set the line marker

        ========== ==========================
        marker     description
        ========== ==========================
        '.'        point
        ','        pixel
        'o'        circle
        'v'        triangle_down
        '^'        triangle_up
        '<'        triangle_left
        '>'        triangle_right
        '1'        tri_down
        '2'        tri_up
        '3'        tri_left
        '4'        tri_right
        's'        square
        'p'        pentagon
        '*'        star
        'h'        hexagon1
        'H'        hexagon2
        '+'        plus
        'x'        x
        'D'        diamond
        'd'        thin_diamond
        '|'        vline
        '_'        hline
        TICKLEFT   tickleft
        TICKRIGHT  tickright
        TICKUP     tickup
        TICKDOWN   tickdown
        CARETLEFT  caretleft
        CARETRIGHT caretright
        CARETUP    caretup
        CARETDOWN  caretdown
        'None'     nothing
        ' '        nothing
        ''         nothing
        ========== ==========================



        ACCEPTS: [ '+' | '*' | ',' | '.' | '1' | '2' | '3' | '4'
                 | '<' | '>' | 'D' | 'H' | '^' | '_' | 'd'
                 | 'h' | 'o' | 'p' | 's' | 'v' | 'x' | '|'
                 | TICKUP | TICKDOWN | TICKLEFT | TICKRIGHT
                 | 'None' | ' ' | '' ]

        """
        if marker not in self._markers:
            verbose.report('Unrecognized marker style %s, %s' %
                                            (marker, type(marker)))
        if marker in [' ','']:
            marker = 'None'
        self._marker = marker
        self._markerFunc = self._markers[marker]

    def set_markeredgecolor(self, ec):
        """
        Set the marker edge color

        ACCEPTS: any matplotlib color
        """
        if ec is None :
            ec = 'auto'
        self._markeredgecolor = ec

    def set_markeredgewidth(self, ew):
        """
        Set the marker edge width in points

        ACCEPTS: float value in points
        """
        if ew is None :
            ew = rcParams['lines.markeredgewidth']
        self._markeredgewidth = ew

    def set_markerfacecolor(self, fc):
        """
        Set the marker face color

        ACCEPTS: any matplotlib color
        """
        if fc is None :
            fc = 'auto'
        self._markerfacecolor = fc

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
        x = np.asarray(x)
        self.set_data(x, self._yorig)

    def set_ydata(self, y):
        """
        Set the data np.array for y

        ACCEPTS: 1D array
        """
        y = np.asarray(y)
        self.set_data(self._xorig, y)

    def set_dashes(self, seq):
        """
        Set the dash sequence, sequence of dashes with on off ink in
        points.  If seq is empty or if seq = (None, None), the
        linestyle will be set to solid.

        ACCEPTS: sequence of on/off ink in points
        """
        if seq == (None, None) or len(seq)==0:
            self.set_linestyle('-')
        else:
            self.set_linestyle('--')
        self._dashSeq = seq  # TODO: offset ignored for now


    def _draw_lines(self, renderer, gc, path, trans):
        self._lineFunc(renderer, gc, path, trans)

    def _draw_steps_pre(self, renderer, gc, path, trans):
        vertices = self._xy
        steps = ma.zeros((2*len(vertices)-1, 2), np.float_)

        steps[0::2, 0], steps[1::2, 0] = vertices[:, 0], vertices[:-1, 0]
        steps[0::2, 1], steps[1:-1:2, 1] = vertices[:, 1], vertices[1:, 1]

        path = Path(steps)
        path = path.transformed(self.get_transform())
        self._lineFunc(renderer, gc, path, IdentityTransform())

    def _draw_steps_post(self, renderer, gc, path, trans):
        vertices = self._xy
        steps = ma.zeros((2*len(vertices)-1, 2), np.float_)

        steps[::2, 0], steps[1:-1:2, 0] = vertices[:, 0], vertices[1:, 0]
        steps[0::2, 1], steps[1::2, 1] = vertices[:, 1], vertices[:-1, 1]

        path = Path(steps)
        path = path.transformed(self.get_transform())
        self._lineFunc(renderer, gc, path, IdentityTransform())

    def _draw_steps_mid(self, renderer, gc, path, trans):
        vertices = self._xy
        steps = ma.zeros((2*len(vertices), 2), np.float_)

        steps[1:-1:2, 0] = 0.5 * (vertices[:-1, 0] + vertices[1:, 0])
        steps[2::2, 0] = 0.5 * (vertices[:-1, 0] + vertices[1:, 0])
        steps[0, 0] = vertices[0, 0]
        steps[-1, 0] = vertices[-1, 0]
        steps[0::2, 1], steps[1::2, 1] = vertices[:, 1], vertices[:, 1]

        path = Path(steps)
        path = path.transformed(self.get_transform())
        self._lineFunc(renderer, gc, path, IdentityTransform())


    def _draw_nothing(self, *args, **kwargs):
        pass

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


    def _draw_point(self, renderer, gc, path, path_trans):
        w = renderer.points_to_pixels(self._markersize) * \
            self._point_size_reduction * 0.5
        gc.set_snap(renderer.points_to_pixels(self._markersize) > 3.0)
        rgbFace = self._get_rgb_face()
        transform = Affine2D().scale(w)
        renderer.draw_markers(
            gc, Path.unit_circle(), transform, path, path_trans,
            rgbFace)

    _draw_pixel_transform = Affine2D().translate(-0.5, -0.5)
    def _draw_pixel(self, renderer, gc, path, path_trans):
        rgbFace = self._get_rgb_face()
        gc.set_snap(False)
        renderer.draw_markers(gc, Path.unit_rectangle(),
                              self._draw_pixel_transform,
                              path, path_trans, rgbFace)


    def _draw_circle(self, renderer, gc, path, path_trans):
        w = renderer.points_to_pixels(self._markersize) * 0.5
        gc.set_snap(renderer.points_to_pixels(self._markersize) > 3.0)
        rgbFace = self._get_rgb_face()
        transform = Affine2D().scale(w, w)
        renderer.draw_markers(
            gc, Path.unit_circle(), transform, path, path_trans,
            rgbFace)


    _triangle_path = Path([[0.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]])
    def _draw_triangle_up(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset, offset)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, self._triangle_path, transform,
                              path, path_trans, rgbFace)


    def _draw_triangle_down(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset, -offset)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, self._triangle_path, transform,
                              path, path_trans, rgbFace)


    def _draw_triangle_left(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset, offset).rotate_deg(90)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, self._triangle_path, transform,
                              path, path_trans, rgbFace)


    def _draw_triangle_right(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset, offset).rotate_deg(-90)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, self._triangle_path, transform,
                              path, path_trans, rgbFace)


    def _draw_square(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 2.0)
        side = renderer.points_to_pixels(self._markersize)
        transform = Affine2D().translate(-0.5, -0.5).scale(side)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, Path.unit_rectangle(), transform,
                              path, path_trans, rgbFace)


    def _draw_diamond(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        side = renderer.points_to_pixels(self._markersize)
        transform = Affine2D().translate(-0.5, -0.5).rotate_deg(45).scale(side)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, Path.unit_rectangle(), transform,
                              path, path_trans, rgbFace)


    def _draw_thin_diamond(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 3.0)
        offset = renderer.points_to_pixels(self._markersize)
        transform = Affine2D().translate(-0.5, -0.5) \
            .rotate_deg(45).scale(offset * 0.6, offset)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, Path.unit_rectangle(), transform,
                              path, path_trans, rgbFace)


    def _draw_pentagon(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5 * renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, Path.unit_regular_polygon(5), transform,
                              path, path_trans, rgbFace)

    def _draw_star(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5 * renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset)
        rgbFace = self._get_rgb_face()
        _starpath = Path.unit_regular_star(5, innerCircle=0.381966)
        renderer.draw_markers(gc, _starpath, transform,
                              path, path_trans, rgbFace)


    def _draw_hexagon1(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5 * renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, Path.unit_regular_polygon(6), transform,
                              path, path_trans, rgbFace)


    def _draw_hexagon2(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5 * renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset).rotate_deg(30)
        rgbFace = self._get_rgb_face()
        renderer.draw_markers(gc, Path.unit_regular_polygon(6), transform,
                              path, path_trans, rgbFace)


    _line_marker_path = Path([[0.0, -1.0], [0.0, 1.0]])
    def _draw_vline(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 1.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset)
        renderer.draw_markers(gc, self._line_marker_path, transform,
                              path, path_trans)


    def _draw_hline(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 1.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset).rotate_deg(90)
        renderer.draw_markers(gc, self._line_marker_path, transform,
                              path, path_trans)


    _tickhoriz_path = Path([[0.0, 0.0], [1.0, 0.0]])
    def _draw_tickleft(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 1.0)
        offset = renderer.points_to_pixels(self._markersize)
        marker_transform = Affine2D().scale(-offset, 1.0)
        renderer.draw_markers(gc, self._tickhoriz_path, marker_transform,
                              path, path_trans)


    def _draw_tickright(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 1.0)
        offset = renderer.points_to_pixels(self._markersize)
        marker_transform = Affine2D().scale(offset, 1.0)
        renderer.draw_markers(gc, self._tickhoriz_path, marker_transform,
                              path, path_trans)


    _tickvert_path = Path([[-0.0, 0.0], [-0.0, 1.0]])
    def _draw_tickup(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 1.0)
        offset = renderer.points_to_pixels(self._markersize)
        marker_transform = Affine2D().scale(1.0, offset)
        renderer.draw_markers(gc, self._tickvert_path, marker_transform,
                              path, path_trans)


    def _draw_tickdown(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 1.0)
        offset = renderer.points_to_pixels(self._markersize)
        marker_transform = Affine2D().scale(1.0, -offset)
        renderer.draw_markers(gc, self._tickvert_path, marker_transform,
                              path, path_trans)


    _plus_path = Path([[-1.0, 0.0], [1.0, 0.0],
                       [0.0, -1.0], [0.0, 1.0]],
                      [Path.MOVETO, Path.LINETO,
                       Path.MOVETO, Path.LINETO])
    def _draw_plus(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 3.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset)
        renderer.draw_markers(gc, self._plus_path, transform,
                              path, path_trans)


    _tri_path = Path([[0.0, 0.0], [0.0, -1.0],
                      [0.0, 0.0], [0.8, 0.5],
                      [0.0, 0.0], [-0.8, 0.5]],
                     [Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO,
                      Path.MOVETO, Path.LINETO])
    def _draw_tri_down(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset)
        renderer.draw_markers(gc, self._tri_path, transform,
                              path, path_trans)


    def _draw_tri_up(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset).rotate_deg(180)
        renderer.draw_markers(gc, self._tri_path, transform,
                              path, path_trans)


    def _draw_tri_left(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset).rotate_deg(90)
        renderer.draw_markers(gc, self._tri_path, transform,
                              path, path_trans)


    def _draw_tri_right(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 5.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset).rotate_deg(270)
        renderer.draw_markers(gc, self._tri_path, transform,
                              path, path_trans)


    _caret_path = Path([[-1.0, 1.5], [0.0, 0.0], [1.0, 1.5]])
    def _draw_caretdown(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 3.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset)
        renderer.draw_markers(gc, self._caret_path, transform,
                              path, path_trans)


    def _draw_caretup(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 3.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset).rotate_deg(180)
        renderer.draw_markers(gc, self._caret_path, transform,
                              path, path_trans)


    def _draw_caretleft(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 3.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset).rotate_deg(270)
        renderer.draw_markers(gc, self._caret_path, transform,
                              path, path_trans)


    def _draw_caretright(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 3.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset).rotate_deg(90)
        renderer.draw_markers(gc, self._caret_path, transform,
                              path, path_trans)


    _x_path = Path([[-1.0, -1.0], [1.0, 1.0],
                    [-1.0, 1.0], [1.0, -1.0]],
                   [Path.MOVETO, Path.LINETO,
                    Path.MOVETO, Path.LINETO])
    def _draw_x(self, renderer, gc, path, path_trans):
        gc.set_snap(renderer.points_to_pixels(self._markersize) >= 3.0)
        offset = 0.5*renderer.points_to_pixels(self._markersize)
        transform = Affine2D().scale(offset)
        renderer.draw_markers(gc, self._x_path, transform,
                              path, path_trans)


    def update_from(self, other):
        'copy properties from other to self'
        Artist.update_from(self, other)
        self._linestyle = other._linestyle
        self._linewidth = other._linewidth
        self._color = other._color
        self._markersize = other._markersize
        self._markerfacecolor = other._markerfacecolor
        self._markeredgecolor = other._markeredgecolor
        self._markeredgewidth = other._markeredgewidth
        self._dashSeq = other._dashSeq
        self._dashcapstyle = other._dashcapstyle
        self._dashjoinstyle = other._dashjoinstyle
        self._solidcapstyle = other._solidcapstyle
        self._solidjoinstyle = other._solidjoinstyle

        self._linestyle = other._linestyle
        self._marker = other._marker
        self._drawstyle = other._drawstyle


    def _get_rgb_face(self):
        facecolor = self.get_markerfacecolor()
        if is_string_like(facecolor) and facecolor.lower()=='none':
            rgbFace = None
        else:
            rgbFace = colorConverter.to_rgb(facecolor)
        return rgbFace

    # some aliases....
    def set_aa(self, val):
        'alias for set_antialiased'
        self.set_antialiased(val)

    def set_c(self, val):
        'alias for set_color'
        self.set_color(val)


    def set_ls(self, val):
        'alias for set_linestyle'
        self.set_linestyle(val)


    def set_lw(self, val):
        'alias for set_linewidth'
        self.set_linewidth(val)


    def set_mec(self, val):
        'alias for set_markeredgecolor'
        self.set_markeredgecolor(val)


    def set_mew(self, val):
        'alias for set_markeredgewidth'
        self.set_markeredgewidth(val)


    def set_mfc(self, val):
        'alias for set_markerfacecolor'
        self.set_markerfacecolor(val)


    def set_ms(self, val):
        'alias for set_markersize'
        self.set_markersize(val)

    def get_aa(self):
        'alias for get_antialiased'
        return self.get_antialiased()

    def get_c(self):
        'alias for get_color'
        return self.get_color()


    def get_ls(self):
        'alias for get_linestyle'
        return self.get_linestyle()


    def get_lw(self):
        'alias for get_linewidth'
        return self.get_linewidth()


    def get_mec(self):
        'alias for get_markeredgecolor'
        return self.get_markeredgecolor()


    def get_mew(self):
        'alias for get_markeredgewidth'
        return self.get_markeredgewidth()


    def get_mfc(self):
        'alias for get_markerfacecolor'
        return self.get_markerfacecolor()


    def get_ms(self):
        'alias for get_markersize'
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
            raise RuntimeError('You must first set the picker property of the line')

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
        'When the line is picked, update the set of selected indicies.'
        if event.artist is not self.line: return

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
lineMarkers = Line2D._markers
drawStyles = Line2D.drawStyles

artist.kwdocd['Line2D'] = artist.kwdoc(Line2D)

# You can not set the docstring of an instancemethod,
# but you can on the underlying function.  Go figure.
Line2D.__init__.im_func.__doc__ = dedent(Line2D.__init__.__doc__) % artist.kwdocd
