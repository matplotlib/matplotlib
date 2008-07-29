# -*- coding: utf-8 -*-

from __future__ import division
import math

import matplotlib as mpl
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.artist as artist
import matplotlib.colors as colors
import matplotlib.transforms as transforms
import matplotlib.artist as artist
from matplotlib.path import Path

# these are not available for the object inspector until after the
# class is built so we define an initial set here for the init
# function and they will be overridden after object definition
artist.kwdocd['Patch'] = """

          =================   ==============================================
          Property            Description
          =================   ==============================================
          alpha               float
          animated            [True | False]
          antialiased or aa   [True | False]
          clip_box            a matplotlib.transform.Bbox instance
          clip_on             [True | False]
          edgecolor or ec     any matplotlib color
          facecolor or fc     any matplotlib color
          figure              a matplotlib.figure.Figure instance
          fill                [True | False]
          hatch               unknown
          label               any string
          linewidth or lw     float
          lod                 [True | False]
          transform           a matplotlib.transform transformation instance
          visible             [True | False]
          zorder              any number
          =================   ==============================================

          """

class Patch(artist.Artist):
    """
    A patch is a 2D thingy with a face color and an edge color.

    If any of *edgecolor*, *facecolor*, *linewidth*, or *antialiased*
    are *None*, they default to their rc params setting.
    """
    zorder = 1
    def __str__(self):
        return str(self.__class__).split('.')[-1]

    def __init__(self,
                 edgecolor=None,
                 facecolor=None,
                 linewidth=None,
                 linestyle=None,
                 antialiased = None,
                 hatch = None,
                 fill=True,
                 **kwargs
                 ):
        """
        The following kwarg properties are supported
        %(Patch)s
        """
        artist.Artist.__init__(self)

        if linewidth is None: linewidth = mpl.rcParams['patch.linewidth']
        if linestyle is None: linestyle = "solid"
        if antialiased is None: antialiased = mpl.rcParams['patch.antialiased']

        self.set_edgecolor(edgecolor)
        self.set_facecolor(facecolor)
        self.set_linewidth(linewidth)
        self.set_linestyle(linestyle)
        self.set_antialiased(antialiased)
        self.set_hatch(hatch)
        self.fill = fill
        self._combined_transform = transforms.IdentityTransform()

        if len(kwargs): artist.setp(self, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd


    def get_verts(self):
        """
        Return a copy of the vertices used in this patch

        If the patch contains Bézier curves, the curves will be
        interpolated by line segments.  To access the curves as
        curves, use :meth:`get_path`.
        """
        trans = self.get_transform()
        path = self.get_path()
        polygons = path.to_polygons(trans)
        if len(polygons):
            return polygons[0]
        return []

    def contains(self, mouseevent):
        """Test whether the mouse event occurred in the patch.

        Returns T/F, {}
        """
        # This is a general version of contains that should work on any
        # patch with a path.  However, patches that have a faster
        # algebraic solution to hit-testing should override this
        # method.
        if callable(self._contains): return self._contains(self,mouseevent)

        inside = self.get_path().contains_point(
            (mouseevent.x, mouseevent.y), self.get_transform())
        return inside, {}

    def update_from(self, other):
        artist.Artist.update_from(self, other)
        self.set_edgecolor(other.get_edgecolor())
        self.set_facecolor(other.get_facecolor())
        self.set_fill(other.get_fill())
        self.set_hatch(other.get_hatch())
        self.set_linewidth(other.get_linewidth())
        self.set_linestyle(other.get_linestyle())
        self.set_transform(other.get_data_transform())
        self.set_figure(other.get_figure())
        self.set_alpha(other.get_alpha())

    def get_extents(self):
        return self.get_path().get_extents(self.get_transform())

    def get_transform(self):
        return self.get_patch_transform() + artist.Artist.get_transform(self)

    def get_data_transform(self):
        return artist.Artist.get_transform(self)

    def get_patch_transform(self):
        return transforms.IdentityTransform()

    def get_antialiased(self):
        return self._antialiased
    get_aa = get_antialiased

    def get_edgecolor(self):
        return self._edgecolor
    get_ec = get_edgecolor

    def get_facecolor(self):
        return self._facecolor
    get_fc = get_facecolor

    def get_linewidth(self):
        return self._linewidth
    get_lw = get_linewidth

    def get_linestyle(self):
        return self._linestyle
    get_ls = get_linestyle

    def set_antialiased(self, aa):
        """
        Set whether to use antialiased rendering

        ACCEPTS: [True | False]  or None for default
        """
        if aa is None: aa = mpl.rcParams['patch.antialiased']
        self._antialiased = aa
    set_aa = set_antialiased

    def set_edgecolor(self, color):
        """
        Set the patch edge color

        ACCEPTS: mpl color spec, or None for default, or 'none' for no color
        """
        if color is None: color = mpl.rcParams['patch.edgecolor']
        self._edgecolor = color
    set_ec = set_edgecolor

    def set_facecolor(self, color):
        """
        Set the patch face color

        ACCEPTS: mpl color spec, or None for default, or 'none' for no color
        """
        if color is None: color = mpl.rcParams['patch.facecolor']
        self._facecolor = color
    set_fc = set_facecolor

    def set_linewidth(self, w):
        """
        Set the patch linewidth in points

        ACCEPTS: float or None for default
        """
        if w is None: w = mpl.rcParams['patch.linewidth']
        self._linewidth = w
    set_lw = set_linewidth

    def set_linestyle(self, ls):
        """
        Set the patch linestyle

        ACCEPTS: ['solid' | 'dashed' | 'dashdot' | 'dotted']
        """
        if ls is None: ls = "solid"
        self._linestyle = ls
    set_ls = set_linestyle

    def set_fill(self, b):
        """
        Set whether to fill the patch

        ACCEPTS: [True | False]
        """
        self.fill = b

    def get_fill(self):
        'return whether fill is set'
        return self.fill

    def set_hatch(self, h):
        """
        Set the hatching pattern

        hatch can be one of::

          /   - diagonal hatching
          \   - back diagonal
          |   - vertical
          -   - horizontal
          #   - crossed
          x   - crossed diagonal

        Letters can be combined, in which case all the specified
        hatchings are done.  If same letter repeats, it increases the
        density of hatching in that direction.

        CURRENT LIMITATIONS:

        1. Hatching is supported in the PostScript backend only.

        2. Hatching is done with solid black lines of width 0.

        """
        self._hatch = h

    def get_hatch(self):
        'return the current hatching pattern'
        return self._hatch


    def draw(self, renderer):
        if not self.get_visible(): return
        #renderer.open_group('patch')
        gc = renderer.new_gc()

        if cbook.is_string_like(self._edgecolor) and self._edgecolor.lower()=='none':
            gc.set_linewidth(0)
        else:
            gc.set_foreground(self._edgecolor)
            gc.set_linewidth(self._linewidth)
            gc.set_linestyle(self._linestyle)

        gc.set_alpha(self._alpha)
        gc.set_antialiased(self._antialiased)
        self._set_gc_clip(gc)
        gc.set_capstyle('projecting')

        if (not self.fill or self._facecolor is None or
            (cbook.is_string_like(self._facecolor) and self._facecolor.lower()=='none')):
            rgbFace = None
        else:
            rgbFace = colors.colorConverter.to_rgb(self._facecolor)

        if self._hatch:
            gc.set_hatch(self._hatch )

        path = self.get_path()
        transform = self.get_transform()
        tpath = transform.transform_path_non_affine(path)
        affine = transform.get_affine()

        renderer.draw_path(gc, tpath, affine, rgbFace)

        #renderer.close_group('patch')

    def get_path(self):
        """
        Return the path of this patch
        """
        raise NotImplementedError('Derived must override')

    def get_window_extent(self, renderer=None):
        return self.get_path().get_extents(self.get_transform())


class Shadow(Patch):
    def __str__(self):
        return "Shadow(%s)"%(str(self.patch))

    def __init__(self, patch, ox, oy, props=None, **kwargs):
        """
        Create a shadow of the given *patch* offset by *ox*, *oy*.
        *props*, if not *None*, is a patch property update dictionary.
        If *None*, the shadow will have have the same color as the face,
        but darkened.

        kwargs are
        %(Patch)s
        """
        Patch.__init__(self)
        self.patch = patch
        self.props = props
        self._ox, self._oy = ox, oy
        self._update_transform()
        self._update()
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def _update(self):
        self.update_from(self.patch)
        if self.props is not None:
            self.update(self.props)
        else:
            r,g,b,a = colors.colorConverter.to_rgba(self.patch.get_facecolor())
            rho = 0.3
            r = rho*r
            g = rho*g
            b = rho*b

            self.set_facecolor((r,g,b,0.5))
            self.set_edgecolor((r,g,b,0.5))

    def _update_transform(self):
        self._shadow_transform = transforms.Affine2D().translate(self._ox, self._oy)

    def _get_ox(self):
        return self._ox
    def _set_ox(self, ox):
        self._ox = ox
        self._update_transform()

    def _get_oy(self):
        return self._oy
    def _set_oy(self, oy):
        self._oy = oy
        self._update_transform()

    def get_path(self):
        return self.patch.get_path()

    def get_patch_transform(self):
        return self.patch.get_patch_transform() + self._shadow_transform

class Rectangle(Patch):
    """
    Draw a rectangle with lower left at *xy*=(*x*, *y*) with specified
    width and height
    """

    def __str__(self):
        return self.__class__.__name__ \
            + "(%g,%g;%gx%g)" % (self._x, self._y, self._width, self._height)

    def __init__(self, xy, width, height, **kwargs):
        """

        *fill* is a boolean indicating whether to fill the rectangle

        Valid kwargs are:
        %(Patch)s
        """

        Patch.__init__(self, **kwargs)

        self._x = xy[0]
        self._y = xy[1]
        self._width = width
        self._height = height
        self._rect_transform = transforms.IdentityTransform()
        self._update_patch_transform()
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def get_path(self):
        """
        Return the vertices of the rectangle
        """
        return Path.unit_rectangle()

    def _update_patch_transform(self):
        x = self.convert_xunits(self._x)
        y = self.convert_yunits(self._y)
        width = self.convert_xunits(self._width)
        height = self.convert_yunits(self._height)
        bbox = transforms.Bbox.from_bounds(x, y, width, height)
        self._rect_transform = transforms.BboxTransformTo(bbox)

    def get_patch_transform(self):
        self._update_patch_transform()
        return self._rect_transform

    def contains(self, mouseevent):
        # special case the degernate rectangle
        if self._width==0 or self._height==0:
            return False, {}

        x, y = self.get_transform().inverted().transform_point(
            (mouseevent.x, mouseevent.y))
        return (x >= 0.0 and x <= 1.0 and y >= 0.0 and y <= 1.0), {}

    def get_x(self):
        "Return the left coord of the rectangle"
        return self._x

    def get_y(self):
        "Return the bottom coord of the rectangle"
        return self._y

    def get_width(self):
        "Return the width of the  rectangle"
        return self._width

    def get_height(self):
        "Return the height of the rectangle"
        return self._height

    def set_x(self, x):
        """
        Set the left coord of the rectangle

        ACCEPTS: float
        """
        self._x = x

    def set_y(self, y):
        """
        Set the bottom coord of the rectangle

        ACCEPTS: float
        """
        self._y = y

    def set_width(self, w):
        """
        Set the width rectangle

        ACCEPTS: float
        """
        self._width = w

    def set_height(self, h):
        """
        Set the width rectangle

        ACCEPTS: float
        """
        self._height = h

    def set_bounds(self, *args):
        """
        Set the bounds of the rectangle: l,b,w,h

        ACCEPTS: (left, bottom, width, height)
        """
        if len(args)==0:
            l,b,w,h = args[0]
        else:
            l,b,w,h = args
        self._x = l
        self._y = b
        self._width = w
        self._height = h

    def get_bbox(self):
        return transforms.Bbox.from_bounds(self._x, self._y, self._width, self._height)

class RegularPolygon(Patch):
    """
    A regular polygon patch.
    """
    def __str__(self):
        return "Poly%d(%g,%g)"%(self._numVertices,self._xy[0],self._xy[1])

    def __init__(self, xy, numVertices, radius=5, orientation=0,
                 **kwargs):
        """
        Constructor arguments:

        *xy*
          A length 2 tuple (*x*, *y*) of the center.

        *numVertices*
          the number of vertices.

        *radius*
          The distance from the center to each of the vertices.

        *orientation*
          rotates the polygon (in radians).

        Valid kwargs are:
        %(Patch)s
        """
        self._xy = xy
        self._numVertices = numVertices
        self._orientation = orientation
        self._radius = radius
        self._path = Path.unit_regular_polygon(numVertices)
        self._poly_transform = transforms.Affine2D()
        self._update_transform()

        Patch.__init__(self, **kwargs)

    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def _update_transform(self):
        self._poly_transform.clear() \
            .scale(self.radius) \
            .rotate(self.orientation) \
            .translate(*self.xy)

    def _get_xy(self):
        return self._xy
    def _set_xy(self, xy):
        self._update_transform()
    xy = property(_get_xy, _set_xy)

    def _get_orientation(self):
        return self._orientation
    def _set_orientation(self, xy):
        self._orientation = xy
    orientation = property(_get_orientation, _set_orientation)

    def _get_radius(self):
        return self._radius
    def _set_radius(self, xy):
        self._radius = xy
    radius = property(_get_radius, _set_radius)

    def _get_numvertices(self):
        return self._numVertices
    def _set_numvertices(self, numVertices):
        self._numVertices = numVertices
    numvertices = property(_get_numvertices, _set_numvertices)

    def get_path(self):
        return self._path

    def get_patch_transform(self):
        self._update_transform()
        return self._poly_transform

class PathPatch(Patch):
    """
    A general polycurve path patch.
    """
    def __str__(self):
        return "Poly((%g, %g) ...)" % tuple(self._path.vertices[0])

    def __init__(self, path, **kwargs):
        """
        *path* is a :class:`matplotlib.path.Path` object.

        Valid kwargs are:
        %(Patch)s
        See Patch documentation for additional kwargs
        """
        Patch.__init__(self, **kwargs)
        self._path = path
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def get_path(self):
        return self._path

class Polygon(Patch):
    """
    A general polygon patch.
    """
    def __str__(self):
        return "Poly((%g, %g) ...)" % tuple(self._path.vertices[0])

    def __init__(self, xy, closed=True, **kwargs):
        """
        *xy* is a numpy array with shape Nx2.

        If *closed* is *True*, the polygon will be closed so the
        starting and ending points are the same.

        Valid kwargs are:
        %(Patch)s
        See Patch documentation for additional kwargs
        """
        Patch.__init__(self, **kwargs)
        xy = np.asarray(xy, np.float_)
        self._path = Path(xy)
        self.set_closed(closed)

    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def get_path(self):
        return self._path

    def get_closed(self):
        return self._closed

    def set_closed(self, closed):
        self._closed = closed
        xy = self._get_xy()
        if closed:
            if len(xy) and (xy[0] != xy[-1]).any():
                xy = np.concatenate([xy, [xy[0]]])
        else:
            if len(xy)>2 and (xy[0]==xy[-1]).all():
                xy = xy[0:-1]
        self._set_xy(xy)

    def get_xy(self):
        return self._path.vertices
    def set_xy(self, vertices):
        self._path = Path(vertices)
    _get_xy = get_xy
    _set_xy = set_xy
    xy = property(
        get_xy, set_xy, None,
        """Set/get the vertices of the polygon.  This property is
           provided for backward compatibility with matplotlib 0.91.x
           only.  New code should use
           :meth:`~matplotlib.patches.Polygon.get_xy` and
           :meth:`~matplotlib.patches.Polygon.set_xy` instead.""")

class Wedge(Patch):
    def __str__(self):
        return "Wedge(%g,%g)"%(self.theta1,self.theta2)

    def __init__(self, center, r, theta1, theta2, **kwargs):
        """
        Draw a wedge centered at *x*, *y* center with radius *r* that
        sweeps *theta1* to *theta2* (in degrees).

        Valid kwargs are:

        %(Patch)s
        """
        Patch.__init__(self, **kwargs)
        self.center = center
        self.r = r
        self.theta1 = theta1
        self.theta2 = theta2
        self._patch_transform = transforms.IdentityTransform()
        self._path = Path.wedge(self.theta1, self.theta2)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def get_path(self):
        return self._path

    def get_patch_transform(self):
        x = self.convert_xunits(self.center[0])
        y = self.convert_yunits(self.center[1])
        rx = self.convert_xunits(self.r)
        ry = self.convert_yunits(self.r)
        self._patch_transform = transforms.Affine2D() \
            .scale(rx, ry).translate(x, y)
        return self._patch_transform

# COVERAGE NOTE: Not used internally or from examples
class Arrow(Patch):
    """
    An arrow patch.
    """
    def __str__(self):
        return "Arrow()"

    _path = Path( [
            [ 0.0,  0.1 ], [ 0.0, -0.1],
            [ 0.8, -0.1 ], [ 0.8, -0.3],
            [ 1.0,  0.0 ], [ 0.8,  0.3],
            [ 0.8,  0.1 ], [ 0.0,  0.1] ] )

    def __init__( self, x, y, dx, dy, width=1.0, **kwargs ):
        """
        Draws an arrow, starting at (*x*, *y*), direction and length
        given by (*dx*, *dy*) the width of the arrow is scaled by *width*.

        Valid kwargs are:
        %(Patch)s
        """
        Patch.__init__(self, **kwargs)
        L = np.sqrt(dx**2+dy**2) or 1 # account for div by zero
        cx = float(dx)/L
        sx = float(dy)/L

        trans1 = transforms.Affine2D().scale(L, width)
        trans2 = transforms.Affine2D.from_values(cx, sx, -sx, cx, 0.0, 0.0)
        trans3 = transforms.Affine2D().translate(x, y)
        trans = trans1 + trans2 + trans3
        self._patch_transform = trans.frozen()
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def get_path(self):
        return self._path

    def get_patch_transform(self):
        return self._patch_transform

class FancyArrow(Polygon):
    """
    Like Arrow, but lets you set head width and head height independently.
    """

    def __str__(self):
        return "FancyArrow()"

    def __init__(self, x, y, dx, dy, width=0.001, length_includes_head=False, \
        head_width=None, head_length=None, shape='full', overhang=0, \
        head_starts_at_zero=False,**kwargs):
        """
        Constructor arguments

            *length_includes_head*:
               *True* if head is counted in calculating the length.

            *shape*: ['full', 'left', 'right']

            *overhang*:
              distance that the arrow is swept back (0 overhang means
              triangular shape).

            *head_starts_at_zero*:
              If *True*, the head starts being drawn at coordinate 0
              instead of ending at coordinate 0.

        Valid kwargs are:
        %(Patch)s

        """
        if head_width is None:
            head_width = 3 * width
        if head_length is None:
            head_length = 1.5 * head_width

        distance = np.sqrt(dx**2 + dy**2)
        if length_includes_head:
            length=distance
        else:
            length=distance+head_length
        if not length:
            verts = [] #display nothing if empty
        else:
            #start by drawing horizontal arrow, point at (0,0)
            hw, hl, hs, lw = head_width, head_length, overhang, width
            left_half_arrow = np.array([
                [0.0,0.0],                  #tip
                [-hl, -hw/2.0],             #leftmost
                [-hl*(1-hs), -lw/2.0], #meets stem
                [-length, -lw/2.0],          #bottom left
                [-length, 0],
            ])
            #if we're not including the head, shift up by head length
            if not length_includes_head:
                left_half_arrow += [head_length, 0]
            #if the head starts at 0, shift up by another head length
            if head_starts_at_zero:
                left_half_arrow += [head_length/2.0, 0]
            #figure out the shape, and complete accordingly
            if shape == 'left':
                coords = left_half_arrow
            else:
                right_half_arrow = left_half_arrow*[1,-1]
                if shape == 'right':
                    coords = right_half_arrow
                elif shape == 'full':
                    coords=np.concatenate([left_half_arrow,right_half_arrow[::-1]])
                else:
                    raise ValueError, "Got unknown shape: %s" % shape
            cx = float(dx)/distance
            sx = float(dy)/distance
            M = np.array([[cx, sx],[-sx,cx]])
            verts = np.dot(coords, M) + (x+dx, y+dy)

        Polygon.__init__(self, map(tuple, verts), **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

class YAArrow(Patch):
    """
    Yet another arrow class.

    This is an arrow that is defined in display space and has a tip at
    *x1*, *y1* and a base at *x2*, *y2*.
    """
    def __str__(self):
        return "YAArrow()"

    def __init__(self, figure, xytip, xybase, width=4, frac=0.1, headwidth=12, **kwargs):
        """
        Constructor arguments:

        *xytip*
          (*x*, *y*) location of arrow tip

        *xybase*
          (*x*, *y*) location the arrow base mid point

        *figure*
          The :class:`~matplotlib.figure.Figure` instance
          (fig.dpi)

        *width*
          The width of the arrow in points

        *frac*
          The fraction of the arrow length occupied by the head

        *headwidth*
          The width of the base of the arrow head in points

        Valid kwargs are:
        %(Patch)s

        """
        self.figure = figure
        self.xytip = xytip
        self.xybase = xybase
        self.width = width
        self.frac = frac
        self.headwidth = headwidth
        Patch.__init__(self, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def get_path(self):
        # Since this is dpi dependent, we need to recompute the path
        # every time.

        # the base vertices
        x1, y1 = self.xytip
        x2, y2 = self.xybase
        k1 = self.width*self.figure.dpi/72./2.
        k2 = self.headwidth*self.figure.dpi/72./2.
        xb1, yb1, xb2, yb2 = self.getpoints(x1, y1, x2, y2, k1)

        # a point on the segment 20% of the distance from the tip to the base
        theta = math.atan2(y2-y1, x2-x1)
        r = math.sqrt((y2-y1)**2. + (x2-x1)**2.)
        xm = x1 + self.frac * r * math.cos(theta)
        ym = y1 + self.frac * r * math.sin(theta)
        xc1, yc1, xc2, yc2 = self.getpoints(x1, y1, xm, ym, k1)
        xd1, yd1, xd2, yd2 = self.getpoints(x1, y1, xm, ym, k2)

        xs = self.convert_xunits([xb1, xb2, xc2, xd2, x1, xd1, xc1, xb1])
        ys = self.convert_yunits([yb1, yb2, yc2, yd2, y1, yd1, yc1, yb1])

        return Path(zip(xs, ys))

    def get_patch_transform(self):
        return transforms.IdentityTransform()

    def getpoints(self, x1,y1,x2,y2, k):
        """
        For line segment defined by (*x1*, *y1*) and (*x2*, *y2*)
        return the points on the line that is perpendicular to the
        line and intersects (*x2*, *y2*) and the distance from (*x2*,
        *y2*) of the returned points is *k*.
        """
        x1,y1,x2,y2,k = map(float, (x1,y1,x2,y2,k))

        if y2-y1 == 0:
            return x2, y2+k, x2, y2-k
        elif x2-x1 == 0:
            return x2+k, y2, x2-k, y2

        m = (y2-y1)/(x2-x1)
        pm = -1./m
        a = 1
        b = -2*y2
        c = y2**2. - k**2.*pm**2./(1. + pm**2.)

        y3a = (-b + math.sqrt(b**2.-4*a*c))/(2.*a)
        x3a = (y3a - y2)/pm + x2

        y3b = (-b - math.sqrt(b**2.-4*a*c))/(2.*a)
        x3b = (y3b - y2)/pm + x2
        return x3a, y3a, x3b, y3b


class CirclePolygon(RegularPolygon):
    """
    A polygon-approximation of a circle patch.
    """
    def __str__(self):
        return "CirclePolygon(%d,%d)"%self.center

    def __init__(self, xy, radius=5,
                 resolution=20,  # the number of vertices
                 **kwargs):
        """
        Create a circle at *xy* = (*x*, *y*) with given *radius*.
        This circle is approximated by a regular polygon with
        *resolution* sides.  For a smoother circle drawn with splines,
        see :class:`~matplotlib.patches.Circle`.

        Valid kwargs are:
        %(Patch)s

        """
        RegularPolygon.__init__(self, xy,
                                resolution,
                                radius,
                                orientation=0,
                                **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd


class Ellipse(Patch):
    """
    A scale-free ellipse.
    """
    def __str__(self):
        return "Ellipse(%s,%s;%sx%s)"%(self.center[0],self.center[1],self.width,self.height)

    def __init__(self, xy, width, height, angle=0.0, **kwargs):
        """
        *xy*
          center of ellipse

        *width*
          length of horizontal axis

        *height*
          length of vertical axis

        *angle*
          rotation in degrees (anti-clockwise)

        Valid kwargs are:
        %(Patch)s
        """
        Patch.__init__(self, **kwargs)

        self.center = xy
        self.width, self.height = width, height
        self.angle = angle
        self._path = Path.unit_circle()
        self._patch_transform = transforms.IdentityTransform()
        self._recompute_transform()
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def _recompute_transform(self):
        center = (self.convert_xunits(self.center[0]),
                  self.convert_yunits(self.center[1]))
        width = self.convert_xunits(self.width)
        height = self.convert_yunits(self.height)
        self._patch_transform = transforms.Affine2D() \
            .scale(width * 0.5, height * 0.5) \
            .rotate_deg(self.angle) \
            .translate(*center)

    def get_path(self):
        """
        Return the vertices of the rectangle
        """
        return self._path

    def get_patch_transform(self):
        self._recompute_transform()
        return self._patch_transform

    def contains(self,ev):
        if ev.x is None or ev.y is None: return False,{}
        x, y = self.get_transform().inverted().transform_point((ev.x, ev.y))
        return (x*x + y*y) <= 1.0, {}


class Circle(Ellipse):
    """
    A circle patch.
    """
    def __str__(self):
        return "Circle((%g,%g),r=%g)"%(self.center[0],self.center[1],self.radius)

    def __init__(self, xy, radius=5, **kwargs):
        """
        Create true circle at center *xy* = (*x*, *y*) with given
        *radius*.  Unlike :class:`~matplotlib.patches.CirclePolygon`
        which is a polygonal approximation, this uses Bézier splines
        and is much closer to a scale-free circle.

        Valid kwargs are:
        %(Patch)s

        """
        if kwargs.has_key('resolution'):
            import warnings
            warnings.warn('Circle is now scale free.  Use CirclePolygon instead!', DeprecationWarning)
            kwargs.pop('resolution')

        self.radius = radius
        Ellipse.__init__(self, xy, radius*2, radius*2, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

class Arc(Ellipse):
    """
    An elliptical arc.  Because it performs various optimizations, it
    can not be filled.

    The arc must be used in an :class:`~matplotlib.axes.Axes`
    instance---it cannot be added directly to a
    :class:`~matplotlib.figure.Figure`---because it is optimized to
    only render the segments that are inside the axes bounding box
    with high resolution.
    """
    def __str__(self):
        return "Arc(%s,%s;%sx%s)"%(self.center[0],self.center[1],self.width,self.height)

    def __init__(self, xy, width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs):
        """
        The following args are supported:

        *xy*
          center of ellipse

        *width*
          length of horizontal axis

        *height*
          length of vertical axis

        *angle*
          rotation in degrees (anti-clockwise)

        *theta1*
          starting angle of the arc in degrees

        *theta2*
          ending angle of the arc in degrees

        If *theta1* and *theta2* are not provided, the arc will form a
        complete ellipse.

        Valid kwargs are:

        %(Patch)s
        """
        fill = kwargs.pop('fill')
        if fill:
            raise ValueError("Arc objects can not be filled")
        kwargs['fill'] = False

        Ellipse.__init__(self, xy, width, height, angle, **kwargs)

        self.theta1 = theta1
        self.theta2 = theta2
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def draw(self, renderer):
        """
        Ellipses are normally drawn using an approximation that uses
        eight cubic bezier splines.  The error of this approximation
        is 1.89818e-6, according to this unverified source:

          Lancaster, Don.  Approximating a Circle or an Ellipse Using
          Four Bezier Cubic Splines.

          http://www.tinaja.com/glib/ellipse4.pdf

        There is a use case where very large ellipses must be drawn
        with very high accuracy, and it is too expensive to render the
        entire ellipse with enough segments (either splines or line
        segments).  Therefore, in the case where either radius of the
        ellipse is large enough that the error of the spline
        approximation will be visible (greater than one pixel offset
        from the ideal), a different technique is used.

        In that case, only the visible parts of the ellipse are drawn,
        with each visible arc using a fixed number of spline segments
        (8).  The algorithm proceeds as follows:

          1. The points where the ellipse intersects the axes bounding
             box are located.  (This is done be performing an inverse
             transformation on the axes bbox such that it is relative
             to the unit circle -- this makes the intersection
             calculation much easier than doing rotated ellipse
             intersection directly).

             This uses the "line intersecting a circle" algorithm
             from:

               Vince, John.  Geometry for Computer Graphics: Formulae,
               Examples & Proofs.  London: Springer-Verlag, 2005.

          2. The angles of each of the intersection points are
             calculated.

          3. Proceeding counterclockwise starting in the positive
             x-direction, each of the visible arc-segments between the
             pairs of vertices are drawn using the bezier arc
             approximation technique implemented in
             :meth:`matplotlib.path.Path.arc`.
        """
        if not hasattr(self, 'axes'):
            raise RuntimeError('Arcs can only be used in Axes instances')

        self._recompute_transform()

        # Get the width and height in pixels
        width = self.convert_xunits(self.width)
        height = self.convert_yunits(self.height)
        width, height = self.get_transform().transform_point(
            (width, height))
        inv_error = (1.0 / 1.89818e-6) * 0.5

        if width < inv_error and height < inv_error:
            self._path = Path.arc(self.theta1, self.theta2)
            return Patch.draw(self, renderer)

        def iter_circle_intersect_on_line(x0, y0, x1, y1):
            dx = x1 - x0
            dy = y1 - y0
            dr2 = dx*dx + dy*dy
            D = x0*y1 - x1*y0
            D2 = D*D
            discrim = dr2 - D2

            # Single (tangential) intersection
            if discrim == 0.0:
                x = (D*dy) / dr2
                y = (-D*dx) / dr2
                yield x, y
            elif discrim > 0.0:
                # The definition of "sign" here is different from
                # np.sign: we never want to get 0.0
                if dy < 0.0:
                    sign_dy = -1.0
                else:
                    sign_dy = 1.0
                sqrt_discrim = np.sqrt(discrim)
                for sign in (1., -1.):
                    x = (D*dy + sign * sign_dy * dx * sqrt_discrim) / dr2
                    y = (-D*dx + sign * np.abs(dy) * sqrt_discrim) / dr2
                    yield x, y

        def iter_circle_intersect_on_line_seg(x0, y0, x1, y1):
            epsilon = 1e-9
            if x1 < x0:
                x0e, x1e = x1, x0
            else:
                x0e, x1e = x0, x1
            if y1 < y0:
                y0e, y1e = y1, y0
            else:
                y0e, y1e = y0, y1
            x0e -= epsilon
            y0e -= epsilon
            x1e += epsilon
            y1e += epsilon
            for x, y in iter_circle_intersect_on_line(x0, y0, x1, y1):
                if x >= x0e and x <= x1e and y >= y0e and y <= y1e:
                    yield x, y

        # Transforms the axes box_path so that it is relative to the unit
        # circle in the same way that it is relative to the desired
        # ellipse.
        box_path = Path.unit_rectangle()
        box_path_transform = transforms.BboxTransformTo(self.axes.bbox) + \
            self.get_transform().inverted()
        box_path = box_path.transformed(box_path_transform)

        PI = np.pi
        TWOPI = PI * 2.0
        RAD2DEG = 180.0 / PI
        DEG2RAD = PI / 180.0
        theta1 = self.theta1
        theta2 = self.theta2
        thetas = {}
        # For each of the point pairs, there is a line segment
        for p0, p1 in zip(box_path.vertices[:-1], box_path.vertices[1:]):
            x0, y0 = p0
            x1, y1 = p1
            for x, y in iter_circle_intersect_on_line_seg(x0, y0, x1, y1):
                theta = np.arccos(x)
                if y < 0:
                    theta = TWOPI - theta
                # Convert radians to angles
                theta *= RAD2DEG
                if theta > theta1 and theta < theta2:
                    thetas[theta] = None

        thetas = thetas.keys()
        thetas.sort()
        thetas.append(theta2)

        last_theta = theta1
        theta1_rad = theta1 * DEG2RAD
        inside = box_path.contains_point((np.cos(theta1_rad), np.sin(theta1_rad)))
        for theta in thetas:
            if inside:
                self._path = Path.arc(last_theta, theta, 8)
                Patch.draw(self, renderer)
                inside = False
            else:
                inside = True
            last_theta = theta

def bbox_artist(artist, renderer, props=None, fill=True):
    """
    This is a debug function to draw a rectangle around the bounding
    box returned by
    :meth:`~matplotlib.artist.Artist.get_window_extent` of an artist,
    to test whether the artist is returning the correct bbox.

    *props* is a dict of rectangle props with the additional property
    'pad' that sets the padding around the bbox in points.
    """
    if props is None: props = {}
    props = props.copy() # don't want to alter the pad externally
    pad = props.pop('pad', 4)
    pad = renderer.points_to_pixels(pad)
    bbox = artist.get_window_extent(renderer)
    l,b,w,h = bbox.bounds
    l-=pad/2.
    b-=pad/2.
    w+=pad
    h+=pad
    r = Rectangle(xy=(l,b),
                  width=w,
                  height=h,
                  fill=fill,
                  )
    r.set_transform(transforms.IdentityTransform())
    r.set_clip_on( False )
    r.update(props)
    r.draw(renderer)


def draw_bbox(bbox, renderer, color='k', trans=None):
    """
    This is a debug function to draw a rectangle around the bounding
    box returned by
    :meth:`~matplotlib.artist.Artist.get_window_extent` of an artist,
    to test whether the artist is returning the correct bbox.
    """

    l,b,w,h = bbox.get_bounds()
    r = Rectangle(xy=(l,b),
                  width=w,
                  height=h,
                  edgecolor=color,
                  fill=False,
                  )
    if trans is not None: r.set_transform(trans)
    r.set_clip_on( False )
    r.draw(renderer)

artist.kwdocd['Patch'] = patchdoc = artist.kwdoc(Patch)
for k in ('Rectangle', 'Circle', 'RegularPolygon', 'Polygon', 'Wedge', 'Arrow',
          'FancyArrow', 'YAArrow', 'CirclePolygon', 'Ellipse', 'Arc'):
    artist.kwdocd[k] = patchdoc
