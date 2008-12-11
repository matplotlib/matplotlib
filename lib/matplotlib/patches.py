# -*- coding: utf-8 -*-

from __future__ import division
import math

import matplotlib as mpl
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.artist as artist
import matplotlib.colors as colors
import matplotlib.transforms as transforms
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
        """
        Updates this :class:`Patch` from the properties of *other*.
        """
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
        """
        Return a :class:`~matplotlib.transforms.Bbox` object defining
        the axis-aligned extents of the :class:`Patch`.
        """
        return self.get_path().get_extents(self.get_transform())

    def get_transform(self):
        """
        Return the :class:`~matplotlib.transforms.Transform` applied
        to the :class:`Patch`.
        """
        return self.get_patch_transform() + artist.Artist.get_transform(self)

    def get_data_transform(self):
        return artist.Artist.get_transform(self)

    def get_patch_transform(self):
        return transforms.IdentityTransform()

    def get_antialiased(self):
        """
        Returns True if the :class:`Patch` is to be drawn with antialiasing.
        """
        return self._antialiased
    get_aa = get_antialiased

    def get_edgecolor(self):
        """
        Return the edge color of the :class:`Patch`.
        """
        return self._edgecolor
    get_ec = get_edgecolor

    def get_facecolor(self):
        """
        Return the face color of the :class:`Patch`.
        """
        return self._facecolor
    get_fc = get_facecolor

    def get_linewidth(self):
        """
        Return the line width in points.
        """
        return self._linewidth
    get_lw = get_linewidth

    def get_linestyle(self):
        """
        Return the linestyle.  Will be one of ['solid' | 'dashed' |
        'dashdot' | 'dotted']
        """
        return self._linestyle
    get_ls = get_linestyle

    def set_antialiased(self, aa):
        """
        Set whether to use antialiased rendering

        ACCEPTS: [True | False]  or None for default
        """
        if aa is None: aa = mpl.rcParams['patch.antialiased']
        self._antialiased = aa

    def set_aa(self, aa):
        """alias for set_antialiased"""
        return self.set_antialiased(aa)

    def set_edgecolor(self, color):
        """
        Set the patch edge color

        ACCEPTS: mpl color spec, or None for default, or 'none' for no color
        """
        if color is None: color = mpl.rcParams['patch.edgecolor']
        self._edgecolor = color

    def set_ec(self, color):
        """alias for set_edgecolor"""
        return self.set_edgecolor(color)

    def set_facecolor(self, color):
        """
        Set the patch face color

        ACCEPTS: mpl color spec, or None for default, or 'none' for no color
        """
        if color is None: color = mpl.rcParams['patch.facecolor']
        self._facecolor = color

    def set_fc(self, color):
        """alias for set_facecolor"""
        return self.set_facecolor(color)

    def set_linewidth(self, w):
        """
        Set the patch linewidth in points

        ACCEPTS: float or None for default
        """
        if w is None: w = mpl.rcParams['patch.linewidth']
        self._linewidth = w

    def set_lw(self, lw):
        """alias for set_linewidth"""
        return self.set_linewidth(lw)

    def set_linestyle(self, ls):
        """
        Set the patch linestyle

        ACCEPTS: ['solid' | 'dashed' | 'dashdot' | 'dotted']
        """
        if ls is None: ls = "solid"
        self._linestyle = ls

    def set_ls(self, ls):
        """alias for set_linestyle"""
        return self.set_linestyle(ls)

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


        ACCEPTS: [ '/' | '\\' | '|' | '-' | '#' | 'x' ]
        """
        self._hatch = h

    def get_hatch(self):
        'Return the current hatching pattern'
        return self._hatch


    def draw(self, renderer):
        'Draw the :class:`Patch` to the given *renderer*.'
        if not self.get_visible(): return
        #renderer.open_group('patch')
        gc = renderer.new_gc()

        if cbook.is_string_like(self._edgecolor) and self._edgecolor.lower()=='none':
            gc.set_linewidth(0)
        else:
            gc.set_foreground(self._edgecolor)
            gc.set_linewidth(self._linewidth)
            gc.set_linestyle(self._linestyle)

        gc.set_antialiased(self._antialiased)
        self._set_gc_clip(gc)
        gc.set_capstyle('projecting')
        gc.set_url(self._url)
        gc.set_snap(self._snap)

        if (not self.fill or self._facecolor is None or
            (cbook.is_string_like(self._facecolor) and self._facecolor.lower()=='none')):
            rgbFace = None
            gc.set_alpha(1.0)
        else:
            r, g, b, a = colors.colorConverter.to_rgba(self._facecolor, self._alpha)
            rgbFace = (r, g, b)
            gc.set_alpha(a)


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

artist.kwdocd['Patch'] = patchdoc = artist.kwdoc(Patch)
for k in ('Rectangle', 'Circle', 'RegularPolygon', 'Polygon', 'Wedge', 'Arrow',
          'FancyArrow', 'YAArrow', 'CirclePolygon', 'Ellipse', 'Arc',
          'FancyBboxPatch'):
    artist.kwdocd[k] = patchdoc

# define Patch.__init__ after the class so that the docstring can be
# auto-generated.
def __patch__init__(self,
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

__patch__init__.__doc__ = cbook.dedent(__patch__init__.__doc__) % artist.kwdocd
Patch.__init__ = __patch__init__

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
    Draw a rectangle with lower left at *xy* = (*x*, *y*) with
    specified *width* and *height*.
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
        # Note: This cannot be calculated until this is added to an Axes
        self._rect_transform = transforms.IdentityTransform()
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def get_path(self):
        """
        Return the vertices of the rectangle
        """
        return Path.unit_rectangle()

    def _update_patch_transform(self):
        """NOTE: This cannot be called until after this has been added
                 to an Axes, otherwise unit conversion will fail. This
                 maxes it very important to call the accessor method and
                 not directly access the transformation member variable.
        """
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
        # special case the degenerate rectangle
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

    def get_xy(self):
        "Return the left and bottom coords of the rectangle"
        return self._x, self._y

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

    def set_xy(self, xy):
        """
        Set the left and bottom coords of the rectangle

        ACCEPTS: 2-item sequence
        """
        self._x, self._y = xy

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

    xy = property(get_xy, set_xy)

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

        .. seealso::
            :class:`Patch`:
                For additional kwargs
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

        .. seealso::
            :class:`Patch`:
                For additional kwargs
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
    """
    Wedge shaped patch.
    """
    def __str__(self):
        return "Wedge(%g,%g)"%(self.theta1,self.theta2)

    def __init__(self, center, r, theta1, theta2, width=None, **kwargs):
        """
        Draw a wedge centered at *x*, *y* center with radius *r* that
        sweeps *theta1* to *theta2* (in degrees).  If *width* is given,
        then a partial wedge is drawn from inner radius *r* - *width*
        to outer radius *r*.

        Valid kwargs are:

        %(Patch)s
        """
        Patch.__init__(self, **kwargs)
        self.center = center
        self.r,self.width = r,width
        self.theta1,self.theta2 = theta1,theta2

        # Inner and outer rings are connected unless the annulus is complete
        delta=theta2-theta1
        if abs((theta2-theta1) - 360) <= 1e-12:
            theta1,theta2 = 0,360
            connector = Path.MOVETO
        else:
            connector = Path.LINETO

        # Form the outer ring
        arc = Path.arc(theta1,theta2)

        if width is not None:
            # Partial annulus needs to draw the outter ring
            # followed by a reversed and scaled inner ring
            v1 = arc.vertices
            v2 = arc.vertices[::-1]*float(r-width)/r
            v = np.vstack([v1,v2,v1[0,:],(0,0)])
            c = np.hstack([arc.codes,arc.codes,connector,Path.CLOSEPOLY])
            c[len(arc.codes)]=connector
        else:
            # Wedge doesn't need an inner ring
            v = np.vstack([arc.vertices,[(0,0),arc.vertices[0,:],(0,0)]])
            c = np.hstack([arc.codes,[connector,connector,Path.CLOSEPOLY]])

        # Shift and scale the wedge to the final location.
        v *= r
        v += np.asarray(center)
        self._path = Path(v,c)
        self._patch_transform = transforms.IdentityTransform()
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def get_path(self):
        return self._path


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
                    # The half-arrows contain the midpoint of the stem,
                    # which we can omit from the full arrow. Including it
                    # twice caused a problem with xpdf.
                    coords=np.concatenate([left_half_arrow[:-1],
                                            right_half_arrow[-2::-1]])
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
        # Note: This cannot be calculated until this is added to an Axes
        self._patch_transform = transforms.IdentityTransform()
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

    def _recompute_transform(self):
        """NOTE: This cannot be called until after this has been added
                 to an Axes, otherwise unit conversion will fail. This
                 maxes it very important to call the accessor method and
                 not directly access the transformation member variable.
        """
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
        if 'resolution' in kwargs:
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
    instance---it can not be added directly to a
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



def _pprint_table(_table, leadingspace=2):
    """
    Given the list of list of strings, return a string of REST table format.
    """
    if leadingspace:
        pad = ' '*leadingspace
    else:
        pad  = ''

    columns = [[] for cell in _table[0]]

    for row in _table:
        for column, cell in zip(columns, row):
            column.append(cell)



    col_len = [max([len(cell) for cell in column]) for column in columns]

    lines = []
    table_formatstr = pad + '   '.join([('=' * cl) for cl in col_len])

    lines.append('')
    lines.append(table_formatstr)
    lines.append(pad + '   '.join([cell.ljust(cl) for cell, cl in zip(_table[0], col_len)]))
    lines.append(table_formatstr)

    lines.extend([(pad + '   '.join([cell.ljust(cl) for cell, cl in zip(row, col_len)]))
                  for row in _table[1:]])

    lines.append(table_formatstr)
    lines.append('')
    return "\n".join(lines)


def _pprint_styles(_styles, leadingspace=2):
    """
    A helper function for the _Style class.  Given the dictionary of
    (stylename : styleclass), return a formatted string listing all the
    styles. Used to update the documentation.
    """
    if leadingspace:
        pad = ' '*leadingspace
    else:
        pad  = ''

    names, attrss, clss = [], [], []

    import inspect

    _table = [["Class", "Name", "Attrs"]]

    for name, cls in sorted(_styles.items()):
        args, varargs, varkw, defaults =  inspect.getargspec(cls.__init__)
        if defaults:
            args = [(argname, argdefault) \
                    for argname, argdefault in zip(args[1:], defaults)]
        else:
            args = None

        if args is None:
            argstr = 'None'
        else:
            argstr =  ",".join([("%s=%s" % (an, av)) for an, av in args])

        #adding quotes for now to work around tex bug treating '-' as itemize
        _table.append([cls.__name__, "'%s'"%name, argstr])

    return _pprint_table(_table)



class _Style(object):
    """
    A base class for the Styles. It is meant to be a container class,
    where actual styles are declared as subclass of it, and it
    provides some helper functions.
    """
    def __new__(self, stylename, **kw):
        """
        return the instance of the subclass with the given style name.
        """

        # the "class" should have the _style_list attribute, which is
        # a dictionary of stylname, style class paie.

        _list = stylename.replace(" ","").split(",")
        _name = _list[0].lower()
        try:
            _cls = self._style_list[_name]
        except KeyError:
            raise ValueError("Unknown style : %s" % stylename)

        try:
            _args_pair = [cs.split("=") for cs in _list[1:]]
            _args = dict([(k, float(v)) for k, v in _args_pair])
        except ValueError:
            raise ValueError("Incorrect style argument : %s" % stylename)
        _args.update(kw)

        return _cls(**_args)


    @classmethod
    def get_styles(klass):
        """
        A class method which returns a dictionary of available styles.
        """
        return klass._style_list

    @classmethod
    def pprint_styles(klass):
        """
        A class method which returns a string of the available styles.
        """
        return _pprint_styles(klass._style_list)




class BoxStyle(_Style):
    """
    :class:`BoxStyle` is a container class which defines several
    boxstyle classes, which are used for :class:`FancyBoxPatch`.

    A style object can be created as::

           BoxStyle.Round(pad=0.2)

    or::

           BoxStyle("Round", pad=0.2)

    or::

           BoxStyle("Round, pad=0.2")

    Following boxstyle classes are defined.

    %(AvailableBoxstyles)s

    An instance of any boxstyle class is an callable object,
    whose call signature is::

       __call__(self, x0, y0, width, height, mutation_size, aspect_ratio=1.)

    and returns a :class:`Path` instance. *x0*, *y0*, *width* and
    *height* specify the location and size of the box to be
    drawn. *mutation_scale* determines the overall size of the
    mutation (by which I mean the transformation of the rectangle to
    the fancy box).  *mutation_aspect* determines the aspect-ratio of
    the mutation.

    .. plot:: mpl_examples/pylab_examples/fancybox_demo2.py
    """

    _style_list = {}


    class _Base(object):
        """
        :class:`BBoxTransmuterBase` and its derivatives are used to make a
        fancy box around a given rectangle. The :meth:`__call__` method
        returns the :class:`~matplotlib.path.Path` of the fancy box. This
        class is not an artist and actual drawing of the fancy box is done
        by the :class:`FancyBboxPatch` class.
        """

        # The derived classes are required to be able to be initialized
        # w/o arguments, i.e., all its argument (except self) must have
        # the default values.

        def __init__(self):
            """
            initializtion.
            """
            super(BoxStyle._Base, self).__init__()




        def transmute(self, x0, y0, width, height, mutation_size):
            """
            The transmute method is a very core of the
            :class:`BboxTransmuter` class and must be overriden in the
            subclasses. It receives the location and size of the
            rectangle, and the mutation_size, with which the amount of
            padding and etc. will be scaled. It returns a
            :class:`~matplotlib.path.Path` instance.
            """
            raise NotImplementedError('Derived must override')



        def __call__(self, x0, y0, width, height, mutation_size,
                     aspect_ratio=1.):
            """
            Given the location and size of the box, return the path of
            the box around it.

              - *x0*, *y0*, *width*, *height* : location and size of the box
              - *mutation_size* : a reference scale for the mutation.
              - *aspect_ratio* : aspect-ration for the mutation.
            """
            # The __call__ method is a thin wrapper around the transmute method
            # and take care of the aspect.

            if aspect_ratio is not None:
                # Squeeze the given height by the aspect_ratio
                y0, height = y0/aspect_ratio, height/aspect_ratio
                # call transmute method with squeezed height.
                path = self.transmute(x0, y0, width, height, mutation_size)
                vertices, codes = path.vertices, path.codes
                # Restore the height
                vertices[:,1] = vertices[:,1] * aspect_ratio
                return Path(vertices, codes)
            else:
                return self.transmute(x0, y0, width, height, mutation_size)



    class Square(_Base):
        """
        A simple square box.
        """

        def __init__(self, pad=0.3):
            """
             *pad*
                amount of padding
            """

            self.pad = pad
            super(BoxStyle.Square, self).__init__()

        def transmute(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # width and height with padding added.
            width, height = width + 2.*pad, \
                            height + 2.*pad,

            # boundary of the padded box
            x0, y0 = x0-pad, y0-pad,
            x1, y1 = x0+width, y0 + height

            cp = [(x0, y0), (x1, y0), (x1, y1), (x0, y1),
                  (x0, y0), (x0, y0)]

            com = [Path.MOVETO,
                   Path.LINETO,
                   Path.LINETO,
                   Path.LINETO,
                   Path.LINETO,
                   Path.CLOSEPOLY]

            path = Path(cp, com)

            return path

    _style_list["square"] = Square


    class LArrow(_Base):
        """
        (left) Arrow Box
        """

        def __init__(self, pad=0.3):
            self.pad = pad
            super(BoxStyle.LArrow, self).__init__()

        def transmute(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # width and height with padding added.
            width, height = width + 2.*pad, \
                            height + 2.*pad,

            # boundary of the padded box
            x0, y0 = x0-pad, y0-pad,
            x1, y1 = x0+width, y0 + height

            dx = (y1-y0)/2.
            dxx = dx*.5
            # adjust x0.  1.4 <- sqrt(2)
            x0 = x0 + pad / 1.4

            cp = [(x0+dxx, y0), (x1, y0), (x1, y1), (x0+dxx, y1),
                  (x0+dxx, y1+dxx), (x0-dx, y0+dx), (x0+dxx, y0-dxx), # arrow
                  (x0+dxx, y0), (x0+dxx, y0)]

            com = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
                   Path.LINETO, Path.LINETO, Path.LINETO,
                   Path.LINETO, Path.CLOSEPOLY]

            path = Path(cp, com)

            return path
    _style_list["larrow"] = LArrow



    class RArrow(LArrow):
        """
        (right) Arrow Box
        """

        def __init__(self, pad=0.3):
            self.pad = pad
            super(BoxStyle.RArrow, self).__init__()

        def transmute(self, x0, y0, width, height, mutation_size):

            p = BoxStyle.LArrow.transmute(self, x0, y0,
                                          width, height, mutation_size)

            p.vertices[:,0] = 2*x0 + width - p.vertices[:,0]

            return p


    _style_list["rarrow"] = RArrow


    class Round(_Base):
        """
        A box with round corners.
        """

        def __init__(self, pad=0.3, rounding_size=None):
            """
            *pad*
              amount of padding

            *rounding_size*
              rounding radius of corners. *pad* if None
            """
            self.pad = pad
            self.rounding_size = rounding_size
            super(BoxStyle.Round, self).__init__()

        def transmute(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # size of the roudning corner
            if self.rounding_size:
                dr = mutation_size * self.rounding_size
            else:
                dr = pad

            width, height = width + 2.*pad, \
                            height + 2.*pad,


            x0, y0 = x0-pad, y0-pad,
            x1, y1 = x0+width, y0 + height

            # Round corners are implemented as quadratic bezier. eg.
            # [(x0, y0-dr), (x0, y0), (x0+dr, y0)] for lower left corner.
            cp = [(x0+dr, y0),
                  (x1-dr, y0),
                  (x1, y0), (x1, y0+dr),
                  (x1, y1-dr),
                  (x1, y1), (x1-dr, y1),
                  (x0+dr, y1),
                  (x0, y1), (x0, y1-dr),
                  (x0, y0+dr),
                  (x0, y0), (x0+dr, y0),
                  (x0+dr, y0)]

            com = [Path.MOVETO,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.LINETO,
                   Path.CURVE3, Path.CURVE3,
                   Path.CLOSEPOLY]

            path = Path(cp, com)

            return path

    _style_list["round"] = Round


    class Round4(_Base):
        """
        Another box with round edges.
        """

        def __init__(self, pad=0.3, rounding_size=None):
            """
            *pad*
              amount of padding

            *rounding_size*
              rounding size of edges. *pad* if None
            """

            self.pad = pad
            self.rounding_size = rounding_size
            super(BoxStyle.Round4, self).__init__()

        def transmute(self, x0, y0, width, height, mutation_size):

            # padding
            pad = mutation_size * self.pad

            # roudning size. Use a half of the pad if not set.
            if self.rounding_size:
                dr = mutation_size * self.rounding_size
            else:
                dr = pad / 2.

            width, height = width + 2.*pad - 2*dr, \
                            height + 2.*pad - 2*dr,


            x0, y0 = x0-pad+dr, y0-pad+dr,
            x1, y1 = x0+width, y0 + height


            cp = [(x0, y0),
                  (x0+dr, y0-dr), (x1-dr, y0-dr), (x1, y0),
                  (x1+dr, y0+dr), (x1+dr, y1-dr), (x1, y1),
                  (x1-dr, y1+dr), (x0+dr, y1+dr), (x0, y1),
                  (x0-dr, y1-dr), (x0-dr, y0+dr), (x0, y0),
                  (x0, y0)]

            com = [Path.MOVETO,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CURVE4, Path.CURVE4, Path.CURVE4,
                   Path.CLOSEPOLY]

            path = Path(cp, com)

            return path

    _style_list["round4"] = Round4


    class Sawtooth(_Base):
        """
        A sawtooth box.
        """

        def __init__(self, pad=0.3, tooth_size=None):
            """
            *pad*
              amount of padding

            *tooth_size*
              size of the sawtooth. pad* if None
            """
            self.pad = pad
            self.tooth_size = tooth_size
            super(BoxStyle.Sawtooth, self).__init__()

        def _get_sawtooth_vertices(self, x0, y0, width, height, mutation_size):


            # padding
            pad = mutation_size * self.pad

            # size of sawtooth
            if self.tooth_size is None:
                tooth_size = self.pad * .5 * mutation_size
            else:
                tooth_size = self.tooth_size * mutation_size

            tooth_size2 = tooth_size / 2.
            width, height = width + 2.*pad - tooth_size, \
                            height + 2.*pad - tooth_size,

            # the sizes of the vertical and horizontal sawtooth are
            # separately adjusted to fit the given box size.
            dsx_n = int(round((width  - tooth_size) / (tooth_size * 2))) * 2
            dsx = (width  - tooth_size) / dsx_n
            dsy_n = int(round((height  - tooth_size) / (tooth_size * 2))) * 2
            dsy = (height  - tooth_size) / dsy_n


            x0, y0 = x0-pad+tooth_size2, y0-pad+tooth_size2
            x1, y1 = x0+width, y0 + height


            bottom_saw_x = [x0] + \
                           [x0 + tooth_size2 + dsx*.5* i for i in range(dsx_n*2)] + \
                           [x1 - tooth_size2]
            bottom_saw_y = [y0] + \
                           [y0 - tooth_size2, y0, y0 + tooth_size2, y0] * dsx_n + \
                           [y0 - tooth_size2]

            right_saw_x = [x1] + \
                          [x1 + tooth_size2, x1, x1 - tooth_size2, x1] * dsx_n + \
                          [x1 + tooth_size2]
            right_saw_y = [y0] + \
                          [y0 + tooth_size2 + dsy*.5* i for i in range(dsy_n*2)] + \
                          [y1 - tooth_size2]

            top_saw_x = [x1] + \
                        [x1 - tooth_size2 - dsx*.5* i for i in range(dsx_n*2)] + \
                        [x0 + tooth_size2]
            top_saw_y = [y1] + \
                        [y1 + tooth_size2, y1, y1 - tooth_size2, y1] * dsx_n + \
                        [y1 + tooth_size2]

            left_saw_x = [x0] + \
                         [x0 - tooth_size2, x0, x0 + tooth_size2, x0] * dsy_n + \
                         [x0 - tooth_size2]
            left_saw_y = [y1] + \
                         [y1 - tooth_size2 - dsy*.5* i for i in range(dsy_n*2)] + \
                         [y0 + tooth_size2]

            saw_vertices = zip(bottom_saw_x, bottom_saw_y) + \
                           zip(right_saw_x, right_saw_y) + \
                           zip(top_saw_x, top_saw_y) + \
                           zip(left_saw_x, left_saw_y) + \
                           [(bottom_saw_x[0], bottom_saw_y[0])]

            return saw_vertices


        def transmute(self, x0, y0, width, height, mutation_size):

            saw_vertices = self._get_sawtooth_vertices(x0, y0, width, height, mutation_size)
            path = Path(saw_vertices)
            return path

    _style_list["sawtooth"] = Sawtooth


    class Roundtooth(Sawtooth):
        """
        A roundtooth(?) box.
        """

        def __init__(self, pad=0.3, tooth_size=None):
            """
            *pad*
              amount of padding

            *tooth_size*
              size of the sawtooth. pad* if None
            """
            super(BoxStyle.Roundtooth, self).__init__(pad, tooth_size)


        def transmute(self, x0, y0, width, height, mutation_size):

            saw_vertices = self._get_sawtooth_vertices(x0, y0, width, height, mutation_size)

            cp = [Path.MOVETO] + ([Path.CURVE3, Path.CURVE3] * ((len(saw_vertices)-1)//2))
            path = Path(saw_vertices, cp)

            return path

    _style_list["roundtooth"] = Roundtooth

    __doc__ = cbook.dedent(__doc__) % \
           {"AvailableBoxstyles": _pprint_styles(_style_list)}


class FancyBboxPatch(Patch):
    """
    Draw a fancy box around a rectangle with lower left at *xy*=(*x*,
    *y*) with specified width and height.

    :class:`FancyBboxPatch` class is similar to :class:`Rectangle`
    class, but it draws a fancy box around the rectangle. The
    transformation of the rectangle box to the fancy box is delegated
    to the :class:`BoxTransmuterBase` and its derived classes.

    """

    def __str__(self):
        return self.__class__.__name__ \
            + "FancyBboxPatch(%g,%g;%gx%g)" % (self._x, self._y, self._width, self._height)

    def __init__(self, xy, width, height,
                 boxstyle="round",
                 bbox_transmuter=None,
                 mutation_scale=1.,
                 mutation_aspect=None,
                 **kwargs):
        """
        *xy* = lower left corner

        *width*, *height*

        *boxstyle* determines what kind of fancy box will be drawn. It
        can be a string of the style name with a comma separated
        attribute, or an instance of :class:`BoxStyle`. Following box
        styles are available.

        %(AvailableBoxstyles)s

        *mutation_scale* : a value with which attributes of boxstyle
        (e.g., pad) will be scaled. default=1.

        *mutation_aspect* : The height of the rectangle will be
        squeezed by this value before the mutation and the mutated
        box will be stretched by the inverse of it. default=None.

        Valid kwargs are:
        %(Patch)s
        """

        Patch.__init__(self, **kwargs)

        self._x = xy[0]
        self._y = xy[1]
        self._width = width
        self._height = height

        if boxstyle == "custom":
            if bbox_transmuter is None:
                raise ValueError("bbox_transmuter argument is needed with custom boxstyle")
            self._bbox_transmuter = bbox_transmuter
        else:
            self.set_boxstyle(boxstyle)

        self._mutation_scale=mutation_scale
        self._mutation_aspect=mutation_aspect


    kwdoc = dict()
    kwdoc["AvailableBoxstyles"]=_pprint_styles(BoxStyle._style_list)
    kwdoc.update(artist.kwdocd)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % kwdoc
    del kwdoc


    def set_boxstyle(self, boxstyle=None, **kw):
        """
        Set the box style.

        *boxstyle* can be a string with boxstyle name with optional
        comma-separated attributes. Alternatively, the attrs can
        be provided as keywords::

            set_boxstyle("round,pad=0.2")
            set_boxstyle("round", pad=0.2)

        Old attrs simply are forgotten.

        Without argument (or with *boxstyle* = None), it returns
        available box styles.

        ACCEPTS: [ %(AvailableBoxstyles)s ]

        """

        if boxstyle==None:
            return BoxStyle.pprint_styles()

        if isinstance(boxstyle, BoxStyle._Base):
            self._bbox_transmuter = boxstyle
        elif callable(boxstyle):
            self._bbox_transmuter = boxstyle
        else:
            self._bbox_transmuter = BoxStyle(boxstyle, **kw)


    kwdoc = dict()
    kwdoc["AvailableBoxstyles"]=_pprint_styles(BoxStyle._style_list)
    kwdoc.update(artist.kwdocd)
    set_boxstyle.__doc__ = cbook.dedent(set_boxstyle.__doc__) % kwdoc
    del kwdoc

    def set_mutation_scale(self, scale):
        """
        Set the mutation scale.

        ACCEPTS: float
        """
        self._mutation_scale=scale

    def get_mutation_scale(self):
        """
        Return the mutation scale.
        """
        return self._mutation_scale

    def set_mutation_aspect(self, aspect):
        """
        Set the aspect ratio of the bbox mutation.

        ACCEPTS: float
        """
        self._mutation_aspect=aspect

    def get_mutation_aspect(self):
        """
        Return the aspect ratio of the bbox mutation.
        """
        return self._mutation_aspect

    def get_boxstyle(self):
        "Return the boxstyle object"
        return self._bbox_transmuter

    def get_path(self):
        """
        Return the mutated path of the rectangle
        """

        _path = self.get_boxstyle()(self._x, self._y,
                                    self._width, self._height,
                                    self.get_mutation_scale(),
                                    self.get_mutation_aspect())
        return _path


    # Following methods are borrowed from the Rectangle class.

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





from matplotlib.bezier import split_bezier_intersecting_with_closedpath
from matplotlib.bezier import get_intersection, inside_circle, get_parallels
from matplotlib.bezier import make_wedged_bezier2
from matplotlib.bezier import split_path_inout, get_cos_sin


class ConnectionStyle(_Style):
    """
    :class:`ConnectionStyle` is a container class which defines
    several connectionstyle classes, which is used to create a path
    between two points. These are mainly used with
    :class:`FancyArrowPatch`.

    A connectionstyle object can be either created as::

           ConnectionStyle.Arc3(rad=0.2)

    or::

           ConnectionStyle("Arc3", rad=0.2)

    or::

           ConnectionStyle("Arc3, rad=0.2")

    The following classes are defined

    %(AvailableConnectorstyles)s


    An instance of any connection style class is an callable object,
    whose call signature is::

        __call__(self, posA, posB, patchA=None, patchB=None, shrinkA=2., shrinkB=2.)

    and it returns a :class:`Path` instance. *posA* and *posB* are
    tuples of x,y coordinates of the two points to be
    connected. *patchA* (or *patchB*) is given, the returned path is
    clipped so that it start (or end) from the boundary of the
    patch. The path is further shrunk by *shrinkA* (or *shrinkB*)
    which is given in points.

    """

    _style_list = {}


    class _Base(object):
        """
        A base class for connectionstyle classes. The dervided needs
        to implement a *connect* methods whose call signature is::

          connect(posA, posB)

        where posA and posB are tuples of x, y coordinates to be
        connected.  The methods needs to return a path connecting two
        points. This base class defines a __call__ method, and few
        helper methods.
        """

        class SimpleEvent:
            def __init__(self, xy):
                self.x, self.y = xy

        def _clip(self, path, patchA, patchB):
            """
            Clip the path to the boundary of the patchA and patchB.
            The starting point of the path needed to be inside of the
            patchA and the end point inside the patch B. The *contains*
            methods of each patch object is utilized to test if the point
            is inside the path.
            """

            if patchA:
                def insideA(xy_display):
                    #xy_display = patchA.get_data_transform().transform_point(xy_data)
                    xy_event =  ConnectionStyle._Base.SimpleEvent(xy_display)
                    return patchA.contains(xy_event)[0]

                try:
                    left, right = split_path_inout(path, insideA)
                except ValueError:
                    right = path

                path = right

            if patchB:
                def insideB(xy_display):
                    #xy_display = patchB.get_data_transform().transform_point(xy_data)
                    xy_event =  ConnectionStyle._Base.SimpleEvent(xy_display)
                    return patchB.contains(xy_event)[0]

                try:
                    left, right = split_path_inout(path, insideB)
                except ValueError:
                    left = path

                path = left

            return path


        def _shrink(self, path, shrinkA, shrinkB):
            """
            Shrink the path by fixed size (in points) with shrinkA and shrinkB
            """
            if shrinkA:
                x, y = path.vertices[0]
                insideA = inside_circle(x, y, shrinkA)

                left, right = split_path_inout(path, insideA)
                path = right

            if shrinkB:
                x, y = path.vertices[-1]
                insideB = inside_circle(x, y, shrinkB)

                left, right = split_path_inout(path, insideB)
                path = left

            return path

        def __call__(self, posA, posB,
                     shrinkA=2., shrinkB=2., patchA=None, patchB=None):
            """
            Calls the *connect* method to create a path between *posA*
             and *posB*. The path is clipped and shrinked.
            """

            path = self.connect(posA, posB)

            clipped_path = self._clip(path, patchA, patchB)
            shrinked_path = self._shrink(clipped_path, shrinkA, shrinkB)

            return shrinked_path


    class Arc3(_Base):
        """
        Creates a simple quadratic bezier curve between two
        points. The curve is created so that the middle contol points
        (C1) is located at the same distance from the start (C0) and
        end points(C2) and the distance of the C1 to the line
        connecting C0-C2 is *rad* times the distance of C0-C2.
        """

        def __init__(self, rad=0.):
            """
            *rad*
              curvature of the curve.
            """
            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB
            x12, y12 = (x1 + x2)/2., (y1 + y2)/2.
            dx, dy = x2 - x1, y2 - y1

            f = self.rad

            cx, cy = x12 + f*dy, y12 - f*dx

            vertices = [(x1, y1),
                        (cx, cy),
                        (x2, y2)]
            codes = [Path.MOVETO,
                     Path.CURVE3,
                     Path.CURVE3]

            return Path(vertices, codes)

    _style_list["arc3"] = Arc3


    class Angle3(_Base):
        """
        Creates a simple quadratic bezier curve between two
        points. The middle control points is placed at the
        intersecting point of two lines which crosses the start (or
        end) point and has a angle of angleA (or angleB).
        """


        def __init__(self, angleA=90, angleB=0):
            """
            *angleA*
              starting angle of the path

            *angleB*
              ending angle of the path
            """

            self.angleA = angleA
            self.angleB = angleB


        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            cosA, sinA = math.cos(self.angleA/180.*math.pi),\
                         math.sin(self.angleA/180.*math.pi),
            cosB, sinB = math.cos(self.angleB/180.*math.pi),\
                         math.sin(self.angleB/180.*math.pi),

            cx, cy = get_intersection(x1, y1, cosA, sinA,
                                      x2, y2, cosB, sinB)

            vertices = [(x1, y1), (cx, cy), (x2, y2)]
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]

            return Path(vertices, codes)

    _style_list["angle3"] = Angle3


    class Angle(_Base):
        """
        Creates a picewise continuous quadratic bezier path between
        two points. The path has a one passing-through point placed at
        the intersecting point of two lines which crosses the start
        (or end) point and has a angle of angleA (or angleB).  The
        connecting edges are rounded with *rad*.
        """

        def __init__(self, angleA=90, angleB=0, rad=0.):
            """
            *angleA*
              starting angle of the path

            *angleB*
              ending angle of the path

            *rad*
              rounding radius of the edge
            """

            self.angleA = angleA
            self.angleB = angleB

            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            cosA, sinA = math.cos(self.angleA/180.*math.pi),\
                         math.sin(self.angleA/180.*math.pi),
            cosB, sinB = math.cos(self.angleB/180.*math.pi),\
                         -math.sin(self.angleB/180.*math.pi),

            cx, cy = get_intersection(x1, y1, cosA, sinA,
                                      x2, y2, cosB, sinB)

            vertices = [(x1, y1)]
            codes = [Path.MOVETO]

            if self.rad == 0.:
                vertices.append((cx, cy))
                codes.append(Path.LINETO)
            else:
                vertices.extend([(cx - self.rad * cosA, cy - self.rad * sinA),
                                 (cx, cy),
                                 (cx + self.rad * cosB, cy + self.rad * sinB)])
                codes.extend([Path.LINETO, Path.CURVE3, Path.CURVE3])

            vertices.append((x2, y2))
            codes.append(Path.LINETO)

            return Path(vertices, codes)

    _style_list["angle"] = Angle


    class Arc(_Base):
        """
        Creates a picewise continuous quadratic bezier path between
        two points. The path can have two passing-through points, a
        point placed at the distance of armA and angle of angleA from
        point A, another point with respect to point B. The edges are
        rounded with *rad*.
        """

        def __init__(self, angleA=0, angleB=0, armA=None, armB=None, rad=0.):
            """
            *angleA* :
              starting angle of the path

            *angleB* :
              ending angle of the path

            *armA* :
              length of the starting arm

            *armB* :
              length of the ending arm

            *rad* :
              rounding radius of the edges
            """

            self.angleA = angleA
            self.angleB = angleB
            self.armA = armA
            self.armB = armB

            self.rad = rad

        def connect(self, posA, posB):
            x1, y1 = posA
            x2, y2 = posB

            vertices = [(x1, y1)]
            rounded = []
            codes = [Path.MOVETO]

            if self.armA:
                cosA = math.cos(self.angleA/180.*math.pi)
                sinA = math.sin(self.angleA/180.*math.pi)
                #x_armA, y_armB
                d = self.armA - self.rad
                rounded.append((x1 + d*cosA, y1 + d*sinA))
                d = self.armA
                rounded.append((x1 + d*cosA, y1 + d*sinA))

            if self.armB:
                cosB = math.cos(self.angleB/180.*math.pi)
                sinB = math.sin(self.angleB/180.*math.pi)
                x_armB, y_armB = x2 + self.armB*cosB, y2 + self.armB*sinB

                if rounded:
                    xp, yp = rounded[-1]
                    dx, dy = x_armB - xp, y_armB - yp
                    dd = (dx*dx + dy*dy)**.5

                    rounded.append((xp + self.rad*dx/dd, yp  + self.rad*dy/dd))
                    vertices.extend(rounded)
                    codes.extend([Path.LINETO,
                                  Path.CURVE3,
                                  Path.CURVE3])
                else:
                    xp, yp = vertices[-1]
                    dx, dy = x_armB - xp, y_armB - yp
                    dd = (dx*dx + dy*dy)**.5

                d = dd - self.rad
                rounded = [(xp + d*dx/dd, yp  + d*dy/dd),
                           (x_armB, y_armB)]

            if rounded:
                xp, yp = rounded[-1]
                dx, dy = x2 - xp, y2 - yp
                dd = (dx*dx + dy*dy)**.5

                rounded.append((xp + self.rad*dx/dd, yp  + self.rad*dy/dd))
                vertices.extend(rounded)
                codes.extend([Path.LINETO,
                              Path.CURVE3,
                              Path.CURVE3])

            vertices.append((x2, y2))
            codes.append(Path.LINETO)

            return Path(vertices, codes)

    _style_list["arc"] = Arc

    __doc__ = cbook.dedent(__doc__) % \
           {"AvailableConnectorstyles": _pprint_styles(_style_list)}



class ArrowStyle(_Style):
    """
    :class:`ArrowStyle` is a container class which defines several
    arrowstyle classes, which is used to create an arrow path along a
    given path. These are mainly used with :class:`FancyArrowPatch`.

    A arrowstyle object can be either created as::

           ArrowStyle.Fancy(head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy", head_length=.4, head_width=.4, tail_width=.4)

    or::

           ArrowStyle("Fancy, head_length=.4, head_width=.4, tail_width=.4")

    The following classes are defined

    %(AvailableArrowstyles)s


    An instance of any arrow style class is an callable object,
    whose call signature is::

        __call__(self, path, mutation_size, linewidth, aspect_ratio=1.)

    and it returns a tuple of a :class:`Path` instance and a boolean
    value. *path* is a :class:`Path` instance along witch the arrow
    will be drawn. *mutation_size* and *aspect_ratio* has a same
    meaning as in :class:`BoxStyle`. *linewidth* is a line width to be
    stroked. This is meant to be used to correct the location of the
    head so that it does not overshoot the destination point, but not all
    classes support it.

    .. plot:: mpl_examples/pylab_examples/fancyarrow_demo.py
    """


    _style_list = {}

    class _Base(object):
        """
        Arrow Transmuter Base class

        ArrowTransmuterBase and its derivatives are used to make a fancy
        arrow around a given path. The __call__ method returns a path
        (which will be used to create a PathPatch instance) and a boolean
        value indicating the path is open therefore is not fillable.  This
        class is not an artist and actual drawing of the fancy arrow is
        done by the FancyArrowPatch class.

        """

        # The derived classes are required to be able to be initialized
        # w/o arguments, i.e., all its argument (except self) must have
        # the default values.

        def __init__(self):
            super(ArrowStyle._Base, self).__init__()

        @staticmethod
        def ensure_quadratic_bezier(path):
            """ Some ArrowStyle class only wokrs with a simple
            quaratic bezier curve (created with Arc3Connetion or
            Angle3Connector). This static method is to check if the
            provided path is a simple quadratic bezier curve and returns
            its control points if true.
            """
            segments = list(path.iter_segments())
            assert len(segments) == 2

            assert segments[0][1] == Path.MOVETO
            assert segments[1][1] == Path.CURVE3

            return list(segments[0][0]) + list(segments[1][0])


        def transmute(self, path, mutation_size, linewidth):
            """
            The transmute method is a very core of the ArrowStyle
            class and must be overriden in the subclasses. It receives the
            path object along which the arrow will be drawn, and the
            mutation_size, with which the amount arrow head and etc. will
            be scaled. It returns a Path instance. The linewidth may be
            used to adjust the the path so that it does not pass beyond
            the given points.
            """

            raise NotImplementedError('Derived must override')



        def __call__(self, path, mutation_size, linewidth,
                     aspect_ratio=1.):
            """
            The __call__ method is a thin wrapper around the transmute method
            and take care of the aspect ratio.
            """

            if aspect_ratio is not None:
                # Squeeze the given height by the aspect_ratio

                vertices, codes = path.vertices[:], path.codes[:]
                # Squeeze the height
                vertices[:,1] = vertices[:,1] / aspect_ratio
                path_shrinked = Path(vertices, codes)
                # call transmute method with squeezed height.
                path_mutated, closed = self.transmute(path_shrinked, linewidth,
                                                      mutation_size)
                vertices, codes = path_mutated.vertices, path_mutated.codes
                # Restore the height
                vertices[:,1] = vertices[:,1] * aspect_ratio
                return Path(vertices, codes), closed
            else:
                return self.transmute(path, mutation_size, linewidth)



    class _Curve(_Base):
        """
        A simple arrow which will work with any path instance. The
        returned path is simply concatenation of the original path + at
        most two paths representing the arrow at the begin point and the
        at the end point. The returned path is not closed and only meant
        to be stroked.
        """

        def __init__(self, beginarrow=None, endarrow=None,
                     head_length=.2, head_width=.1):
            """
            The arrows are drawn if *beginarrow* and/or *endarrow* are
            true. *head_length* and *head_width* determines the size of
            the arrow relative to the *mutation scale*.
            """
            self.beginarrow, self.endarrow = beginarrow, endarrow
            self.head_length, self.head_width = \
                    head_length, head_width
            super(ArrowStyle._Curve, self).__init__()


        def _get_pad_projected(self, x0, y0, x1, y1, linewidth):
            # when no arrow head is drawn

            dx, dy = x0 - x1, y0 - y1
            cp_distance = math.sqrt(dx**2 + dy**2)

            # padx_projected, pady_projected : amount of pad to account
            # projection of the wedge
            padx_projected = (.5*linewidth)
            pady_projected = (.5*linewidth)

            # apply pad for projected edge
            ddx = padx_projected * dx / cp_distance
            ddy = pady_projected * dy / cp_distance

            return ddx, ddy

        def _get_arrow_wedge(self, x0, y0, x1, y1,
                             head_dist, cos_t, sin_t, linewidth
                             ):
            """
            Return the paths for arrow heads. Since arrow lines are
            drawn with capstyle=projected, The arrow is goes beyond the
            desired point. This method also returns the amount of the path
            to be shrinked so that it does not overshoot.
            """

            # arrow from x0, y0 to x1, y1


            dx, dy = x0 - x1, y0 - y1
            cp_distance = math.sqrt(dx**2 + dy**2)

            # padx_projected, pady_projected : amount of pad for account
            # the overshooting of the projection of the wedge
            padx_projected = (.5*linewidth / cos_t)
            pady_projected = (.5*linewidth / sin_t)

            # apply pad for projected edge
            ddx = padx_projected * dx / cp_distance
            ddy = pady_projected * dy / cp_distance

            # offset for arrow wedge
            dx, dy = dx / cp_distance * head_dist, dy / cp_distance * head_dist

            dx1, dy1 = cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy
            dx2, dy2 = cos_t * dx - sin_t * dy, sin_t * dx + cos_t * dy

            vertices_arrow = [(x1+ddx+dx1, y1+ddy+dy1),
                              (x1+ddx, y1++ddy),
                              (x1+ddx+dx2, y1+ddy+dy2)]
            codes_arrow = [Path.MOVETO,
                           Path.LINETO,
                           Path.LINETO]

            return vertices_arrow, codes_arrow, ddx, ddy


        def transmute(self, path, mutation_size, linewidth):

            head_length, head_width = self.head_length * mutation_size, \
                                      self.head_width * mutation_size
            head_dist = math.sqrt(head_length**2 + head_width**2)
            cos_t, sin_t = head_length / head_dist, head_width / head_dist


            # begin arrow
            x0, y0 = path.vertices[0]
            x1, y1 = path.vertices[1]

            if self.beginarrow:
                verticesA, codesA, ddxA, ddyA = \
                           self._get_arrow_wedge(x1, y1, x0, y0,
                                                 head_dist, cos_t, sin_t,
                                                 linewidth)
            else:
                verticesA, codesA = [], []
                #ddxA, ddyA = self._get_pad_projected(x1, y1, x0, y0, linewidth)
                ddxA, ddyA = 0., 0., #self._get_pad_projected(x1, y1, x0, y0, linewidth)

            # end arrow
            x2, y2 = path.vertices[-2]
            x3, y3 = path.vertices[-1]

            if self.endarrow:
                verticesB, codesB, ddxB, ddyB = \
                           self._get_arrow_wedge(x2, y2, x3, y3,
                                                 head_dist, cos_t, sin_t,
                                                 linewidth)
            else:
                verticesB, codesB = [], []
                ddxB, ddyB = 0., 0. #self._get_pad_projected(x2, y2, x3, y3, linewidth)


            # this simple code will not work if ddx, ddy is greater than
            # separation bettern vertices.
            vertices = np.concatenate([verticesA + [(x0+ddxA, y0+ddyA)],
                                       path.vertices[1:-1],
                                       [(x3+ddxB, y3+ddyB)] + verticesB])
            codes = np.concatenate([codesA,
                                    path.codes,
                                    codesB])

            p = Path(vertices, codes)

            return p, False


    class Curve(_Curve):
        """
        A simple curve without any arrow head.
        """

        def __init__(self):
            super(ArrowStyle.Curve, self).__init__( \
                beginarrow=False, endarrow=False)

    _style_list["-"] = Curve



    class CurveA(_Curve):
        """
        An arrow with a head at its begin point.
        """

        def __init__(self, head_length=.4, head_width=.2):
            """
            *head_length*
              length of the arrow head

            *head_width*
              width of the arrow head
            """

            super(ArrowStyle.CurveA, self).__init__( \
                beginarrow=True, endarrow=False,
                head_length=head_length, head_width=head_width )

    _style_list["<-"] = CurveA

    class CurveB(_Curve):
        """
        An arrow with a head at its end point.
        """

        def __init__(self, head_length=.4, head_width=.2):
            """
            *head_length*
              length of the arrow head

            *head_width*
              width of the arrow head
            """

            super(ArrowStyle.CurveB, self).__init__( \
                beginarrow=False, endarrow=True,
                head_length=head_length, head_width=head_width )

    #_style_list["->"] = CurveB
    _style_list["->"] = CurveB


    class CurveAB(_Curve):
        """
        An arrow with heads both at the begin and the end point.
        """

        def __init__(self, head_length=.4, head_width=.2):
            """
            *head_length*
              length of the arrow head

            *head_width*
              width of the arrow head
            """

            super(ArrowStyle.CurveAB, self).__init__( \
                beginarrow=True, endarrow=True,
                head_length=head_length, head_width=head_width )

    #_style_list["<->"] = CurveAB
    _style_list["<->"] = CurveAB


    class _Bracket(_Base):

        def __init__(self, bracketA=None, bracketB=None,
                     widthA=1., widthB=1.,
                     lengthA=0.2, lengthB=0.2,
                     angleA=None, angleB=None,
                     scaleA=None, scaleB=None
                     ):
            self.bracketA, self.bracketB = bracketA, bracketB
            self.widthA, self.widthB = widthA, widthB
            self.lengthA, self.lengthB = lengthA, lengthB
            self.angleA, self.angleB = angleA, angleB
            self.scaleA, self.scaleB= scaleA, scaleB

        def _get_bracket(self, x0, y0,
                         cos_t, sin_t, width, length,
                         ):

            # arrow from x0, y0 to x1, y1
            from matplotlib.bezier import get_normal_points
            x1, y1, x2, y2 = get_normal_points(x0, y0, cos_t, sin_t, width)

            dx, dy = length * cos_t, length * sin_t

            vertices_arrow = [(x1+dx, y1+dy),
                              (x1, y1),
                              (x2, y2),
                              (x2+dx, y2+dy)]
            codes_arrow = [Path.MOVETO,
                           Path.LINETO,
                           Path.LINETO,
                           Path.LINETO]

            return vertices_arrow, codes_arrow


        def transmute(self, path, mutation_size, linewidth):

            if self.scaleA is None:
                scaleA = mutation_size
            else:
                scaleA = self.scaleA

            if self.scaleB is None:
                scaleB = mutation_size
            else:
                scaleB = self.scaleB

            vertices_list, codes_list = [], []

            if self.bracketA:
                x0, y0 = path.vertices[0]
                x1, y1 = path.vertices[1]
                cos_t, sin_t = get_cos_sin(x1, y1, x0, y0)
                verticesA, codesA = self._get_bracket(x0, y0, cos_t, sin_t,
                                                      self.widthA*scaleA,
                                                      self.legnthA*scaleA)
                vertices_list.append(verticesA)
                codes_list.append(codesA)

            vertices_list.append(path.vertices)
            codes_list.append(path.codes)

            if self.bracketB:
                x0, y0 = path.vertices[-1]
                x1, y1 = path.vertices[-2]
                cos_t, sin_t = get_cos_sin(x1, y1, x0, y0)
                verticesB, codesB = self._get_bracket(x0, y0, cos_t, sin_t,
                                                      self.widthB*scaleB,
                                                      self.lengthB*scaleB)
                vertices_list.append(verticesB)
                codes_list.append(codesB)


            vertices = np.concatenate(vertices_list)
            codes = np.concatenate(codes_list)

            p = Path(vertices, codes)

            return p, False


    class BracketB(_Bracket):
        """
        An arrow with a bracket([)  at its end.
        """

        def __init__(self, widthB=1., lengthB=0.2, angleB=None):
            """
            *widthB*
              width of the bracket

            *lengthB*
              length of the bracket

            *angleB*
              angle between the bracket and the line
            """

            super(ArrowStyle.BracketB, self).__init__(None, True,
                     widthB=widthB, lengthB=lengthB, angleB=None )

    #_style_list["-["] = BracketB
    _style_list["-["] = BracketB


    class Simple(_Base):
        """
        A simple arrow. Only works with a quadratic bezier curve.
        """

        def __init__(self, head_length=.5, head_width=.5, tail_width=.2):
            """
            *head_length*
              length of the arrow head

            *head_with*
              width of the arrow head

            *tail_width*
              width of the arrow tail

            """

            self.head_length, self.head_width, self.tail_width = \
                    head_length, head_width, tail_width
            super(ArrowStyle.Simple, self).__init__()

        def transmute(self, path, mutation_size, linewidth):

            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            # divide the path into a head and a tail
            head_length  = self.head_length * mutation_size
            in_f = inside_circle(x2, y2, head_length)
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
            arrow_out, arrow_in = \
                       split_bezier_intersecting_with_closedpath(arrow_path,
                                                                 in_f,
                                                                 tolerence=0.01)

            # head
            head_width = self.head_width * mutation_size
            head_l, head_r = make_wedged_bezier2(arrow_in, head_width/2.,
                                                 wm=.5)



            # tail
            tail_width = self.tail_width * mutation_size
            tail_left, tail_right = get_parallels(arrow_out, tail_width/2.)

            head_right, head_left = head_r, head_l
            patch_path = [(Path.MOVETO, tail_right[0]),
                          (Path.CURVE3, tail_right[1]),
                          (Path.CURVE3, tail_right[2]),
                          (Path.LINETO, head_right[0]),
                          (Path.CURVE3, head_right[1]),
                          (Path.CURVE3, head_right[2]),
                          (Path.CURVE3, head_left[1]),
                          (Path.CURVE3, head_left[0]),
                          (Path.LINETO, tail_left[2]),
                          (Path.CURVE3, tail_left[1]),
                          (Path.CURVE3, tail_left[0]),
                          (Path.LINETO, tail_right[0]),
                          (Path.CLOSEPOLY, tail_right[0]),
                          ]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True

    _style_list["simple"] = Simple


    class Fancy(_Base):
        """
        A fancy arrow. Only works with a quadratic bezier curve.
        """

        def __init__(self, head_length=.4, head_width=.4, tail_width=.4):
            """
            *head_length*
              length of the arrow head

            *head_with*
              width of the arrow head

            *tail_width*
              width of the arrow tail

            """

            self.head_length, self.head_width, self.tail_width = \
                    head_length, head_width, tail_width
            super(ArrowStyle.Fancy, self).__init__()

        def transmute(self, path, mutation_size, linewidth):

            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            # divide the path into a head and a tail
            head_length  = self.head_length * mutation_size
            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]

            # path for head
            in_f = inside_circle(x2, y2, head_length)
            path_out, path_in = \
                       split_bezier_intersecting_with_closedpath(arrow_path,
                                                                 in_f,
                                                                 tolerence=0.01)
            path_head = path_in

            # path for head
            in_f = inside_circle(x2, y2, head_length*.8)
            path_out, path_in = \
                       split_bezier_intersecting_with_closedpath(arrow_path,
                                                                 in_f,
                                                                 tolerence=0.01)
            path_tail = path_out


            # head
            head_width = self.head_width * mutation_size
            head_l, head_r = make_wedged_bezier2(path_head, head_width/2.,
                                                 wm=.6)

            # tail
            tail_width = self.tail_width * mutation_size
            tail_left, tail_right = make_wedged_bezier2(path_tail,
                                                        tail_width*.5,
                                                        w1=1., wm=0.6, w2=0.3)

            # path for head
            in_f = inside_circle(x0, y0, tail_width*.3)
            path_in, path_out = \
                       split_bezier_intersecting_with_closedpath(arrow_path,
                                                                 in_f,
                                                                 tolerence=0.01)
            tail_start = path_in[-1]

            head_right, head_left = head_r, head_l
            patch_path = [(Path.MOVETO, tail_start),
                          (Path.LINETO, tail_right[0]),
                          (Path.CURVE3, tail_right[1]),
                          (Path.CURVE3, tail_right[2]),
                          (Path.LINETO, head_right[0]),
                          (Path.CURVE3, head_right[1]),
                          (Path.CURVE3, head_right[2]),
                          (Path.CURVE3, head_left[1]),
                          (Path.CURVE3, head_left[0]),
                          (Path.LINETO, tail_left[2]),
                          (Path.CURVE3, tail_left[1]),
                          (Path.CURVE3, tail_left[0]),
                          (Path.LINETO, tail_start),
                          (Path.CLOSEPOLY, tail_start),
                          ]
            patch_path2 = [(Path.MOVETO, tail_right[0]),
                          (Path.CURVE3, tail_right[1]),
                          (Path.CURVE3, tail_right[2]),
                          (Path.LINETO, head_right[0]),
                          (Path.CURVE3, head_right[1]),
                          (Path.CURVE3, head_right[2]),
                          (Path.CURVE3, head_left[1]),
                          (Path.CURVE3, head_left[0]),
                          (Path.LINETO, tail_left[2]),
                          (Path.CURVE3, tail_left[1]),
                          (Path.CURVE3, tail_left[0]),
                          (Path.CURVE3, tail_start),
                          (Path.CURVE3, tail_right[0]),
                          (Path.CLOSEPOLY, tail_right[0]),
                          ]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True

    _style_list["fancy"] = Fancy


    class Wedge(_Base):
        """
        Wedge(?) shape. Only wokrs with a quadratic bezier curve.  The
        begin point has a width of the tail_width and the end point has a
        width of 0. At the middle, the width is shrink_factor*tail_width.

        """

        def __init__(self, tail_width=.3, shrink_factor=0.5):
            """
            *tail_width*
              width of the tail

            *shrink_factor*
              fraction of the arrow width at the middle point
            """

            self.tail_width = tail_width
            self.shrink_factor = shrink_factor
            super(ArrowStyle.Wedge, self).__init__()


        def transmute(self, path, mutation_size, linewidth):

            x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

            arrow_path = [(x0, y0), (x1, y1), (x2, y2)]
            b_plus, b_minus = make_wedged_bezier2(arrow_path,
                                                  self.tail_width * mutation_size / 2.,
                                                  wm=self.shrink_factor)


            patch_path = [(Path.MOVETO, b_plus[0]),
                          (Path.CURVE3, b_plus[1]),
                          (Path.CURVE3, b_plus[2]),
                          (Path.LINETO, b_minus[2]),
                          (Path.CURVE3, b_minus[1]),
                          (Path.CURVE3, b_minus[0]),
                          (Path.CLOSEPOLY, b_minus[0]),
                          ]
            path = Path([p for c, p in patch_path], [c for c, p in patch_path])

            return path, True

    _style_list["wedge"] = Wedge

    __doc__ = cbook.dedent(__doc__) % \
           {"AvailableArrowstyles": _pprint_styles(_style_list)}



class FancyArrowPatch(Patch):
    """
    A fancy arrow patch. It draws an arrow using the :class:ArrowStyle.
    """


    def __str__(self):
        return self.__class__.__name__ \
            + "FancyArrowPatch(%g,%g,%g,%g,%g,%g)" % tuple(self._q_bezier)

    def __init__(self, posA=None, posB=None,
                 path=None,
                 arrowstyle="simple",
                 arrow_transmuter=None,
                 connectionstyle="arc3",
                 connector=None,
                 patchA=None,
                 patchB=None,
                 shrinkA=2.,
                 shrinkB=2.,
                 mutation_scale=1.,
                 mutation_aspect=None,
                 **kwargs):
        """
        If *posA* and *posB* is given, a path connecting two point are
        created according to the connectionstyle. The path will be
        clipped with *patchA* and *patchB* and further shirnked by
        *shrinkA* and *shrinkB*. An arrow is drawn along this
        resulting path using the *arrowstyle* parameter. If *path*
        provided, an arrow is drawn along this path and *patchA*,
        *patchB*, *shrinkA*, and *shrinkB* are ignored.

        The *connectionstyle* describes how *posA* and *posB* are
        connected. It can be an instance of the ConnectionStyle class
        (matplotlib.patches.ConnectionStlye) or a string of the
        connectionstyle name, with optional comma-separated
        attributes.  The following connection styles are available.

        %(AvailableConnectorstyles)s


        The *arrowstyle* describes how the fancy arrow will be
        drawn. It can be string of the available arrowstyle names,
        with optional comma-separated attributes, or one of the
        ArrowStyle instance. The optional attributes are meant to be
        scaled with the *mutation_scale*. The following arrow styles are
        available.

        %(AvailableArrowstyles)s

        *mutation_scale* : a value with which attributes of arrowstyle
            (e.g., head_length) will be scaled. default=1.

        *mutation_aspect* : The height of the rectangle will be
            squeezed by this value before the mutation and the mutated
            box will be stretched by the inverse of it. default=None.

        Valid kwargs are:
        %(Patch)s
        """

        if posA is not None and posB is not None and path is None:
            self._posA_posB = [posA, posB]

            if connectionstyle is None:
                connectionstyle = "arc3"
            self.set_connectionstyle(connectionstyle)

        elif posA is None and posB is None and path is not None:
            self._posA_posB = None
            self._connetors = None
        else:
            raise ValueError("either posA and posB, or path need to provided")

        self.patchA = patchA
        self.patchB = patchB
        self.shrinkA = shrinkA
        self.shrinkB = shrinkB

        Patch.__init__(self, **kwargs)

        self._path_original = path

        self.set_arrowstyle(arrowstyle)


        self._mutation_scale=mutation_scale
        self._mutation_aspect=mutation_aspect

        #self._draw_in_display_coordinate = True

    kwdoc = dict()
    kwdoc["AvailableArrowstyles"]=_pprint_styles(ArrowStyle._style_list)
    kwdoc["AvailableConnectorstyles"]=_pprint_styles(ConnectionStyle._style_list)

    kwdoc.update(artist.kwdocd)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % kwdoc
    del kwdoc

    def set_positions(self, posA, posB):
        """ set the begin end end positions of the connecting
        path. Use current vlaue if None.
        """
        if posA is not None: self._posA_posB[0] = posA
        if posB is not None: self._posA_posB[1] = posB


    def set_patchA(self, patchA):
        """ set the begin patch.
        """
        self.patchA = patchA


    def set_patchB(self, patchB):
        """ set the begin patch
        """
        self.patchB = patchB


    def set_connectionstyle(self, connectionstyle, **kw):
        """
        Set the connection style.

        *connectionstyle* can be a string with connectionstyle name with optional
         comma-separated attributes. Alternatively, the attrs can
         be probided as keywords.

         set_connectionstyle("arc,angleA=0,armA=30,rad=10")
         set_connectionstyle("arc", angleA=0,armA=30,rad=10)

        Old attrs simply are forgotten.

        Without argument (or with connectionstyle=None), return
        available styles as a list of strings.
        """

        if connectionstyle==None:
            return ConnectionStyle.pprint_styles()

        if isinstance(connectionstyle, ConnectionStyle._Base):
            self._connector = connectionstyle
        elif callable(connectionstyle):
            # we may need check the calling convention of the given function
            self._connector = connectionstyle
        else:
            self._connector = ConnectionStyle(connectionstyle, **kw)


    def get_connectionstyle(self):
        """
        Return the ConnectionStyle instance
        """
        return self._connector


    def set_arrowstyle(self, arrowstyle=None, **kw):
        """
        Set the arrow style.

        *arrowstyle* can be a string with arrowstyle name with optional
         comma-separated attributes. Alternatively, the attrs can
         be provided as keywords.

         set_arrowstyle("Fancy,head_length=0.2")
         set_arrowstyle("fancy", head_length=0.2)

        Old attrs simply are forgotten.

        Without argument (or with arrowstyle=None), return
        available box styles as a list of strings.
        """

        if arrowstyle==None:
            return ArrowStyle.pprint_styles()

        if isinstance(arrowstyle, ConnectionStyle._Base):
            self._arrow_transmuter = arrowstyle
        else:
            self._arrow_transmuter = ArrowStyle(arrowstyle, **kw)

    def get_arrowstyle(self):
        """
        Return the arrowstyle object
        """
        return self._arrow_transmuter


    def set_mutation_scale(self, scale):
        """
        Set the mutation scale.

        ACCEPTS: float
        """
        self._mutation_scale=scale

    def get_mutation_scale(self):
        """
        Return the mutation scale.
        """
        return self._mutation_scale

    def set_mutation_aspect(self, aspect):
        """
        Set the aspect ratio of the bbox mutation.

        ACCEPTS: float
        """
        self._mutation_aspect=aspect

    def get_mutation_aspect(self):
        """
        Return the aspect ratio of the bbox mutation.
        """
        return self._mutation_aspect


    def get_path(self):
        """
        return the path of the arrow in the data coordinate. Use
        get_path_in_displaycoord() medthod to retrieve the arrow path
        in the disaply coord.
        """
        _path = self.get_path_in_displaycoord()
        return self.get_transform().inverted().transform_path(_path)

    def get_path_in_displaycoord(self):
        """
        Return the mutated path of the arrow in the display coord
        """

        if self._posA_posB is not None:
            posA = self.get_transform().transform_point(self._posA_posB[0])
            posB = self.get_transform().transform_point(self._posA_posB[1])
            _path = self.get_connectionstyle()(posA, posB,
                                               patchA=self.patchA,
                                               patchB=self.patchB,
                                               shrinkA=self.shrinkA,
                                               shrinkB=self.shrinkB
                                               )
        else:
            _path = self.get_transform().transform_path(self._path_original)



        _path, closed = self.get_arrowstyle()(_path,
                                              self.get_mutation_scale(),
                                              self.get_linewidth(),
                                              self.get_mutation_aspect()
                                              )

        if not closed:
            self.fill = False

        return _path



    def draw(self, renderer):
        if not self.get_visible(): return
        #renderer.open_group('patch')
        gc = renderer.new_gc()

        fill_orig = self.fill

        path = self.get_path_in_displaycoord()
        affine = transforms.IdentityTransform()


        if cbook.is_string_like(self._edgecolor) and self._edgecolor.lower()=='none':
            gc.set_linewidth(0)
        else:
            gc.set_foreground(self._edgecolor)
            gc.set_linewidth(self._linewidth)
            gc.set_linestyle(self._linestyle)

        gc.set_antialiased(self._antialiased)
        self._set_gc_clip(gc)
        gc.set_capstyle('round')

        if (not self.fill or self._facecolor is None or
            (cbook.is_string_like(self._facecolor) and self._facecolor.lower()=='none')):
            rgbFace = None
            gc.set_alpha(1.0)
        else:
            r, g, b, a = colors.colorConverter.to_rgba(self._facecolor, self._alpha)
            rgbFace = (r, g, b)
            gc.set_alpha(a)


        if self._hatch:
            gc.set_hatch(self._hatch )


        renderer.draw_path(gc, path, affine, rgbFace)

        self.fill = fill_orig

        #renderer.close_group('patch')
