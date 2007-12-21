from __future__ import division
import math

import matplotlib as mpl
import numpy as npy
import matplotlib.cbook as cbook
import matplotlib.artist as artist
import matplotlib.colors as colors
import matplotlib.lines as lines
import matplotlib.transforms as transforms
import matplotlib.nxutils as nxutils
import matplotlib.mlab as mlab
import matplotlib.artist as artist
from matplotlib import transforms as mtrans
import agg

# these are not available for the object inspector until after the
# class is build so we define an initial set here for the init
# function and they will be overridden after object defn
artist.kwdocd['Patch'] = """\
          alpha: float
          animated: [True | False]
          antialiased or aa: [True | False]
          clip_box: a matplotlib.transform.Bbox instance
          clip_on: [True | False]
          edgecolor or ec: any matplotlib color
          facecolor or fc: any matplotlib color
          figure: a matplotlib.figure.Figure instance
          fill: [True | False]
          hatch: unknown
          label: any string
          linewidth or lw: float
          lod: [True | False]
          transform: a matplotlib.transform transformation instance
          visible: [True | False]
          zorder: any number
          """

class Patch(artist.Artist):
    """
    A patch is a 2D thingy with a face color and an edge color

    If any of edgecolor, facecolor, linewidth, or antialiased are
    None, they default to their rc params setting

    """
    zorder = 1
    def __str__(self):
        return str(self.__class__).split('.')[-1]

    def __init__(self,
                 edgecolor=None,
                 facecolor=None,
                 linewidth=None,
                 antialiased = None,
                 hatch = None,
                 fill=1,
                 **kwargs
                 ):
        """
        The following kwarg properties are supported
        %(Patch)s
        """
        artist.Artist.__init__(self)

        if edgecolor is None: edgecolor = mpl.rcParams['patch.edgecolor']
        if facecolor is None: facecolor = mpl.rcParams['patch.facecolor']
        if linewidth is None: linewidth = mpl.rcParams['patch.linewidth']
        if antialiased is None: antialiased = mpl.rcParams['patch.antialiased']

        self._edgecolor = edgecolor
        self._facecolor = facecolor
        self._linewidth = linewidth
        self._antialiased = antialiased
        self._hatch = hatch
        self.fill = fill

        if len(kwargs): artist.setp(self, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd



    def contains(self, mouseevent):
        """Test whether the mouse event occurred in the patch.

        Returns T/F, {}
        """
        if callable(self._contains): return self._contains(self,mouseevent)

        try:
            # TODO: make this consistent with patch collection algorithm
            x, y = self.get_transform().inverse_xy_tup((mouseevent.x, mouseevent.y))
            xyverts = self.get_verts()
            inside = nxutils.pnpoly(x, y, xyverts)
            #print str(self),"%g,%g is in"%(x,y),xyverts,inside
            return inside,{}
        except ValueError:
            return False,{}

    def update_from(self, other):
        artist.Artist.update_from(self, other)
        self.set_edgecolor(other.get_edgecolor())
        self.set_facecolor(other.get_facecolor())
        self.set_fill(other.get_fill())
        self.set_hatch(other.get_hatch())
        self.set_linewidth(other.get_linewidth())
        self.set_transform(other.get_transform())
        self.set_figure(other.get_figure())
        self.set_alpha(other.get_alpha())


    def get_antialiased(self):
        return self._antialiased

    def get_edgecolor(self):
        return self._edgecolor

    def get_facecolor(self):
        return self._facecolor

    def get_linewidth(self):
        return self._linewidth

    def set_antialiased(self, aa):
        """
        Set whether to use antialiased rendering

        ACCEPTS: [True | False]
        """
        self._antialiased = aa

    def set_edgecolor(self, color):
        """
        Set the patch edge color

        ACCEPTS: any matplotlib color
        """
        self._edgecolor = color

    def set_facecolor(self, color):
        """
        Set the patch face color

        ACCEPTS: any matplotlib color
        """
        self._facecolor = color

    def set_linewidth(self, w):
        """
        Set the patch linewidth in points

        ACCEPTS: float
        """
        self._linewidth = w

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

        hatch can be one of:
        /   - diagonal hatching
        \   - back diagonal
        |   - vertical
        -   - horizontal
        #   - crossed
        x   - crossed diagonal
        letters can be combined, in which case all the specified
        hatchings are done
        if same letter repeats, it increases the density of hatching
        in that direction

        CURRENT LIMITATIONS:
        1. Hatching is supported in the PostScript
        backend only.

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

        verts = self.get_verts()
        tverts = self.get_transform().seq_xy_tups(verts)

        renderer.draw_polygon(gc, rgbFace, tverts)


        #renderer.close_group('patch')

    def get_verts(self):
        """
        Return the vertices of the patch
        """
        raise NotImplementedError('Derived must override')


    def get_window_extent(self, renderer=None):
        verts = self.get_verts()
        tverts = self.get_transform().seq_xy_tups(verts)
        return transforms.bound_vertices(tverts)



    def set_lw(self, val):
        'alias for set_linewidth'
        self.set_linewidth(val)


    def set_ec(self, val):
        'alias for set_edgecolor'
        self.set_edgecolor(val)


    def set_fc(self, val):
        'alias for set_facecolor'
        self.set_facecolor(val)


    def get_aa(self):
        'alias for get_antialiased'
        return self.get_antialiased()


    def get_lw(self):
        'alias for get_linewidth'
        return self.get_linewidth()


    def get_ec(self):
        'alias for get_edgecolor'
        return self.get_edgecolor()


    def get_fc(self):
        'alias for get_facecolor'
        return self.get_facecolor()


class Shadow(Patch):
    def __str__(self):
        return "Shadow(%s)"%(str(self.patch))

    def __init__(self, patch, ox, oy, props=None, **kwargs):
        """
        Create a shadow of the patch offset by ox, oy.  props, if not None is
        a patch property update dictionary.  If None, the shadow will have
        have the same color as the face, but darkened

        kwargs are
        %(Patch)s
        """
        Patch.__init__(self)
        self.ox, self.oy = ox, oy
        self.patch = patch
        self.props = props
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

            self.set_facecolor((r,g,b))
            self.set_edgecolor((r,g,b))

    def get_verts(self):
        verts = self.patch.get_verts()
        xs = self.convert_xunits([x+self.ox for x,y in verts])
        ys = self.convert_yunits([y+self.oy for x,y in verts])
        return zip(xs, ys)

    def _draw(self, renderer):
        'draw the shadow'
        self._update()
        Patch.draw(self, renderer)

class Rectangle(Patch):
    """
    Draw a rectangle with lower left at xy=(x,y) with specified
    width and height

    """

    def __str__(self):
        return str(self.__class__).split('.')[-1] \
            + "(%g,%g;%gx%g)"%(self.xy[0],self.xy[1],self.width,self.height)

    def __init__(self, xy, width, height,
                 **kwargs):
        """
        xy is an x,y tuple lower, left

        width and height are width and height of rectangle

        fill is a boolean indicating whether to fill the rectangle

        Valid kwargs are:
        %(Patch)s
        """

        Patch.__init__(self, **kwargs)

        self.xy  = list(xy)
        self.width, self.height = width, height
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd


    def get_verts(self):
        """
        Return the vertices of the rectangle
        """
        x, y = self.xy
        left = self.convert_xunits(x)
        right = self.convert_xunits(x + self.width)
        bottom = self.convert_yunits(y)
        top = self.convert_yunits(y+self.height)

        return ( (left, bottom), (left, top),
                 (right, top), (right, bottom),
                 )

    def get_x(self):
        "Return the left coord of the rectangle"
        return self.xy[0]

    def get_y(self):
        "Return the bottom coord of the rectangle"
        return self.xy[1]

    def get_width(self):
        "Return the width of the  rectangle"
        return self.width

    def get_height(self):
        "Return the height of the rectangle"
        return self.height

    def set_x(self, x):
        """
        Set the left coord of the rectangle

        ACCEPTS: float
        """
        self.xy[0] = x

    def set_y(self, y):
        """
        Set the bottom coord of the rectangle

        ACCEPTS: float
        """
        self.xy[1] = y

    def set_width(self, w):
        """
        Set the width rectangle

        ACCEPTS: float
        """
        self.width = w

    def set_height(self, h):
        """
        Set the width rectangle

        ACCEPTS: float
        """
        self.height = h

    def set_bounds(self, *args):
        """
        Set the bounds of the rectangle: l,b,w,h

        ACCEPTS: (left, bottom, width, height)
        """
        if len(args)==0:
            l,b,w,h = args[0]
        else:
            l,b,w,h = args
        self.xy = [l,b]
        self.width = w
        self.height = h


class RegularPolygon(Patch):
    """
    A regular polygon patch.
    """
    def __str__(self):
        return "Poly%d(%g,%g)"%(self.numVertices,self.xy[0],self.xy[1])

    def __init__(self, xy, numVertices, radius=5, orientation=0,
                 **kwargs):
        """
        xy is a length 2 tuple (the center)
        numVertices is the number of vertices.
        radius is the distance from the center to each of the vertices.
        orientation is in radians and rotates the polygon.

        Valid kwargs are:
        %(Patch)s
        """
        Patch.__init__(self, **kwargs)

        self.xy = list(xy)
        self.numVertices = numVertices
        self.radius = radius
        self.orientation = orientation

    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd



    def get_verts(self):
        theta = 2*npy.pi/self.numVertices*npy.arange(self.numVertices) + \
                self.orientation
        r = float(self.radius)
        x, y = map(float, self.xy)

        xs = x + r*npy.cos(theta)
        ys = y + r*npy.sin(theta)

        #xs = self.convert_xunits(xs)
        #ys = self.convert_yunits(ys)


        self.verts = zip(xs, ys)

        return self.verts

class Polygon(Patch):
    """
    A general polygon patch.
    """
    def __str__(self):
        return "Poly(%g,%g)"%self.xy[0]

    def __init__(self, xy, **kwargs):
        """
        xy is a sequence of (x,y) 2 tuples

        Valid kwargs are:
        %(Patch)s
        See Patch documentation for additional kwargs
        """

        Patch.__init__(self, **kwargs)
        if not isinstance(xy, list):
            xy = list(xy)
        self.xy = xy
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd



    def get_verts(self):
        xs, ys = zip(*self.xy)[:2]
        xs = self.convert_xunits(xs)
        ys = self.convert_yunits(ys)
        return zip(xs, ys)


class Wedge(Polygon):
    def __str__(self):
        return "Wedge(%g,%g)"%self.xy[0]
    def __init__(self, center, r, theta1, theta2,
                 dtheta=5.0, **kwargs):
        """
        Draw a wedge centered at x,y tuple center with radius r that
        sweeps theta1 to theta2 (angles)

        dtheta is the resolution in degrees

        Valid kwargs are:
        %(Patch)s

        """
        xc, yc = center
        theta1 = float(theta1)
        theta2 = float(theta2)
        dtheta = float(dtheta)
        num_points = abs(theta2 - theta1) / dtheta
        rads = (npy.pi/180.) * npy.linspace(theta1, theta2, num_points, endpoint=True)
        xs = r*npy.cos(rads)+xc
        ys = r*npy.sin(rads)+yc
        verts = [center]
        verts.extend([(x,y) for x,y in zip(xs,ys)])

        Polygon.__init__(self, verts, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

class Arrow(Polygon):
    """
    An arrow patch
    """
    def __str__(self):
        x1,y1 = self.xy[0]
        x2,y2 = self.xy[1]
        cx,cy = (x1+x2)/2.,(y1+y2)/2.
        return "Arrow(%g,%g)"%(cx,cy)

    def __init__( self, x, y, dx, dy, width=1.0, **kwargs ):
        """Draws an arrow, starting at (x,y), direction and length
        given by (dx,dy) the width of the arrow is scaled by width

        Valid kwargs are:
        %(Patch)s
          """
        arrow = npy.array( [
            [ 0.0,  0.1 ], [ 0.0, -0.1],
            [ 0.8, -0.1 ], [ 0.8, -0.3],
            [ 1.0,  0.0 ], [ 0.8,  0.3],
            [ 0.8,  0.1 ] ] )
        L = npy.sqrt(dx**2+dy**2) or 1 # account for div by zero
        arrow[:,0] *= L
        arrow[:,1] *= width
        cx = float(dx)/L
        sx = float(dy)/L
        M = npy.array( [ [ cx, sx],[ -sx, cx ] ] )
        verts = npy.dot( arrow, M )+ [x,y]
        Polygon.__init__( self, [ tuple(t) for t in verts ], **kwargs )
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

class FancyArrow(Polygon):
    """Like Arrow, but lets you set head width and head height independently."""

    def __str__(self):
        x1,y1 = self.xy[0]
        x2,y2 = self.xy[1]
        cx,cy = (x1+x2)/2.,(y1+y2)/2.
        return "FancyArrow(%g,%g)"%(cx,cy)

    def __init__(self, x, y, dx, dy, width=0.001, length_includes_head=False, \
        head_width=None, head_length=None, shape='full', overhang=0, \
        head_starts_at_zero=False,**kwargs):
        """Returns a new Arrow.

        length_includes_head: True if head is counted in calculating the length.

        shape: ['full', 'left', 'right']

        overhang: distance that the arrow is swept back (0 overhang means
        triangular shape).

        head_starts_at_zero: if True, the head starts being drawn at coordinate
        0 instead of ending at coordinate 0.

        Valid kwargs are:
        %(Patch)s

        """
        if head_width is None:
            head_width = 3 * width
        if head_length is None:
            head_length = 1.5 * head_width

        distance = npy.sqrt(dx**2 + dy**2)
        if length_includes_head:
            length=distance
        else:
            length=distance+head_length
        if not length:
            verts = [] #display nothing if empty
        else:
            #start by drawing horizontal arrow, point at (0,0)
            hw, hl, hs, lw = head_width, head_length, overhang, width
            left_half_arrow = npy.array([
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
                    coords=npy.concatenate([left_half_arrow,right_half_arrow[::-1]])
                else:
                    raise ValueError, "Got unknown shape: %s" % shape
            cx = float(dx)/distance
            sx = float(dy)/distance
            M = npy.array([[cx, sx],[-sx,cx]])
            verts = npy.dot(coords, M) + (x+dx, y+dy)

        Polygon.__init__(self, map(tuple, verts), **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd

class YAArrow(Polygon):
    """
    Yet another arrow class

    This is an arrow that is defined in display space and has a tip at
    x1,y1 and a base at x2, y2.
    """
    def __str__(self):
        x1,y1 = self.xy[0]
        x2,y2 = self.xy[1]
        cx,cy = (x1+x2)/2.,(y1+y2)/2.
        return "YAArrow(%g,%g)"%(cx,cy)

    def __init__(self, dpi, xytip, xybase, width=4, frac=0.1, headwidth=12, **kwargs):
        """
        xytip : (x,y) location of arrow tip
        xybase : (x,y) location the arrow base mid point
        dpi : the figure dpi instance (fig.dpi)
        width : the width of the arrow in points
        frac  : the fraction of the arrow length occupied by the head
        headwidth : the width of the base of the arrow head in points

        Valid kwargs are:
        %(Patch)s

        """
        self.dpi = dpi
        self.xytip = xytip
        self.xybase = xybase
        self.width = width
        self.frac = frac
        self.headwidth = headwidth
        verts = self.get_verts()
        Polygon.__init__(self, verts, **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd



    def get_verts(self):
        # the base vertices
        x1, y1 = self.xytip
        x2, y2 = self.xybase
        k1 = self.width*self.dpi.get()/72./2.
        k2 = self.headwidth*self.dpi.get()/72./2.
        xb1, yb1, xb2, yb2 = self.getpoints(x1, y1, x2, y2, k1)

        # a point on the segment 20% of the distance from the tip to the base
        theta = math.atan2(y2-y1, x2-x1)
        r = math.sqrt((y2-y1)**2. + (x2-x1)**2.)
        xm = x1 + self.frac * r * math.cos(theta)
        ym = y1 + self.frac * r * math.sin(theta)
        xc1, yc1, xc2, yc2 = self.getpoints(x1, y1, xm, ym, k1)
        xd1, yd1, xd2, yd2 = self.getpoints(x1, y1, xm, ym, k2)


        xs = self.convert_xunits([xb1, xb2, xc2, xd2, x1, xd1, xc1])
        ys = self.convert_yunits([yb1, yb2, yc2, yd2, y1, yd1, yc1])
        return zip(xs, ys)


    def getpoints(self, x1,y1,x2,y2, k):
        """
        for line segment defined by x1,y1 and x2,y2, return the points on
        the line that is perpendicular to the line and intersects x2,y2
        and the distance from x2,y2 ot the returned points is k
        """
        x1,y1,x2,y2,k = map(float, (x1,y1,x2,y2,k))
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
    A circle patch
    """
    def __str__(self):
        return "CirclePolygon(%d,%d)"%self.center

    def __init__(self, xy, radius=5,
                 resolution=20,  # the number of vertices
                 **kwargs):
        """
        Create a circle at xy=(x,y) with radius given by 'radius'

        Valid kwargs are:
        %(Patch)s

        """
        self.center = xy
        self.radius = radius
        RegularPolygon.__init__(self, xy,
                                resolution,
                                radius,
                                orientation=0,
                                **kwargs)
    __init__.__doc__ = cbook.dedent(__init__.__doc__) % artist.kwdocd


def inellipse(x,y,cx,cy,a,b,angle):
    x,y = x-cx,y-cy
    theta = math.atan2(x,y) + math.radians(angle)
    rsq = x*x+y*y
    asin = a * math.sin(theta)
    bcos = b * math.cos(theta)
    Rsq = b*b*a*a / (bcos*bcos + asin*asin)
    return Rsq > rsq;

class Ellipse(Patch):
    """
    A scale-free ellipse
    """
    MAGIC = 0.2652031
    SQRTHALF = npy.sqrt(0.5)
    MAGIC45 = npy.sqrt((MAGIC*MAGIC) / 2.0)

    circle = npy.array(
        [[0.0, -1.0],

         [MAGIC, -1.0],
         [SQRTHALF-MAGIC45, -SQRTHALF-MAGIC45],
         [SQRTHALF, -SQRTHALF],

         [SQRTHALF+MAGIC45, -SQRTHALF+MAGIC45],
         [1.0, -MAGIC],
         [1.0, 0.0],

         [1.0, MAGIC],
         [SQRTHALF+MAGIC45, SQRTHALF-MAGIC45],
         [SQRTHALF, SQRTHALF],

         [SQRTHALF-MAGIC45, SQRTHALF+MAGIC45],
         [MAGIC, 1.0],
         [0.0, 1.0],

         [-MAGIC, 1.0],
         [-SQRTHALF+MAGIC45, SQRTHALF+MAGIC45],
         [-SQRTHALF, SQRTHALF],

         [-SQRTHALF-MAGIC45, SQRTHALF-MAGIC45],
         [-1.0, MAGIC],
         [-1.0, 0.0],

         [-1.0, -MAGIC],
         [-SQRTHALF-MAGIC45, -SQRTHALF+MAGIC45],
         [-SQRTHALF, -SQRTHALF],

         [-SQRTHALF+MAGIC45, -SQRTHALF-MAGIC45],
         [-MAGIC, -1.0],
         [0.0, -1.0]],
        npy.float_)

    def __str__(self):
        return "Ellipse(%d,%d;%dx%d)"%(self.center[0],self.center[1],self.width,self.height)

    def __init__(self, xy, width, height, angle=0.0, **kwargs):
        """
        xy - center of ellipse
        width - length of horizontal axis
        height - length of vertical axis
        angle - rotation in degrees (anti-clockwise)

        Valid kwargs are:
        %(Patch)s
        """
        Patch.__init__(self, **kwargs)

        # self.center  = npy.array(xy, npy.float)
        self.center = xy
        self.width, self.height = width, height
        self.angle = angle

    def contains(self,ev):
        if ev.xdata is None or ev.ydata is None: return False,{}
        inside = inellipse(ev.xdata,ev.ydata,
                           self.center[0],self.center[1],
                           self.height*0.5,self.width*0.5,self.angle)
        return inside,{}

    def get_verts(self):

        xcenter, ycenter = self.center
        width, height = self.width, self.height

        xcenter = self.convert_xunits(xcenter)
        width = self.convert_xunits(width)
        ycenter = self.convert_yunits(ycenter)
        height = self.convert_xunits(height)



        angle = self.angle

        theta = npy.arange(0.0, 360.0, 1.0)*npy.pi/180.0
        x = width/2. * npy.cos(theta)
        y = height/2. * npy.sin(theta)

        rtheta = angle*npy.pi/180.
        R = npy.array([
            [npy.cos(rtheta),  -npy.sin(rtheta)],
            [npy.sin(rtheta), npy.cos(rtheta)],
            ])


        x, y = npy.dot(R, npy.array([x, y]))
        x += xcenter
        y += ycenter

        return zip(x, y)

    def draw(self, renderer):
        if not self.get_visible(): return
        #renderer.open_group('patch')
        gc = renderer.new_gc()
        gc.set_foreground(self._edgecolor)
        gc.set_linewidth(self._linewidth)
        gc.set_alpha(self._alpha)
        gc.set_antialiased(self._antialiased)
        self._set_gc_clip(gc)

        gc.set_capstyle('projecting')

        if not self.fill or self._facecolor is None: rgbFace = None
        else: rgbFace = colors.colorConverter.to_rgb(self._facecolor)

        if self._hatch:
            gc.set_hatch(self._hatch )


        if not hasattr(renderer, 'draw_path'):
            mpl.verbose.report('patches.Ellipse renderer does not support path drawing; falling back on vertex approximation for nonlinear transformation')
            renderer.draw_polygon(gc, rgbFace, self.get_verts())
            return


        x, y = self.center
        x = self.convert_xunits(x)
        y = self.convert_yunits(y)
        w = self.convert_xunits(self.width)/2.
        h = self.convert_yunits(self.height)/2.

        theta = self.angle * npy.pi/180.
        T = npy.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1]])




        S = npy.array([
            [w, 0, 0],
            [0, h, 0],
            [0, 0, 1]])



        # rotate by theta
        R = npy.array([
            [npy.cos(theta),  -npy.sin(theta), 0],
            [npy.sin(theta), npy.cos(theta), 0],
            [0,           0,          1]])

        # transform unit circle into ellipse
        E = npy.dot(T, npy.dot(R, S))


        # Apply the display affine
        sx, b, c, sy, tx, ty = self.get_transform().as_vec6_val()

        # display coords
        D = npy.array([
            [sx, b, tx],
            [c, sy, ty],
            [0, 0, 1]], npy.float_)

        M = npy.dot(D,E)

        C = npy.ones((3, len(self.circle)))
        C[0:2,:] = self.circle.T

        ellipse = npy.dot(M, C).T[:,:2]

        path =  agg.path_storage()
        path.move_to(*ellipse[0])
        for i in range(1, 25, 3):
            path.curve4(*ellipse[i:i+3].flat)
        path.close_polygon()

        renderer.draw_path(gc, rgbFace, path)



class Circle(Ellipse):
    """
    A circle patch
    """
    def __str__(self):
        return "Circle((%g,%g),r=%g)"%(self.center[0],self.center[1],self.radius)

    def __init__(self, xy, radius=5,
                 **kwargs):
        """
        Create true circle at center xy=(x,y) with given radius;
        unlike circle polygon which is a polygonal approcimation, this
        uses splines and is much closer to a scale free circle

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

    The arc must be used in an Axes instance it cannot be added
    directly to a Figure) because it is optimized to only render the
    segments that are inside the axes bounding box with high
    resolution.
    """
    def __str__(self):

        return "Arc(%s,%s;%sx%s)"%(self.center[0],self.center[1],self.width,self.height)

    def __init__(self, xy, width, height, angle=0.0, theta1=0.0, theta2=360.0, **kwargs):
        """
        xy - center of ellipse
        width - length of horizontal axis
        height - length of vertical axis
        angle - rotation in degrees (anti-clockwise)
        theta1 - starting angle of the arc in degrees
        theta2 - ending angle of the arc in degrees

        If theta1 and theta2 are not provided, the arc will form a
        complete ellipse.

        Valid kwargs are:
        %(Patch)s
        """
        fill = kwargs.pop('fill')
        if fill:
            raise ValueError("Arc objects can not be filled")
        kwargs['fill'] = False

        Ellipse.__init__(self, xy, width, height, angle, **kwargs)

        self._theta1 = theta1
        self._theta2 = theta2

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
        (8), which should be adequate when the number of pixels across
        the image is less than 5e5.  The algorithm proceeds as
        follows:

          1. The points where the ellipse intersects the axes bounding
          box are located.  (This is done be performing an inverse
          transformation on the axes bbox such that it is relative to
          the unit circle -- this makes the intersection calculation
          much easier than doing rotated ellipse intersection
          directly).

          This uses the "line intersecting a circle" algorithm from:

            Vince, John.  Geometry for Computer Graphics: Formulae,
            Examples & Proofs.  London: Springer-Verlag, 2005.

          2. The angles of each of the intersection points are
          calculated.

          3. Proceeding counterclockwise starting in the positive
          x-direction, each of the visible arc-segments between each
          pair of intersections are drawn using the bezier arc
          approximation technique implemented in arc().
        """
        # Do the usual GC handling stuff
        if not self.get_visible(): return

        if not hasattr(self, 'axes'):
            raise RuntimeError('Arcs can only be used in Axes instances')

        gc = renderer.new_gc()
        gc.set_foreground(self._edgecolor)
        gc.set_linewidth(self._linewidth)
        gc.set_alpha(self._alpha)
        gc.set_antialiased(self._antialiased)
        self._set_gc_clip(gc)
        gc.set_capstyle('projecting')
        if not self.fill or self._facecolor is None: rgbFace = None
        else: rgbFace = colors.colorConverter.to_rgb(self._facecolor)
        if self._hatch:
            gc.set_hatch(self._hatch )

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
                # npy.sign: we never want to get 0.0
                if dy < 0.0:
                    sign_dy = -1.0
                else:
                    sign_dy = 1.0
                sqrt_discrim = npy.sqrt(discrim)
                for sign in (1., -1.):
                    x = (D*dy + sign * sign_dy * dx * sqrt_discrim) / dr2
                    y = (-D*dx + sign * npy.abs(dy) * sqrt_discrim) / dr2
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

        def arc(theta1, theta2, trans, n=None):
            """
            Returns an arc on the unit circle from angle theta1 to
            angle theta2 (in degrees).  The returned arc is already
            transformed using the affine transformation matrix trans.
            The arc is returned as an agg::path_storage object.

            If n is provided, it is the number of spline segments to make.
            If n is not provided, the number of spline segments is determined
            based on the delta between theta1 and theta2.
            """
            # From Masionobe, L.  2003.  "Drawing an elliptical arc using
            # polylines, quadratic or cubic Bezier curves".
            #
            # http://www.spaceroots.org/documents/ellipse/index.html

            # degrees to radians
            theta1 *= npy.pi / 180.0
            theta2 *= npy.pi / 180.0

            twopi  = npy.pi * 2.0
            halfpi = npy.pi * 0.5

            eta1 = npy.arctan2(npy.sin(theta1), npy.cos(theta1))
            eta2 = npy.arctan2(npy.sin(theta2), npy.cos(theta2))
            eta2 -= twopi * npy.floor((eta2 - eta1) / twopi)
            if (theta2 - theta1 > npy.pi) and (eta2 - eta1 < npy.pi):
                eta2 += twopi

            # number of curve segments to make
            if n is None:
                n = int(2 ** npy.ceil((eta2 - eta1) / halfpi))

            deta = (eta2 - eta1) / n
            t = npy.tan(0.5 * deta)
            alpha = npy.sin(deta) * (npy.sqrt(4.0 + 3.0 * t * t) - 1) / 3.0

            steps = npy.linspace(eta1, eta2, n + 1, True)
            cos_eta = npy.cos(steps)
            sin_eta = npy.sin(steps)

            xA = cos_eta[:-1]
            yA = sin_eta[:-1]
            xA_dot = -yA
            yA_dot = xA

            xB = cos_eta[1:]
            yB = sin_eta[1:]
            xB_dot = -yB
            yB_dot = xB

            length = n * 3 + 1
            vertices = npy.zeros((length, 2), npy.float_)
            vertices[0] = [xA[0], yA[0]]
            end = length

            vertices[1::3, 0] = xA + alpha * xA_dot
            vertices[1::3, 1] = yA + alpha * yA_dot
            vertices[2::3, 0] = xB - alpha * xB_dot
            vertices[2::3, 1] = yB - alpha * yB_dot
            vertices[3::3, 0] = xB
            vertices[3::3, 1] = yB

            vertices = affine_transform(vertices, trans)

            path = agg.path_storage()
            path.move_to(*vertices[0])
            for i in range(1, length, 3):
                path.curve4(*vertices[i:i+3].flat)
            return path

        def point_in_polygon(x, y, poly):
            inside = False
            for i in range(len(poly) - 1):
                p1x, p1y = poly[i]
                p2x, p2y = poly[i+1]
                if p1x < p2x:
                    xmin, xmax = p1x, p2x
                else:
                    xmin, xmax = p2x, p1x
                if p1y < p2y:
                    ymin, ymax = p1y, p2y
                else:
                    ymin, ymax = p2y, p1y
                if (y > ymin and
                    y <= ymax and
                    x <= xmax):
                    xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            return inside

        def affine_transform(vertices, transform):
            # This may seem silly, but it's faster than expanding the
            # vertices array to Nx3 and then back to Nx2
            transform = transform.copy()
            transform[0, 1], transform[1, 0] = transform[1, 0], transform[0, 1]
            vertices = npy.dot(vertices, transform[0:2, 0:2])
            vertices += transform[0:2, 2:].flat
            return vertices

        # Set up the master transform from unit circle, all the way to
        # display space.

        centerx, centery = self.center
        centerx = self.convert_xunits(centerx)
        centery = self.convert_yunits(centery)
        width = self.convert_xunits(self.width)
        height = self.convert_yunits(self.height)

        trans = self.get_transform()
        scale = npy.array(
            [[width * 0.5, 0.0, 0.0],
             [0.0, height * 0.5, 0.0],
             [0.0, 0.0, 1.0]], npy.float_)
        theta = (self.angle / 180.0) * npy.pi
        rotate = npy.array(
            [[npy.cos(theta), -npy.sin(theta), 0.0],
             [npy.sin(theta), npy.cos(theta), 0.0],
             [0.0, 0.0, 1.0]], npy.float_)
        translate = npy.array(
            [[1.0, 0.0, centerx],
             [0.0, 1.0, centery],
             [0.0, 0.0, 1.0]], npy.float_)
        sx, b, c, sy, tx, ty = trans.as_vec6_val()
        dataTrans = npy.array(
            [[sx, b, tx],
             [c, sy, ty],
             [0, 0, 1]], npy.float_)
        mainTrans = \
            npy.dot(
              npy.dot(
                npy.dot(dataTrans, translate), rotate), scale)

        # Determine the size of the ellipse in pixels, and use
        # that as a threshold to use the fast (whole ellipse)
        # technique or accurate (partial arcs) technique.
        size = affine_transform(
            npy.array([[width, height]], npy.float_),
            mainTrans)
        width = size[0,0]
        height = size[0,1]
        # We divide the error in half, to just be *really*
        # conservative
        inv_error = (1.0 / 1.89818e-6) * 0.5

        if width < inv_error and height < inv_error:
            path = arc(self._theta1, self._theta2, mainTrans)
            renderer.draw_path(gc, rgbFace, path)
            return

        # Transforms the axes box_path so that it is relative to the unit
        # circle in the same way that it is relative to the desired
        # ellipse.
        axes_bbox = self.axes.bbox
        box_path = npy.array(
            [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]],
            npy.float_)
        axesTrans = npy.array(
            [[axes_bbox.width(), 0.0, axes_bbox.xmin()],
             [0.0, axes_bbox.height(), axes_bbox.ymin()],
             [0.0, 0.0, 1.0]], npy.float_)
        boxTrans = npy.dot(npy.linalg.inv(mainTrans), axesTrans)
        box_path = affine_transform(box_path, boxTrans)

        PI = npy.pi
        TWOPI = PI * 2.0
        RAD2DEG = 180.0 / PI
        DEG2RAD = PI / 180.0
        theta1 = self._theta1
        theta2 = self._theta2
        thetas = {}
        # For each of the point pairs, there is a line segment
        for p0, p1 in zip(box_path[:-1], box_path[1:]):
            x0, y0 = p0
            x1, y1 = p1
            for x, y in iter_circle_intersect_on_line_seg(x0, y0, x1, y1):
                theta = npy.arccos(x)
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
        inside = point_in_polygon(npy.cos(theta1_rad), npy.sin(theta1_rad), box_path)
        for theta in thetas:
            if inside:
                path = arc(last_theta, theta, mainTrans, 8)
                renderer.draw_path(gc, rgbFace, path)
                inside = False
            else:
                inside = True
            last_theta = theta


class PolygonInteractor:
    """
    An polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __str__(self):
        return "PolygonInteractor"

    def __init__(self, poly):
        if poly.figure is None:
            raise RuntimeError('You must first add the polygon to a figure or canvas before defining the interactor')
        canvas = poly.figure.canvas
        self.poly = poly
        self.poly.verts = list(self.poly.get_verts())
        x, y = zip(*self.poly.verts)
        self.line = lines.Line2D(x,y,marker='o', markerfacecolor='r')
        #self._update_line(poly)

        cid = self.poly.add_callback(self.poly_changed)
        self._ind = None # the active vert

        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        self.canvas = canvas


    def poly_changed(self, poly):
        'this method is called whenever the polygon object is called'
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        artist.Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state


    def get_ind_under_point(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        x, y = zip(*self.poly.verts)

        # display coords
        xt, yt = self.poly.get_transform().numerix_x_y(x, y)
        d = npy.sqrt((xt-event.x)**2 + (yt-event.y)**2)
        ind, = npy.nonzero(npy.equal(d, npy.amin(d)))

        if d[ind]>=self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        'whenever a mouse button is pressed'
        if not self.showverts: return
        if event.inaxes==None: return
        if event.button != 1: return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        'whenever a mouse button is released'
        if not self.showverts: return
        if event.button != 1: return
        self._ind = None

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes: return
        if event.key=='t':
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts: self._ind = None
        elif event.key=='d':
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.verts = [tup for i,tup in enumerate(self.poly.verts) if i!=ind]
                self.line.set_data(zip(*self.poly.verts))
        elif event.key=='i':
            xys = self.poly.get_transform().seq_xy_tups(self.poly.verts)
            p = event.x, event.y # display coords
            for i in range(len(xys)-1):
                s0 = xys[i]
                s1 = xys[i+1]
                d = mlab.dist_point_to_segment(p, s0, s1)
                if d<=self.epsilon:
                    self.poly.verts.insert(i+1, (event.xdata, event.ydata))
                    self.line.set_data(zip(*self.poly.verts))
                    break


        self.canvas.draw()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showverts: return
        if self._ind is None: return
        if event.inaxes is None: return
        if event.button != 1: return
        x,y = event.xdata, event.ydata
        self.poly.verts[self._ind] = x,y
        self.line.set_data(zip(*self.poly.verts))
        self.canvas.draw_idle()


def bbox_artist(artist, renderer, props=None, fill=True):
    """
    This is a debug function to draw a rectangle around the bounding
    box returned by get_window_extent of an artist, to test whether
    the artist is returning the correct bbox

    props is a dict of rectangle props with the additional property
    'pad' that sets the padding around the bbox in points
    """
    if props is None: props = {}
    props = props.copy() # don't want to alter the pad externally
    pad = props.pop('pad', 4)
    pad = renderer.points_to_pixels(pad)
    bbox = artist.get_window_extent(renderer)
    l,b,w,h = bbox.get_bounds()
    l-=pad/2.
    b-=pad/2.
    w+=pad
    h+=pad
    r = Rectangle(xy=(l,b),
                  width=w,
                  height=h,
                  fill=fill,
                  )
    r.set_clip_on( False )
    r.update(props)
    r.draw(renderer)


def draw_bbox(bbox, renderer, color='k', trans=None):
    """
    This is a debug function to draw a rectangle around the bounding
    box returned by get_window_extent of an artist, to test whether
    the artist is returning the correct bbox
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
          'FancyArrow', 'YAArrow', 'CirclePolygon', 'Ellipse'):
    artist.kwdocd[k] = patchdoc


