"""
Support for plotting vector fields.

Presently this contains Quiver and Barb. Quiver plots an arrow in the
direction of the vector, with the size of the arrow related to the
magnitude of the vector.

Barbs are like quiver in that they point along a vector, but
the magnitude of the vector is given schematically by the presence of barbs
or flags on the barb.

This will also become a home for things such as standard
deviation ellipses, which can and will be derived very easily from
the Quiver code.
"""


import numpy as np
from numpy import ma
import matplotlib.collections as collections
import matplotlib.transforms as transforms
import matplotlib.text as mtext
import matplotlib.artist as martist
import matplotlib.font_manager as font_manager
from matplotlib.cbook import delete_masked_points
from matplotlib.patches import CirclePolygon
import math


_quiver_doc = """
Plot a 2-D field of arrows.

call signatures::

  quiver(U, V, **kw)
  quiver(U, V, C, **kw)
  quiver(X, Y, U, V, **kw)
  quiver(X, Y, U, V, C, **kw)

Arguments:

  *X*, *Y*:

    The x and y coordinates of the arrow locations (default is tail of
    arrow; see *pivot* kwarg)

  *U*, *V*:

    give the *x* and *y* components of the arrow vectors

  *C*:
    an optional array used to map colors to the arrows

All arguments may be 1-D or 2-D arrays or sequences. If *X* and *Y*
are absent, they will be generated as a uniform grid.  If *U* and *V*
are 2-D arrays but *X* and *Y* are 1-D, and if len(*X*) and len(*Y*)
match the column and row dimensions of *U*, then *X* and *Y* will be
expanded with :func:`numpy.meshgrid`.

*U*, *V*, *C* may be masked arrays, but masked *X*, *Y* are not
supported at present.

Keyword arguments:

  *units*: ['width' | 'height' | 'dots' | 'inches' | 'x' | 'y' ]
    arrow units; the arrow dimensions *except for length* are in
    multiples of this unit.

    * 'width' or 'height': the width or height of the axes

    * 'dots' or 'inches': pixels or inches, based on the figure dpi

    * 'x' or 'y': *X* or *Y* data units

    The arrows scale differently depending on the units.  For
    'x' or 'y', the arrows get larger as one zooms in; for other
    units, the arrow size is independent of the zoom state.  For
    'width or 'height', the arrow size increases with the width and
    height of the axes, respectively, when the the window is resized;
    for 'dots' or 'inches', resizing does not change the arrows.

   *angles*: ['uv' | 'xy' | array]
    With the default 'uv', the arrow aspect ratio is 1, so that
    if *U*==*V* the angle of the arrow on the plot is 45 degrees
    CCW from the *x*-axis.
    With 'xy', the arrow points from (x,y) to (x+u, y+v).
    Alternatively, arbitrary angles may be specified as an array
    of values in degrees, CCW from the *x*-axis.

  *scale*: [ None | float ]
    data units per arrow unit, e.g. m/s per plot width; a smaller
    scale parameter makes the arrow longer.  If *None*, a simple
    autoscaling algorithm is used, based on the average vector length
    and the number of vectors.

  *width*:
    shaft width in arrow units; default depends on choice of units,
    above, and number of vectors; a typical starting value is about
    0.005 times the width of the plot.

  *headwidth*: scalar
    head width as multiple of shaft width, default is 3

  *headlength*: scalar
    head length as multiple of shaft width, default is 5

  *headaxislength*: scalar
    head length at shaft intersection, default is 4.5

  *minshaft*: scalar
    length below which arrow scales, in units of head length. Do not
    set this to less than 1, or small arrows will look terrible!
    Default is 1

  *minlength*: scalar
    minimum length as a multiple of shaft width; if an arrow length
    is less than this, plot a dot (hexagon) of this diameter instead.
    Default is 1.

  *pivot*: [ 'tail' | 'middle' | 'tip' ]
    The part of the arrow that is at the grid point; the arrow rotates
    about this point, hence the name *pivot*.

  *color*: [ color | color sequence ]
    This is a synonym for the
    :class:`~matplotlib.collections.PolyCollection` facecolor kwarg.
    If *C* has been set, *color* has no effect.

The defaults give a slightly swept-back arrow; to make the head a
triangle, make *headaxislength* the same as *headlength*. To make the
arrow more pointed, reduce *headwidth* or increase *headlength* and
*headaxislength*. To make the head smaller relative to the shaft,
scale down all the head parameters. You will probably do best to leave
minshaft alone.

linewidths and edgecolors can be used to customize the arrow
outlines. Additional :class:`~matplotlib.collections.PolyCollection`
keyword arguments:

%(PolyCollection)s
""" % martist.kwdocd

_quiverkey_doc = """
Add a key to a quiver plot.

call signature::

  quiverkey(Q, X, Y, U, label, **kw)

Arguments:

  *Q*:
    The Quiver instance returned by a call to quiver.

  *X*, *Y*:
    The location of the key; additional explanation follows.

  *U*:
    The length of the key

  *label*:
    a string with the length and units of the key

Keyword arguments:

  *coordinates* = [ 'axes' | 'figure' | 'data' | 'inches' ]
    Coordinate system and units for *X*, *Y*: 'axes' and 'figure' are
    normalized coordinate systems with 0,0 in the lower left and 1,1
    in the upper right; 'data' are the axes data coordinates (used for
    the locations of the vectors in the quiver plot itself); 'inches'
    is position in the figure in inches, with 0,0 at the lower left
    corner.

  *color*:
    overrides face and edge colors from *Q*.

  *labelpos* = [ 'N' | 'S' | 'E' | 'W' ]
    Position the label above, below, to the right, to the left of the
    arrow, respectively.

  *labelsep*:
    Distance in inches between the arrow and the label.  Default is
    0.1

  *labelcolor*:
    defaults to default :class:`~matplotlib.text.Text` color.

  *fontproperties*:
    A dictionary with keyword arguments accepted by the
    :class:`~matplotlib.font_manager.FontProperties` initializer:
    *family*, *style*, *variant*, *size*, *weight*

Any additional keyword arguments are used to override vector
properties taken from *Q*.

The positioning of the key depends on *X*, *Y*, *coordinates*, and
*labelpos*.  If *labelpos* is 'N' or 'S', *X*, *Y* give the position
of the middle of the key arrow.  If *labelpos* is 'E', *X*, *Y*
positions the head, and if *labelpos* is 'W', *X*, *Y* positions the
tail; in either of these two cases, *X*, *Y* is somewhere in the
middle of the arrow+label key object.
"""


class QuiverKey(martist.Artist):
    """ Labelled arrow for use as a quiver plot scale key.
    """
    halign = {'N': 'center', 'S': 'center', 'E': 'left',   'W': 'right'}
    valign = {'N': 'bottom', 'S': 'top',    'E': 'center', 'W': 'center'}
    pivot  = {'N': 'mid',    'S': 'mid',    'E': 'tip',    'W': 'tail'}

    def __init__(self, Q, X, Y, U, label, **kw):
        martist.Artist.__init__(self)
        self.Q = Q
        self.X = X
        self.Y = Y
        self.U = U
        self.coord = kw.pop('coordinates', 'axes')
        self.color = kw.pop('color', None)
        self.label = label
        self._labelsep_inches = kw.pop('labelsep', 0.1)
        self.labelsep = (self._labelsep_inches * Q.ax.figure.dpi)

        def on_dpi_change(fig):
            self.labelsep = (self._labelsep_inches * fig.dpi)
            self._initialized = False # simple brute force update
                                      # works because _init is called
                                      # at the start of draw.

        Q.ax.figure.callbacks.connect('dpi_changed', on_dpi_change)

        self.labelpos = kw.pop('labelpos', 'N')
        self.labelcolor = kw.pop('labelcolor', None)
        self.fontproperties = kw.pop('fontproperties', dict())
        self.kw = kw
        _fp = self.fontproperties
        #boxprops = dict(facecolor='red')
        self.text = mtext.Text(text=label,  #      bbox=boxprops,
                       horizontalalignment=self.halign[self.labelpos],
                       verticalalignment=self.valign[self.labelpos],
                       fontproperties=font_manager.FontProperties(**_fp))
        if self.labelcolor is not None:
            self.text.set_color(self.labelcolor)
        self._initialized = False
        self.zorder = Q.zorder + 0.1


    __init__.__doc__ = _quiverkey_doc

    def _init(self):
        if True: ##not self._initialized:
            self._set_transform()
            _pivot = self.Q.pivot
            self.Q.pivot = self.pivot[self.labelpos]
            self.verts = self.Q._make_verts(np.array([self.U]),
                                                        np.zeros((1,)))
            self.Q.pivot = _pivot
            kw = self.Q.polykw
            kw.update(self.kw)
            self.vector = collections.PolyCollection(self.verts,
                                         offsets=[(self.X,self.Y)],
                                         transOffset=self.get_transform(),
                                         **kw)
            if self.color is not None:
                self.vector.set_color(self.color)
            self.vector.set_transform(self.Q.get_transform())
            self._initialized = True

    def _text_x(self, x):
        if self.labelpos == 'E':
            return x + self.labelsep
        elif self.labelpos == 'W':
            return x - self.labelsep
        else:
            return x

    def _text_y(self, y):
        if self.labelpos == 'N':
            return y + self.labelsep
        elif self.labelpos == 'S':
            return y - self.labelsep
        else:
            return y

    def draw(self, renderer):
        self._init()
        self.vector.draw(renderer)
        x, y = self.get_transform().transform_point((self.X, self.Y))
        self.text.set_x(self._text_x(x))
        self.text.set_y(self._text_y(y))
        self.text.draw(renderer)


    def _set_transform(self):
        if self.coord == 'data':
            self.set_transform(self.Q.ax.transData)
        elif self.coord == 'axes':
            self.set_transform(self.Q.ax.transAxes)
        elif self.coord == 'figure':
            self.set_transform(self.Q.ax.figure.transFigure)
        elif self.coord == 'inches':
            self.set_transform(self.Q.ax.figure.dpi_scale_trans)
        else:
            raise ValueError('unrecognized coordinates')

    def set_figure(self, fig):
        martist.Artist.set_figure(self, fig)
        self.text.set_figure(fig)

    def contains(self, mouseevent):
        # Maybe the dictionary should allow one to
        # distinguish between a text hit and a vector hit.
        if (self.text.contains(mouseevent)[0]
                or self.vector.contains(mouseevent)[0]):
            return True, {}
        return False, {}

    quiverkey_doc = _quiverkey_doc


class Quiver(collections.PolyCollection):
    """
    Specialized PolyCollection for arrows.

    The only API method is set_UVC(), which can be used
    to change the size, orientation, and color of the
    arrows; their locations are fixed when the class is
    instantiated.  Possibly this method will be useful
    in animations.

    Much of the work in this class is done in the draw()
    method so that as much information as possible is available
    about the plot.  In subsequent draw() calls, recalculation
    is limited to things that might have changed, so there
    should be no performance penalty from putting the calculations
    in the draw() method.
    """
    def __init__(self, ax, *args, **kw):
        self.ax = ax
        X, Y, U, V, C = self._parse_args(*args)
        self.X = X
        self.Y = Y
        self.XY = np.hstack((X[:,np.newaxis], Y[:,np.newaxis]))
        self.N = len(X)
        self.scale = kw.pop('scale', None)
        self.headwidth = kw.pop('headwidth', 3)
        self.headlength = float(kw.pop('headlength', 5))
        self.headaxislength = kw.pop('headaxislength', 4.5)
        self.minshaft = kw.pop('minshaft', 1)
        self.minlength = kw.pop('minlength', 1)
        self.units = kw.pop('units', 'width')
        self.angles = kw.pop('angles', 'uv')
        self.width = kw.pop('width', None)
        self.color = kw.pop('color', 'k')
        self.pivot = kw.pop('pivot', 'tail')
        kw.setdefault('facecolors', self.color)
        kw.setdefault('linewidths', (0,))
        collections.PolyCollection.__init__(self, [], offsets=self.XY,
                                            transOffset=ax.transData,
                                            closed=False,
                                            **kw)
        self.polykw = kw
        self.set_UVC(U, V, C)
        self._initialized = False

        self.keyvec = None
        self.keytext = None

        def on_dpi_change(fig):
            self._new_UV = True # vertices depend on width, span
                                # which in turn depend on dpi
            self._initialized = False # simple brute force update
                                      # works because _init is called
                                      # at the start of draw.

        self.ax.figure.callbacks.connect('dpi_changed', on_dpi_change)


    __init__.__doc__ = """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pylab interface documentation:
        %s""" % _quiver_doc

    def _parse_args(self, *args):
        X, Y, U, V, C = [None]*5
        args = list(args)
        if len(args) == 3 or len(args) == 5:
            C = ma.asarray(args.pop(-1)).ravel()
        V = ma.asarray(args.pop(-1))
        U = ma.asarray(args.pop(-1))
        nn = np.shape(U)
        nc = nn[0]
        nr = 1
        if len(nn) > 1:
            nr = nn[1]
        if len(args) == 2: # remaining after removing U,V,C
            X, Y = [np.array(a).ravel() for a in args]
            if len(X) == nc and len(Y) == nr:
                X, Y = [a.ravel() for a in np.meshgrid(X, Y)]
        else:
            indexgrid = np.meshgrid(np.arange(nc), np.arange(nr))
            X, Y = [np.ravel(a) for a in indexgrid]
        return X, Y, U, V, C

    def _init(self):
        """initialization delayed until first draw;
        allow time for axes setup.
        """
        # It seems that there are not enough event notifications
        # available to have this work on an as-needed basis at present.
        if True: ##not self._initialized:
            trans = self._set_transform()
            ax = self.ax
            sx, sy = trans.inverted().transform_point(
                                            (ax.bbox.width, ax.bbox.height))
            self.span = sx
            sn = max(8, min(25, math.sqrt(self.N)))
            if self.width is None:
                self.width = 0.06 * self.span / sn

    def draw(self, renderer):
        self._init()
        if self._new_UV or self.angles == 'xy':
            verts = self._make_verts(self.U, self.V)
            self.set_verts(verts, closed=False)
            self._new_UV = False
        collections.PolyCollection.draw(self, renderer)

    def set_UVC(self, U, V, C=None):
        self.U = U.ravel()
        self.V = V.ravel()
        if C is not None:
            self.set_array(C.ravel())
        self._new_UV = True

    def _set_transform(self):
        ax = self.ax
        if self.units in ('x', 'y'):
            if self.units == 'x':
                dx0 = ax.viewLim.width
                dx1 = ax.bbox.width
            else:
                dx0 = ax.viewLim.height
                dx1 = ax.bbox.height
            dx = dx1/dx0
        else:
            if self.units == 'width':
                dx = ax.bbox.width
            elif self.units == 'height':
                dx = ax.bbox.height
            elif self.units == 'dots':
                dx = 1.0
            elif self.units == 'inches':
                dx = ax.figure.dpi
            else:
                raise ValueError('unrecognized units')
        trans = transforms.Affine2D().scale(dx)
        self.set_transform(trans)
        return trans

    def _angles(self, U, V, eps=0.001):
        xy = self.ax.transData.transform(self.XY)
        uv = ma.hstack((U[:,np.newaxis], V[:,np.newaxis])).filled(0)
        xyp = self.ax.transData.transform(self.XY + eps * uv)
        dxy = xyp - xy
        ang = ma.arctan2(dxy[:,1], dxy[:,0])
        return ang

    def _make_verts(self, U, V):
        uv = ma.asarray(U+V*1j)
        a = ma.absolute(uv)
        if self.scale is None:
            sn = max(10, math.sqrt(self.N))
            scale = 1.8 * a.mean() * sn / self.span # crude auto-scaling
            self.scale = scale
        length = a/(self.scale*self.width)
        X, Y = self._h_arrows(length)
        if self.angles == 'xy':
            theta = self._angles(U, V).filled(0)[:,np.newaxis]
        elif self.angles == 'uv':
            theta = np.angle(ma.asarray(uv[..., np.newaxis]).filled(0))
        else:
            theta = ma.asarray(self.angles*np.pi/180.0).filled(0)
        xy = (X+Y*1j) * np.exp(1j*theta)*self.width
        xy = xy[:,:,np.newaxis]
        XY = ma.concatenate((xy.real, xy.imag), axis=2)
        return XY


    def _h_arrows(self, length):
        """ length is in arrow width units """
        # It might be possible to streamline the code
        # and speed it up a bit by using complex (x,y)
        # instead of separate arrays; but any gain would be slight.
        minsh = self.minshaft * self.headlength
        N = len(length)
        length = length.reshape(N, 1)
        # x, y: normal horizontal arrow
        x = np.array([0, -self.headaxislength,
                        -self.headlength, 0], np.float64)
        x = x + np.array([0,1,1,1]) * length
        y = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        y = np.repeat(y[np.newaxis,:], N, axis=0)
        # x0, y0: arrow without shaft, for short vectors
        x0 = np.array([0, minsh-self.headaxislength,
                        minsh-self.headlength, minsh], np.float64)
        y0 = 0.5 * np.array([1, 1, self.headwidth, 0], np.float64)
        ii = [0,1,2,3,2,1,0]
        X = x.take(ii, 1)
        Y = y.take(ii, 1)
        Y[:, 3:] *= -1
        X0 = x0.take(ii)
        Y0 = y0.take(ii)
        Y0[3:] *= -1
        shrink = length/minsh
        X0 = shrink * X0[np.newaxis,:]
        Y0 = shrink * Y0[np.newaxis,:]
        short = np.repeat(length < minsh, 7, axis=1)
        #print 'short', length < minsh
        # Now select X0, Y0 if short, otherwise X, Y
        X = ma.where(short, X0, X)
        Y = ma.where(short, Y0, Y)
        if self.pivot[:3] == 'mid':
            X -= 0.5 * X[:,3, np.newaxis]
        elif self.pivot[:3] == 'tip':
            X = X - X[:,3, np.newaxis]   #numpy bug? using -= does not
                                         # work here unless we multiply
                                         # by a float first, as with 'mid'.
        tooshort = length < self.minlength
        if tooshort.any():
            # Use a heptagonal dot:
            th = np.arange(0,7,1, np.float64) * (np.pi/3.0)
            x1 = np.cos(th) * self.minlength * 0.5
            y1 = np.sin(th) * self.minlength * 0.5
            X1 = np.repeat(x1[np.newaxis, :], N, axis=0)
            Y1 = np.repeat(y1[np.newaxis, :], N, axis=0)
            tooshort = ma.repeat(tooshort, 7, 1)
            X = ma.where(tooshort, X1, X)
            Y = ma.where(tooshort, Y1, Y)
        return X, Y

    quiver_doc = _quiver_doc

_barbs_doc = """
Plot a 2-D field of barbs.

call signatures::

  barb(U, V, **kw)
  barb(U, V, C, **kw)
  barb(X, Y, U, V, **kw)
  barb(X, Y, U, V, C, **kw)

Arguments:

  *X*, *Y*:
    The x and y coordinates of the barb locations
    (default is head of barb; see *pivot* kwarg)

  *U*, *V*:
    give the *x* and *y* components of the barb shaft

  *C*:
    an optional array used to map colors to the barbs

All arguments may be 1-D or 2-D arrays or sequences. If *X* and *Y*
are absent, they will be generated as a uniform grid.  If *U* and *V*
are 2-D arrays but *X* and *Y* are 1-D, and if len(*X*) and len(*Y*)
match the column and row dimensions of *U*, then *X* and *Y* will be
expanded with :func:`numpy.meshgrid`.

*U*, *V*, *C* may be masked arrays, but masked *X*, *Y* are not
supported at present.

Keyword arguments:

  *length*:
    Length of the barb in points; the other parts of the barb
    are scaled against this.
    Default is 9

  *pivot*: [ 'tip' | 'middle' ]
    The part of the arrow that is at the grid point; the arrow rotates
    about this point, hence the name *pivot*.  Default is 'tip'

  *barbcolor*: [ color | color sequence ]
    Specifies the color all parts of the barb except any flags.  This
    parameter is analagous to the *edgecolor* parameter for polygons,
    which can be used instead. However this parameter will override
    facecolor.

  *flagcolor*: [ color | color sequence ]
    Specifies the color of any flags on the barb.  This parameter is
    analagous to the *facecolor* parameter for polygons, which can be
    used instead. However this parameter will override facecolor.  If
    this is not set (and *C* has not either) then *flagcolor* will be
    set to match *barbcolor* so that the barb has a uniform color. If
    *C* has been set, *flagcolor* has no effect.

  *sizes*:
    A dictionary of coefficients specifying the ratio of a given
    feature to the length of the barb. Only those values one wishes to
    override need to be included.  These features include:

        - 'spacing' - space between features (flags, full/half barbs)

        - 'height' - height (distance from shaft to top) of a flag or
          full barb

        - 'width' - width of a flag, twice the width of a full barb

        - 'emptybarb' - radius of the circle used for low magnitudes

  *fill_empty*:
    A flag on whether the empty barbs (circles) that are drawn should
    be filled with the flag color.  If they are not filled, they will
    be drawn such that no color is applied to the center.  Default is
    False

  *rounding*:
    A flag to indicate whether the vector magnitude should be rounded
    when allocating barb components.  If True, the magnitude is
    rounded to the nearest multiple of the half-barb increment.  If
    False, the magnitude is simply truncated to the next lowest
    multiple.  Default is True

  *barb_increments*:
    A dictionary of increments specifying values to associate with
    different parts of the barb. Only those values one wishes to
    override need to be included.

        - 'half' - half barbs (Default is 5)

        - 'full' - full barbs (Default is 10)

        - 'flag' - flags (default is 50)

  *flip_barb*:
    Either a single boolean flag or an array of booleans.  Single
    boolean indicates whether the lines and flags should point
    opposite to normal for all barbs.  An array (which should be the
    same size as the other data arrays) indicates whether to flip for
    each individual barb.  Normal behavior is for the barbs and lines
    to point right (comes from wind barbs having these features point
    towards low pressure in the Northern Hemisphere.)  Default is
    False

Barbs are traditionally used in meteorology as a way to plot the speed
and direction of wind observations, but can technically be used to
plot any two dimensional vector quantity.  As opposed to arrows, which
give vector magnitude by the length of the arrow, the barbs give more
quantitative information about the vector magnitude by putting slanted
lines or a triangle for various increments in magnitude, as show
schematically below::

 :     /\    \\
 :    /  \    \\
 :   /    \    \    \\
 :  /      \    \    \\
 : ------------------------------

.. note the double \\ at the end of each line to make the figure
.. render correctly

The largest increment is given by a triangle (or "flag"). After those
come full lines (barbs). The smallest increment is a half line.  There
is only, of course, ever at most 1 half line.  If the magnitude is
small and only needs a single half-line and no full lines or
triangles, the half-line is offset from the end of the barb so that it
can be easily distinguished from barbs with a single full line.  The
magnitude for the barb shown above would nominally be 65, using the
standard increments of 50, 10, and 5.

linewidths and edgecolors can be used to customize the barb.
Additional :class:`~matplotlib.collections.PolyCollection` keyword
arguments:

%(PolyCollection)s
""" % martist.kwdocd

class Barbs(collections.PolyCollection):
    '''
    Specialized PolyCollection for barbs.

    The only API method is :meth:`set_UVC`, which can be used to
    change the size, orientation, and color of the arrows.  Locations
    are changed using the :meth:`set_offsets` collection method.
    Possibly this method will be useful in animations.

    There is one internal function :meth:`_find_tails` which finds
    exactly what should be put on the barb given the vector magnitude.
    From there :meth:`_make_barbs` is used to find the vertices of the
    polygon to represent the barb based on this information.
    '''
    #This may be an abuse of polygons here to render what is essentially maybe
    #1 triangle and a series of lines.  It works fine as far as I can tell
    #however.
    def __init__(self, ax, *args, **kw):
        self._pivot = kw.pop('pivot', 'tip')
        self._length = kw.pop('length', 7)
        barbcolor = kw.pop('barbcolor', None)
        flagcolor = kw.pop('flagcolor', None)
        self.sizes = kw.pop('sizes', dict())
        self.fill_empty = kw.pop('fill_empty', False)
        self.barb_increments = kw.pop('barb_increments', dict())
        self.rounding = kw.pop('rounding', True)
        self.flip = kw.pop('flip_barb', False)

        #Flagcolor and and barbcolor provide convenience parameters for setting
        #the facecolor and edgecolor, respectively, of the barb polygon.  We
        #also work here to make the flag the same color as the rest of the barb
        #by default
        if None in (barbcolor, flagcolor):
            kw['edgecolors'] = 'face'
            if flagcolor:
                kw['facecolors'] = flagcolor
            elif barbcolor:
                kw['facecolors'] = barbcolor
            else:
                #Set to facecolor passed in or default to black
                kw.setdefault('facecolors', 'k')
        else:
            kw['edgecolors'] = barbcolor
            kw['facecolors'] = flagcolor

        #Parse out the data arrays from the various configurations supported
        x, y, u, v, c = self._parse_args(*args)
        self.x = x
        self.y = y
        xy = np.hstack((x[:,np.newaxis], y[:,np.newaxis]))

        #Make a collection
        barb_size = self._length**2 / 4 #Empirically determined
        collections.PolyCollection.__init__(self, [], (barb_size,), offsets=xy,
            transOffset=ax.transData, **kw)
        self.set_transform(transforms.IdentityTransform())

        self.set_UVC(u, v, c)

    __init__.__doc__ = """
        The constructor takes one required argument, an Axes
        instance, followed by the args and kwargs described
        by the following pylab interface documentation:
        %s""" % _barbs_doc

    def _find_tails(self, mag, rounding=True, half=5, full=10, flag=50):
        '''
        Find how many of each of the tail pieces is necessary.  Flag
        specifies the increment for a flag, barb for a full barb, and half for
        half a barb. Mag should be the magnitude of a vector (ie. >= 0).

        This returns a tuple of:

            (*number of flags*, *number of barbs*, *half_flag*, *empty_flag*)

        *half_flag* is a boolean whether half of a barb is needed,
        since there should only ever be one half on a given
        barb. *empty_flag* flag is an array of flags to easily tell if
        a barb is empty (too low to plot any barbs/flags.
        '''

        #If rounding, round to the nearest multiple of half, the smallest
        #increment
        if rounding:
            mag = half * (mag / half + 0.5).astype(np.int)

        num_flags = np.floor(mag / flag).astype(np.int)
        mag = np.mod(mag, flag)

        num_barb = np.floor(mag / full).astype(np.int)
        mag = np.mod(mag, full)

        half_flag = mag >= half
        empty_flag = ~(half_flag | (num_flags > 0) | (num_barb > 0))

        return num_flags, num_barb, half_flag, empty_flag

    def _make_barbs(self, u, v, nflags, nbarbs, half_barb, empty_flag, length,
        pivot, sizes, fill_empty, flip):
        '''
        This function actually creates the wind barbs.  *u* and *v*
        are components of the vector in the *x* and *y* directions,
        respectively.

        *nflags*, *nbarbs*, and *half_barb*, empty_flag* are,
        *respectively, the number of flags, number of barbs, flag for
        *half a barb, and flag for empty barb, ostensibly obtained
        *from :meth:`_find_tails`.

        *length* is the length of the barb staff in points.

        *pivot* specifies the point on the barb around which the
        entire barb should be rotated.  Right now, valid options are
        'head' and 'middle'.

        *sizes* is a dictionary of coefficients specifying the ratio
        of a given feature to the length of the barb. These features
        include:

            - *spacing*: space between features (flags, full/half
               barbs)

            - *height*: distance from shaft of top of a flag or full
               barb

            - *width* - width of a flag, twice the width of a full barb

            - *emptybarb* - radius of the circle used for low
               magnitudes

        *fill_empty* specifies whether the circle representing an
        empty barb should be filled or not (this changes the drawing
        of the polygon).

        *flip* is a flag indicating whether the features should be flipped to
        the other side of the barb (useful for winds in the southern
        hemisphere.

        This function returns list of arrays of vertices, defining a polygon for
        each of the wind barbs.  These polygons have been rotated to properly
        align with the vector direction.
        '''

        #These control the spacing and size of barb elements relative to the
        #length of the shaft
        spacing = length * sizes.get('spacing', 0.125)
        full_height = length * sizes.get('height', 0.4)
        full_width = length * sizes.get('width', 0.25)
        empty_rad = length * sizes.get('emptybarb', 0.15)

        #Controls y point where to pivot the barb.
        pivot_points = dict(tip=0.0, middle=-length/2.)

        #Check for flip
        if flip: full_height = -full_height

        endx = 0.0
        endy = pivot_points[pivot.lower()]

        #Get the appropriate angle for the vector components.  The offset is due
        #to the way the barb is initially drawn, going down the y-axis.  This
        #makes sense in a meteorological mode of thinking since there 0 degrees
        #corresponds to north (the y-axis traditionally)
        angles = -(ma.arctan2(v, u) + np.pi/2)

        #Used for low magnitude.  We just get the vertices, so if we make it
        #out here, it can be reused.  The center set here should put the
        #center of the circle at the location(offset), rather than at the
        #same point as the barb pivot; this seems more sensible.
        circ = CirclePolygon((0,0), radius=empty_rad).get_verts()
        if fill_empty:
            empty_barb = circ
        else:
            #If we don't want the empty one filled, we make a degenerate polygon
            #that wraps back over itself
            empty_barb = np.concatenate((circ, circ[::-1]))

        barb_list = []
        for index, angle in np.ndenumerate(angles):
            #If the vector magnitude is too weak to draw anything, plot an
            #empty circle instead
            if empty_flag[index]:
                #We can skip the transform since the circle has no preferred
                #orientation
                barb_list.append(empty_barb)
                continue

            poly_verts = [(endx, endy)]
            offset = length

            #Add vertices for each flag
            for i in range(nflags[index]):
                #The spacing that works for the barbs is a little to much for
                #the flags, but this only occurs when we have more than 1 flag.
                if offset != length: offset += spacing / 2.
                poly_verts.extend([[endx, endy + offset],
                    [endx + full_height, endy - full_width/2 + offset],
                    [endx, endy - full_width + offset]])

                offset -= full_width + spacing

            #Add vertices for each barb.  These really are lines, but works
            #great adding 3 vertices that basically pull the polygon out and
            #back down the line
            for i in range(nbarbs[index]):
                poly_verts.extend([(endx, endy + offset),
                    (endx + full_height, endy + offset + full_width/2),
                    (endx, endy + offset)])

                offset -= spacing

            #Add the vertices for half a barb, if needed
            if half_barb[index]:
                #If the half barb is the first on the staff, traditionally it is
                #offset from the end to make it easy to distinguish from a barb
                #with a full one
                if offset == length:
                    poly_verts.append((endx, endy + offset))
                    offset -= 1.5 * spacing
                poly_verts.extend([(endx, endy + offset),
                    (endx + full_height/2, endy + offset + full_width/4),
                    (endx, endy + offset)])

            #Rotate the barb according the angle. Making the barb first and then
            #rotating it made the math for drawing the barb really easy.  Also,
            #the transform framework makes doing the rotation simple.
            poly_verts = transforms.Affine2D().rotate(-angle).transform(
                poly_verts)
            barb_list.append(poly_verts)

        return barb_list

    #Taken shamelessly from Quiver
    def _parse_args(self, *args):
        X, Y, U, V, C = [None]*5
        args = list(args)
        if len(args) == 3 or len(args) == 5:
            C = ma.asarray(args.pop(-1)).ravel()
        V = ma.asarray(args.pop(-1))
        U = ma.asarray(args.pop(-1))
        nn = np.shape(U)
        nc = nn[0]
        nr = 1
        if len(nn) > 1:
            nr = nn[1]
        if len(args) == 2: # remaining after removing U,V,C
            X, Y = [np.array(a).ravel() for a in args]
            if len(X) == nc and len(Y) == nr:
                X, Y = [a.ravel() for a in np.meshgrid(X, Y)]
        else:
            indexgrid = np.meshgrid(np.arange(nc), np.arange(nr))
            X, Y = [np.ravel(a) for a in indexgrid]
        return X, Y, U, V, C

    def set_UVC(self, U, V, C=None):
        self.u = ma.asarray(U).ravel()
        self.v = ma.asarray(V).ravel()
        if C is not None:
            c = ma.asarray(C).ravel()
            x,y,u,v,c = delete_masked_points(self.x.ravel(), self.y.ravel(),
                self.u, self.v, c)
        else:
            x,y,u,v = delete_masked_points(self.x.ravel(), self.y.ravel(),
                self.u, self.v)

        magnitude = np.sqrt(u*u + v*v)
        flags, barbs, halves, empty = self._find_tails(magnitude,
            self.rounding, **self.barb_increments)

        #Get the vertices for each of the barbs

        plot_barbs = self._make_barbs(u, v, flags, barbs, halves, empty,
            self._length, self._pivot, self.sizes, self.fill_empty, self.flip)
        self.set_verts(plot_barbs)

        #Set the color array
        if C is not None:
            self.set_array(c)

        #Update the offsets in case the masked data changed
        xy = np.hstack((x[:,np.newaxis], y[:,np.newaxis]))
        self._offsets = xy

    def set_offsets(self, xy):
        '''
        Set the offsets for the barb polygons.  This saves the offets passed in
        and actually sets version masked as appropriate for the existing U/V
        data. *offsets* should be a sequence.

        ACCEPTS: sequence of pairs of floats
        '''
        self.x = xy[:,0]
        self.y = xy[:,1]
        x,y,u,v = delete_masked_points(self.x.ravel(), self.y.ravel(), self.u,
            self.v)
        xy = np.hstack((x[:,np.newaxis], y[:,np.newaxis]))
        collections.PolyCollection.set_offsets(self, xy)
    set_offsets.__doc__ = collections.PolyCollection.set_offsets.__doc__

    barbs_doc = _barbs_doc
