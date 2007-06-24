"""
The transforms module is broken into two parts, a collection of
classes written in the extension module _transforms to handle
efficient transformation of data, and some helper functions in
transforms to make it easy to instantiate and use those objects.
Hence the core of this module lives in _transforms.

The transforms class is built around the idea of a LazyValue.  A
LazyValue is a base class that defines a method get that returns the
value.  The concrete derived class Value wraps a float, and simply
returns the value of that float.  The concrete derived class BinOp
allows binary operations on LazyValues, so you can add them, multiply
them, etc.  When you do something like

  inches = Value(8)
  dpi    = Value(72)
  width  = inches * dpi

width is a BinOp instance (that tells you the width of the figure in
pixels).  Later, if the figure size in changed, ie we call

  inches.set(10)

The width variable is automatically updated because it stores a
pointer to the inches variable, not the value.  Since a BinOp is also
a lazy value, you can define binary operations on BinOps as well, such
as

  middle = Value(0.5) * width

Pairs of LazyValue instances can occur as instances of two classes:

    pt = Point( Value(x), Value(y)) # where x, y are numbers
         pt.x(), pt.y() return  Value(x), Value(y))

    iv = Interval( Value(x), Value(y))
         iv.contains(z) returns True if z is in the closed interval
         iv.contains_open(z): same for open interval
         iv.span() returns y-x as a float
         iv.get_bounds() returns (x,y) as a tuple of floats
         iv.set_bounds(x, y) allows input of new floats
         iv.update(seq) updates the bounds to include all elements
              in a sequence of floats
         iv.shift(s) shifts the interval by s, a float

The bounding box class Bbox is also heavily used, and is defined by a
lower left point ll and an upper right point ur.  The points ll and ur
are given by Point(x, y) instances, where x and y are LazyValues.  So
you can represent a point such as

  ll = Point( Value(0), Value(0)  )  # the origin
  ur = Point( width, height )        # the upper right of the figure

where width and height are defined as above, using the product of the
figure width in inches and the dpi.  This is, in face, how the Figure
bbox is defined

  bbox = Bbox(ll, ur)

A bbox basically defines an x,y coordinate system, with ll giving the
lower left of the coordinate system and ur giving the upper right.

The bbox methods are

  ll()                - return the lower left Point
  ur()                - return the upper right Point
  contains(x,y)       - return True if self contains point
  overlaps(bbox)      - return True if self overlaps bbox
  overlapsx(bbox)     - return True if self overlaps bbox in the x interval
  overlapsy(bbox)     - return True if self overlaps bbox in the y interval
  intervalx()         - return the x Interval instance
  intervaly()         - return the y interval instance
  get_bounds()        - get the left, bottom, width, height bounding tuple
  update(xys, ignore) - update the bbox to bound all the xy tuples in
      xys; if ignore is true ignore the current contents of bbox and
      just bound the tuples.  If ignore is false, bound self + tuples
  width()             - return the width of the bbox
  height()            - return the height of the bbox
  xmax()              - return the x coord of upper right
  ymax()              - return the y coord of upper right
  xmin()              - return the x coord of lower left
  ymin()              - return the y coord of lower left
  scale(sx,sy)        - scale the bbox by sx, sy
  deepcopy()          - return a deep copy of self (pointers are lost)


The basic transformation maps one bbox to another, with an optional
nonlinear transformation of one of coordinates (eg log scaling).

The base class for transformations is Transformation, and the concrete
derived classes are SeparableTransformation and Affine.  Earlier
versions of matplotlib handled transformation of x and y separately
(ie we assumed all transformations were separable) but this makes it
difficult to do rotations or polar transformations, for example.  All
artists contain their own transformation, defaulting to the identity
transform.

The signature of a separable transformation instance is

  trans = SeparableTransformation(bbox1, bbox2, funcx, funcy)

where funcx and funcy operate on x and y.  The typical linear
coordinate transformation maps one bounding box to another, with funcx
and funcy both identity.  Eg,

  transData = Transformation(viewLim, displayLim,
                             Func(IDENTITY), Func(IDENTITY))

maps the axes view limits to display limits.  If the xaxis scaling is
changed to log, one simply calls

  transData.get_funcx().set_type(LOG10)

For more general transformations including rotation, the Affine class
is provided, which is constructed with 6 LazyValue instances:
a, b, c, d, tx, ty.  These give the values of the matrix transformation

  [xo  =  |a  c| [xi  + [tx
   yo]    |b  d|  yi]    ty]

where if sx, sy are the scaling components, tx, y are the translation
components, and alpha is the rotation

   a = sx*cos(alpha);
   b = -sx*sin(alpha);
   c = sy*sin(alpha);
   d = sy*cos(alpha);

The affine transformation can be accomplished for row vectors with a
single matrix multiplication
            X_new = X_old * M
where
       M =     [ a  b  0
                 c  d  0
                 tx ty 1]
and each X is the row vector [x, y, 1].  Hence M is
the transpose of the matrix representation given in
http://en.wikipedia.org/wiki/Affine_transformation,
which is for the more usual column-vector representation
of the position.)



From a user perspective, the most important Tranformation methods are

All transformations
-------------------
  freeze()              - eval and freeze the lazy objects
  thaw()                - release the lazy objects

  xy_tup(xy)            - transform the tuple (x,y)
  seq_x_y(x, y)         - transform the python sequences x and y
  numerix_x_y(x, y)     - x and y are numerix 1D arrays
  numerix_xy(xy)        - xy is a numerix array of shape (N,2)
  inverse_numerix_xy(xy)- inverse of the above
  seq_xy_tups(seq)      - seq is a sequence of xy tuples or a (N,2) array
  inverse_xy_tup(xy)    - apply the inverse transformation to tuple xy

  set_offset(xy, trans) - xy is an x,y tuple and trans is a
    Transformation instance.  This will apply a post transformational
    offset of all future transformations by xt,yt = trans.xy_tup(xy[0], xy[1])

  deepcopy()            - returns a deep copy; references are lost
  shallowcopy()         - returns a shallow copy excluding the offset

Separable transformations
-------------------------

  get_bbox1() - return the input bbox
  get_bbox2() - return the output bbox
  set_bbox1() - set the input bbox
  set_bbox2() - set the output bbox
  get_funcx() - return the Func instance on x
  get_funcy() - return the Func instance on y
  set_funcx() - set the Func instance on x
  set_funcy() - set the Func instance on y


Affine transformations
----------------------

  as_vec6() - return the affine as length 6 list of Values


In general, you shouldn't need to construct your own transformations,
but should use the helper functions defined in this module.


  zero                        - return Value(0)
  one                         - return Value(1)
  origin                      - return Point(zero(), zero())
  unit_bbox                   - return the 0,0 to 1,1 bounding box
  identity_affine             - An affine identity transformation
  identity_transform          - An identity separable transformation
  translation_transform       - a pure translational affine
  scale_transform             - a pure scale affine
  scale_sep_transform         - a pure scale separable transformation
  scale_translation_transform - a scale and translate affine
  bound_vertices              - return the bbox that bounds all the xy tuples
  bbox_all                    - return the bbox that bounds all the bboxes
  lbwh_to_bbox                - build a bbox from tuple
                                left, bottom, width, height tuple

  multiply_affines            - return the affine that is the matrix product of
                                the two affines

  get_bbox_transform          - return a SeparableTransformation instance that
                                transforms one bbox to another

  blend_xy_sep_transform      - mix the x and y components of two separable
                                transformations into a new transformation.
                                This allows you to specify x and y in
                                different coordinate systems

  transform_bbox              - apply a transformation to a bbox and return the
                                transformed bbox

  inverse_transform_bbox      - apply the inverse transformation of a bbox
                                and return the inverse transformed bbox

  offset_copy                 - make a copy with an offset


The units/transform_unit.py code has many examples.

A related and partly overlapping class, PBox, has been added to the
original transforms module to facilitate Axes repositioning and resizing.
At present, the differences between Bbox and PBox include:

    Bbox works with the bounding box, the coordinates of the lower-left
    and upper-right corners; PBox works with the lower-left coordinates
    and the width, height pair (left, bottom, width, height, or 'lbwh').
    Obviously, these are equivalent, but lbwh is what is used by
    Axes._position, and it is the natural specification for the types of
    manipulations for which the PBox class was made.

    Bbox uses LazyValues grouped in pairs as 'll' and 'ur' Point objects;
    PBox uses a 4-element list, subclassed from the python list.

    Bbox and PBox methods are mostly quite different, reflecting their
    different original purposes.  Similarly, the CXX implementation of
    Bbox is good for methods such as update and for lazy evaluation, but
    for PBox intended uses, involving very little calculation, pure
    python probably is adequate.

In the future we may reimplement the PBox using Bbox
and transforms, or eliminate it entirely by adding its methods
and attributes to Bbox and/or putting them elsewhere in this module.
"""
from __future__ import division
import math
import numpy as npy

from matplotlib._transforms import Value, Point, Interval, Bbox, Affine
from matplotlib._transforms import IDENTITY, LOG10, POLAR, Func, FuncXY
from matplotlib._transforms import SeparableTransformation
from matplotlib._transforms import NonseparableTransformation

def nonsingular(vmin, vmax, expander=0.001, tiny=1e-15, increasing=True):
    '''
    Ensure the endpoints of a range are not too close together.

    "too close" means the interval is smaller than 'tiny' times
            the maximum absolute value.

    If they are too close, each will be moved by the 'expander'.
    If 'increasing' is True and vmin > vmax, they will be swapped,
    regardless of whether they are too close.
    '''
    swapped = False
    if vmax < vmin:
        vmin, vmax = vmax, vmin
        swapped = True
    if vmax - vmin <= max(abs(vmin), abs(vmax)) * tiny:
        if vmin==0.0:
            vmin = -expander
            vmax = expander
        else:
            vmin -= expander*abs(vmin)
            vmax += expander*abs(vmax)
    if swapped and not increasing:
        vmin, vmax = vmax, vmin
    return vmin, vmax


def zero(): return Value(0)

def one() : return Value(1)

def origin():
    return Point( zero(), zero() )

def unit_bbox():
    """
    Get a 0,0 -> 1,1 Bbox instance
    """
    return  Bbox( origin(), Point( one(), one() ) )

def identity_affine():
    """
    Get an affine transformation that maps x,y -> x,y
    """

    return Affine(one(), zero(), zero(), one(), zero(), zero())

def identity_transform():
    """
    Get an affine transformation that maps x,y -> x,y
    """
    return SeparableTransformation(unit_bbox(), unit_bbox(),
                                   Func(IDENTITY),
                                   Func(IDENTITY))

def translation_transform(tx, ty):
    """
    return a pure tranlational transformation tx and ty are LazyValue
    instances (Values or binary operations on values)
    """
    return Affine(one(), zero(), zero(), one(), tx, ty)

def scale_transform(sx, sy):
    """
    Return a pure scale transformation as an Affine instance; sx and
    sy are LazyValue instances (Values or binary opertations on
    values)
    """
    return Affine(sx, zero(), zero(), sy, zero(), zero())

def scale_sep_transform(sx, sy):
    """
    Return a pure scale transformation as a SeparableTransformation;
    sx and sy are LazyValue instances (Values or binary opertations on
    values)
    """

    bboxin = unit_bbox()
    bboxout = Bbox( Point( zero(), zero() ),
                    Point( sx, sy ) )
    return SeparableTransformation(
        bboxin, bboxout,
        Func(IDENTITY), Func(IDENTITY))



def bound_vertices(verts):
    """
    Return the Bbox of the sequence of x,y tuples in verts
    """
    # this is a good candidate to move to _transforms
    bbox = unit_bbox()
    bbox.update(verts, 1)
    return bbox

def bbox_all(bboxes):
    """
    Return the Bbox that bounds all bboxes
    """
    # this is a good candidate to move to _transforms
    assert(len(bboxes))

    if len(bboxes)==1: return bboxes[0]

    bbox = bboxes[0]
    minx = bbox.xmin()
    miny = bbox.ymin()
    maxx = bbox.xmax()
    maxy = bbox.ymax()

    for bbox in bboxes[1:]:
        thisminx = bbox.xmin()
        thisminy = bbox.ymin()
        thismaxx = bbox.xmax()
        thismaxy = bbox.ymax()

        if thisminx < minx : minx = thisminx
        if thismaxx > maxx : maxx = thismaxx
        if thisminy < miny : miny = thisminy
        if thismaxy > maxy : maxy = thismaxy

    return Bbox( Point( Value(minx), Value(miny) ),
                 Point( Value(maxx), Value(maxy) )
                 )

def lbwh_to_bbox(l,b,w,h):
    return Bbox( Point( Value(l), Value(b)),
                 Point( Value(l+w), Value(b + h) ) )


def invert_vec6(v):
    """
    v is a,b,c,d,tx,ty vec6 repr of an affine transformation.
    Return the inverse of v as a vec6
    """
    a,b,c,d,tx,ty = v
    M = npy.array([ [a,b,0], [c,d,0], [tx,ty,1]], dtype=npy.float64)
    Mi = npy.linalg.inv(M)
    return Mi[:,:2].ravel()

def multiply_affines( v1, v2):
    """
    v1 and v2 are Affine instances
    """

    a1, b1, c1, d1, tx1, ty1 = v1.as_vec6()
    a2, b2, c2, d2, tx2, ty2 = v2.as_vec6()

    a  = a1 * a2  + c1 * b2
    b  = b1 * a2  + d1 * b2
    c  = a1 * c2  + c1 * d2
    d  = b1 * c2  + d1 * d2
    tx = a1 * tx2 + c1 * ty2 + tx1
    ty = b1 * tx2 + d1 * ty2 + ty1
    return Affine(a,b,c,d,tx,ty)

def get_bbox_transform(boxin, boxout):
    """
    return the transform that maps transform one bounding box to
    another
    """
    return SeparableTransformation(
        boxin, boxout, Func(IDENTITY), Func( IDENTITY))


def blend_xy_sep_transform(trans1, trans2):
    """
    If trans1 and trans2 are SeparableTransformation instances, you can
    build a new SeparableTransformation from them by extracting the x and y
    bounding points and functions and recomposing a new SeparableTransformation

    This function extracts all the relevant bits from trans1 and
    trans2 and returns the new Transformation instance.  This is
    useful, for example, if you want to specify x in data coordinates
    and y in axes coordinates.
    """

    inboxx = trans1.get_bbox1()
    inboxy = trans2.get_bbox1()

    outboxx = trans1.get_bbox2()
    outboxy = trans2.get_bbox2()

    xminIn  =  inboxx.ll().x()
    xmaxIn  =  inboxx.ur().x()
    xminOut = outboxx.ll().x()
    xmaxOut = outboxx.ur().x()

    yminIn  =  inboxy.ll().y()
    ymaxIn  =  inboxy.ur().y()
    yminOut = outboxy.ll().y()
    ymaxOut = outboxy.ur().y()

    funcx = trans1.get_funcx()
    funcy = trans2.get_funcy()

    boxin  = Bbox( Point(xminIn,  yminIn),  Point(xmaxIn,  ymaxIn)  )
    boxout = Bbox( Point(xminOut, yminOut), Point(xmaxOut, ymaxOut) )

    return SeparableTransformation(boxin, boxout, funcx, funcy)


def transform_bbox(trans, bbox):
    'transform the bbox to a new bbox'
    xmin, xmax = bbox.intervalx().get_bounds()
    ymin, ymax = bbox.intervaly().get_bounds()

    xmin, ymin = trans.xy_tup((xmin, ymin))
    xmax, ymax = trans.xy_tup((xmax, ymax))

    return Bbox(Point(Value(xmin), Value(ymin)),
                Point(Value(xmax), Value(ymax)))

def inverse_transform_bbox(trans, bbox):
    'inverse transform the bbox'
    xmin, xmax = bbox.intervalx().get_bounds()
    ymin, ymax = bbox.intervaly().get_bounds()

    xmin, ymin = trans.inverse_xy_tup((xmin, ymin))
    xmax, ymax = trans.inverse_xy_tup((xmax, ymax))
    return Bbox(Point(Value(xmin), Value(ymin)),
                Point(Value(xmax), Value(ymax)))


def offset_copy(trans, fig=None, x=0, y=0, units='inches'):
    '''
    Return a shallow copy of a transform with an added offset.
      args:
        trans is any transform
      kwargs:
        fig is the current figure; it can be None if units are 'dots'
        x, y give the offset
        units is 'inches', 'points' or 'dots'
    '''
    newtrans = trans.shallowcopy()
    if units == 'dots':
        newtrans.set_offset((x,y), identity_transform())
        return newtrans
    if fig is None:
        raise ValueError('For units of inches or points a fig kwarg is needed')
    if units == 'points':
        x /= 72.0
        y /= 72.0
    elif not units == 'inches':
        raise ValueError('units must be dots, points, or inches')
    tx = Value(x) * fig.dpi
    ty = Value(y) * fig.dpi
    newtrans.set_offset((0,0), translation_transform(tx, ty))
    return newtrans


def get_vec6_scales(v):
    'v is an affine vec6 a,b,c,d,tx,ty; return sx, sy'
    a,b,c,d = v[:4]
    sx = math.sqrt(a**2 + b**2)
    sy = math.sqrt(c**2 + d**2)
    return sx, sy

def get_vec6_rotation(v):
    'v is an affine vec6 a,b,c,d,tx,ty; return rotation in degrees'
    sx, sy = get_vec6_scales(v)
    c,d = v[2:4]
    angle = math.atan2(c,d)/math.pi*180
    return angle



class PBox(list):
    '''
    A left-bottom-width-height (lbwh) specification of a bounding box,
    such as is used to specify the position of an Axes object within
    a Figure.
    It is a 4-element list with methods for changing the size, shape,
    and position relative to its container.
    '''
    coefs = {'C':  (0.5, 0.5),
             'SW': (0,0),
             'S':  (0.5, 0),
             'SE': (1.0, 0),
             'E':  (1.0, 0.5),
             'NE': (1.0, 1.0),
             'N':  (0.5, 1.0),
             'NW': (0, 1.0),
             'W':  (0, 0.5)}
    def __init__(self, box, container=None, llur=False):
        if len(box) != 4:
            raise ValueError("Argument must be iterable of length 4")
        if llur:
            box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        list.__init__(self, box)
        self.set_container(container)

    def as_llur(self):
        return [self[0], self[1], self[0]+self[2], self[1]+self[3]]

    def set_container(self, box=None):
        if box is None:
            box = self
        if len(box) != 4:
            raise ValueError("Argument must be iterable of length 4")
        self._container = list(box)

    def get_container(self, box):
        return self._container

    def anchor(self, c, container=None):
        '''
        Shift to position c within its container.

        c can be a sequence (cx, cy) where cx, cy range from 0 to 1,
        where 0 is left or bottom and 1 is right or top.

        Alternatively, c can be a string: C for centered,
        S for bottom-center, SE for bottom-left, E for left, etc.

        Optional arg container is the lbwh box within which the
        PBox is positioned; it defaults to the initial
        PBox.
        '''
        if container is None:
            container = self._container
        l,b,w,h = container
        if isinstance(c, str):
            cx, cy = self.coefs[c]
        else:
            cx, cy = c
        W,H = self[2:]
        self[:2] = l + cx * (w-W), b + cy * (h-H)
        return self

    def shrink(self, mx, my):
        '''
        Shrink the box by mx in the x direction and my in the y direction.
        The lower left corner of the box remains unchanged.
        Normally mx and my will be <= 1, but this is not enforced.
        '''
        assert mx >= 0 and my >= 0
        self[2:] = mx * self[2], my * self[3]
        return self

    def shrink_to_aspect(self, box_aspect, fig_aspect = 1):
        '''
        Shrink the box so that it is as large as it can be while
        having the desired aspect ratio, box_aspect.
        If the box coordinates are relative--that is, fractions of
        a larger box such as a figure--then the physical aspect
        ratio of that figure is specified with fig_aspect, so
        that box_aspect can also be given as a ratio of the
        absolute dimensions, not the relative dimensions.
        '''
        assert box_aspect > 0 and fig_aspect > 0
        l,b,w,h = self._container
        H = w * box_aspect/fig_aspect
        if H <= h:
            W = w
        else:
            W = h * fig_aspect/box_aspect
            H = h
        self[2:] = W,H
        return self

    def splitx(self, *args):
        '''
        e.g., PB.splitx(f1, f2, ...)

        Returns a list of new PBoxes formed by
        splitting the original one (PB) with vertical lines
        at fractional positions f1, f2, ...
        '''
        boxes = []
        xf = [0] + list(args) + [1]
        l,b,w,h = self[:]
        for xf0, xf1 in zip(xf[:-1], xf[1:]):
            boxes.append(PBox([l+xf0*w, b, (xf1-xf0)*w, h]))
        return boxes

    def splity(self, *args):
        '''
        e.g., PB.splity(f1, f2, ...)

        Returns a list of new PBoxes formed by
        splitting the original one (PB) with horizontal lines
        at fractional positions f1, f2, ..., with y measured
        positive up.
        '''
        boxes = []
        yf = [0] + list(args) + [1]
        l,b,w,h = self[:]
        for yf0, yf1 in zip(yf[:-1], yf[1:]):
            boxes.append(PBox([l, b+yf0*h, w, (yf1-yf0)*h]))
        return boxes



