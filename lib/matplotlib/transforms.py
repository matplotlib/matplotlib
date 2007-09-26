"""
A set of classes to handle transformations.

2007 Michael Droettboom
"""

import numpy as npy
from numpy import ma as ma
from numpy.linalg import inv
from sets import Set
from weakref import WeakKeyDictionary

from path import Path

DEBUG = False

# MGDTODO: This creates a ton of cyclical references.  We may want to
# consider using weak references

# MGDTODO: deep copying is probably incorrect wrt the parent/child
# relationships

class TransformNode(object):
    _gid = 0
    
    def __init__(self):
        # MGDTODO: I'd like to use a WeakKeyDictionary here, but it makes
        # these instances uncopyable.  As it stands, _parents grows
        # unboundedly...  Not a good idea.
        self._parents = Set()
        self._children = Set()
        self._id = TransformNode._gid
        
    def invalidate(self, affine_only=None):
        if affine_only is None:
            affine_only = self.is_affine() or self.is_bbox()
        if not self._do_invalidation(affine_only):
            self._id = TransformNode._gid
            TransformNode._gid += 1
            for parent in self._parents:
                parent.invalidate(affine_only)

    def _do_invalidation(self, affine_only):
        return False
        
    def set_children(self, children):
        for child in children:
            getattr(self, child)._parents.add(self)
        self._children = children

    def make_graphviz(self, fobj):
        seen = Set()

        def recurse(root):
            if root in seen:
                return
            seen.add(root)
            fobj.write('%s [label="%s"];\n' %
                       (hash(root), root.__class__.__name__))
            if root.is_affine():
                fobj.write('%s [style=filled, color=".7 .7 .9"];\n' %
                           hash(root))
            elif root.is_bbox():
                fobj.write('%s [style=filled, color=".9 .9 .7"];\n' %
                           hash(root))
            for child_name in root._children:
                child = getattr(root, child_name)
                fobj.write('%s -> %s [label="%s"];\n' % (
                        hash(root),
                        hash(child),
                        child_name))
                recurse(child)

        fobj.write("digraph G {\n")
        recurse(self)
        fobj.write("}\n")
        
    def is_affine(self):
        return False

    def is_bbox(self):
        return False

    def get_id(self):
        return self._id
    
    
class BboxBase(TransformNode):
    '''
    This is the read-only part of a bounding-box
    '''
    
    def __init__(self):
        TransformNode.__init__(self)

    def is_bbox(self):
        return True
        
    def __array__(self):
        return self.get_points()

    if DEBUG:
        def invalidate(self):
            points = self.get_points()
            assert points[0, 0] <= points[1, 0]
            assert points[0, 1] <= points[1, 1]
            TransformNode.invalidate(self)
    
    # MGDTODO: Probably a more efficient ways to do this...
    def _get_xmin(self):
        return self.get_points()[0, 0]
    xmin = property(_get_xmin)
    
    def _get_ymin(self):
        return self.get_points()[0, 1]
    ymin = property(_get_ymin)

    def _get_xmax(self):
        return self.get_points()[1, 0]
    xmax = property(_get_xmax)

    def _get_ymax(self):
        return self.get_points()[1, 1]
    ymax = property(_get_ymax)

    def _get_min(self):
        return self.get_points()[0]
    min = property(_get_min)
    
    def _get_max(self):
        return self.get_points()[1]
    max = property(_get_max)
    
    def _get_intervalx(self):
        return self.get_points()[:, 0]
    intervalx = property(_get_intervalx)

    def _get_intervaly(self):
        return self.get_points()[:, 1]
    intervaly = property(_get_intervaly)
                         
    def _get_width(self):
	points = self.get_points()
	return points[1, 0] - points[0, 0]
    width = property(_get_width)

    def _get_height(self):
	points = self.get_points()
	return points[1, 1] - points[0, 1]
    height = property(_get_height)

    def _get_bounds(self):
        ((xmin, ymin), (xmax, ymax)) = self.get_points()
        return (xmin, ymin, xmax - xmin, ymax - ymin)
    bounds = property(_get_bounds)

    def _get_lbrt(self):
        return self.get_points().flatten().copy()
    lbrt = property(_get_lbrt)
    
    def get_points(self):
        return NotImplementedError()
    
    # MGDTODO: Optimize
    def containsx(self, x):
        xmin, xmax = self.intervalx
        return x >= xmin and x <= xmax

    def containsy(self, y):
        ymin, ymax = self.intervaly
        return y >= ymin and y <= ymax
    
    def contains(self, x, y):
        return self.containsx(x) and self.containsy(y)

    def overlapsx(self, other):
        xmin, xmax = other.intervalx
        return self.containsx(xmin) \
            or self.containsx(xmax)

    def overlapsy(self, other):
        ymin, ymax = other.intervaly
        return self.containsy(ymin) \
            or self.containsx(ymax)
    
    def overlaps(self, other):
        return self.overlapsx(other) \
            and self.overlapsy(other)
    
    def fully_containsx(self, x):
        xmin, xmax = self.intervalx
        return x > xmin and x < xmax

    def fully_containsy(self, y):
        ymin, ymax = self.intervaly
        return y > ymin and y < ymax
    
    def fully_contains(self, x, y):
        return self.fully_containsx(x) \
            and self.fully_containsy(y)

    def fully_overlapsx(self, other):
        xmin, xmax = other.intervalx
        return self.fully_containsx(xmin) \
            or self.fully_containsx(xmax)

    def fully_overlapsy(self, other):
        ymin, ymax = other.intervaly
        return self.fully_containsy(ymin) \
            or self.fully_containsx(ymax)
    
    def fully_overlaps(self, other):
        return self.fully_overlapsx(other) and \
            self.fully_overlapsy(other)

    def transformed(self, transform):
        return Bbox(transform.transform(self.get_points()))

    def inverse_transformed(self, transform):
        return Bbox(transform.inverted().transform(self.get_points()))
    
    
class Bbox(BboxBase):
    def __init__(self, points):
        BboxBase.__init__(self)
        self._points = npy.asarray(points, npy.float_)
        self._minpos = npy.array([0.0000001, 0.0000001])
        self._invalid = False

    #@staticmethod
    def unit():
        return Bbox.from_lbrt(0., 0., 1., 1.)
    unit = staticmethod(unit)

    #@staticmethod
    def from_lbwh(left, bottom, width, height):
        return Bbox.from_lbrt(left, bottom, left + width, bottom + height)
    from_lbwh = staticmethod(from_lbwh)

    #@staticmethod
    def from_lbrt(*args):
        points = npy.array(args, dtype=npy.float_).reshape(2, 2)
        return Bbox(points)
    from_lbrt = staticmethod(from_lbrt)
    
    def __repr__(self):
        return 'Bbox(%s)' % repr(self._points)
    __str__ = __repr__

    def _do_invalidation(self, affine_only):
        result = self._invalid
        self._invalid = True
        return result
    
    def update_from_data(self, x, y, ignore=True):
        if ignore:
            self._points = npy.array(
                [[x.min(), y.min()], [x.max(), y.max()]],
                npy.float_)
            self._minpos = npy.array(
                [npy.where(x > 0.0, x, npy.inf).min(), npy.where(y > 0.0, y, npy.inf).min()],
                npy.float_)
        else:
	    self._points = npy.array(
		[[min(x.min(), self.xmin), min(y.min(), self.ymin)],
		 [max(x.max(), self.xmax), max(y.max(), self.ymax)]],
		 npy.float_)
            minpos = npy.array(
                [npy.where(x > 0.0, x, npy.inf).min(), npy.where(y > 0.0, y, npy.inf).min()],
                npy.float_)
            self._minpos = npy.minimum(minpos, self._minpos)
        self.invalidate()
        
    def _set_xmin(self, val):
        self._points[0, 0] = val
        self.invalidate()
    xmin = property(BboxBase._get_xmin, _set_xmin)

    def _set_ymin(self, val):
        self._points[0, 1] = val
        self.invalidate()
    ymin = property(BboxBase._get_ymin, _set_ymin)

    def _set_xmax(self, val):
        self._points[1, 0] = val
        self.invalidate()
    xmax = property(BboxBase._get_xmax, _set_xmax)

    def _set_ymax(self, val):
        self._points[1, 1] = val
        self.invalidate()
    ymax = property(BboxBase._get_ymax, _set_ymax)

    def _set_min(self, val):
        self._points[0] = val
        self.invalidate()
    min = property(BboxBase._get_min, _set_min)
    
    def _set_max(self, val):
        self._points[1] = val
        self.invalidate()
    max = property(BboxBase._get_max, _set_max)
    
    def _set_intervalx(self, interval):
        self._points[:, 0] = interval
        self.invalidate()
    intervalx = property(BboxBase._get_intervalx, _set_intervalx)

    def _set_intervaly(self, interval):
        self._points[:, 1] = interval
        self.invalidate()
    intervaly = property(BboxBase._get_intervaly, _set_intervaly)

    def _set_bounds(self, bounds):
        l,b,w,h = bounds
        self._points = npy.array([[l, b], [l+w, b+h]], npy.float_)
        self.invalidate()
    bounds = property(BboxBase._get_bounds, _set_bounds)

    def _get_minpos(self):
        return self._minpos
    minpos = property(_get_minpos)

    def _get_minposx(self):
        return self._minpos[0]
    minposx = property(_get_minposx)

    def _get_minposy(self):
        return self._minpos[1]
    minposy = property(_get_minposy)
    
    def get_points(self):
        self._invalid = False
        return self._points

    def set_points(self, points):
        self._points = points
        self.invalidate()

    def set(self, other):
        self._points = other.get_points()
        self.invalidate()
        
    def expanded(self, sw, sh):
        width = self.width
        height = self.height
        deltaw = (sw * width - width) / 2.0
        deltah = (sh * height - height) / 2.0
        a = npy.array([[-deltaw, -deltah], [deltaw, deltah]])
        return Bbox(self._points + a)

    #@staticmethod
    def union(bboxes):
        """
        Return the Bbox that bounds all bboxes
        """
        # MGDTODO: There's got to be a way to utilize numpy here
        # to make this faster...
        assert(len(bboxes))

        if len(bboxes) == 1:
            return bboxes[0]

        bbox = bboxes[0]
        xmin = bbox.xmin
        ymin = bbox.ymin
        xmax = bbox.xmax
        ymax = bbox.ymax

        for bbox in bboxes[1:]:
            xmin = min(xmin, bbox.xmin)
            ymin = min(ymin, bbox.ymin)
            xmax = max(xmax, bbox.xmax)
            ymax = max(ymax, bbox.ymax)

        return Bbox.from_lbrt(xmin, ymin, xmax, ymax)
    union = staticmethod(union)

    
class TransformedBbox(BboxBase):
    def __init__(self, bbox, transform):
        assert bbox.is_bbox()
        assert isinstance(transform, Transform)
        assert transform.input_dims == 2
        assert transform.output_dims == 2

        BboxBase.__init__(self)
        self._bbox = bbox
        self._transform = transform
        self.set_children(['_bbox', '_transform'])
        self._points = None

    def __repr__(self):
        return "TransformedBbox(%s, %s)" % (self._bbox, self._transform)
    __str__ = __repr__
    
    def _do_invalidation(self, affine_only):
        result = self._points is None
        self._points = None
        return result

    def get_points(self):
        if self._points is None:
            self._points = self._transform.transform(self._bbox.get_points())
        return self._points

    
class Transform(TransformNode):
    def __init__(self):
        TransformNode.__init__(self)

    def is_separable(self):
        return False
        
    def __add__(self, other):
        if isinstance(other, Transform):
            return composite_transform_factory(self, other)
        raise TypeError(
            "Can not add Transform to object of type '%s'" % type(other))

    def __radd__(self, other):
        if isinstance(other, Transform):
            return composite_transform_factory(other, self)
        raise TypeError(
            "Can not add Transform to object of type '%s'" % type(other))

    def transform(self, points):
        raise NotImplementedError

    def transform_affine(self, points):
        raise NotImplementedError

    def transform_non_affine(self, points):
        raise NotImplementedError

    def get_affine(self):
        raise NotImplementedError
    
    def transform_point(self, point):
        return self.transform(npy.asarray([point]))[0]

    def has_inverse(self):
        raise NotImplementedError()
    
    def inverted(self):
        raise NotImplementedError()


class TransformWrapper(Transform):
    def __init__(self, child):
        assert isinstance(child, Transform)
        
        Transform.__init__(self)
        self.input_dims = child.input_dims
        self.output_dims = child.output_dims
        self._child = child
        self.set_children(['_child'])

    def __repr__(self):
        return "TransformWrapper(%r)" % self._child
    __str__ = __repr__
        
    def set(self, child):
        assert child.input_dims == self.input_dims
        assert child.output_dims == self.output_dims
        self._child = child
        self.set_children(['_child'])
        self.invalidate()
        
    def is_separable(self):
        return self._child.is_separable()

    def is_affine(self):
        return self._child.is_affine()
    
    def transform(self, points):
        return self._child.transform(points)

    def transform_affine(self, points):
        return self._child.transform_affine(points)

    def transform_non_affine(self, points):
        return self._child.transform_non_affine(points)

    def get_affine(self):
        return self._child.get_affine()
    
    def inverted(self):
        return self._child.inverted()
    
    
class AffineBase(Transform):
    def __init__(self):
        Transform.__init__(self)
        self._inverted = None

    def is_affine(self):
        return True
        
    def __array__(self, *args, **kwargs):
	return self.get_matrix()
	
    #@staticmethod
    def _concat(a, b):
        return npy.dot(b, a)
    _concat = staticmethod(_concat)

    #@staticmethod
    def concat(a, b):
        return Affine1D(Affine1D._concat(a.get_matrix(), b.get_matrix()))
    concat = staticmethod(concat)

    def get_matrix(self):
        raise NotImplementedError()

    def transform_non_affine(self, points):
        return points

    def get_affine(self):
        return self
    

class Affine1DBase(AffineBase):
    input_dims = 1
    output_dims = 1

    def __init__(self):
        AffineBase.__init__(self)

    def is_separable(self):
        return True
        
    def __array__(self, *args, **kwargs):
	return self.get_matrix()

    def to_values(self):
        mtx = self.get_matrix()
        return mtx[0]
    
    #@staticmethod
    def matrix_from_values(a, b):
        affine = npy.zeros((2, 2), npy.float_)
        affine[0, :] = (a, b)
        affine[1, 1] = 1
        return affine
    matrix_from_values = staticmethod(matrix_from_values)

    def transform(self, values):
        """
        Applies the transformation to an array of values and
        returns the result.
        """
        # MGDTODO: The major speed trap here is just converting to
        # the points to an array in the first place.  If we can use
        # more arrays upstream, that should help here.
#         if not isinstance(points, npy.ndarray):
#             import traceback
#             print '-' * 60
#             print 'A non-numpy array was passed in for transformation.  Please '
#             print 'correct this.'
#             print "".join(traceback.format_stack())
#             print points
        mtx = self.get_matrix()
        # points = npy.asarray(values, npy.float_)
        return points * mtx[0,0] + mtx[0,1]

    transform_affine = transform
    
    def inverted(self):
        if self._inverted is None:
            mtx = self.get_matrix()
            self._inverted = Affine1D(inv(mtx))
        return self._inverted
    

class Affine1D(Affine1DBase):
    def __init__(self, matrix = None):
        """
        Initialize an Affine transform from a 2x2 numpy float array.

        a b
        0 1
        """
        Affine1DBase.__init__(self)
        if matrix is None:
            matrix = npy.identity(2)
        else:
	    matrix = npy.asarray(matrix, npy.float_)
            assert matrix.shape == (2, 2)
        self._mtx = matrix

    def __repr__(self):
        return "Affine1D(%s)" % repr(self._mtx)
    __str__ = __repr__

    def __cmp__(self, other):
        if (isinstance(other, Affine1D) and
            (self.get_matrix() == other.get_matrix()).all()):
            return 0
        return -1
    
    #@staticmethod
    def from_values(a, b):
        return Affine1D(Affine1D.matrix_from_values(a, b))
    from_values = staticmethod(from_values)

    def get_matrix(self):
        return self._mtx

    def set_matrix(self, mtx):
        self._mtx = mtx
        self.invalidate()
        
    def set(self, other):
        self._mtx = other.get_matrix()
        self.invalidate()
    
    #@staticmethod
    def identity():
        return Affine1D(npy.identity(2))
    identity = staticmethod(identity)

    def clear(self):
        self._mtx = npy.identity(2)
        self.invalidate()
        return self
    
    def translate(self, t):
        self._mtx[0, 1] += t
        self.invalidate()
        return self

    def scale(self, s):
        self._mtx[0, 0] *= s
        self.invalidate()
        return self

    def is_separable(self):
        mtx = self.get_matrix()
        return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0

    
class IntervalTransform(Affine1DBase):
    def __init__(self, bbox, direction):
        assert direction in ('x', 'y')
        assert bbox.is_bbox()
        
        Affine1DBase.__init__(self)
        self._bbox = bbox
        self._direction = "interval" + direction
        self.set_children(['_bbox'])
        self._mtx = None
        
    def __repr__(self):
        return "IntervalTransform(%s)" % (getattr(self._bbox, self._direction))
    __str__ = __repr__

    def _do_invalidation(self, affine_only):
        result = self._mtx is None
        self._mtx = None
        self._inverted = None
        return result

    def get_matrix(self):
        if self._mtx is None:
            min, max = getattr(self._bbox, self._direction)
            self._mtx = inv(npy.array([[max - min, min],
                                       [0.0, 1.0]], npy.float_))
        return self._mtx
    
    
class Affine2DBase(AffineBase):
    input_dims = 2
    output_dims = 2

    def __init__(self):
        AffineBase.__init__(self)

    def is_separable(self):
        mtx = self.get_matrix()
        return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0
        
    def __array__(self, *args, **kwargs):
	return self.get_matrix()
    
    def to_values(self):
        mtx = self.get_matrix()
        return tuple(mtx[:2].swapaxes(0, 1).flatten())
    
    #@staticmethod
    def matrix_from_values(a, b, c, d, e, f):
        affine = npy.zeros((3, 3), npy.float_)
        affine[0,  ] = a, c, e
        affine[1,  ] = b, d, f
        affine[2, 2] = 1
        return affine
    matrix_from_values = staticmethod(matrix_from_values)

    def transform(self, points):
        """
        Applies the transformation to an array of 2D points and
        returns the result.

        points must be a numpy array of shape (N, 2), where N is the
        number of points.
        """
        # MGDTODO: The major speed trap here is just converting to
        # the points to an array in the first place.  If we can use
        # more arrays upstream, that should help here.
#         if not isinstance(points, npy.ndarray):
#             import traceback
#             print '-' * 60
#             print 'A non-numpy array was passed in for transformation.  Please '
#             print 'correct this.'
#             print "".join(traceback.format_stack())
#             print points
        mtx = self.get_matrix()
        if ma.isarray(points):
            points = points.transpose()
            points = ma.dot(mtx[0:2, 0:2], points)
            points = points + mtx[0:2, 2:]
        else:
            points = npy.asarray(points, npy.float_)
            points = points.transpose()
            points = npy.dot(mtx[0:2, 0:2], points)
            points = points + mtx[0:2, 2:]
        return points.transpose()

    transform_affine = transform
    
    def inverted(self):
        if self._inverted is None:
            mtx = self.get_matrix()
            self._inverted = Affine2D(inv(mtx))
        return self._inverted

        
class Affine2D(Affine2DBase):
    def __init__(self, matrix = None):
        """
        Initialize an Affine transform from a 3x3 numpy float array.

        a c e
        b d f
        0 0 1
        """
        Affine2DBase.__init__(self)
        if matrix is None:
            matrix = npy.identity(3)
        else:
	    matrix = npy.asarray(matrix, npy.float_)
            assert matrix.shape == (3, 3)
        self._mtx = matrix

    def __repr__(self):
        return "Affine2D(%s)" % repr(self._mtx)
    __str__ = __repr__

    def __cmp__(self, other):
        if (isinstance(other, Affine2D) and
            (self.get_matrix() == other.get_matrix()).all()):
            return 0
        return -1
    
    #@staticmethod
    def from_values(a, b, c, d, e, f):
        return Affine2D(Affine2D.matrix_from_values(a, b, c, d, e, f))
    from_values = staticmethod(from_values)

    def get_matrix(self):
        return self._mtx

    def set_matrix(self, mtx):
        self._mtx = mtx
        self.invalidate()

    def set(self, other):
        self._mtx = other.get_matrix()
        self.invalidate()
    
    #@staticmethod
    def identity():
        return Affine2D(npy.identity(3))
    identity = staticmethod(identity)

    def clear(self):
        self._mtx = npy.identity(3)
        self.invalidate()
        return self
    
    def rotate(self, theta):
        a = npy.cos(theta)
        b = npy.sin(theta)
        rotate_mtx = self.matrix_from_values(a, b, -b, a, 0, 0)
        self._mtx = self._concat(self._mtx, rotate_mtx)
        self.invalidate()
        return self

    def rotate_deg(self, degrees):
        return self.rotate(degrees*npy.pi/180.)

    def rotate_around(self, x, y, theta):
        return self.translate(-x, -y).rotate(theta).translate(x, y)

    def rotate_deg_around(self, x, y, degrees):
        return self.translate(-x, -y).rotate_deg(degrees).translate(x, y)
    
    def translate(self, tx, ty):
        translate_mtx = self.matrix_from_values(1., 0., 0., 1., tx, ty)
        self._mtx = self._concat(self._mtx, translate_mtx)
        self.invalidate()
        return self

    def scale(self, sx, sy=None):
        if sy is None:
            sy = sx
        scale_mtx = self.matrix_from_values(sx, 0., 0., sy, 0., 0.)
        self._mtx = self._concat(self._mtx, scale_mtx)
        self.invalidate()
        return self

    def is_separable(self):
        mtx = self.get_matrix()
        return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0


class IdentityTransform(Affine2DBase):
    """
    A special class that does the identity transform quickly.
    """
    _mtx = npy.identity(3)

    def __repr__(self):
        return "IdentityTransform()"
    __str__ = __repr__
    
    def __cmp__(self, other):
        if (isinstance(other, Affine2D) and
            (other == IDENTITY)):
            return 0
        return -1

    def get_matrix(self):
        return self._mtx
    
    def transform(self, points):
        return points
    transform_affine = transform_non_affine = transform

    def get_affine(self):
        return self
    inverted = get_affine
    
    
class BlendedGenericTransform(Transform):
    input_dims = 2
    output_dims = 2

    def __init__(self, x_transform, y_transform):
	# Here we ask: "Does it blend?"
        assert x_transform.is_separable()
        assert y_transform.is_separable()
        
        Transform.__init__(self)
        self._x = x_transform
        self._y = y_transform
        self.set_children(['_x', '_y'])

    def is_affine(self):
        return self._x.is_affine() and self._y.is_affine()
        
    def is_separable(self):
        return True
        
    def __repr__(self):
        return "BlendedGenericTransform(%s,%s)" % (self._x, self._y)
    __str__ = __repr__
        
    def transform(self, points):
        x = self._x
        y = self._y
        if x == y and x.input_dims == 2:
            return self._x.transform(points)

        if x.input_dims == 2:
            x_points = x.transform(points)[:, 0:1]
        else:
            x_points = x.transform(points[:, 0])
            x_points = x_points.reshape((len(x_points), 1))
            
        if y.input_dims == 2:
            y_points = y.transform(points)[:, 1:]
        else:
            y_points = y.transform(points[:, 1])
            y_points = y_points.reshape((len(y_points), 1))

        return ma.concatenate((x_points, y_points), 1)
    transform_non_affine = transform
    
    def transform_affine(self, points):
        return points
        
    def get_affine(self):
        return IdentityTransform()
    
    def inverted(self):
        return BlendedGenericTransform(self._x.inverted(), self._y.inverted())


class BlendedAffine1D(Affine2DBase, Transform):
    def __init__(self, x_transform, y_transform):
        assert isinstance(x_transform, Affine1DBase)
        assert isinstance(y_transform, Affine1DBase)

        Transform.__init__(self)
        self._x = x_transform
        self._y = y_transform
        self.set_children(['_x', '_y'])
        
        Affine2DBase.__init__(self)
        self._mtx = None

    def is_separable(self):
        return True
        
    def __repr__(self):
        return "BlendedAffine1D(%s,%s)" % (self._x, self._y)
    __str__ = __repr__
        
    def _do_invalidation(self, affine_only):
        result = self._mtx is None
        self._mtx = None
        self._inverted = None

    def get_matrix(self):
        if self._mtx is None:
            x_mtx = self._x.get_matrix()
            y_mtx = self._y.get_matrix()
            self._mtx = npy.array([[x_mtx[0, 0], 0.0, x_mtx[0, 1]],
                                   [0.0, y_mtx[0, 0], y_mtx[0, 1]],
                                   [0.0, 0.0, 1.0]])
        return self._mtx

    
class BlendedAffine2D(Affine2DBase, Transform):
    def __init__(self, x_transform, y_transform):
        assert x_transform.is_affine()
        assert y_transform.is_affine()
        assert x_transform.is_separable()
        assert y_transform.is_separable()

        Transform.__init__(self)
        self._x = x_transform
        self._y = y_transform
        self.set_children(['_x', '_y'])
        
        Affine2DBase.__init__(self)
        self._mtx = None

    def is_separable(self):
        return True
        
    def __repr__(self):
        return "BlendedAffine2D(%s,%s)" % (self._x, self._y)
    __str__ = __repr__
        
    def _do_invalidation(self, affine_only):
        result = self._mtx is None
        self._mtx = None
        self._inverted = None
        return result

    def get_matrix(self):
        if self._mtx is None:
            if self._x == self._y:
                self._mtx = self._x.get_matrix()
            else:
                x_mtx = self._x.get_matrix()
                y_mtx = self._y.get_matrix()
                # This works because we already know the transforms are
                # separable, though normally one would want to set b and
                # c to zero.
                self._mtx = npy.vstack((x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]))
        return self._mtx

    
def blended_transform_factory(x_transform, y_transform):
    if isinstance(x_transform, Affine2DBase) and isinstance(y_transform, Affine2DBase):
        return BlendedAffine2D(x_transform, y_transform)
    elif isinstance(x_transform, Affine1DBase) and isinstance(y_transform, Affine1DBase):
        return BlendedAffine1D(x_transform, y_transform)
    return BlendedGenericTransform(x_transform, y_transform)


class CompositeGenericTransform(Transform):
    def __init__(self, a, b):
        assert a.output_dims == b.input_dims
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims
        
        Transform.__init__(self)
        self._a = a
        self._b = b
        self.set_children(['_a', '_b'])

    def is_affine(self):
        return self._a.is_affine() and self._b.is_affine()
        
    def is_separable(self):
        return self._a.is_separable() and self._b.is_separable()
        
    def __repr__(self):
        return "CompositeGenericTransform(%s, %s)" % (self._a, self._b)
    __str__ = __repr__
        
    def transform(self, points):
        return self._b.transform(self._a.transform(points))

    def transform_affine(self, points):
        return self._b.transform_affine(self._a.transform_affine(points))

    def transform_non_affine(self, points):
        return self._b.transform_non_affine(self._a.transform_non_affine(points))

    def get_affine(self):
        return CompositeAffine2D(self._a.get_affine(), self._b.get_affine())
    
    def inverted(self):
        return CompositeGenericTransform(self._b.inverted(), self._a.inverted())

    
class CompositeAffine2D(Affine2DBase):
    def __init__(self, a, b):
        assert a.output_dims == b.input_dims
        self.input_dims = a.input_dims
        self.output_dims = b.output_dims
        assert a.is_affine()
        assert b.is_affine()

        Affine2DBase.__init__(self)
        self._a = a
        self._b = b
        self.set_children(['_a', '_b'])
        self._mtx = None

    def __repr__(self):
        return "CompositeAffine2D(%s, %s)" % (self._a, self._b)
    __str__ = __repr__

    def _do_invalidation(self, affine_only):
        result = self._mtx is None
        self._mtx = None
        self._inverted = None
        return result
    
    def get_matrix(self):
        if self._mtx is None:
            self._mtx = self._concat(
                self._a.get_matrix(),
                self._b.get_matrix())
        return self._mtx

    
def composite_transform_factory(a, b):
#     if isinstance(a, BboxTransform) and isinstance(b, BboxTransform):
#         return BboxTransform(a._boxin, b._boxout)
    if isinstance(a, AffineBase) and isinstance(b, AffineBase):
        return CompositeAffine2D(a, b)
    return CompositeGenericTransform(a, b)


class LogTransform(Transform):
    input_dims = 1
    output_dims = 1
    
    def transform(self, a):
        m = ma.masked_where(a < 0, a)
        return npy.log10(m)


class TestPolarTransform(Transform):
    input_dims = 2
    output_dims = 2

    def __init__(self, limits):
        assert limits.is_bbox()

        Transform.__init__(self)
        self._limits = limits
        self.set_children(['_limits'])
    
    def transform(self, xy):
        debug = len(xy) > 4
        limmin, limmax = self._limits.intervaly
        mask = (xy[:, 1:] < limmin) | (xy[:, 1:] > limmax)
        mask = ma.concatenate((mask, mask), 1)
        masked_xy = npy.ma.masked_where(mask, xy)
        x = masked_xy[:, 0:1]
        y = masked_xy[:, 1:2]
        if x.shape == () or y.shape == ():
            return masked_xy
        y = (y - limmin) / (limmax - limmin)
        x, y = y * ma.cos(x), y * ma.sin(x)
        result = ma.concatenate((x, y), 1)
        result = result * 0.5 + 0.5
        return result

    def inverted(self):
        return TestInvertPolarTransform(self._limits)
    
    def is_separable(self):
        return False


class TestInvertPolarTransform(Transform):
    input_dims = 2
    output_dims = 2

    def __init__(self, limits):
        assert limits.is_bbox()

        Transform.__init__(self)
        self._limits = limits
        self.set_children(['_limits'])
    
    def transform(self, xy):
        limmin, limmax = self._limits.intervaly
        xy = (xy - 0.5) * 2.0
        x = xy[:, 0:1]
        y = xy[:, 1:]
        r = ma.sqrt(ma.power(x, 2) + ma.power(y, 2))
        theta = ma.arccos(x / r)
        theta = ma.where(y < 0, 2 * npy.pi - theta, theta)
        r = r * (limmax - limmin) + limmin
        return ma.concatenate((theta, r), 1)

    def inverted(self):
        return TestInvertPolarTransform(self._limits)
    
    def is_separable(self):
        return False
    
    
class BboxTransform(Affine2DBase):
    def __init__(self, boxin, boxout):
        assert boxin.is_bbox()
        assert boxout.is_bbox()

        Affine2DBase.__init__(self)
        self._boxin = boxin
        self._boxout = boxout
        self.set_children(['_boxin', '_boxout'])
        self._mtx = None
        self._inverted = None

    def __repr__(self):
        return "BboxTransform(%s, %s)" % (self._boxin, self._boxout)
    __str__ = __repr__
        
    def _do_invalidation(self, affine_only):
        result = self._mtx is None
        self._mtx = None
        self._inverted = None
        return result
        
    def is_separable(self):
        return True

    def get_matrix(self):
        if self._mtx is None:
            boxin = self._boxin
            boxout = self._boxout
            x_scale = boxout.width / boxin.width
            y_scale = boxout.height / boxin.height

            # MGDTODO: Optimize
            affine = Affine2D() \
                .translate(-boxin.xmin, -boxin.ymin) \
                .scale(x_scale, y_scale) \
                .translate(boxout.xmin, boxout.ymin)

            self._mtx = affine._mtx
        return self._mtx


class TransformedPath(TransformNode):
    def __init__(self, path, transform):
        assert isinstance(transform, Transform)
        TransformNode.__init__(self)
        
        self._path = path
        self._transform = transform
        self.set_children(['_transform'])
        self._transformed_path = None

    def _do_invalidation(self, affine_only):
        if not affine_only:
            self._transformed_path = None
        return True
        
    def get_path_and_affine(self):
        if self._transformed_path is None:
            vertices = self._transform.transform_non_affine(self._path.vertices)
            self._transformed_path = Path(vertices, self._path.codes)

        return self._transformed_path, self._transform.get_affine()

    def get_path(self):
        if self._transformed_path is None:
            vertices = self._tranform.transform_non_affine(self._path.vertices)
            self._transformed_path = Path(vertices, self._path.codes)
        vertices = self._transform.transform_affine(self._transformed_path.vertices)
        return Path(vertices, self._transformed_path.codes)

    def get_affine(self):
        return self._transform.get_affine()
    
    
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
        if vmin == 0.0:
            vmin = -expander
            vmax = expander
        else:
            vmin -= expander*abs(vmin)
            vmax += expander*abs(vmax)
    if swapped and not increasing:
        vmin, vmax = vmax, vmin
    return vmin, vmax


# MGDTODO: Optimize (perhaps in an extension)
def interval_contains(interval, val):
    return interval[0] <= val and interval[1] >= val

def interval_contains_open(interval, val):
    return interval[0] < val and interval[1] > val
    
if __name__ == '__main__':
    import copy
    from random import random
    import timeit

    bbox = Bbox.from_lbrt(10., 15., 20., 25.)
    assert bbox.xmin == 10
    assert bbox.ymin == 15
    assert bbox.xmax == 20
    assert bbox.ymax == 25

    assert npy.all(bbox.min == [10, 15])
    assert npy.all(bbox.max == [20, 25])
    assert npy.all(bbox.intervalx == (10, 20))
    assert npy.all(bbox.intervaly == (15, 25))

    assert bbox.width == 10
    assert bbox.height == 10

    assert bbox.bounds == (10, 15, 10, 10)

    assert tuple(npy.asarray(bbox).flatten()) == (10, 15, 20, 25)
    
    bbox.intervalx = (11, 21)
    bbox.intervaly = (16, 26)
    
    assert bbox.bounds == (11, 16, 10, 10)

    bbox.xmin = 12
    bbox.ymin = 17
    bbox.xmax = 22
    bbox.ymax = 27

    assert bbox.bounds == (12, 17, 10, 10)

    bbox = Bbox.from_lbwh(10, 11, 12, 13)
    assert bbox.bounds == (10, 11, 12, 13)

    bbox_copy = copy.deepcopy(bbox)
    assert (bbox.lbrt == bbox_copy.lbrt).all()
    bbox_copy.max = (14, 15)
    assert bbox.bounds == (10, 11, 12, 13)
    assert bbox_copy.bounds == (10, 11, 4, 4)
    
    bbox1 = Bbox([[10., 15.], [20., 25.]])
    bbox2 = Bbox([[30., 35.], [40., 45.]])
    trans = BboxTransform(bbox1, bbox2)
    bbox3 = bbox1.transformed(trans)
    assert (bbox3.lbrt == bbox2.lbrt).all()

    translation = Affine2D().translate(10, 20)
    assert translation.to_values() == (1, 0, 0, 1, 10, 20)
    scale = Affine2D().scale(10, 20)
    assert scale.to_values() == (10, 0, 0, 20, 0, 0)
    rotation = Affine2D().rotate_deg(30)
    assert rotation.to_values() == (0.86602540378443871, 0.49999999999999994,
                                   -0.49999999999999994, 0.86602540378443871,
                                   0.0, 0.0)
    
    points = npy.array([[1,2],[3,4],[5,6],[7,8]], npy.float_)
    translated_points = translation.transform(points)
    assert (translated_points == [[11., 22.], [13., 24.], [15., 26.], [17., 28.]]).all()
    scaled_points = scale.transform(points)
    print scaled_points
    rotated_points = rotation.transform(points)
    print rotated_points

    tpoints1 = rotation.transform(translation.transform(scale.transform(points)))
    trans_sum = scale + translation + rotation
    tpoints2 = trans_sum.transform(points)
    # Need to do some sort of fuzzy comparison here?
    assert (tpoints1.round() == tpoints2.round()).all()

    print points
    
    # Here are some timing tests
    points = npy.asarray([(random(), random()) for i in xrange(10000)])
    t = timeit.Timer("trans_sum.transform(points)", "from __main__ import trans_sum, points")
    print "Time to transform 10000 x 10 points:", t.timeit(10)
    
__all__ = ['Transform', 'Affine2D']
