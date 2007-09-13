"""
A set of classes to handle transformations.

2007 Michael Droettboom
"""

import numpy as npy
from numpy.linalg import inv
from sets import Set

# MGDTODO: The name of this module is bad, since it deals with
# non-affine transformations as well.  It should probably just be
# "transforms", but we already had one of those... ;)

# MGDTODO: This creates a ton of cyclical references.  We may want to
# consider using weak references

# MGDTODO: deep copying is probably incorrect wrt the parent/child
# relationships

class TransformNode(object):
    def __init__(self):
	self._parents = Set()
	
    def invalidate(self):
	if not self._do_invalidation():
	    for parent in self._parents:
		parent.invalidate()

    def _do_invalidation(self):
	return False
	    
    def add_children(self, children):
	for child in children:
	    child._parents.add(self)

class Bbox(TransformNode):
    def __init__(self, points):
	TransformNode.__init__(self)
	self._points = npy.asarray(points, npy.float_)
	self.track = False

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
    
    def __copy__(self):
	return Bbox(self._points.copy())

    def __deepcopy__(self, memo):
	return Bbox(self._points.copy())
    
    def __cmp__(self, other):
	# MGDTODO: Totally suboptimal
	if isinstance(other, Bbox) and (self._points == other._points).all():
	    return 0
	return -1

    def __repr__(self):
	return 'Bbox(%s)' % repr(self._points)
    __str__ = __repr__

    def __array__(self):
	return self._points
    
    # JDH: the update method will update the box limits from the
    # existing limits and the new data; it appears here you are just
    # using the new data.  We use an "ignore" flag to specify whether
    # you want to include the existing data or not in the update
    def update_from_data(self, x, y, ignore=True):
	self._points = npy.array([[x.min(), y.min()], [x.max(), y.max()]], npy.float_)
	self.invalidate()

    # MGDTODO: Probably a more efficient ways to do this...
    def _get_xmin(self):
	return self._points[0, 0]
    def _set_xmin(self, val):
	self._points[0, 0] = val
	self.invalidate()
    xmin = property(_get_xmin, _set_xmin)

    def _get_ymin(self):
	return self._points[0, 1]
    def _set_ymin(self, val):
	self._points[0, 1] = val
	self.invalidate()
    ymin = property(_get_ymin, _set_ymin)

    def _get_xmax(self):
	return self._points[1, 0]
    def _set_xmax(self, val):
	self._points[1, 0] = val
	self.invalidate()
    xmax = property(_get_xmax, _set_xmax)

    def _get_ymax(self):
	return self._points[1, 1]
    def _set_ymax(self, val):
	self._points[1, 1] = val
	self.invalidate()
    ymax = property(_get_ymax, _set_ymax)

    def _get_min(self):
	return self._points[0]
    def _set_min(self, val):
	self._points[0] = val
	self.invalidate()
    min = property(_get_min, _set_min)
    
    def _get_max(self):
	return self._points[1]
    def _set_max(self, val):
	self._points[1] = val
	self.invalidate()
    max = property(_get_max, _set_max)
    
    def _get_intervalx(self):
	return self._points[:,0]
    def _set_intervalx(self, interval):
	self._points[:,0] = interval
	self.invalidate()
    intervalx = property(_get_intervalx, _set_intervalx)

    def _get_intervaly(self):
	return self._points[:,1]
    def _set_intervaly(self, interval):
	self._points[:,1] = interval
	self.invalidate()
    intervaly = property(_get_intervaly, _set_intervaly)

    def _get_width(self):
	return self.xmax - self.xmin
    width = property(_get_width)

    def _get_height(self):
	return self.ymax - self.ymin
    height = property(_get_height)

    def _get_bounds(self):
	return (self.xmin, self.ymin,
		self.xmax - self.xmin, self.ymax - self.ymin)
    def _set_bounds(self, bounds):
	l,b,w,h = bounds
	self._points = npy.array([[l, b], [l+w, b+h]], npy.float_)
	self.invalidate()
    bounds = property(_get_bounds, _set_bounds)
	
    def transformed(self, transform):
	return Bbox(transform(self._points))

    def inverse_transformed(self, transform):
	return Bbox(transform.inverted()(self._points))
    
    def expanded(self, sw, sh):
	width = self.width
	height = self.height
	deltaw = (sw * width - width) / 2.0
	deltah = (sh * height - height) / 2.0
	a = npy.array([[-deltaw, -deltah], [deltaw, deltah]])
	return Bbox(self._points + a)

    def contains(self, x, y):
	return (x >= self.xmin and x <= self.xmax and
		y >= self.ymin and y <= self.ymax)

    #@staticmethod
    def union(bboxes):
	"""
	Return the Bbox that bounds all bboxes
	"""
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
    
class Transform(TransformNode):
    def __init__(self):
	TransformNode.__init__(self)
    
    def __call__(self, points):
	raise NotImplementedError()

    def __add__(self, other):
	if isinstance(other, Transform):
	    return composite_transform_factory(self, other)
	raise TypeError("Can not add Transform to object of type '%s'" % type(other))

    def __radd__(self, other):
	if isinstance(other, Transform):
	    return composite_transform_factory(other, self)
	raise TypeError("Can not add Transform to object of type '%s'" % type(other))

    def transform_point(self, point):
	return self.__call__(npy.asarray([point]))[0]
    
    def has_inverse(self):
	raise NotImplementedError()
    
    def inverted(self):
	raise NotImplementedError()

    def is_separable(self):
	return False

    def is_affine(self):
	return False

class Affine2DBase(Transform):
    input_dims = 2
    output_dims = 2

    def __init__(self):
	Transform.__init__(self)
	self._inverted = None

    def _do_invalidation(self):
	result = self._inverted is None
	self._inverted = None
	return result

    #@staticmethod
    def _concat(a, b):
        return npy.dot(b, a)
    _concat = staticmethod(_concat)
    
    def to_values(self):
	mtx = self.get_matrix()
	return tuple(mtx[:2].swapaxes(0, 1).flatten())
    
    #@staticmethod
    def matrix_from_values(a, b, c, d, e, f):
	affine = npy.zeros((3,3), npy.float_)
	affine[0,] = a, c, e
	affine[1,] = b, d, f
	affine[2,2] = 1
	return affine
    matrix_from_values = staticmethod(matrix_from_values)

    def get_matrix(self):
	raise NotImplementedError()
    
    def __call__(self, points):
        """
        Applies the transformation to an array of 2D points and
	returns the result.

	points must be a numpy array of shape (N, 2), where N is the
	number of points.
	"""
	# MGDTODO: The major speed trap here is just converting to
	# the points to an array in the first place.  If we can use
	# more arrays upstream, that should help here.
	mtx = self.get_matrix()
	points = npy.asarray(points, npy.float_)
	points = points.transpose()
	points = npy.dot(mtx[0:2, 0:2], points)
	points = points + mtx[0:2, 2:]
	return points.transpose()
    
    def inverted(self):
	if self._inverted is None:
	    mtx = self.get_matrix()
	    self._inverted = Affine2D(inv(mtx))
	return self._inverted
    
    def is_separable(self):
	mtx = self.get_matrix()
	return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0

    def is_affine(self):
	return True

	
class Affine2D(Affine2DBase):
    input_dims = 2
    output_dims = 2
    
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
	    assert matrix.shape == (3, 3)
	self._mtx = matrix
	self._inverted = None

    def __repr__(self):
	return "Affine2D(%s)" % repr(self._mtx)
    __str__ = __repr__

    def __cmp__(self, other):
	if (isinstance(other, Affine2D) and
	    (self.get_matrix() == other.get_matrix()).all()):
	    return 0
	return -1
    
    def __copy__(self):
	return Affine2D(self._mtx.copy())
    
    def __deepcopy__(self, memo):
	return Affine2D(self._mtx.copy())
    
    #@staticmethod
    def from_values(a, b, c, d, e, f):
        return Affine2D(Affine2D.matrix_from_values(a, b, c, d, e, f))
    from_values = staticmethod(from_values)

    def get_matrix(self):
	return self._mtx
    
    #@staticmethod
    def concat(a, b):
	return Affine2D(Affine2D._concat(a._mtx, b._mtx))
    concat = staticmethod(concat)
    
    #@staticmethod
    def identity():
        return Affine2D(npy.identity(3))
    identity = staticmethod(identity)

    def rotate(self, theta):
        a = npy.cos(theta)
        b = npy.sin(theta)
        rotate_mtx = self.matrix_from_values(a, b, -b, a, 0, 0)
        self._mtx = self._concat(self._mtx, rotate_mtx)
	self.invalidate()
	return self

    def rotate_deg(self, degrees):
        return self.rotate(degrees*npy.pi/180.)

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

    def inverted(self):
	if self._inverted is None:
	    mtx = self.get_matrix()
	    self._inverted = Affine2D(inv(mtx))
	return self._inverted
    
    def is_separable(self):
	mtx = self.get_matrix()
	return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0

    def is_affine(self):
	return True
    
class BlendedAffine2D(Affine2DBase):
    def __init__(self, x_transform, y_transform):
	assert x_transform.is_affine()
	assert y_transform.is_affine()
	assert x_transform.is_separable()
	assert y_transform.is_separable()

	Affine2DBase.__init__(self)
	self.add_children([x_transform, y_transform])
	self._x = x_transform
	self._y = y_transform
	self._mtx = None

    def __repr__(self):
	return "BlendedAffine2D(%s,%s)" % (self._x, self._y)
    __str__ = __repr__
	
    def _do_invalidation(self):
	if self._mtx is not None:
	    self._mtx = None
	    Affine2DBase._do_invalidation(self)
	    return False
	return True

    def _make__mtx(self):
	if self._mtx is None:
	    x_mtx = self._x.get_matrix()
	    y_mtx = self._y.get_matrix()
	    # This works because we already know the transforms are
	    # separable, though normally one would want to set b and
	    # c to zero.
	    self._mtx = npy.vstack((x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]))
	
    def is_separable(self):
	return True

    def get_matrix(self):
	self._make__mtx()
	return self._mtx
    
class BlendedTransform(Transform):
    def __init__(self, x_transform, y_transform):
	assert x_transform.is_separable()
	assert y_transform.is_separable()

	Transform.__init__(self)
	self.add_children([x_transform, y_transform])
	self._x = x_transform
	self._y = y_transform

    def __call__(self, points):
	x_points = self._x(points)
	y_points = self._y(points)
	# This works because we already know the transforms are
	# separable
	return npy.hstack((x_points[:, 0:1], y_points[:, 1:2]))

class CompositeAffine2D(Affine2DBase):
    def __init__(self, a, b):
	assert a.is_affine()
	assert b.is_affine()

	Affine2DBase.__init__(self)
	self.add_children([a, b])
	self._a = a
	self._b = b
	self._mtx = None

    def __repr__(self):
	return "CompositeAffine2D(%s, %s)" % (self._a, self._b)
    __str__ = __repr__

    def _do_invalidation(self):
	self._mtx = None
	Affine2DBase._do_invalidation(self)
    
    def _make__mtx(self):
	if self._mtx is None:
	    self._mtx = self._concat(
		self._a.get_matrix(),
		self._b.get_matrix())

    def get_matrix(self):
	self._make__mtx()
	return self._mtx
	
class CompositeTransform(Transform):
    def __init__(self, a, b):
	assert a.output_dims == b.input_dims
	self.input_dims = a.input_dims
	self.output_dims = b.output_dims
	
	Transform.__init__(self)
	self.add_children([a, b])
	self._a = a
	self._b = b

    def __call__(self, points):
	return self._b(self._a(points))

class BboxTransform(Affine2DBase):
    def __init__(self, boxin, boxout):
	assert isinstance(boxin, Bbox)
	assert isinstance(boxout, Bbox)

	Affine2DBase.__init__(self)
	self.add_children([boxin, boxout])
	self._boxin = boxin
	self._boxout = boxout
	self._mtx = None
	self._inverted = None

    def __repr__(self):
	return "BboxTransform(%s, %s)" % (self._boxin, self._boxout)
    __str__ = __repr__
	
    def _do_invalidation(self):
	if self._mtx is not None:
	    self._mtx = None
	    Affine2DBase._do_invalidation(self)
	    return False
	return True

    def _make__mtx(self):
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
	
    def is_separable(self):
	return True

    def get_matrix(self):
	self._make__mtx()
	return self._mtx
    
def blend_xy_sep_transform(x_transform, y_transform):
    if x_transform.is_affine() and y_transform.is_affine():
	return BlendedAffine2D(x_transform, y_transform)
    return BlendedTransform(x_transform, y_transform)

def composite_transform_factory(a, b):
    if a.is_affine() and b.is_affine():
	return CompositeAffine2D(a, b)
    return CompositeTransform(a, b)

# MGDTODO: There's probably a better place for this
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

# MGDTODO: Optimize
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

    print npy.asarray(bbox)
    
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

    bbox_copy = copy.copy(bbox)
    assert bbox == bbox_copy
    bbox_copy.max = (14, 15)
    assert bbox.bounds == (10, 11, 12, 13)
    assert bbox_copy.bounds == (10, 11, 4, 4)
    
    bbox1 = Bbox([[10., 15.], [20., 25.]])
    bbox2 = Bbox([[30., 35.], [40., 45.]])
    trans = BboxTransform(bbox1, bbox2)
    bbox3 = bbox1.transformed(trans)
    assert bbox3 == bbox2

    translation = Affine2D().translate(10, 20)
    assert translation.to_values() == (1, 0, 0, 1, 10, 20)
    scale = Affine2D().scale(10, 20)
    assert scale.to_values() == (10, 0, 0, 20, 0, 0)
    rotation = Affine2D().rotate_deg(30)
    print rotation.to_values() == (0.86602540378443871, 0.49999999999999994,
				   -0.49999999999999994, 0.86602540378443871,
				   0.0, 0.0)
    
    points = npy.array([[1,2],[3,4],[5,6],[7,8]], npy.float_)
    translated_points = translation(points)
    assert (translated_points == [[11., 22.], [13., 24.], [15., 26.], [17., 28.]]).all()
    scaled_points = scale(points)
    print scaled_points
    rotated_points = rotation(points)
    print rotated_points

    tpoints1 = rotation(translation(scale(points)))
    trans_sum = scale + translation + rotation
    tpoints2 = trans_sum(points)
    print tpoints1, tpoints2
    print tpoints1 == tpoints2
    # Need to do some sort of fuzzy comparison here?
    # assert (tpoints1 == tpoints2).all()

    # Here are some timing tests
    points = [(random(), random()) for i in xrange(10000)]
    t = timeit.Timer("trans_sum(points)", "from __main__ import trans_sum, points")
    print "Time to transform 10000 x 10 points as tuples:", t.timeit(10)

    points2 = npy.asarray(points)
    t = timeit.Timer("trans_sum(points2)", "from __main__ import trans_sum, points2")
    print "Time to transform 10000 x 10 points as numpy array:", t.timeit(10)
    
__all__ = ['Transform', 'Affine2D']
