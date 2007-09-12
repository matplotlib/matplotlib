"""
A set of classes to handle transformations.

2007 Michael Droettboom
"""

import numpy as N
from numpy.linalg import inv
from sets import Set

# MGDTODO: This creates a ton of cyclical references.  We may want to
# consider using weak references

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
	self._points = N.asarray(points, N.float_)
	self.track = False

    def __del__(self):
	if self.track:
	    print "Bbox::__del__"
	
    #@staticmethod
    def unit():
	return Bbox([[0,0], [1,1]])
    unit = staticmethod(unit)

    #@staticmethod
    def from_lbwh(left, bottom, width, height):
	return Bbox([[left, bottom], [left + width, bottom + height]])
    from_lbwh = staticmethod(from_lbwh)

    #@staticmethod
    def from_lbrt(left, bottom, right, top):
	return Bbox([[left, bottom], [right, top]])
    from_lbrt = staticmethod(from_lbrt)

    def update_from_data(self, x, y):
	self._points = N.array([[x.min(), y.min()], [x.max(), y.max()]], N.float_)
	self.invalidate()
	if self.track:
	    print "Bbox::update_from_data", self._points
    
    def copy(self):
	if self.track:
	    print "Bbox::copy"
	return Bbox(self._points.copy())

    def __repr__(self):
	return 'Bbox(%s)' % repr(self._points)
    __str__ = __repr__

    def __cmp__(self, other):
	# MGDTODO: Totally suboptimal
	if isinstance(other, Bbox):
	    return (self._points == other._points).all()
	return -1
    
    # MGDTODO: Probably a more efficient ways to do this...
    def _get_xmin(self):
	if self.track:
	    print "Bbox::_get_xmin"
	return self._points[0, 0]
    def _set_xmin(self, val):
	print "Bbox::_set_xmin"
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

    def transformed(self, transform):
	return Bbox(self.transform(self._points))

    def inverse_transformed(self, transform):
	return Bbox(self.transform.inverted()(self._points))
    
    def get_bounds(self):
	return (self.xmin, self.ymin,
		self.xmax - self.xmin, self.ymax - self.ymin)

    def expanded(self, sw, sh):
	width = self.width()
	height = self.height()
	deltaw = (sw * width - width) / 2.0
	deltah = (sh * height - height) / 2.0
	a = N.array([[-deltaw, -deltah], [deltaw, deltah]])
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
    
    def has_inverse(self):
	raise NotImplementedError()
    
    def inverted(self):
	raise NotImplementedError()

    def is_separable(self):
	return False

    def is_affine(self):
	return False
	
class Affine2D(Transform):
    input_dims = 2
    output_dims = 2
    
    def __init__(self, matrix = None):
        """
        Initialize an Affine transform from a 3x3 numpy float array.

        a c e
        b d f
        0 0 1
        """
	Transform.__init__(self)
	if matrix is None:
	    matrix = N.identity(3)
	else:
	    assert matrix.shape == (3, 3)
	self._mtx = matrix
	self._inverted = None

    def __repr__(self):
	return "Affine2D(%s)" % repr(self._mtx)
    __str__ = __repr__

    def _do_invalidation(self):
	result = self._inverted is None
	self._inverted = None
	return result
    
    #@staticmethod
    def from_values(a, b, c, d, e, f):
        return Affine2D(Affine2D.matrix_from_values(a, b, c, d, e, f))
    from_values = staticmethod(from_values)

    def to_values(self):
	mtx = self.get_matrix()
	return tuple(mtx[:2].swapaxes(0, 1).flatten())
    
    #@staticmethod
    def matrix_from_values(a, b, c, d, e, f):
	affine = N.zeros((3,3), N.float_)
	affine[0,] = a, c, e
	affine[1,] = b, d, f
	affine[2,2] = 1
	return affine
    matrix_from_values = staticmethod(matrix_from_values)

    def get_matrix(self):
	return self._mtx
    
    def __call__(self, points):
        """
        Applies the transformation to a set of 2D points and
	returns the result.

	points must be a numpy array of shape (N, 2), where N is the
	number of points.
	"""
	# MGDTODO: This involves a copy.  We may need to do something like
	# http://neuroimaging.scipy.org/svn/ni/ni/trunk/neuroimaging/core/reference/mapping.py
	# to separate the matrix out into the translation and scale components
	# and apply each separately (which is still sub-optimal)

	# This is easier for now, however, since we can just keep a
	# regular affine matrix around
	# MGDTODO: Trap cases where this isn't an array and fix there
	mtx = self.get_matrix()
	points = N.asarray(points, N.float_)
	new_points = points.swapaxes(0, 1)
	new_points = N.vstack((new_points, N.ones((1, points.shape[0]))))
	result = N.dot(mtx, new_points)[:2]
	return result.swapaxes(0, 1)
    
    #@staticmethod
    def _concat(a, b):
        return N.dot(b, a)
    _concat = staticmethod(_concat)

    #@staticmethod
    def concat(a, b):
	return Affine2D(Affine2D._concat(a._mtx, b._mtx))
    concat = staticmethod(concat)
    
    #@staticmethod
    def identity():
        return Affine2D(N.identity(3))
    identity = staticmethod(identity)

    def rotate(self, theta):
        a = N.cos(theta)
        b = N.sin(theta)
        rotate_mtx = self.matrix_from_values(a, b, -b, a, 0, 0)
        self._mtx = self._concat(self._mtx, rotate_mtx)
	self.invalidate()
	return self

    def rotate_deg(self, degrees):
        return self.rotate(degrees*N.pi/180.)

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
    
class BlendedAffine2D(Affine2D):
    def __init__(self, x_transform, y_transform):
	assert x_transform.is_affine()
	assert y_transform.is_affine()
	assert x_transform.is_separable()
	assert y_transform.is_separable()

	Transform.__init__(self)
	self.add_children([x_transform, y_transform])
	self._x = x_transform
	self._y = y_transform
	self._mtx = None
	self._inverted = None

    def __repr__(self):
	return "BlendedAffine2D(%s,%s)" % (self._x, self._y)
    __str__ = __repr__
	
    def _do_invalidation(self):
	if self._mtx is not None:
	    self._mtx = None
	    Affine2D._do_invalidation(self)
	    return False
	return True

    def _make__mtx(self):
	if self._mtx is None:
	    x_mtx = self._x.get_matrix()
	    y_mtx = self._y.get_matrix()
	    self._mtx = N.vstack([x_mtx[0], y_mtx[1], [0.0, 0.0, 1.0]])
# 	    self._mtx = self.matrix_from_values(
# 		x_mtx[0,0], 0.0, 0.0, y_mtx[1,1], x_mtx[0,2], y_mtx[1,2])
	    print "Blended", x_mtx, y_mtx, self._mtx
	
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
	# MGDTODO: Implement me
	pass

class CompositeAffine2D(Affine2D):
    def __init__(self, a, b):
	assert a.is_affine()
	assert b.is_affine()

	Transform.__init__(self)
	self.add_children([a, b])
	self._a = a
	self._b = b
	self._mtx = None
	self._inverted = None

    def __repr__(self):
	return "CompositeAffine2D(%s, %s)" % (self._a, self._b)
    __str__ = __repr__

    def _do_invalidation(self):
	self._mtx = None
	Affine2D._do_invalidation(self)
    
    def _make__mtx(self):
	if self._mtx is None:
	    self._mtx = self._concat(
		self._b.get_matrix(),
		self._a.get_matrix())

    def get_matrix(self):
	self._make__mtx()
	return self._mtx
	
class CompositeTransform(Transform):
    def __init__(self, a, b):
	assert a.output_dims == b.input_dims

	Transform.__init__(self)
	self.add_children([a, b])
	self._a = a
	self._b = b

    def __call__(self, points):
	# MGDTODO: Optimize here by concatenating affines if possible
	return self._b(self._a(points))

class BboxTransform(Affine2D):
    def __init__(self, boxin, boxout):
	assert isinstance(boxin, Bbox)
	assert isinstance(boxout, Bbox)

	Transform.__init__(self)
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
	    Affine2D._do_invalidation(self)
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
	
    def __call__(self, points):
	self._make__mtx()
	return Affine2D.__call__(self, points)

    def inverted(self):
	self._make__mtx()
	return Affine2D.inverted(self)

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
    bbox1 = Bbox([[10., 15.], [20., 25.]])
    bbox2 = Bbox([[30., 35.], [40., 45.]])
    trans = BboxTransform(bbox1, bbox2)
    print trans(bbox1._points)

    bbox2.intervalx = 50, 55
    print trans(bbox1._points)
    
__all__ = ['Transform', 'Affine2D']
