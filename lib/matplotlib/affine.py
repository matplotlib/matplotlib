"""
A set of classes to handle transformations.

2007 Michael Droettboom
"""

import numpy as N
from numpy.linalg import inv

class Transform(object):
    def __call__(self, points):
	raise NotImplementedError()

    def __add__(self, other):
	if isinstance(other, Transform):
	    return CompositeTransform(self, other)
	raise TypeError("Can not add Transform to object of type '%s'" % type(other))

    def __radd__(self, other):
	if isinstance(other, Transform):
	    return CompositeTransform(other, self)
	raise TypeError("Can not add Transform to object of type '%s'" % type(other))
    
    def has_inverse(self):
	raise NotImplementedError()
    
    def inverted(self):
	raise NotImplementedError()

    def is_separable(self):
	return False

class CompositeTransform(Transform):
    def __init__(self, a, b):
	assert a.output_dims == b.input_dims
	self.a = a
	self.b = b

    def __call__(self, points):
	return self.b(self.a(points))
    
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
	if matrix is None:
	    matrix = N.identity(3)
	else:
	    assert matrix.shape == (3, 3)
	self.mtx = matrix

    def __repr__(self):
	return "Affine2D(%s)" % repr(self.mtx)
    __str__ = __repr__
    
    #@staticmethod
    def from_values(a, b, c, d, e, f):
        return Affine2D(Affine2D.matrix_from_values(a, b, c, d, e, f))
    from_values = staticmethod(from_values)

    def to_values(self):
	return tuple(self.mtx[:2].swapaxes(0, 1).flatten())
    
    #@staticmethod
    def matrix_from_values(a, b, c, d, e, f):
	affine = N.zeros((3,3), N.float_)
	affine[0,] = a, c, e
	affine[1,] = b, d, f
	affine[2,2] = 1
	return affine
    matrix_from_values = staticmethod(matrix_from_values)

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

	# This is nicer for now, however, since we can just keep a
	# regular affine matrix around
	# MGDTODO: Trap cases where this isn't an array and fix there
	points = N.array(points, N.float_)
	new_points = points.swapaxes(0, 1)
	new_points = N.vstack((new_points, N.ones((1, points.shape[0]))))
	result = N.dot(self.mtx, new_points)[:2]
	return result.swapaxes(0, 1)
    
    #@staticmethod
    def _concat(a, b):
        return N.dot(b, a)
    _concat = staticmethod(_concat)

    def concat(a, b):
	return Affine2D(Affine2D._concat(a.mtx, b.mtx))
    concat = staticmethod(concat)
    
    #@staticmethod
    def identity():
        return Affine2D(N.identity(3))
    identity = staticmethod(identity)

    def __add__(self, other):
        if isinstance(other, Affine2D):
	    return Affine2D.concat(self, other)
	return Transform.__add__(self, other)

    def __radd__(self, other):
        if isinstance(other, Affine2D):
	    return Affine2D.concat(other, self)
	return Transform.__radd__(self, other)
	
    def rotated(self, theta):
        a = N.cos(theta)
        b = N.sin(theta)
        rotate_mtx = self.matrix_from_values(a, b, -b, a, 0, 0)
        return Affine2D(self._concat(self.mtx, rotate_mtx))

    def rotated_deg(self, degrees):
        return self.rotated(degrees*N.pi/180.)

    def translated(self, tx, ty):
        translate_mtx = self.matrix_from_values(1., 0., 0., 1., tx, ty)
        return Affine2D(self._concat(self.mtx, translate_mtx))

    def scaled(self, sx, sy=None):
	if sy is None:
	    sy = sx
	scale_mtx = self.matrix_from_values(sx, 0., 0., sy, 0., 0.)
        return Affine2D(self._concat(self.mtx, scale_mtx))

    def inverted(self):
	# MGDTODO: We may want to optimize by storing the inverse
	# of the transform with every transform
	return Affine2D(inv(self.mtx))
    
    def is_separable(self):
	mtx = self.mtx
	return mtx[0, 1] == 0.0 and mtx[1, 0] == 0.0

class BlendedAffine2D(Affine2D):
    def __init__(self, x_transform, y_transform):
#	assert isinstance(x_transform, Affine2D)
#	assert isinstance(y_transform, Affine2D)
	assert x_transform.is_separable()
	assert y_transform.is_separable()
	x_mtx = x_transform.mtx
	y_mtx = y_transform.mtx
	self.mtx = self.matrix_from_values(
	    x_mtx[0,0], 0.0, 0.0, y_mtx[1,1], x_mtx[0,2], y_mtx[1,2])

# This is a placeholder since eventually we may need to handle the
# more general case of two transforms that aren't affines
BlendedTransform = BlendedAffine2D

def blend_xy_sep_transform(x_transform, y_transform):
    return BlendedAffine2D(x_transform, y_transform)

def get_bbox_transform(boxin, boxout):
    x_scale = boxout.width() / boxin.width()
    y_scale = boxout.height() / boxin.height()
    
    # MGDTODO: Optimize
    return Affine2D() \
	.translated(-boxin.xmin(), -boxin.ymin()) \
	.scaled(x_scale, y_scale) \
	.translated(boxout.xmin(), boxout.ymin())
    
if __name__ == '__main__':
    print Affine2D.from_values(1., 0, 0, 1, 0, 0)
    
    print "translated", Affine2D.identity().translated(5, 4)
    print "rotated", Affine2D.identity().rotated_deg(30)
    print "scaled", Affine2D.identity().scaled(5, 4)
    
    transform = Affine2D.identity().rotated_deg(30).translated(5, 4)

    points = N.array([[1, 2], [3, 4], [5, 6]])

    print inv(transform.mtx)
    
    print transform(points)

    transform = Affine2D.identity().scaled(5., 1.).translated(10, 0)
    print transform
    print transform.inverted()

    from bbox import Bbox
    print "BBOX"
    boxin = Bbox([[10, 10], [320, 240]])
    boxout = Bbox([[25, 25], [640, 400]])
    print boxin._points, boxin.xmin(), boxin.ymin(), boxin.xmax(), boxin.ymax()
    print boxout._points, boxout.xmin(), boxout.ymin(), boxout.xmax(), boxout.ymax()
    trans = get_bbox_transform(boxin, boxout)
    print trans
    print trans(N.array([[10, 10], [320, 240]]))
    print trans([[10, 10]])
    
__all__ = ['Transform', 'Affine2D']
