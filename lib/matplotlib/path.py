"""
Contains a class for managing paths (polylines).

October 2007 Michael Droettboom
"""

import math
from weakref import WeakValueDictionary

import numpy as npy
from matplotlib.numerix import npyma as ma

from matplotlib._path import point_in_path, get_path_extents, \
    point_in_path_collection
import matplotlib._path as _path
from matplotlib.cbook import simple_linear_interpolation

KAPPA = 4.0 * (npy.sqrt(2) - 1) / 3.0

class Path(object):
    """
    Path represents a series of possibly disconnected, possibly
    closed, line and curve segments.

    The underlying storage is made up of two parallel numpy arrays:
      vertices: an Nx2 float array of vertices
      codes: an N-length uint8 array of vertex types

    These two arrays always have the same length in the first
    dimension.  Therefore, to represent a cubic curve, you must
    provide three vertices as well as three codes "CURVE3".

    The code types are:

       STOP   :  1 vertex (ignored)
          A marker for the end of the entire path (currently not
          required and ignored)

       MOVETO :  1 vertex
          Pick up the pen and move to the given vertex.
          
       LINETO :  1 vertex
          Draw a line from the current position to the given vertex.
          
       CURVE3 :  1 control point, 1 endpoint
          Draw a quadratic Bezier curve from the current position,
          with the given control point, to the given end point.

       CURVE4 :  2 control points, 1 endpoint
          Draw a cubic Bezier curve from the current position, with
          the given control points, to the given end point.

       CLOSEPOLY : 1 vertex (ignored)
          Draw a line segment to the start point of the current
          polyline.
    """
    
    # Path codes
    STOP      = 0 # 1 vertex
    MOVETO    = 1 # 1 vertex
    LINETO    = 2 # 1 vertex
    CURVE3    = 3 # 2 vertices
    CURVE4    = 4 # 3 vertices
    CLOSEPOLY = 5 # 1 vertex

    NUM_VERTICES = [1, 1, 1, 2, 3, 1]
    
    code_type = npy.uint8

    _open_codes_cache = WeakValueDictionary()
    _closed_codes_cache = WeakValueDictionary()
    
    def __init__(self, vertices, codes=None, closed=False):
        """
        Create a new path with the given vertices and codes.

        vertices is an Nx2 numpy float array, masked array or Python
        sequence.

        codes is an N-length numpy array or Python sequence of type
        Path.code_type.

        See the docstring of Path for a description of the various
        codes.

        These two arrays must have the same length in the first
        dimension.

        If codes is None, vertices will be treated as a series of line
        segments.  Additionally, if closed is also True, the polyline
        will closed.  If vertices contains masked values, the
        resulting path will be compressed, with MOVETO codes inserted
        in the correct places to jump over the masked regions.
        """
        if ma.isMaskedArray(vertices):
            mask = ma.getmask(vertices)
        else:
            vertices = npy.asarray(vertices, npy.float_)
            mask = ma.nomask
            
	if codes is None:
            if closed:
                # MGDTODO: Remove me once efficiency concerns are
                # taken care of.
                warnings.warn("""
EFFICIENCY CONCERN: Having the Path constructor create a closed
polyline automatically is not always the most efficient way to do
things, since it causes a memory copy of the vertices array.  If the
caller can easily close the polygon itself it should do so.
""")
                codes = self._closed_codes_cache.get(len(vertices))
                if codes is None:
                    codes = self.LINETO * npy.ones(
                        vertices.shape[0] + 1, self.code_type)
                    codes[0] = self.MOVETO
                    codes[-1] = self.CLOSEPOLY
                    self._closed_codes_cache[len(vertices)] = codes
                vertices = npy.concatenate((vertices, [vertices[0]]))
            else:
                codes = self._open_codes_cache.get(len(vertices))
                if codes is None:
                    codes = self.LINETO * npy.ones(
                        vertices.shape[0], self.code_type)
                    codes[0] = self.MOVETO
                    self._open_codes_cache[len(vertices)] = codes
        else:
	    codes = npy.asarray(codes, self.code_type)
            assert codes.ndim == 1
            assert len(codes) == len(vertices)

        # The path being passed in may have masked values.  However,
        # the backends (and any affine transformations in matplotlib
        # itself), are not expected to deal with masked arrays, so we
        # must remove them from the array (using compressed), and add
        # MOVETO commands to the codes array accordingly.
        if mask is not ma.nomask:
            mask1d = ma.mask_or(mask[:, 0], mask[:, 1])
            vertices = ma.compress(npy.invert(mask1d), vertices, 0)
            codes = npy.where(npy.concatenate((mask1d[-1:], mask1d[:-1])),
                              self.MOVETO, codes)
            codes = ma.masked_array(codes, mask=mask1d).compressed()
            codes = npy.asarray(codes, self.code_type)

        assert vertices.ndim == 2
        assert vertices.shape[1] == 2
        
        self._codes = codes
	self._vertices = vertices

    def __repr__(self):
	return "Path(%s, %s)" % (self.vertices, self.codes)

    def __len__(self):
        return len(self._vertices)
    
    def _get_codes(self):
	return self._codes
    codes = property(_get_codes)

    def _get_vertices(self):
	return self._vertices
    vertices = property(_get_vertices)

    def iter_segments(self):
        """
        Iterates over all of the endpoints in the path.  Unlike
        iterating directly over the vertices array, curve control
        points are skipped over.
        """
	i = 0
	NUM_VERTICES = self.NUM_VERTICES
	vertices = self.vertices
        codes = self.codes
        
	while i < len(vertices):
            code = codes[i]
            if code == self.CLOSEPOLY:
                yield [], code
                i += 1
            elif code == self.STOP:
                return
            else:
                num_vertices = NUM_VERTICES[code]
                yield vertices[i:i+num_vertices].flatten(), code
                i += num_vertices
                
    def transformed(self, transform):
        """
        Return a transformed copy of the path.

        See transforms.TransformedPath for a path that will cache the
        transformed result and automatically update when the transform
        changes.
        """
        return Path(transform.transform(self.vertices), self.codes)

    def contains_point(self, point, transform=None):
        """
        Returns True if the path contains the given point.

        If transform is not None, the path will be transformed before
        performing the test.
        """
        if transform is None:
            from transforms import IdentityTransform
            transform = IdentityTransform
        return point_in_path(point[0], point[1], self, transform.frozen())

    def get_extents(self, transform=None):
        """
        Returns the extents (xmin, ymin, xmax, ymax) of the path.

        Unlike computing the extents on the vertices alone, this
        algorithm will take into account the curves and deal with
        control points appropriately.
        """
        from transforms import Affine2D, Bbox
        if transform is None:
            transform = Affine2D()
        return Bbox.from_extents(*get_path_extents(self, transform))

    def interpolated(self, steps):
        """
        Returns a new path resampled to length N x steps.
        Does not currently handle interpolating curves.
        """
        vertices = simple_linear_interpolation(self.vertices, steps)
        codes = self.codes
        new_codes = Path.LINETO * npy.ones(((len(codes) - 1) * steps + 1, ))
        new_codes[0::steps] = codes
        return Path(vertices, new_codes)
        
    _unit_rectangle = None
    #@classmethod
    def unit_rectangle(cls):
        """
        Returns a Path of the unit rectangle from (0, 0) to (1, 1).
        """
	if cls._unit_rectangle is None:
	    cls._unit_rectangle = \
		Path([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        return cls._unit_rectangle
    unit_rectangle = classmethod(unit_rectangle)

    _unit_regular_polygons = {}
    #@classmethod
    def unit_regular_polygon(cls, numVertices):
        """
        Returns a Path for a unit regular polygon with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
	path = cls._unit_regular_polygons.get(numVertices)
	if path is None:
	    theta = (2*npy.pi/numVertices *
                     npy.arange(numVertices).reshape((numVertices, 1)))
	    # This initial rotation is to make sure the polygon always
            # "points-up"
	    theta += npy.pi / 2.0
	    verts = npy.concatenate((npy.cos(theta), npy.sin(theta)), 1)
	    path = Path(verts)
	    cls._unit_regular_polygons[numVertices] = path
	return path
    unit_regular_polygon = classmethod(unit_regular_polygon)

    _unit_regular_stars = {}
    #@classmethod
    def unit_regular_star(cls, numVertices, innerCircle=0.5):
        """
        Returns a Path for a unit regular star with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
	path = cls._unit_regular_stars.get((numVertices, innerCircle))
	if path is None:
            ns2 = numVertices * 2
	    theta = (2*npy.pi/ns2 * npy.arange(ns2))
	    # This initial rotation is to make sure the polygon always
            # "points-up"
	    theta += npy.pi / 2.0
            r = npy.ones(ns2)
            r[1::2] = innerCircle
	    verts = npy.vstack((r*npy.cos(theta), r*npy.sin(theta))).transpose()
	    path = Path(verts)
	    cls._unit_regular_polygons[(numVertices, innerCircle)] = path
	return path
    unit_regular_star = classmethod(unit_regular_star)

    #@classmethod
    def unit_regular_asterisk(cls, numVertices):
        """
        Returns a Path for a unit regular asterisk with the given
        numVertices and radius of 1.0, centered at (0, 0).
        """
	return cls.unit_regular_star(numVertices, 0.0)
    unit_regular_asterisk = classmethod(unit_regular_asterisk)
    
    _unit_circle = None
    #@classmethod
    def unit_circle(cls):
        """
        Returns a Path of the unit circle.  The circle is approximated
        using cubic Bezier curves.
        """
	if cls._unit_circle is None:
            offset = KAPPA
	    vertices = npy.array(
		[[-1.0, 0.0],
		 
		 [-1.0, offset],
		 [-offset, 1.0],
		 [0.0, 1.0],
		 
		 [offset, 1.0],
		 [1.0, offset],
		 [1.0, 0.0],
		 
		 [1.0, -offset],
		 [offset, -1.0],
		 [0.0, -1.0],
		 
		 [-offset, -1.0],
		 [-1.0, -offset],
		 [-1.0, 0.0],

                 [-1.0, 0.0]],
                npy.float_)

            codes = cls.CURVE4 * npy.ones(14)
	    codes[0] = cls.MOVETO
            codes[-1] = cls.CLOSEPOLY

	    cls._unit_circle = Path(vertices, codes)
	return cls._unit_circle
    unit_circle = classmethod(unit_circle)

    #@classmethod
    def arc(cls, theta1, theta2, is_wedge=False):
        """
        Returns an arc on the unit circle from angle theta1 to angle
        theta2 (in degrees).
        """
        # From Masionobe, L.  2003.  "Drawing an elliptical arc using
        # polylines, quadratic or cubic Bezier curves".
        #
        # http://www.spaceroots.org/documents/ellipse/index.html

        # degrees to radians
        theta1 *= math.pi / 180.0
        theta2 *= math.pi / 180.0
        
        twopi = math.pi * 2.0
        halfpi = math.pi * 0.5
        
        eta1 = math.atan2(math.sin(theta1), math.cos(theta1))
        eta2 = math.atan2(math.sin(theta2), math.cos(theta2))
        eta2 -= twopi * math.floor((eta2 - eta1) / twopi)
        if (theta2 - theta1 > math.pi) and (eta2 - eta1 < math.pi):
            eta2 += twopi

        # number of curve segments to make
        n = int(2 ** math.ceil((eta2 - eta1) / halfpi))

        deta = (eta2 - eta1) / n
        etaB = eta1

        cos_etaB = math.cos(etaB)
        sin_etaB = math.sin(etaB)
        xB = cos_etaB
        yB = sin_etaB
        xB_dot = -sin_etaB
        yB_dot = cos_etaB

        if is_wedge:
            length = n * 3 + 4
            vertices = npy.zeros((length, 2), npy.float_)
            codes = Path.CURVE4 * npy.ones((length, ), Path.code_type)
            vertices[1] = [xB, yB]
            codes[0:2] = [Path.MOVETO, Path.LINETO]
            vertex_offset = 2
        else:
            length = n * 3 + 1
            vertices = npy.zeros((length, 2), npy.float_)
            codes = Path.CURVE4 * npy.ones((length, ), Path.code_type)
            vertices[0] = [xB, yB]
            codes[0] = Path.MOVETO
            vertex_offset = 1

        t = math.tan(0.5 * deta)
        alpha = math.sin(deta) * (math.sqrt(4.0 + 3.0 * t * t) - 1) / 3.0
        
        for i in xrange(n):
            xA = xB
            yA = yB
            xA_dot = xB_dot
            yA_dot = yB_dot
            
            etaB += deta
            cos_etaB = math.cos(etaB)
            sin_etaB = math.sin(etaB)
            xB = cos_etaB
            yB = sin_etaB
            xB_dot = -sin_etaB
            yB_dot = cos_etaB

            offset = i*3 + vertex_offset
            vertices[offset:offset+3] = [
                [xA + alpha * xA_dot, yA + alpha * yA_dot],
                [xB - alpha * xB_dot, yB - alpha * yB_dot],
                [xB, yB]]

        if is_wedge:
            codes[-2:] = [Path.LINETO, Path.CLOSEPOLY]
            
        return Path(vertices, codes)
    arc = classmethod(arc)

    def wedge(cls, theta1, theta2):
        """
        Returns a wedge of the unit circle from angle theta1 to angle
        theta2 (in degrees).
        """
        return cls.arc(theta1, theta2, True)
    wedge = classmethod(wedge)

def get_path_collection_extents(*args):
    from transforms import Bbox
    if len(args[1]) == 0:
        raise ValueError("No paths provided")
    return Bbox.from_extents(*_path.get_path_collection_extents(*args))
