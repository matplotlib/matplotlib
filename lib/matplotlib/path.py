import math

import numpy as npy
from numpy import ma as ma

KAPPA = 4.0 * (npy.sqrt(2) - 1) / 3.0

class Path(object):
    # Path codes
    STOP      = 0 # 1 vertex
    MOVETO    = 1 # 1 vertex
    LINETO    = 2 # 1 vertex
    CURVE3    = 3 # 2 vertices
    CURVE4    = 4 # 3 vertices
    CLOSEPOLY = 5 # 1 vertex
    ###
    # MGDTODO: I'm not sure these are supported by PS/PDF/SVG,
    # so if they don't, we probably shouldn't
    CURVEN    = 6
    CATROM    = 7
    UBSPLINE  = 8
    ####

    NUM_VERTICES = [1, 1, 1, 2, 3, 1]
    
    code_type = npy.uint8
    
    def __init__(self, vertices, codes=None, closed=True):
        vertices = ma.asarray(vertices, npy.float_)

	if codes is None:
	    if closed:
		codes = self.LINETO * npy.ones(
		    vertices.shape[0] + 1, self.code_type)
		codes[0] = self.MOVETO
                codes[-1] = self.CLOSEPOLY
                vertices = npy.concatenate((vertices, [[0.0, 0.0]]))
	    else:
		codes = self.LINETO * npy.ones(
		    vertices.shape[0], self.code_type)
		codes[0] = self.MOVETO
        else:
	    codes = npy.asarray(codes, self.code_type)
            assert codes.ndim == 1
            assert len(codes) == len(vertices)

        # The path being passed in may have masked values.  However,
        # the backends (and any affine transformations in matplotlib
        # itself), are not expected to deal with masked arrays, so we
        # must remove them from the array (using compressed), and add
        # MOVETO commands to the codes array accordingly.
        mask = ma.getmask(vertices)
        if mask is not ma.nomask:
            mask1d = ma.mask_or(mask[:, 0], mask[:, 1])
            vertices = ma.compress(npy.invert(mask1d), vertices, 0)
            codes = npy.where(npy.concatenate((mask1d[-1:], mask1d[:-1])),
                              self.MOVETO, codes)
            codes = ma.masked_array(codes, mask=mask1d).compressed()
            codes = npy.asarray(codes, self.code_type)

        vertices = npy.asarray(vertices, npy.float_)

        assert vertices.ndim == 2
        assert vertices.shape[1] == 2
	assert codes.ndim == 1
        
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

    def iter_endpoints(self):
	i = 0
	NUM_VERTICES = self.NUM_VERTICES
	vertices = self.vertices
	for code in self.codes:
            if code == self.CLOSEPOLY:
                i += 1
            else:
                num_vertices = NUM_VERTICES[code]
                i += num_vertices - 1
                yield vertices[i]
                i += 1

    def transformed(self, transform):
        return Path(transform.transform(self.vertices), self.codes)
                
    _unit_rectangle = None
    #@classmethod
    def unit_rectangle(cls):
	if cls._unit_rectangle is None:
	    cls._unit_rectangle = \
		Path([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
	return cls._unit_rectangle
    unit_rectangle = classmethod(unit_rectangle)

    _unit_regular_polygons = {}
    #@classmethod
    def unit_regular_polygon(cls, numVertices):
	path = cls._unit_regular_polygons.get(numVertices)
	if path is None:
	    theta = 2*npy.pi/numVertices * npy.arange(numVertices).reshape((numVertices, 1))
	    # This initial rotation is to make sure the polygon always
            # "points-up"
	    theta += npy.pi / 2.0
	    verts = npy.concatenate((npy.cos(theta), npy.sin(theta)), 1)
	    path = Path(verts)
	    cls._unit_regular_polygons[numVertices] = path
	return path
    unit_regular_polygon = classmethod(unit_regular_polygon)

    _unit_circle = None
    #@classmethod
    def unit_circle(cls):
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

            codes = cls.CURVE4 * npy.ones((len(vertices)))
	    codes[0] = cls.MOVETO
            codes[-1] = cls.CLOSEPOLY

	    cls._unit_circle = Path(vertices, codes)
	return cls._unit_circle
    unit_circle = classmethod(unit_circle)

    #@classmethod
    def arc(cls, theta1, theta2, is_wedge=False):
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
        return cls.arc(theta1, theta2, True)
    wedge = classmethod(wedge)
