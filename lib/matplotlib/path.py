import numpy as npy

class Path:
    # Path codes
    STOP      = 0
    MOVETO    = 1 # 1 vertex
    LINETO    = 2 # 1 vertex
    CURVE3    = 3 # 2 vertices
    CURVE4    = 4 # 3 vertices
    ###
    # MGDTODO: I'm not sure these are supported by PS/PDF/SVG,
    # so if they don't, we probably shouldn't
    CURVEN    = 5
    CATROM    = 6
    UBSPLINE  = 7
    ####
    CLOSEPOLY = 0x0F # 0 vertices

    code_type = npy.uint8
    
    def __init__(self, vertices, codes=None, closed=True):
	self._vertices = npy.asarray(vertices, npy.float_)
	assert self._vertices.ndim == 2
	assert self._vertices.shape[1] == 2

	if codes is None:
	    if closed:
		codes = self.LINETO * npy.ones(
		    self._vertices.shape[0] + 1, self.code_type)
		codes[0] = self.MOVETO
		codes[-1] = self.CLOSEPOLY
	    else:
		codes = self.LINETO * npy.ones(
		    self._vertices.shape[0], self.code_type)
		codes[0] = self.MOVETO
        else:
	    codes = npy.asarray(codes, self.code_type)
	self._codes = codes
	    
	assert self._codes.ndim == 1
	# MGDTODO: Maybe we should add some quick-and-dirty check that
	# the number of vertices is correct for the code array

    def _get_codes(self):
	return self._codes
    codes = property(_get_codes)

    def _get_vertices(self):
	return self._vertices
    vertices = property(_get_vertices)
