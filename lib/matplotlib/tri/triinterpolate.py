from __future__ import print_function
from matplotlib.tri import Triangulation
from matplotlib.tri.trifinder import TriFinder
import numpy as np


class TriInterpolator(object):
    """
    Abstract base class for classes used to perform interpolation on
    triangular grids.

    Derived classes implement __call__(x,y) where x,y are array_like point
    coordinates of the same shape, and that returns a masked array of the same
    shape containing the interpolated z-values.
    """
    def __init__(self, triangulation, z, trifinder=None):
        if not isinstance(triangulation, Triangulation):
            raise ValueError('Expected a Triangulation object')
        self._triangulation = triangulation

        self._z = np.asarray(z)
        if self._z.shape != self._triangulation.x.shape:
            raise ValueError('z array must have same length as triangulation x'
                             ' and y arrays')

        if trifinder is not None and not isinstance(trifinder, TriFinder):
            raise ValueError('Expected a TriFinder object')
        self._trifinder = trifinder or self._triangulation.get_trifinder()


class LinearTriInterpolator(TriInterpolator):
    """
    A LinearTriInterpolator performs linear interpolation on a triangular grid.

    Each triangle is represented by a plane so that an interpolated value at
    point (x,y) lies on the plane of the triangle containing (x,y).
    Interpolated values are therefore continuous across the triangulation, but
    their first derivatives are discontinuous at edges between triangles.
    """
    def __init__(self, triangulation, z, trifinder=None):
        """
        *triangulation*: the :class:`~matplotlib.tri.Triangulation` to
        interpolate over.

        *z*: array_like of shape (npoints).
          Array of values, defined at grid points, to interpolate between.

        *trifinder*: optional :class:`~matplotlib.tri.TriFinder` object.
          If this is not specified, the Triangulation's default TriFinder will
          be used by calling :func:`matplotlib.tri.Triangulation.get_trifinder`.
        """
        TriInterpolator.__init__(self, triangulation, z, trifinder)

        # Store plane coefficients for fast interpolation calculations.
        self._plane_coefficients = \
            self._triangulation.calculate_plane_coefficients(self._z)

        # Store vectorized interpolation function, so can pass in arbitrarily
        # shape arrays of x, y and tri and the _single_interp function is
        # called in turn with scalar x, y and tri.
        self._multi_interp = np.vectorize(self._single_interp,
                                          otypes=[np.float])

    def __call__(self, x, y):
        """
        Return a masked array containing linearly interpolated values at the
        specified x,y points.

        *x*, *y* are array_like x and y coordinates of the same shape and any
        number of dimensions.

        Returned masked array has the same shape as *x* and *y*; values
        corresponding to (x,y) points outside of the triangulation are masked
        out.
        """
        # Check arguments.
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must be equal-shaped arrays")

        # Indices of triangles containing x, y points, or -1 for no triangles.
        tris = self._trifinder(x, y)

        # Perform interpolation.
        z = self._multi_interp(x, y, tris)

        # Return masked array.
        return np.ma.masked_invalid(z, copy=False)

    def _single_interp(self, x, y, tri):
        """
        Return single interpolated value at specified (*x*, *y*) coordinates
        within triangle with index *tri*.  Returns np.nan if tri == -1.
        """
        if tri == -1:
            return np.nan
        else:
            return (self._plane_coefficients[tri,0] * x +
                    self._plane_coefficients[tri,1] * y +
                    self._plane_coefficients[tri,2])
