from __future__ import print_function
import numpy as np

from matplotlib._delaunay import compute_planes, linear_interpolate_grid
from matplotlib._delaunay import nn_interpolate_grid
from matplotlib._delaunay import nn_interpolate_unstructured

__all__ = ['LinearInterpolator', 'NNInterpolator']


def slice2gridspec(key):
    """Convert a 2-tuple of slices to start,stop,steps for x and y.

    key -- (slice(ystart,ystop,ystep), slice(xtart, xstop, xstep))

    For now, the only accepted step values are imaginary integers (interpreted
    in the same way numpy.mgrid, etc. do).
    """
    if ((len(key) != 2) or
        (not isinstance(key[0], slice)) or
        (not isinstance(key[1], slice))):
        raise ValueError("only 2-D slices, please")

    x0 = key[1].start
    x1 = key[1].stop
    xstep = key[1].step
    if not isinstance(xstep, complex) or int(xstep.real) != xstep.real:
        raise ValueError("only the [start:stop:numsteps*1j] form supported")
    xstep = int(xstep.imag)
    y0 = key[0].start
    y1 = key[0].stop
    ystep = key[0].step
    if not isinstance(ystep, complex) or int(ystep.real) != ystep.real:
        raise ValueError("only the [start:stop:numsteps*1j] form supported")
    ystep = int(ystep.imag)

    return x0, x1, xstep, y0, y1, ystep


class LinearInterpolator(object):
    """Interpolate a function defined on the nodes of a triangulation by
    using the planes defined by the three function values at each corner of
    the triangles.

    LinearInterpolator(triangulation, z, default_value=numpy.nan)

    triangulation -- Triangulation instance
    z -- the function values at each node of the triangulation
    default_value -- a float giving the default value should the interpolating
      point happen to fall outside of the convex hull of the triangulation

    At the moment, the only regular rectangular grids are supported for
    interpolation.

        vals = interp[ystart:ystop:ysteps*1j, xstart:xstop:xsteps*1j]

    vals would then be a (ysteps, xsteps) array containing the interpolated
    values. These arguments are interpreted the same way as numpy.mgrid.

    Attributes:
      planes -- (ntriangles, 3) array of floats specifying the plane for each
        triangle.

    Linear Interpolation
    --------------------
    Given the Delauany triangulation (or indeed *any* complete triangulation)
    we can interpolate values inside the convex hull by locating the enclosing
    triangle of the interpolation point and returning the value at that point
    of the plane defined by the three node values.

        f = planes[tri,0]*x + planes[tri,1]*y + planes[tri,2]

    The interpolated function is C0 continuous across the convex hull of the
    input points. It is C1 continuous across the convex hull except for the
    nodes and the edges of the triangulation.
    """
    def __init__(self, triangulation, z, default_value=np.nan):
        self.triangulation = triangulation
        self.z = np.asarray(z, dtype=np.float64)
        self.default_value = default_value

        self.planes = compute_planes(triangulation.x, triangulation.y, self.z,
            triangulation.triangle_nodes)

    def __getitem__(self, key):
        x0, x1, xstep, y0, y1, ystep = slice2gridspec(key)
        grid = linear_interpolate_grid(
            x0, x1, xstep, y0, y1, ystep, self.default_value,
            self.planes, self.triangulation.x, self.triangulation.y,
            self.triangulation.triangle_nodes,
            self.triangulation.triangle_neighbors)
        return grid


class NNInterpolator(object):
    """Interpolate a function defined on the nodes of a triangulation by
    the natural neighbors method.

    NNInterpolator(triangulation, z, default_value=numpy.nan)

    triangulation -- Triangulation instance
    z -- the function values at each node of the triangulation
    default_value -- a float giving the default value should the interpolating
      point happen to fall outside of the convex hull of the triangulation

    At the moment, the only regular rectangular grids are supported for
    interpolation.

        vals = interp[ystart:ystop:ysteps*1j, xstart:xstop:xsteps*1j]

    vals would then be a (ysteps, xsteps) array containing the interpolated
    values. These arguments are interpreted the same way as numpy.mgrid.

    Natural Neighbors Interpolation
    -------------------------------
    One feature of the Delaunay triangulation is that for each triangle, its
    circumcircle contains no other point (although in degenerate cases, like
    squares, other points may be *on* the circumcircle). One can also
    construct what is called the Voronoi diagram from a Delaunay triangulation
    by connecting the circumcenters of the triangles to those of their
    neighbors to form a tesselation of irregular polygons covering the plane
    and containing only one node from the triangulation. Each point in one
    node's Voronoi polygon is closer to that node than any other node.

    To compute the Natural Neighbors interpolant, we consider adding the
    interpolation point to the triangulation. We define the natural neighbors
    of this point as the set of nodes participating in Delaunay triangles
    whose circumcircles contain the point. To restore the Delaunay-ness of the
    triangulation, one would only have to alter those triangles and Voronoi
    polygons. The new Voronoi diagram would have a polygon around the
    inserted point. This polygon would "steal" area from the original Voronoi
    polygons. For each node i in the natural neighbors set, we compute the
    area stolen from its original Voronoi polygon, stolen[i]. We define the
    natural neighbors coordinates

        phi[i] = stolen[i] / sum(stolen,axis=0)

    We then use these phi[i] to weight the corresponding function values from
    the input data z to compute the interpolated value.

    The interpolated surface is C1-continuous except at the nodes themselves
    across the convex hull of the input points. One can find the set of points
    that a given node will affect by computing the union of the areas covered
    by the circumcircles of each Delaunay triangle that node participates in.
    """

    def __init__(self, triangulation, z, default_value=np.nan):
        self.triangulation = triangulation
        self.z = np.asarray(z, dtype=np.float64)
        self.default_value = default_value

    def __getitem__(self, key):
        x0, x1, xstep, y0, y1, ystep = slice2gridspec(key)
        grid = nn_interpolate_grid(
            x0, x1, xstep, y0, y1, ystep, self.default_value,
            self.triangulation.x, self.triangulation.y, self.z,
            self.triangulation.circumcenters,
            self.triangulation.triangle_nodes,
            self.triangulation.triangle_neighbors)
        return grid

    def __call__(self, intx, inty):
        intz = nn_interpolate_unstructured(intx, inty, self.default_value,
            self.triangulation.x, self.triangulation.y, self.z,
            self.triangulation.circumcenters,
            self.triangulation.triangle_nodes,
            self.triangulation.triangle_neighbors)
        return intz
