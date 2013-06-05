"""
This module contains a routine to create a Voronoi diagram of an unstructured
set of points. The Voronoi diagram is calculated from the Delaunay
triangulation of this unstructured mesh.

The approach here roughly follows the one provided by Nicolas Rougier:

https://github.com/rougier/gallery/blob/master/voronoi/voronoi/voronoi.py

"""

from math import atan2

from numpy import matrix, array
from numpy.linalg import solve
from numpy.linalg.linalg import LinAlgError

from triangulation import Triangulation


def circumscribed_circle_center(a, b, c):
    """
    This function calculates the center the circle that circumscribes a triangle
    described by its three corners *a*, *b* and *c*, which are supposed to be
    2-tuples.

    In degenerate cases, when there is no circumscribed circle or when it is
    not unique, an exception is raised.

    Examples:

    >>> circumscribed_circle_center((1, 1), (1, 2), (2, 1))
    (1.5, 1.5)
    >>> circumscribed_circle_center((0, 0), (2, 0), (1, -1))
    (1.0, 0.0)
    >>> circumscribed_circle_center((0, 0), (2, 0), (1, 0))
    Traceback (most recent call last):
    ...
    ValueError: there is no unique circumscribed circle for the given triangle

    """

    # Solve system of two linear equations resulting from trying to find the
    # intersection of the perpendicular bisectors of two of the triangle sides,
    # which is identical to the center of the circumscribed circle.
    # TODO: explain geometry and why this works

    # Define matrix and vector of the system of equations
    M = matrix([[b[1] - c[1], b[1] - a[1]], [c[0] - b[0], a[0] - b[0]]])
    q = array([0.5 * (a[0] - c[0]), 0.5 * (a[1] - c[1])])

    # Determine line parameters
    try:
        t, s = solve(M, q)
    # In case this can't be inverted, raise an exception
    except LinAlgError:
        raise ValueError, 'there is no unique circumscribed circle for the given triangle'
    
    # Calculate the intersection point corresponding to the center
    center = array([b[1] - c[1], c[0] - b[0]]) * t + (array(b) + array(c)) * 0.5
    return tuple(center)


def compute_voronoi_cells(x, y):
    '''
    Given a set of points specified by their coordinates given by 1-D arrays
    *x* and *y*, this function computes the corresponding Voronoi cells using
    a Delaunay triangulation.

    '''

    # Check array lengths
    assert x.shape == y.shape, ValueError('shape mismatch')
    assert len(x.shape) == 1, ValueError('only 1D-arrays for x, y')

    # Get a Delaunay triangulation
    triangulation = Triangulation(x, y)
    p = zip(x, y)

    # Compute triangle centers that are the Voronoi cell corners
    cells = [[] for i in xrange(x.shape[0])]
    for i, j, k in triangulation.triangles:
        center = circumscribed_circle_center(p[i], p[j], p[k])
        for index in i, j, k:
            cells[index].append(center)

    # Sort the polygon corners clockwise
    for i, cell in enumerate(cells):
        cell.sort(key=lambda cell: atan2(cell[1] - y[i], cell[0] - x[i]))

    return cells

