"""
This module contains a routine to create a Voronoi diagram of an unstructured
set of points. The Voronoi diagram is calculated from the Delaunay
triangulation of this unstructured mesh.

The approach here roughly follows the one provided by Nicolas Rougier:

https://github.com/rougier/gallery/blob/master/voronoi/voronoi/voronoi.py

"""

from numpy import matrix, array
from numpy.linalg import solve
from numpy.linalg.linalg import LinAlgError


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

