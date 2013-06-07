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

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.delaunay import delaunay


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

    # Add some fake cell centers at a certain distance from the other points
    # This ensures the creation of corner and border cells, that otherwise
    # might end up having too few corners
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_mid, x_diff = 0.5 * (x_min + x_max), x_max - x_min
    y_mid, y_diff = 0.5 * (y_min + y_max), y_max - y_min
    x_fake = list(x) + [x_mid - x_diff, x_mid - x_diff, x_mid + x_diff, x_mid + x_diff]
    y_fake = list(y) + [y_mid - y_diff, y_mid + y_diff, y_mid - y_diff, y_mid + y_diff]

    # Calculate a Delaunay triangulation
    circumcenters, edges, tri_points, tri_neighbors = delaunay(x_fake, y_fake)

    # Compute triangle centers that are the Voronoi cell corners
    cells = [[] for i in xrange(x.shape[0])]
    for center, tri in zip(circumcenters, tri_points):
        for index in tri:
            if index < len(cells):
                cells[index].append(center)

    # Sort the polygon corners clockwise
    for i, cell in enumerate(cells):
        cells[i] = sorted(cell, key=lambda cell: atan2(cell[1] - y[i], cell[0] - x[i]))

    return cells


def voronoi(X, Y, Z, **kwargs):
    '''
    Draws Voronoi diagram.

    '''
    # Check sizes
    assert X.shape == Y.shape and X.shape == Z.shape, 'shape mismatch'
    assert len(X.shape) == 1, '1D arrays required'

    # Compute Voronoi cells
    cells = compute_voronoi_cells(X, Y)

    # Assemble patches
    patches = []
    for cell in cells:
        codes = [Path.MOVETO] + [Path.LINETO] * (len(cell) - 1) + [Path.CLOSEPOLY]
        path = Path(cell + [cell[0]], codes)
        patches.append(PathPatch(path))

    # Create collection
    voronoi_collection = PatchCollection(patches, **kwargs)
    voronoi_collection.set_array(Z)
    return voronoi_collection

