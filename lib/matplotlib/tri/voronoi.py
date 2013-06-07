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


def voronoi(X, Y, Z=None, **kwargs):
    '''
    Draws Voronoi diagram.

    '''
    # Check sizes
    assert X.shape == Y.shape and (X.shape == Z.shape if Z is not None else True), 'shape mismatch'
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
    if Z is not None:
        voronoi_collection.set_array(Z)
    return voronoi_collection

