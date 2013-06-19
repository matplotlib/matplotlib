'''
This module contains a routine to create a Voronoi diagram of an unstructured
set of points. The Voronoi diagram is calculated from the Delaunay
triangulation of this unstructured mesh.

The approach here roughly follows the one provided by Nicolas Rougier:

https://github.com/rougier/gallery/blob/master/voronoi/voronoi/voronoi.py

'''

from math import atan2

import numpy as np
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
    x_fake = list(x) + [x_mid - 2 * x_diff, x_mid - 2 * x_diff,
        x_mid + 2 * x_diff, x_mid + 2 * x_diff]
    y_fake = list(y) + [y_mid - 2 * y_diff, y_mid + 2 * y_diff,
        y_mid - 2 * y_diff, y_mid + 2 * y_diff]

    # Calculate a Delaunay triangulation
    circumcenters, edges, tri_points, tri_neighbors = delaunay(x_fake, y_fake)

    # Compute triangle centers that are the Voronoi cell corners
    cells = [[] for i in xrange(x.shape[0])]
    for center, tri in zip(circumcenters, tri_points):
        for index in tri:
            if index < len(cells):
                cells[index].append(tuple(center))

    # Sort the polygon corners clockwise
    for i, cell in enumerate(cells):
        cells[i] = sorted(cell, key=lambda cell: atan2(cell[1] - y[i], cell[0] - x[i]))

    return cells


class VoronoiCollection(PatchCollection):
    '''
    A collection of patches specific for a Voronoi diagram.

    *X* and *Y* are arrays containing the coordinates of the cell-forming
    points.

    This class hosts a method `set_grid` to conveniently recalculate the cells
    from a new unstructured grid after creation. Thus, it is advised to use
    this method instead of `set_paths` to change the collections geometry.

    As for :class:`~matplotlib.collections.PatchCollection`, the use of
    :class:`matplotlib.cm.ScalarMappable` is optional.

    '''

    def __init__(self, X, Y, **kwargs):
        '''Create VoronoiCollection.'''
        PatchCollection.__init__(self, [], **kwargs)
        self.set_grid(X, Y)

    def get_grid(self):
        '''Returns a 2-tuple *(X, Y)* of the grid point coordinates.'''
        return self._grid_x, self._grid_y

    def set_grid(self, X, Y):
        '''
        Modifies the geometry of the collection by considering a new
        unstructured grid as specified by their coordiantes *X* and *Y*.

        The arrays must be 1-dimensional array-likes of equal length.

        '''
        # Check and convert arrays
        self._grid_x = np.array(X)
        self._grid_y = np.array(Y)
        assert self._grid_x.shape == self._grid_y.shape, ValueError('shape mismatch')
        assert len(self._grid_x.shape) == 1, ValueError('1D arrays required')

        # Compute Voronoi cells
        cells = compute_voronoi_cells(self._grid_x, self._grid_y)

        # Assemble patches
        patches = []
        for cell in cells:
            codes = [Path.MOVETO] + [Path.LINETO] * (len(cell) - 1) + [Path.CLOSEPOLY]
            path = Path(cell + [cell[0]], codes)
            patches.append(PathPatch(path))
        self.set_paths(patches)


def voronoi(ax, X, Y, C=None, **kwargs):
    '''
    Create a Voronoi diagram of an unstructured grid.

    This function uses a Delaunay triangulation and derives the Voronoi cells
    from the centers of the circumscribed circles around the Delaunay
    triangles.

    Call signature:

    ::

      voronoi(X, Y, ...)
      voronoi(X, Y, C, ...)

    *X* and *Y* are arrays containing the coordinates of the cell-forming
    points.

    *C* is an array of mappable scalars. If it is provided, the Voronoi cells
    associated with a certain point will be colored according to the colormap.

    The remaining kwargs are the same as for
    :class:`~matplotlib.collections.Collection`.

    **Example:**

        .. plot:: mpl_examples/pylab_examples/voronoi_demo.py

    '''
    # Reset axes
    if not ax._hold:
        ax.cla()

    # Create collection and set mappable if required
    voronoi_collection = VoronoiCollection(X, Y, **kwargs)
    x_arr, y_arr = voronoi_collection.get_grid()
    if C is not None:
        c_arr = np.array(C)
        assert c_arr.shape == x_arr.shape, ValueError('shape mismatch')
        voronoi_collection.set_array(c_arr)

    # Handle axes
    ax.grid(False)
    ax.update_datalim(((x_arr.min(), y_arr.min()), (x_arr.max(), y_arr.max())))
    ax.autoscale_view()
    ax.add_collection(voronoi_collection, autolim=False)

    # Return collection
    return voronoi_collection

