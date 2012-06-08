import numpy as np
import matplotlib.tri as mtri
import matplotlib.delaunay as mdel
from nose.tools import assert_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal

def test_delaunay():
    # No duplicate points.
    x = [0,1,1,0]
    y = [0,0,1,1]
    npoints = 4
    ntriangles = 2
    nedges = 5

    # Without duplicate points, mpl calls delaunay triangulation and
    # does not modify it.
    mpl_triang = mtri.Triangulation(x,y)
    del_triang = mdel.Triangulation(x,y)

    # Points - floating point.
    assert_array_almost_equal(mpl_triang.x, x)
    assert_array_almost_equal(mpl_triang.x, del_triang.x)
    assert_array_almost_equal(mpl_triang.y, y)
    assert_array_almost_equal(mpl_triang.y, del_triang.y)

    # Triangles - integers.
    assert_equal(len(mpl_triang.triangles), ntriangles)
    assert_equal(np.min(mpl_triang.triangles), 0)
    assert_equal(np.max(mpl_triang.triangles), npoints-1)
    assert_array_equal(mpl_triang.triangles, del_triang.triangle_nodes)

    # Edges - integers.
    assert_equal(len(mpl_triang.edges), nedges)
    assert_equal(np.min(mpl_triang.edges), 0)
    assert_equal(np.max(mpl_triang.edges), npoints-1)
    assert_array_equal(mpl_triang.edges, del_triang.edge_db)

def test_delaunay_duplicate_points():
    # Issue 838.
    import warnings

    # Index 2 is the same as index 0.
    x = [0,1,0,1,0]
    y = [0,0,0,1,1]
    duplicate_index = 2
    npoints = 4        # Number of non-duplicate points.
    nduplicates = 1
    ntriangles = 2
    nedges = 5

    # With duplicate points, mpl calls delaunay triangulation but
    # modified returned arrays.
    warnings.simplefilter("ignore")   # Ignore DuplicatePointWarning.
    mpl_triang = mtri.Triangulation(x,y)
    del_triang = mdel.Triangulation(x,y)
    warnings.resetwarnings()

    # Points - floating point.
    assert_equal(len(mpl_triang.x), npoints + nduplicates)
    assert_equal(len(del_triang.x), npoints)
    assert_array_almost_equal(mpl_triang.x, x)
    assert_array_almost_equal(del_triang.x[:duplicate_index], x[:duplicate_index])
    assert_array_almost_equal(del_triang.x[duplicate_index:], x[duplicate_index+1:])

    assert_equal(len(mpl_triang.y), npoints + nduplicates)
    assert_equal(len(del_triang.y), npoints)
    assert_array_almost_equal(mpl_triang.y, y)
    assert_array_almost_equal(del_triang.y[:duplicate_index], y[:duplicate_index])
    assert_array_almost_equal(del_triang.y[duplicate_index:], y[duplicate_index+1:])

    # Triangles - integers.
    assert_equal(len(mpl_triang.triangles), ntriangles)
    assert_equal(np.min(mpl_triang.triangles), 0)
    assert_equal(np.max(mpl_triang.triangles), npoints-1 + nduplicates)
    assert_equal(len(del_triang.triangle_nodes), ntriangles)
    assert_equal(np.min(del_triang.triangle_nodes), 0)
    assert_equal(np.max(del_triang.triangle_nodes), npoints-1)
    # Convert mpl triangle point indices to delaunay's.
    converted_indices = np.where(mpl_triang.triangles > duplicate_index,
                                 mpl_triang.triangles - nduplicates,
                                 mpl_triang.triangles)
    assert_array_equal(del_triang.triangle_nodes, converted_indices)

    # Edges - integers.
    assert_equal(len(mpl_triang.edges), nedges)
    assert_equal(np.min(mpl_triang.edges), 0)
    assert_equal(np.max(mpl_triang.edges), npoints-1 + nduplicates)
    assert_equal(len(del_triang.edge_db), nedges)
    assert_equal(np.min(del_triang.edge_db), 0)
    assert_equal(np.max(del_triang.edge_db), npoints-1)
    # Convert mpl edge point indices to delaunay's.
    converted_indices = np.where(mpl_triang.edges > duplicate_index,
                                 mpl_triang.edges - nduplicates,
                                 mpl_triang.edges)
    assert_array_equal(del_triang.edge_db, converted_indices)

