import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.delaunay as mdel
from nose.tools import assert_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib.testing.decorators import image_comparison

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

@image_comparison(baseline_images=['tripcolor1'])
def test_tripcolor():
    x = np.asarray([0, 0.5, 1, 0,   0.5, 1,   0, 0.5, 1, 0.75])
    y = np.asarray([0, 0,   0, 0.5, 0.5, 0.5, 1, 1,   1, 0.75])
    triangles = np.asarray([
        [0, 1, 3], [1, 4, 3],
        [1, 2, 4], [2, 5, 4],
        [3, 4, 6], [4, 7, 6],
        [4, 5, 9], [7, 4, 9], [8, 7, 9], [5, 8, 9] ])

    # Triangulation with same number of points and triangles.
    triang = mtri.Triangulation(x, y, triangles)

    Cpoints = x + 0.5*y

    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    Cfaces = 0.5*xmid + ymid

    plt.subplot(121)
    plt.tripcolor(triang, Cpoints, edgecolors='k')
    plt.title('point colors')

    plt.subplot(122)
    plt.tripcolor(triang, facecolors=Cfaces, edgecolors='k')
    plt.title('facecolors')

def test_no_modify():
    triangles = np.array([[3, 2, 0], [3, 1, 0]], dtype=np.int32)
    points = np.array([(0, 0), (0, 1.1), (1, 0), (1, 1)])

    old_triangles = triangles.copy()
    tri = mtri.Triangulation(points[:,0], points[:,1], triangles)
    edges = tri.edges
    assert_array_equal(old_triangles, triangles)

def test_trifinder():
    # Test points within triangles of masked triangulation.
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    x = x.ravel()
    y = y.ravel()
    triangles = [[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6],
                 [3, 7, 6], [4, 5, 8], [5, 9, 8], [5, 6, 9], [6, 10, 9],
                 [6, 7, 10], [7, 11, 10], [8, 9, 12], [9, 13, 12], [9, 10, 13],
                 [10, 14, 13], [10, 11, 14], [11, 15, 14]]
    mask = np.zeros(len(triangles))
    mask[8:10] = 1
    triang = mtri.Triangulation(x, y, triangles, mask)
    trifinder = triang.get_trifinder()

    xs = [0.25, 1.25, 2.25, 3.25]
    ys = [0.25, 1.25, 2.25, 3.25]
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.ravel()
    ys = ys.ravel()
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [0, 2, 4, -1, 6, -1, 10, -1,
                              12, 14, 16, -1, -1, -1, -1, -1])
    tris = trifinder(xs-0.5, ys-0.5)
    assert_array_equal(tris, [-1, -1, -1, -1, -1, 1, 3, 5,
                              -1, 7, -1, 11, -1, 13, 15, 17])

    # Test points exactly on boundary edges of masked triangulation.
    xs = [0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 1.5, 1.5, 0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.0, 2.0, 1.5, 1.5, 1.5, 1.5]
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [0, 2, 4, 13, 15, 17, 3, 14, 6, 7, 10, 11])

    # Test points exactly on boundary corners of masked triangulation.
    xs = [0.0, 3.0]
    ys = [0.0, 3.0]
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [0, 17])

    # Test triangles with horizontal colinear points.  These are not valid
    # triangulations, but we try to deal with the simplest violations.
    delta = 0.0  # If +ve, triangulation is OK, if -ve triangulation invalid,
                 # if zero have colinear points but should pass tests anyway.
    x = [1.5, 0,  1,  2, 3, 1.5,   1.5]
    y = [-1,  0,  0,  0, 0, delta, 1]
    triangles = [[0, 2, 1], [0, 3, 2], [0, 4, 3], [1, 2, 5], [2, 3, 5],
                 [3, 4, 5], [1, 5, 6], [4, 6, 5]]
    triang = mtri.Triangulation(x, y, triangles)
    trifinder = triang.get_trifinder()

    xs = [-0.1, 0.4, 0.9, 1.4, 1.9, 2.4, 2.9]
    ys = [-0.1, 0.1]
    xs, ys = np.meshgrid(xs, ys)
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [[-1, 0, 0, 1, 1, 2, -1],
                              [-1, 6, 6, 6, 7, 7, -1]])

    # Test triangles with vertical colinear points.  These are not valid
    # triangulations, but we try to deal with the simplest violations.
    delta = 0.0  # If +ve, triangulation is OK, if -ve triangulation invalid,
                 # if zero have colinear points but should pass tests anyway.
    x = [-1, -delta, 0,  0,  0, 0, 1]
    y = [1.5, 1.5,   0,  1,  2, 3, 1.5]
    triangles = [[0, 1, 2], [0, 1, 5], [1, 2, 3], [1, 3, 4], [1, 4, 5],
                 [2, 6, 3], [3, 6, 4], [4, 6, 5]]
    triang = mtri.Triangulation(x, y, triangles)
    trifinder = triang.get_trifinder()

    xs = [-0.1, 0.1]
    ys = [-0.1, 0.4, 0.9, 1.4, 1.9, 2.4, 2.9]
    xs, ys = np.meshgrid(xs, ys)
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [[-1, -1], [0, 5], [0, 5], [0, 6], [1, 6], [1, 7],
                              [-1, -1]])

    # Test that changing triangulation by setting a mask causes the trifinder
    # to be reinitialised.
    x = [0, 1, 0, 1]
    y = [0, 0, 1, 1]
    triangles = [[0, 1, 2], [1, 3, 2]]
    triang = mtri.Triangulation(x, y, triangles)
    trifinder = triang.get_trifinder()

    xs = [-0.2, 0.2, 0.8, 1.2]
    ys = [0.5, 0.5, 0.5, 0.5]
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [-1, 0, 1, -1])

    triang.set_mask([1, 0])
    assert_equal(trifinder, triang.get_trifinder())
    tris = trifinder(xs, ys)
    assert_array_equal(tris, [-1, -1, 1, -1])

def test_triinterp():
    # Test points within triangles of masked triangulation.
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    x = x.ravel()
    y = y.ravel()
    z = 1.23*x - 4.79*y
    triangles = [[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5], [2, 3, 6],
                 [3, 7, 6], [4, 5, 8], [5, 9, 8], [5, 6, 9], [6, 10, 9],
                 [6, 7, 10], [7, 11, 10], [8, 9, 12], [9, 13, 12], [9, 10, 13],
                 [10, 14, 13], [10, 11, 14], [11, 15, 14]]
    mask = np.zeros(len(triangles))
    mask[8:10] = 1
    triang = mtri.Triangulation(x, y, triangles, mask)
    linear_interp = mtri.LinearTriInterpolator(triang, z)

    xs = np.linspace(0.25, 2.75, 6)
    ys = [0.25, 0.75, 2.25, 2.75]
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.ravel()
    ys = ys.ravel()
    zs = linear_interp(xs, ys)
    assert_array_almost_equal(zs, (1.23*xs - 4.79*ys))

    # Test points outside triangulation.
    xs = [-0.25, 1.25, 1.75, 3.25]
    ys = xs
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.ravel()
    ys = ys.ravel()
    zs = linear_interp(xs, ys)
    assert_array_equal(zs.mask, [True]*16)
