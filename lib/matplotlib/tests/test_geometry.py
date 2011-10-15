import numpy as np
import matplotlib.geometry as g
from matplotlib.transforms import Affine2D
from nose.tools import assert_raises

def test_orientation():
    p1, p2, p3 = np.asarray([[-1,3], [2,0], [0,1]])
    for offset in [[0,0], [-10,0], [0,10], [1000,-10000]]:
        assert g.orientation(p1 + offset, p2 + offset, p3 + offset) == -1
        assert g.orientation(p2 + offset, p3 + offset, p1 + offset) == -1
        assert g.orientation(p3 + offset, p1 + offset, p2 + offset) == -1
        assert g.orientation(p1 + offset, p3 + offset, p2 + offset) == 1
        assert g.orientation(p2 + offset, p1 + offset, p3 + offset) == 1
        assert g.orientation(p3 + offset, p2 + offset, p1 + offset) == 1
    assert g.orientation([0,-1],[0,0],[0,4]) == 0

def test_projection():
    proj = g.project_point_on_line_through
    points = np.asarray([[0,0], [10,0],
                         [3,7], [7,-3], [11, 2], [-1, -3],
                         [3,0], [7, 0], [11, 0], [-1,  0]])
    for mat in [ [[ 1, 0, 0], [0, 1, 0]],
                 [[ 0, 0, 3], [0, 0,-8]],
                 [[-1, 2, 5], [3,-2,-2]],
                 [[ 0, 1, 0], [1, 0, 0]] ]:
        trans = Affine2D(np.r_[mat, [[0.0, 0.0, 1.0]]])
        tpoints = trans.transform(points)
        p1, p2 = points[:2]
        for p0, result in zip(points[2:6], points[6:10]):
            assert np.allclose(proj(p0, p1, p2), result)
        for p0 in points[4:6]:
            assert_raises(ValueError, proj, p0, p1, p2, True)
