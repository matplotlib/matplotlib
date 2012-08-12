from numpy.testing import assert_array_equal
from matplotlib.patches import Polygon

def test_Polygon_close():
    #1018
    xy = [[0,0], [0,1], [1,1]]
    xyclosed = xy + [[0,0]]
    p = Polygon(xy, closed=True)
    assert_array_equal(p.get_xy(), xyclosed)
    p.set_xy(xy)
    assert_array_equal(p.get_xy(), xyclosed)
    p = Polygon(xy, closed=False)
    assert_array_equal(p.get_xy(), xy)
    p.set_xy(xyclosed)
    assert_array_equal(p.get_xy(), xy)

