"""
Tests specific to the patches module.
"""

from numpy.testing import assert_array_equal
from matplotlib.patches import Polygon

def test_Polygon_close():
    """
    Github issue #1018 identified a bug in the Polygon handling
    of the closed attribute; the path was not getting closed
    when set_xy was used to set the vertices.
    """
    # open set of vertices:
    xy = [[0,0], [0,1], [1,1]]
    # closed set:
    xyclosed = xy + [[0,0]]

    # start with open path and close it:
    p = Polygon(xy, closed=True)
    assert_array_equal(p.get_xy(), xyclosed)
    p.set_xy(xy)
    assert_array_equal(p.get_xy(), xyclosed)

    # start with closed path and open it:
    p = Polygon(xyclosed, closed=False)
    assert_array_equal(p.get_xy(), xy)
    p.set_xy(xyclosed)
    assert_array_equal(p.get_xy(), xy)

    # start with open path and leave it open:
    p = Polygon(xy, closed=False)
    assert_array_equal(p.get_xy(), xy)
    p.set_xy(xy)
    assert_array_equal(p.get_xy(), xy)

    # start with closed path and leave it closed:
    p = Polygon(xyclosed, closed=True)
    assert_array_equal(p.get_xy(), xyclosed)
    p.set_xy(xyclosed)
    assert_array_equal(p.get_xy(), xyclosed)

