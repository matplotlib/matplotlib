"""
Tests specific to the patches module.
"""

import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_almost_equal
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import path as mpath
from matplotlib import transforms as mtrans


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


def test_rotate_rect():
    loc = np.asarray([1.0, 2.0])
    width = 2
    height = 3
    angle = 30.0

    # A rotated rectangle
    rect1 = Rectangle(loc, width, height, angle=angle)

    # A non-rotated rectangle
    rect2 = Rectangle(loc, width, height)

    # Set up an explicit rotation matrix (in radians)
    angle_rad = np.pi * angle / 180.0
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad),  np.cos(angle_rad)]])

    # Translate to origin, rotate each vertex, and then translate back
    new_verts = np.inner(rotation_matrix, rect2.get_verts() - loc).T + loc

    # They should be the same
    assert_almost_equal(rect1.get_verts(), new_verts)


@image_comparison(baseline_images=['clip_to_bbox'])
def test_clip_to_bbox():
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_xlim([-18, 20])
    ax.set_ylim([-150, 100])

    star = mpath.Path.unit_regular_star(8)
    path = mpath.Path(star.vertices.copy(), star.codes)
    path.vertices *= [10, 100]
    path.vertices -= [5, 25]

    circle = mpath.Path.unit_circle()
    path2 = mpath.Path(circle.vertices.copy(), circle.codes)
    path2.vertices *= [10, 100]
    path2.vertices += [10, -25]

    combined = mpath.Path.make_compound_path(path, path2)

    patch = mpatches.PathPatch(
        combined, alpha=0.5, facecolor='coral', edgecolor='none')
    ax.add_patch(patch)

    bbox = mtrans.Bbox([[-12, -77.5], [50, -110]])
    result_path = combined.clip_to_bbox(bbox)
    result_patch = mpatches.PathPatch(
        result_path, alpha=0.5, facecolor='green', lw=4, edgecolor='black')

    ax.add_patch(result_patch)
