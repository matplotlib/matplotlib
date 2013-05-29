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
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections
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

    path = mpath.Path.unit_regular_star(8).deepcopy()
    path.vertices *= [10, 100]
    path.vertices -= [5, 25]

    path2 = mpath.Path.unit_circle().deepcopy()
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


@image_comparison(baseline_images=['patch_alpha_coloring'], remove_text=True)
def test_patch_alpha_coloring():
    """
    Test checks that the patch and collection are rendered with the specified
    alpha values in their facecolor and edgecolor.
    """
    star = mpath.Path.unit_regular_star(6)
    circle = mpath.Path.unit_circle()
    # concatenate the star with an internal cutout of the circle
    verts = np.concatenate([circle.vertices, star.vertices[::-1]])
    codes = np.concatenate([circle.codes, star.codes])
    cut_star1 = mpath.Path(verts, codes)
    cut_star2 = mpath.Path(verts + 1, codes)

    ax = plt.axes()
    patch = mpatches.PathPatch(cut_star1,
                               linewidth=5, linestyle='dashdot',
                               facecolor=(1, 0, 0, 0.5),
                               edgecolor=(0, 0, 1, 0.75))
    ax.add_patch(patch)

    col = mcollections.PathCollection([cut_star2],
                                      linewidth=5, linestyles='dashdot',
                                      facecolor=(1, 0, 0, 0.5),
                                      edgecolor=(0, 0, 1, 0.75))
    ax.add_collection(col)

    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])



@image_comparison(baseline_images=['patch_alpha_override'], remove_text=True)
def test_patch_alpha_override():
    """
    Test checks that specifying an alpha attribute for a patch or collection
    will override any alpha component of the facecolor or edgecolor.
    """
    star = mpath.Path.unit_regular_star(6)
    circle = mpath.Path.unit_circle()
    # concatenate the star with an internal cutout of the circle
    verts = np.concatenate([circle.vertices, star.vertices[::-1]])
    codes = np.concatenate([circle.codes, star.codes])
    cut_star1 = mpath.Path(verts, codes)
    cut_star2 = mpath.Path(verts + 1, codes)

    ax = plt.axes()
    patch = mpatches.PathPatch(cut_star1,
                               linewidth=5, linestyle='dashdot',
                               alpha=0.25,
                               facecolor=(1, 0, 0, 0.5),
                               edgecolor=(0, 0, 1, 0.75))
    ax.add_patch(patch)

    col = mcollections.PathCollection([cut_star2],
                                      linewidth=5, linestyles='dashdot',
                                      alpha=0.25,
                                      facecolor=(1, 0, 0, 0.5),
                                      edgecolor=(0, 0, 1, 0.75))
    ax.add_collection(col)

    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])



@image_comparison(baseline_images=['patch_custom_linestyle'], remove_text=True)
def test_patch_custom_linestyle():
    """
    A test to check that patches and collections accept custom dash patterns
    as linestyle and that they display correctly.
    """
    star = mpath.Path.unit_regular_star(6)
    circle = mpath.Path.unit_circle()
    # concatenate the star with an internal cutout of the circle
    verts = np.concatenate([circle.vertices, star.vertices[::-1]])
    codes = np.concatenate([circle.codes, star.codes])
    cut_star1 = mpath.Path(verts, codes)
    cut_star2 = mpath.Path(verts + 1, codes)

    ax = plt.axes()
    patch = mpatches.PathPatch(cut_star1,
                               linewidth=5, linestyle=(0.0, (5.0, 7.0, 10.0, 7.0)),
                               facecolor=(1, 0, 0),
                               edgecolor=(0, 0, 1))
    ax.add_patch(patch)

    col = mcollections.PathCollection([cut_star2],
                                      linewidth=5, linestyles=[(0.0, (5.0, 7.0, 10.0, 7.0))],
                                      facecolor=(1, 0, 0),
                                      edgecolor=(0, 0, 1))
    ax.add_collection(col)

    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])





if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
