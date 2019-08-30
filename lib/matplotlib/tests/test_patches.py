"""
Tests specific to the patches module.
"""
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest

from matplotlib.cbook import MatplotlibDeprecationWarning
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
import matplotlib.pyplot as plt
from matplotlib import (
    collections as mcollections, colors as mcolors, patches as mpatches,
    path as mpath, style as mstyle, transforms as mtransforms)

import sys
on_win = (sys.platform == 'win32')


def test_Polygon_close():
    #: GitHub issue #1018 identified a bug in the Polygon handling
    #: of the closed attribute; the path was not getting closed
    #: when set_xy was used to set the vertices.

    # open set of vertices:
    xy = [[0, 0], [0, 1], [1, 1]]
    # closed set:
    xyclosed = xy + [[0, 0]]

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


def test_negative_rect():
    # These two rectangles have the same vertices, but starting from a
    # different point.  (We also drop the last vertex, which is a duplicate.)
    pos_vertices = Rectangle((-3, -2), 3, 2).get_verts()[:-1]
    neg_vertices = Rectangle((0, 0), -3, -2).get_verts()[:-1]
    assert_array_equal(np.roll(neg_vertices, 2, 0), pos_vertices)


@image_comparison(['clip_to_bbox'])
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

    bbox = mtransforms.Bbox([[-12, -77.5], [50, -110]])
    result_path = combined.clip_to_bbox(bbox)
    result_patch = mpatches.PathPatch(
        result_path, alpha=0.5, facecolor='green', lw=4, edgecolor='black')

    ax.add_patch(result_patch)


@image_comparison(['patch_alpha_coloring'], remove_text=True)
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


@image_comparison(['patch_alpha_override'], remove_text=True)
def test_patch_alpha_override():
    #: Test checks that specifying an alpha attribute for a patch or
    #: collection will override any alpha component of the facecolor
    #: or edgecolor.
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


@pytest.mark.style('default')
def test_patch_color_none():
    # Make sure the alpha kwarg does not override 'none' facecolor.
    # Addresses issue #7478.
    c = plt.Circle((0, 0), 1, facecolor='none', alpha=1)
    assert c.get_facecolor()[0] == 0


@image_comparison(['patch_custom_linestyle'], remove_text=True)
def test_patch_custom_linestyle():
    #: A test to check that patches and collections accept custom dash
    #: patterns as linestyle and that they display correctly.
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


def test_patch_linestyle_accents():
    #: Test if linestyle can also be specified with short mnemonics like "--"
    #: c.f. GitHub issue #2136
    star = mpath.Path.unit_regular_star(6)
    circle = mpath.Path.unit_circle()
    # concatenate the star with an internal cutout of the circle
    verts = np.concatenate([circle.vertices, star.vertices[::-1]])
    codes = np.concatenate([circle.codes, star.codes])

    linestyles = ["-", "--", "-.", ":",
                  "solid", "dashed", "dashdot", "dotted"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, ls in enumerate(linestyles):
        star = mpath.Path(verts + i, codes)
        patch = mpatches.PathPatch(star,
                                   linewidth=3, linestyle=ls,
                                   facecolor=(1, 0, 0),
                                   edgecolor=(0, 0, 1))
        ax.add_patch(patch)

    ax.set_xlim([-1, i + 1])
    ax.set_ylim([-1, i + 1])
    fig.canvas.draw()


def test_wedge_movement():
    param_dict = {'center': ((0, 0), (1, 1), 'set_center'),
                  'r': (5, 8, 'set_radius'),
                  'width': (2, 3, 'set_width'),
                  'theta1': (0, 30, 'set_theta1'),
                  'theta2': (45, 50, 'set_theta2')}

    init_args = {k: v[0] for k, v in param_dict.items()}

    w = mpatches.Wedge(**init_args)
    for attr, (old_v, new_v, func) in param_dict.items():
        assert getattr(w, attr) == old_v
        getattr(w, func)(new_v)
        assert getattr(w, attr) == new_v


# png needs tol>=0.06, pdf tol>=1.617
@image_comparison(['wedge_range'], remove_text=True, tol=1.65 if on_win else 0)
def test_wedge_range():
    ax = plt.axes()

    t1 = 2.313869244286224

    args = [[52.31386924, 232.31386924],
            [52.313869244286224, 232.31386924428622],
            [t1, t1 + 180.0],
            [0, 360],
            [90, 90 + 360],
            [-180, 180],
            [0, 380],
            [45, 46],
            [46, 45]]

    for i, (theta1, theta2) in enumerate(args):
        x = i % 3
        y = i // 3

        wedge = mpatches.Wedge((x * 3, y * 3), 1, theta1, theta2,
                               facecolor='none', edgecolor='k', lw=3)

        ax.add_artist(wedge)

    ax.set_xlim([-2, 8])
    ax.set_ylim([-2, 9])


def test_patch_str():
    """
    Check that patches have nice and working `str` representation.

    Note that the logic is that `__str__` is defined such that:
    str(eval(str(p))) == str(p)
    """
    p = mpatches.Circle(xy=(1, 2), radius=3)
    assert str(p) == 'Circle(xy=(1, 2), radius=3)'

    p = mpatches.Ellipse(xy=(1, 2), width=3, height=4, angle=5)
    assert str(p) == 'Ellipse(xy=(1, 2), width=3, height=4, angle=5)'

    p = mpatches.Rectangle(xy=(1, 2), width=3, height=4, angle=5)
    assert str(p) == 'Rectangle(xy=(1, 2), width=3, height=4, angle=5)'

    p = mpatches.Wedge(center=(1, 2), r=3, theta1=4, theta2=5, width=6)
    assert str(p) == 'Wedge(center=(1, 2), r=3, theta1=4, theta2=5, width=6)'

    p = mpatches.Arc(xy=(1, 2), width=3, height=4, angle=5, theta1=6, theta2=7)
    expected = 'Arc(xy=(1, 2), width=3, height=4, angle=5, theta1=6, theta2=7)'
    assert str(p) == expected

    p = mpatches.RegularPolygon((1, 2), 20, radius=5)
    assert str(p) == "RegularPolygon((1, 2), 20, radius=5, orientation=0)"

    p = mpatches.CirclePolygon(xy=(1, 2), radius=5, resolution=20)
    assert str(p) == "CirclePolygon((1, 2), radius=5, resolution=20)"

    p = mpatches.FancyBboxPatch((1, 2), width=3, height=4)
    assert str(p) == "FancyBboxPatch((1, 2), width=3, height=4)"

    # Further nice __str__ which cannot be `eval`uated:
    path_data = [([1, 2], mpath.Path.MOVETO), ([2, 2], mpath.Path.LINETO),
                 ([1, 2], mpath.Path.CLOSEPOLY)]
    p = mpatches.PathPatch(mpath.Path(*zip(*path_data)))
    assert str(p) == "PathPatch3((1, 2) ...)"

    data = [[1, 2], [2, 2], [1, 2]]
    p = mpatches.Polygon(data)
    assert str(p) == "Polygon3((1, 2) ...)"

    p = mpatches.FancyArrowPatch(path=mpath.Path(*zip(*path_data)))
    assert str(p)[:27] == "FancyArrowPatch(Path(array("

    p = mpatches.FancyArrowPatch((1, 2), (3, 4))
    assert str(p) == "FancyArrowPatch((1, 2)->(3, 4))"

    p = mpatches.ConnectionPatch((1, 2), (3, 4), 'data')
    assert str(p) == "ConnectionPatch((1, 2), (3, 4))"

    s = mpatches.Shadow(p, 1, 1)
    assert str(s) == "Shadow(ConnectionPatch((1, 2), (3, 4)))"

    # Not testing Arrow, FancyArrow here
    # because they seem to exist only for historical reasons.


@image_comparison(['multi_color_hatch'], remove_text=True, style='default')
def test_multi_color_hatch():
    fig, ax = plt.subplots()

    rects = ax.bar(range(5), range(1, 6))
    for i, rect in enumerate(rects):
        rect.set_facecolor('none')
        rect.set_edgecolor('C{}'.format(i))
        rect.set_hatch('/')

    ax.autoscale_view()
    ax.autoscale(False)

    for i in range(5):
        with mstyle.context({'hatch.color': 'C{}'.format(i)}):
            r = Rectangle((i - .8 / 2, 5), .8, 1, hatch='//', fc='none')
        ax.add_patch(r)


@image_comparison(['units_rectangle.png'])
def test_units_rectangle():
    import matplotlib.testing.jpl_units as U
    U.register()

    p = mpatches.Rectangle((5*U.km, 6*U.km), 1*U.km, 2*U.km)

    fig, ax = plt.subplots()
    ax.add_patch(p)
    ax.set_xlim([4*U.km, 7*U.km])
    ax.set_ylim([5*U.km, 9*U.km])


@image_comparison(['connection_patch.png'], style='mpl20', remove_text=True)
def test_connection_patch():
    fig, (ax1, ax2) = plt.subplots(1, 2)

    con = mpatches.ConnectionPatch(xyA=(0.1, 0.1), xyB=(0.9, 0.9),
                                   coordsA='data', coordsB='data',
                                   axesA=ax2, axesB=ax1,
                                   arrowstyle="->")
    ax2.add_artist(con)

    xyA = (0.6, 1.0)  # in axes coordinates
    xyB = (0.0, 0.2)  # x in axes coordinates, y in data coordinates
    coordsA = "axes fraction"
    coordsB = ax2.get_yaxis_transform()
    con = mpatches.ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=coordsA,
                                    coordsB=coordsB, arrowstyle="-")
    ax2.add_artist(con)


def test_connection_patch_fig():
    # Test that connection patch can be added as figure artist
    fig, (ax1, ax2) = plt.subplots(1, 2)
    xy = (0.3, 0.2)
    con = mpatches.ConnectionPatch(xyA=xy, xyB=xy,
                                   coordsA="data", coordsB="data",
                                   axesA=ax1, axesB=ax2,
                                   arrowstyle="->", shrinkB=5)
    fig.add_artist(con)
    fig.canvas.draw()


def test_datetime_rectangle():
    # Check that creating a rectangle with timedeltas doesn't fail
    from datetime import datetime, timedelta

    start = datetime(2017, 1, 1, 0, 0, 0)
    delta = timedelta(seconds=16)
    patch = mpatches.Rectangle((start, 0), delta, 1)

    fig, ax = plt.subplots()
    ax.add_patch(patch)


def test_datetime_datetime_fails():
    from datetime import datetime

    start = datetime(2017, 1, 1, 0, 0, 0)
    dt_delta = datetime(1970, 1, 5)    # Will be 5 days if units are done wrong

    with pytest.raises(TypeError):
        mpatches.Rectangle((start, 0), dt_delta, 1)

    with pytest.raises(TypeError):
        mpatches.Rectangle((0, start), 1, dt_delta)


def test_contains_point():
    ell = mpatches.Ellipse((0.5, 0.5), 0.5, 1.0, 0)
    points = [(0.0, 0.5), (0.2, 0.5), (0.25, 0.5), (0.5, 0.5)]
    path = ell.get_path()
    transform = ell.get_transform()
    radius = ell._process_radius(None)
    expected = np.array([path.contains_point(point,
                                             transform,
                                             radius) for point in points])
    result = np.array([ell.contains_point(point) for point in points])
    assert np.all(result == expected)


def test_contains_points():
    ell = mpatches.Ellipse((0.5, 0.5), 0.5, 1.0, 0)
    points = [(0.0, 0.5), (0.2, 0.5), (0.25, 0.5), (0.5, 0.5)]
    path = ell.get_path()
    transform = ell.get_transform()
    radius = ell._process_radius(None)
    expected = path.contains_points(points, transform, radius)
    result = ell.contains_points(points)
    assert np.all(result == expected)


# Currently fails with pdf/svg, probably because some parts assume a dpi of 72.
@check_figures_equal(extensions=["png"])
def test_shadow(fig_test, fig_ref):
    xy = np.array([.2, .3])
    dxy = np.array([.1, .2])
    # We need to work around the nonsensical (dpi-dependent) interpretation of
    # offsets by the Shadow class...
    plt.rcParams["savefig.dpi"] = "figure"
    # Test image.
    a1 = fig_test.subplots()
    rect = mpatches.Rectangle(xy=xy, width=.5, height=.5)
    shadow = mpatches.Shadow(rect, ox=dxy[0], oy=dxy[1])
    a1.add_patch(rect)
    a1.add_patch(shadow)
    # Reference image.
    a2 = fig_ref.subplots()
    rect = mpatches.Rectangle(xy=xy, width=.5, height=.5)
    shadow = mpatches.Rectangle(
        xy=xy + fig_ref.dpi / 72 * dxy, width=.5, height=.5,
        fc=np.asarray(mcolors.to_rgb(rect.get_facecolor())) * .3,
        ec=np.asarray(mcolors.to_rgb(rect.get_facecolor())) * .3,
        alpha=.5)
    a2.add_patch(shadow)
    a2.add_patch(rect)


def test_fancyarrow_units():
    from datetime import datetime
    # Smoke test to check that FancyArrowPatch works with units
    dtime = datetime(2000, 1, 1)
    fig, ax = plt.subplots()
    arrow = FancyArrowPatch((0, dtime), (0.01, dtime))
    ax.add_patch(arrow)
