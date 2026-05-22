import numpy as np
import numpy.testing as nptest
import pytest

import matplotlib.pyplot as plt

from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.mplot3d.art3d import (
    get_dir_vector,
    Line3DCollection,
    Poly3DCollection,
    _all_points_on_plane,
    _clip_line_segment_to_box,
    _clip_polygon_to_box,
)


@pytest.mark.parametrize("zdir, expected", [
    ("x", (1, 0, 0)),
    ("y", (0, 1, 0)),
    ("z", (0, 0, 1)),
    (None, (0, 0, 0)),
    ((1, 2, 3), (1, 2, 3)),
    (np.array([4, 5, 6]), (4, 5, 6)),
])
def test_get_dir_vector(zdir, expected):
    res = get_dir_vector(zdir)
    assert isinstance(res, np.ndarray)
    nptest.assert_array_equal(res, expected)


def test_scatter_3d_projection_conservation():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # fix axes3d projection
    ax.roll = 0
    ax.elev = 0
    ax.azim = -45
    ax.stale = True

    x = [0, 1, 2, 3, 4]
    scatter_collection = ax.scatter(x, x, x)
    fig.canvas.draw_idle()

    # Get scatter location on canvas and freeze the data
    scatter_offset = scatter_collection.get_offsets()
    scatter_location = ax.transData.transform(scatter_offset)

    # Yaw -44 and -46 are enough to produce two set of scatter
    # with opposite z-order without moving points too far
    for azim in (-44, -46):
        ax.azim = azim
        ax.stale = True
        fig.canvas.draw_idle()

        for i in range(5):
            # Create a mouse event used to locate and to get index
            # from each dots
            event = MouseEvent("button_press_event", fig.canvas,
                               *scatter_location[i, :])
            contains, ind = scatter_collection.contains(event)
            assert contains is True
            assert len(ind["ind"]) == 1
            assert ind["ind"][0] == i


def test_zordered_error():
    # Smoke test for https://github.com/matplotlib/matplotlib/issues/26497
    lc = [(np.fromiter([0.0, 0.0, 0.0], dtype="float"),
           np.fromiter([1.0, 1.0, 1.0], dtype="float"))]
    pc = [np.fromiter([0.0, 0.0], dtype="float"),
          np.fromiter([0.0, 1.0], dtype="float"),
          np.fromiter([1.0, 1.0], dtype="float")]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.add_collection(Line3DCollection(lc), autolim="_datalim_only")
    ax.scatter(*pc, visible=False)
    plt.draw()


def test_all_points_on_plane():
    # Non-coplanar points
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert not _all_points_on_plane(*points.T)

    # Duplicate points
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert _all_points_on_plane(*points.T)

    # NaN values
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, np.nan]])
    assert _all_points_on_plane(*points.T)

    # Less than 3 unique points
    points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert _all_points_on_plane(*points.T)

    # All points lie on a line
    points = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]])
    assert _all_points_on_plane(*points.T)

    # All points lie on two lines, with antiparallel vectors
    points = np.array([[-2, 2, 0], [-1, 1, 0], [1, -1, 0],
                       [0, 0, 0], [2, 0, 0], [1, 0, 0]])
    assert _all_points_on_plane(*points.T)

    # All points lie on a plane
    points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0]])
    assert _all_points_on_plane(*points.T)



def test_clip_polygon_to_box_xmin():
    poly = np.array([
        [-1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [-1, 1, 0],
    ], dtype=float)

    clipped = _clip_polygon_to_box(
        poly,
        xlim=(0, 1),
        ylim=(-1, 2),
        zlim=(-1, 1),
    )

    expected = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ], dtype=float)

    nptest.assert_allclose(clipped, expected)


def test_clip_polygon_to_box_fully_outside():
    poly = np.array([
        [-2, 0, 0],
        [-1, 0, 0],
        [-1, 1, 0],
        [-2, 1, 0],
    ], dtype=float)

    clipped = _clip_polygon_to_box(
        poly,
        xlim=(0, 1),
        ylim=(-1, 2),
        zlim=(-1, 1),
    )

    assert clipped.shape == (0, 3)


def test_clip_polygon_to_box_zmin():
    poly = np.array([
        [0, 0, -1],
        [1, 0, 1],
        [0, 1, 1],
    ], dtype=float)

    clipped = _clip_polygon_to_box(
        poly,
        xlim=(-1, 2),
        ylim=(-1, 2),
        zlim=(0, 2),
    )

    assert clipped.shape[1] == 3
    assert len(clipped) >= 3
    assert np.all(clipped[:, 2] >= 0)
    assert np.all(clipped[:, 2] <= 2)




def test_clip_line_segment_to_box_crossing():
    segment = _clip_line_segment_to_box(
        [-1, 0.5, 0.5], [2, 0.5, 0.5],
        xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),
    )

    expected = np.array([
        [0, 0.5, 0.5],
        [1, 0.5, 0.5],
    ], dtype=float)

    nptest.assert_allclose(segment, expected)


def test_clip_line_segment_to_box_fully_outside():
    segment = _clip_line_segment_to_box(
        [-2, 0.5, 0.5], [-1, 0.5, 0.5],
        xlim=(0, 1), ylim=(0, 1), zlim=(0, 1),
    )

    assert segment.shape == (0, 3)


def test_generate_normals():
    # Smoke test for https://github.com/matplotlib/matplotlib/issues/29156
    vertices = ((0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0))
    shape = Poly3DCollection([vertices], edgecolors='r', shade=True)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(shape)
    plt.draw()


def test_clip_polygon_to_box_near_parallel():
    # Ensure near-parallel edge (very small denom) does not error and
    # produced vertices lie within the box.
    poly = np.array([
        [-1e-12, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [-1, 1, 0],
    ], dtype=float)

    clipped = _clip_polygon_to_box(
        poly,
        xlim=(0, 1),
        ylim=(-1, 2),
        zlim=(-1, 1),
    )

    assert clipped.shape[1] == 3
    assert len(clipped) >= 3
    assert np.all(clipped[:, 0] >= 0)
    assert np.all(clipped[:, 0] <= 1)


def test_clip_polygon_to_box_degenerate_collinear():
    # Collinear polygon should be considered degenerate after clipping
    # and yield fewer than 3 vertices.
    poly = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ], dtype=float)

    clipped = _clip_polygon_to_box(
        poly,
        xlim=(0.5, 1.5),
        ylim=(-1, 1),
        zlim=(-1, 1),
    )

    assert clipped.shape[0] < 3


def test_clip_polygon_to_box_with_nan():
    # Inputs containing NaNs must be treated as invalid and return empty.
    poly = np.array([
        [0, 0, 0],
        [1, np.nan, 0],
        [0, 1, 0],
    ], dtype=float)

    clipped = _clip_polygon_to_box(
        poly,
        xlim=(-1, 2),
        ylim=(-1, 2),
        zlim=(-1, 2),
    )

    assert clipped.shape == (0, 3)
