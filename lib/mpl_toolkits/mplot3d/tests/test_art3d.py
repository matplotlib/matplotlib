import numpy as np
import numpy.testing as nptest
import pytest

import matplotlib.pyplot as plt

from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.transforms import Affine2D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import (
    get_dir_vector,
    Line3DCollection,
    Patch3DCollection,
    Poly3DCollection,
    pathpatch_2d_to_3d,
    _all_points_on_plane,
)


def test_line3d_set_data_3d_list_with_nan():
    # Regression test for #32127: set_data_3d stores array-like as given, and
    # draw must not require ndarray .shape when a coordinate is non-finite.
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    line, = ax.plot([0.0], [0.0], [0.0])
    xs, ys, zs = [float("nan")], [float("nan")], [float("nan")]
    line.set_data_3d(xs, ys, zs)
    assert line.get_data_3d() == (xs, ys, zs)
    fig.canvas.draw()


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


def test_generate_normals():
    # Smoke test for https://github.com/matplotlib/matplotlib/issues/29156
    vertices = ((0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0))
    shape = Poly3DCollection([vertices], edgecolors='r', shade=True)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.add_collection3d(shape)
    plt.draw()


@pytest.mark.parametrize(
    'views', [(-60, 33), (-60, 33, -60)], ids=['changed', 'restored'])
@check_figures_equal()
def test_patch3d_collection_reprojects_paths(fig_test, fig_ref, views):
    depths = [0, 0.8]
    for fig in (fig_test, fig_ref):
        ax = fig.add_subplot(projection='3d')
        ax.set(xlim=(-1, 2), ylim=(-1, 2), zlim=(-1, 2))
        ax.set_axis_off()
        patches = [
            Circle((0, 0), 0.25, facecolor='red', edgecolor='none',
                   antialiased=False),
            Rectangle((1, 1), 0.5, 0.5, facecolor='blue', edgecolor='none',
                      antialiased=False),
        ]
        if fig is fig_test:
            collection = Patch3DCollection(
                patches, zs=depths, zdir='x', match_original=True,
                depthshade=False)
            original = [path.deepcopy() for path in collection.get_paths()]
            ax.add_collection3d(collection)
            for azim in views[:-1]:
                ax.view_init(elev=28, azim=azim)
                fig.draw_without_rendering()
            for path, source in zip(collection.get_paths(), original, strict=True):
                nptest.assert_array_equal(path.vertices, source.vertices)
                nptest.assert_array_equal(path.codes, source.codes)
        else:
            for patch, depth in zip(patches, depths):
                ax.add_patch(patch)
                pathpatch_2d_to_3d(patch, z=depth, zdir='x')
        ax.view_init(elev=28, azim=views[-1])


@check_figures_equal()
def test_patch3d_collection_restores_path_projection(fig_test, fig_ref):
    for fig in (fig_test, fig_ref):
        ax = fig.add_subplot(projection='3d')
        ax.set(xlim=(-1, 2), ylim=(-1, 2), zlim=(-1, 2))
        ax.view_init(elev=28, azim=-60)
        ax.set_axis_off()
        patch = Rectangle((0, 0), 0.5, 0.5, facecolor='black', edgecolor='none',
                          antialiased=False)
        collection = Patch3DCollection([patch], match_original=True, depthshade=False)
        ax.add_collection3d(collection)
        if fig is fig_test:
            fig.draw_without_rendering()
            identity = collection.get_offset_transform()
            collection.set_offset_transform(Affine2D().translate(5, 10))
            fig.draw_without_rendering()
            collection.set_offset_transform(identity)


@pytest.mark.parametrize(
    'offsets, translate, set_after_draw', [
        ([(0, 0)], False, False),
        ([(0.25, 0.5)], True, False),
        (None, True, False),
        (None, False, True),
    ],
    ids=['explicit-origin', 'transformed-offset', 'offset-transform-only', 'setter'])
def test_patch3d_collection_preserves_offset_projection(
        offsets, translate, set_after_draw):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=(-1, 2), ylim=(-1, 2), zlim=(-1, 2))
    offset_transform = Affine2D().translate(5, 10) if translate else None
    collection = Patch3DCollection(
        [Rectangle((0, 0), 0.5, 0.5)], zs=0.5, offsets=offsets,
        offset_transform=offset_transform, depthshade=False)
    input_offsets = collection.get_offsets().copy()
    vertices = collection.get_paths()[0].vertices.copy()
    ax.add_collection3d(collection)
    if set_after_draw:
        fig.canvas.draw()
        collection.set_offsets([(0.25, 0.5)])
    fig.canvas.draw()

    x, y, _ = proj3d.proj_transform(*input_offsets.T, [0.5], ax.get_proj())
    nptest.assert_allclose(collection.get_offsets(), np.column_stack([x, y]))
    nptest.assert_array_equal(collection.get_paths()[0].vertices, vertices)
    if translate:
        assert collection.get_offset_transform() is offset_transform


@pytest.mark.parametrize('tail', [
    [],
    [((10, 10), Path.STOP)],
    [((0.5, 0.5), Path.STOP), ((10, 10), Path.MOVETO)],
], ids=['closepoly', 'stop', 'after-stop'])
@check_figures_equal()
def test_patch3d_collection_clips_whole_patch(fig_test, fig_ref, tail):
    for fig in (fig_test, fig_ref):
        ax = fig.add_subplot(projection='3d')
        ax.set(xlim=(0, 2), ylim=(0, 2), zlim=(-1, 1))
        ax.view_init(elev=45, azim=-60)
        ax.set_axis_off()
        path = Path.unit_rectangle().transformed(
            Affine2D().scale(0.5).translate(0.5, 0.5))
        # The ignored CLOSEPOLY vertex can lie outside the view limits.
        path.vertices[-1] = (-10, -10)
        if fig is fig_test and tail:
            vertices, codes = zip(*tail)
            path = Path([*path.vertices, *vertices], [*path.codes, *codes])
        inside = PathPatch(path, facecolor='red', edgecolor='none',
                           antialiased=False)
        if fig is fig_test:
            outside = Rectangle((1.75, 0.5), 0.5, 0.5, facecolor='blue',
                                edgecolor='none', antialiased=False)
            ax.add_collection3d(Patch3DCollection(
                [outside, inside], match_original=True, depthshade=False,
                axlim_clip=True))
        else:
            ax.add_patch(inside)
            pathpatch_2d_to_3d(inside)


@pytest.mark.parametrize('tail', [
    [((10, -10), Path.STOP)],
    [((0, 0), Path.STOP), ((10, -10), Path.MOVETO)],
], ids=['stop', 'after-stop'])
def test_patch3d_collection_ignores_terminator_for_depth(tail):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=(-1, 2), ylim=(-1, 2), zlim=(-1, 2))
    ax.view_init(elev=28, azim=-60)
    ax.set_proj_type('ortho')
    rectangle = Path.unit_rectangle()
    vertices, codes = zip(*tail)
    path = Path([*rectangle.vertices, *vertices], [*rectangle.codes, *codes])
    collection = Patch3DCollection([PathPatch(path)], depthshade=False)
    ax.add_collection3d(collection)
    fig.canvas.draw()

    xs, ys = rectangle.vertices[:-1].T
    _, _, depths = proj3d.proj_transform(xs, ys, np.zeros_like(xs), ax.get_proj())
    assert collection.do_3d_projection() == pytest.approx(depths.min())


@pytest.mark.parametrize('ignored_index', [4, 5, 6],
                         ids=['closepoly', 'stop', 'after-stop'])
def test_patch3d_collection_does_not_project_ignored_coordinates(ignored_index):
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(projection='3d')
    ax.set(xlim=(-0.03, 0.12), ylim=(-0.03, 0.12), zlim=(-1, 2))
    ax.view_init(elev=28, azim=-60)
    ax.set_axis_off()
    collection = Patch3DCollection([], facecolor='black', edgecolor='none',
                                   antialiased=False, axlim_clip=True,
                                   depthshade=False)
    ax.add_collection(collection, autolim=False)
    vertices = np.array([
        (0, 0), (0.06, 0), (0.06, 0.06), (0, 0.06), (0, 0), (0, 0), (0, 0)])
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
             Path.CLOSEPOLY, Path.STOP, Path.MOVETO]

    images, depths = [], []
    for ignored in [(0, 0), (1e308, -1e308)]:
        vertices[ignored_index] = ignored
        collection.set_paths([PathPatch(Path(vertices.copy(), codes))])
        with np.errstate(over='raise', invalid='raise'):
            fig.canvas.draw()
            depths.append(collection.do_3d_projection())
        images.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        source = collection.get_paths()[0]
        nptest.assert_array_equal(source.vertices, vertices)
        nptest.assert_array_equal(source.codes, codes)

    assert np.any(images[0][..., :3] != 255)
    assert np.isfinite(depths[0])
    assert depths[0] == depths[1]
    nptest.assert_array_equal(*images)
