from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import numpy as np

from matplotlib.path import Path
from matplotlib.patches import Polygon
from nose.tools import assert_raises, assert_equal
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib import transforms


def test_readonly_path():
    path = Path.unit_circle()

    def modify_vertices():
        path.vertices = path.vertices * 2.0

    assert_raises(AttributeError, modify_vertices)


def test_point_in_path():
    # Test #1787
    verts2 = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]

    path = Path(verts2, closed=True)
    points = [(0.5, 0.5), (1.5, 0.5)]

    assert np.all(path.contains_points(points) == [True, False])


def test_contains_points_negative_radius():
    path = Path.unit_circle()

    points = [(0.0, 0.0), (1.25, 0.0), (0.9, 0.9)]
    expected = [True, False, False]
    result = path.contains_points(points, radius=-0.5)

    assert result.dtype == np.bool
    assert np.all(result == expected)


@image_comparison(baseline_images=['path_clipping'],
                  extensions=['svg'], remove_text=True)
def test_path_clipping():
    fig = plt.figure(figsize=(6.0, 6.2))

    for i, xy in enumerate([
            [(200, 200), (200, 350), (400, 350), (400, 200)],
            [(200, 200), (200, 350), (400, 350), (400, 100)],
            [(200, 100), (200, 350), (400, 350), (400, 100)],
            [(200, 100), (200, 415), (400, 350), (400, 100)],
            [(200, 100), (200, 415), (400, 415), (400, 100)],
            [(200, 415), (400, 415), (400, 100), (200, 100)],
            [(400, 415), (400, 100), (200, 100), (200, 415)]]):
        ax = fig.add_subplot(4, 2, i+1)
        bbox = [0, 140, 640, 260]
        ax.set_xlim(bbox[0], bbox[0] + bbox[2])
        ax.set_ylim(bbox[1], bbox[1] + bbox[3])
        ax.add_patch(Polygon(
            xy, facecolor='none', edgecolor='red', closed=True))


def test_point_in_path_nan():
    box = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    p = Path(box)
    test = np.array([[np.nan, 0.5]])
    contains = p.contains_points(test)
    assert len(contains) == 1
    assert not contains[0]


@image_comparison(baseline_images=['semi_log_with_zero'], extensions=['png'])
def test_log_transform_with_zero():
    x = np.arange(-10, 10)
    y = (1.0 - 1.0/(x**2+1))**20

    fig, ax = plt.subplots()
    ax.semilogy(x, y, "-o", lw=15)
    ax.grid(True)


def test_make_compound_path_empty():
    # We should be able to make a compound path with no arguments.
    # This makes it easier to write generic path based code.
    r = Path.make_compound_path()
    assert_equal(r.vertices.shape, (0, 2))


@image_comparison(baseline_images=['xkcd'], remove_text=True)
def test_xkcd():
    x = np.linspace(0, 2.0 * np.pi, 100.0)
    y = np.sin(x)

    with plt.xkcd():
        fig, ax = plt.subplots()
        ax.plot(x, y)


@image_comparison(baseline_images=['marker_paths'], extensions=['pdf'],
                  remove_text=True)
def test_marker_paths_pdf():
    N = 7

    plt.errorbar(np.arange(N),
                 np.ones(N) + 4,
                 np.ones(N))
    plt.xlim(-1, N)
    plt.ylim(-1, 7)


def test_path_no_doubled_point_in_to_polygon():
    hand = np.array(
        [[1.64516129, 1.16145833],
         [1.64516129, 1.59375],
         [1.35080645, 1.921875],
         [1.375, 2.18229167],
         [1.68548387, 1.9375],
         [1.60887097, 2.55208333],
         [1.68548387, 2.69791667],
         [1.76209677, 2.56770833],
         [1.83064516, 1.97395833],
         [1.89516129, 2.75],
         [1.9516129, 2.84895833],
         [2.01209677, 2.76041667],
         [1.99193548, 1.99479167],
         [2.11290323, 2.63020833],
         [2.2016129, 2.734375],
         [2.25403226, 2.60416667],
         [2.14919355, 1.953125],
         [2.30645161, 2.36979167],
         [2.39112903, 2.36979167],
         [2.41532258, 2.1875],
         [2.1733871, 1.703125],
         [2.07782258, 1.16666667]])

    (r0, c0, r1, c1) = (1.0, 1.5, 2.1, 2.5)

    poly = Path(np.vstack((hand[:, 1], hand[:, 0])).T, closed=True)
    clip_rect = transforms.Bbox([[r0, c0], [r1, c1]])
    poly_clipped = poly.clip_to_bbox(clip_rect).to_polygons()[0]

    assert np.all(poly_clipped[-2] != poly_clipped[-1])
    assert np.all(poly_clipped[-1] == poly_clipped[0])


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
