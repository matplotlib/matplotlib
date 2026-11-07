import math

import numpy as np
from numpy.testing import assert_array_equal

import pytest

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal
from matplotlib import (
    collections as mcollections, patches as mpatches, path as mpath)


pytestmark = pytest.mark.backend('cairo')


@check_figures_equal()
def test_patch_alpha_coloring(fig_test, fig_ref):
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

    # Reference: two separate patches
    ax = fig_ref.subplots()
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    patch = mpatches.PathPatch(cut_star1,
                               linewidth=5, linestyle='dashdot',
                               facecolor=(1, 0, 0, 0.5),
                               edgecolor=(0, 0, 1, 0.75))
    ax.add_patch(patch)
    patch = mpatches.PathPatch(cut_star2,
                               linewidth=5, linestyle='dashdot',
                               facecolor=(1, 0, 0, 0.5),
                               edgecolor=(0, 0, 1, 0.75))
    ax.add_patch(patch)

    # Test: path collection
    ax = fig_test.subplots()
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    col = mcollections.PathCollection([cut_star1, cut_star2],
                                      linewidth=5, linestyles='dashdot',
                                      facecolor=(1, 0, 0, 0.5),
                                      edgecolor=(0, 0, 1, 0.75))
    ax.add_collection(col)

    # Have pyplot manage the figures to ensure the cairo backend is used
    plt.figure(fig_ref)
    plt.figure(fig_test)


def test_copy_from_bbox_fractional():
    # A fractional bbox must save every pixel it touches, including the edge
    # pixels it only partially covers; otherwise blitting never repairs them.
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100, layout='constrained')
    surface = fig.canvas._get_printed_image_surface()
    assert ax.bbox.x1 % 1 and ax.bbox.y1 % 1  # fractional edges
    sw, sh = surface.get_width(), surface.get_height()
    data = np.frombuffer(surface.get_data(), np.uint32).reshape((sh, sw))
    before = data.copy()

    region = fig.canvas.copy_from_bbox(ax.bbox)
    data[:] = 0  # scribble over the whole surface, then repair the Axes
    surface.mark_dirty()
    fig.canvas.restore_region(region)

    sl = (slice(sh - math.ceil(ax.bbox.y1), sh - math.floor(ax.bbox.y0)),
          slice(math.floor(ax.bbox.x0), math.ceil(ax.bbox.x1)))
    assert_array_equal(data[sl], before[sl])
