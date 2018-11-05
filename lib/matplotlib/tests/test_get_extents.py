import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_equal, assert_array_almost_equal)

from matplotlib import rc_context
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


def color_boxes(fig, axs):
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    bbaxis = []
    for nn, axx in enumerate([axs.xaxis, axs.yaxis]):
        bb = axx.get_tightbbox(renderer)
        if bb:
            axisr = plt.Rectangle((bb.x0, bb.y0), width=bb.width,
                     height=bb.height, linewidth=0.7, edgecolor='y',
                    facecolor="none", transform=None, zorder=3)
            fig.add_artist(axisr)
        bbaxis += [bb]

    bbspines = []
    for nn, a in enumerate(['bottom', 'top', 'left', 'right']):
        bb = axs.spines[a].get_window_extent(renderer)
        if bb:
            spiner = plt.Rectangle((bb.x0, bb.y0), width=bb.width,
                                  height=bb.height, linewidth=0.7,
                                  edgecolor="green", facecolor="none",
                                  transform=None, zorder=3)
            fig.add_artist(spiner)
        bbspines += [bb]

    bb = axs.get_window_extent()
    if bb:
        rect2 = plt.Rectangle((bb.x0, bb.y0), width=bb.width, height=bb.height,
                             linewidth=1.5, edgecolor="magenta",
                             facecolor="none", transform=None, zorder=2)
        fig.add_artist(rect2)
    bbax = bb

    bb2 = axs.get_tightbbox(renderer)
    if bb2:
        rect2 = plt.Rectangle((bb2.x0, bb2.y0), width=bb2.width,
                             height=bb2.height, linewidth=3, edgecolor="red",
                             facecolor="none", transform=None, zorder=1)
        fig.add_artist(rect2)
    bbtb = bb2
    return bbaxis, bbspines, bbax, bbtb


def test_normal_axes():
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        fig.canvas.draw()
        plt.close(fig)
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)

    # test the axis bboxes
    target = [
        [123.375, 75.88888888888886, 983.25, 33.0],
        [85.51388888888889, 99.99999999999997, 53.375, 993.0]
    ]
    for nn, b in enumerate(bbaxis):
        targetbb = mtransforms.Bbox.from_bounds(*target[nn])
        assert_array_almost_equal(b.bounds, targetbb.bounds, decimal=2)

    target = [
        [150.0, 119.999, 930.0, 11.111],
        [150.0, 1080.0, 930.0, 0.0],
        [150.0, 119.9999, 11.111, 960.0],
        [1068.8888, 119.9999, 11.111, 960.0]
    ]
    for nn, b in enumerate(bbspines):
        targetbb = mtransforms.Bbox.from_bounds(*target[nn])
        assert_array_almost_equal(b.bounds, targetbb.bounds, decimal=2)

    target = [150.0, 119.99999999999997, 930.0, 960.0]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    assert_array_almost_equal(bbax.bounds, targetbb.bounds, decimal=2)

    target = [85.5138, 75.88888, 1021.11, 1017.11]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    assert_array_almost_equal(bbtb.bounds, targetbb.bounds, decimal=2)

    # test that get_position roundtrips to get_window_extent
    axbb = ax.get_position().transformed(fig.transFigure).bounds
    assert_array_almost_equal(axbb, ax.get_window_extent().bounds, decimal=2)


def test_nodecorator():
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        fig.canvas.draw()
        ax.set(xticklabels=[], yticklabels=[])
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)

    # test the axis bboxes
    target = [
        None,
        None
    ]
    for nn, b in enumerate(bbaxis):
        assert b is None

    target = [
        [150.0, 119.999, 930.0, 11.111],
        [150.0, 1080.0, 930.0, 0.0],
        [150.0, 119.9999, 11.111, 960.0],
        [1068.8888, 119.9999, 11.111, 960.0]
    ]
    for nn, b in enumerate(bbspines):
        targetbb = mtransforms.Bbox.from_bounds(*target[nn])
        assert_allclose(b.bounds, targetbb.bounds, atol=1e-2)

    target = [150.0, 119.99999999999997, 930.0, 960.0]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    assert_allclose(bbax.bounds, targetbb.bounds, atol=1e-2)

    target = [150., 120., 930., 960.]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    assert_allclose(bbtb.bounds, targetbb.bounds, atol=1e-2)


def test_displaced_spine():
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        ax.set(xticklabels=[], yticklabels=[])
        ax.spines['bottom'].set_position(('axes', -0.1))
        fig.canvas.draw()
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)

    target = [
        [150., 24., 930., 11.111111],
        [150.0, 1080.0, 930.0, 0.0],
        [150.0, 119.9999, 11.111, 960.0],
        [1068.8888, 119.9999, 11.111, 960.0]
    ]
    for nn, b in enumerate(bbspines):
        targetbb = mtransforms.Bbox.from_bounds(*target[nn])

    target = [150.0, 119.99999999999997, 930.0, 960.0]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    assert_allclose(bbax.bounds, targetbb.bounds, atol=1e-2)

    target = [150., 24., 930., 1056.]
    targetbb = mtransforms.Bbox.from_bounds(*target)
    assert_allclose(bbtb.bounds, targetbb.bounds, atol=1e-2)


def test_tickdirs():
    """
    Switch the tickdirs and make sure the bboxes switch with them
    """
    targets = [[[150.0, 120.0, 930.0, 11.1111],
                [150.0, 120.0, 11.111, 960.0]],
               [[150.0, 108.8889, 930.0, 11.111111111111114],
                [138.889, 120, 11.111, 960.0]],
               [[150.0, 114.44444444444441, 930.0, 11.111111111111114],
                [144.44444444444446, 119.999, 11.111, 960.0]]]
    for dnum, dirs in enumerate(['in', 'out', 'inout']):
        with rc_context({'_internal.classic_mode': False}):
            fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
            ax.tick_params(direction=dirs)
            fig.canvas.draw()
            bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
            for nn, num in enumerate([0, 2]):
                targetbb = mtransforms.Bbox.from_bounds(*targets[dnum][nn])
                assert_allclose(bbspines[num].bounds, targetbb.bounds,
                               atol=1e-2)


def test_minor_accountedfor():
    with rc_context({'_internal.classic_mode': False}):
        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        fig.canvas.draw()
        ax.tick_params(which='both', direction='out')

        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
        targets = [[150.0, 108.88888888888886, 930.0, 11.111111111111114],
                   [138.8889, 119.9999, 11.1111, 960.0]]
        for n in range(2):
            targetbb = mtransforms.Bbox.from_bounds(*targets[n])
            assert_allclose(bbspines[n * 2].bounds, targetbb.bounds,
                           atol=1e-2)

        fig, ax = plt.subplots(dpi=200, figsize=(6, 6))
        fig.canvas.draw()
        ax.tick_params(which='both', direction='out')
        ax.minorticks_on()
        ax.tick_params(axis='both', which='minor', length=30)
        fig.canvas.draw()
        bbaxis, bbspines, bbax, bbtb = color_boxes(fig, ax)
        targets = [[150.0, 36.66666666666663, 930.0, 83.33333333333334],
                   [66.6667, 120.0, 83.3333, 960.0]]

        for n in range(2):
            targetbb = mtransforms.Bbox.from_bounds(*targets[n])
            assert_allclose(bbspines[n * 2].bounds, targetbb.bounds,
                           atol=1e-2)
