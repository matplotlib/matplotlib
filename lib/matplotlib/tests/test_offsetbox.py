from collections import namedtuple
import numpy.testing as nptest
import pytest
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.offsetbox import (
        AnchoredOffsetbox, DrawingArea, _get_packed_offsets)


@image_comparison(['offsetbox_clipping'], remove_text=True)
def test_offsetbox_clipping():
    # - create a plot
    # - put an AnchoredOffsetbox with a child DrawingArea
    #   at the center of the axes
    # - give the DrawingArea a gray background
    # - put a black line across the bounds of the DrawingArea
    # - see that the black line is clipped to the edges of
    #   the DrawingArea.
    fig, ax = plt.subplots()
    size = 100
    da = DrawingArea(size, size, clip=True)
    bg = mpatches.Rectangle((0, 0), size, size,
                            facecolor='#CCCCCC',
                            edgecolor='None',
                            linewidth=0)
    line = mlines.Line2D([-size*.5, size*1.5], [size/2, size/2],
                         color='black',
                         linewidth=10)
    anchored_box = AnchoredOffsetbox(
        loc='center',
        child=da,
        pad=0.,
        frameon=False,
        bbox_to_anchor=(.5, .5),
        bbox_transform=ax.transAxes,
        borderpad=0.)

    da.add_artist(bg)
    da.add_artist(line)
    ax.add_artist(anchored_box)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))


def test_offsetbox_clip_children():
    # - create a plot
    # - put an AnchoredOffsetbox with a child DrawingArea
    #   at the center of the axes
    # - give the DrawingArea a gray background
    # - put a black line across the bounds of the DrawingArea
    # - see that the black line is clipped to the edges of
    #   the DrawingArea.
    fig, ax = plt.subplots()
    size = 100
    da = DrawingArea(size, size, clip=True)
    bg = mpatches.Rectangle((0, 0), size, size,
                            facecolor='#CCCCCC',
                            edgecolor='None',
                            linewidth=0)
    line = mlines.Line2D([-size*.5, size*1.5], [size/2, size/2],
                         color='black',
                         linewidth=10)
    anchored_box = AnchoredOffsetbox(
        loc='center',
        child=da,
        pad=0.,
        frameon=False,
        bbox_to_anchor=(.5, .5),
        bbox_transform=ax.transAxes,
        borderpad=0.)

    da.add_artist(bg)
    da.add_artist(line)
    ax.add_artist(anchored_box)

    fig.canvas.draw()
    assert not fig.stale
    da.clip_children = True
    assert fig.stale


def test_offsetbox_loc_codes():
    # Check that valid string location codes all work with an AnchoredOffsetbox
    codes = {'upper right': 1,
             'upper left': 2,
             'lower left': 3,
             'lower right': 4,
             'right': 5,
             'center left': 6,
             'center right': 7,
             'lower center': 8,
             'upper center': 9,
             'center': 10,
             }
    fig, ax = plt.subplots()
    da = DrawingArea(100, 100)
    for code in codes:
        anchored_box = AnchoredOffsetbox(loc=code, child=da)
        ax.add_artist(anchored_box)
    fig.canvas.draw()


def test_expand_with_tight_layout():
    # Check issue reported in #10476, and updated due to #10784
    fig, ax = plt.subplots()

    d1 = [1, 2]
    d2 = [2, 1]
    ax.plot(d1, label='series 1')
    ax.plot(d2, label='series 2')
    ax.legend(ncol=2, mode='expand')

    fig.tight_layout()  # where the crash used to happen


@pytest.mark.parametrize('wd_list',
                         ([(150, 1)], [(150, 1)]*3, [(0.1, 1)], [(0.1, 1)]*2))
@pytest.mark.parametrize('total', (250, 100, 0, -1, None))
@pytest.mark.parametrize('sep', (250, 1, 0, -1))
@pytest.mark.parametrize('mode', ("expand", "fixed", "equal"))
def test_get_packed_offsets(wd_list, total, sep, mode):
    # Check a (rather arbitrary) set of parameters due to successive similar
    # issue tickets (at least #10476 and #10784) related to corner cases
    # triggered inside this function when calling higher-level functions
    # (e.g. `Axes.legend`).
    # These are just some additional smoke tests. The output is untested.
    _get_packed_offsets(wd_list, total, sep, mode=mode)


_Params = namedtuple('_params', 'wd_list, total, sep, expected')


@pytest.mark.parametrize('wd_list, total, sep, expected', [
    _Params(  # total=None
        [(3, 0), (1, 0), (2, 0)], total=None, sep=1, expected=(8, [0, 4, 6])),
    _Params(  # total larger than required
        [(3, 0), (1, 0), (2, 0)], total=10, sep=1, expected=(10, [0, 4, 6])),
    _Params(  # total smaller than required
        [(3, 0), (1, 0), (2, 0)], total=5, sep=1, expected=(5, [0, 4, 6])),
])
def test_get_packed_offsets_fixed(wd_list, total, sep, expected):
    result = _get_packed_offsets(wd_list, total, sep, mode='fixed')
    assert result[0] == expected[0]
    nptest.assert_allclose(result[1], expected[1])


@pytest.mark.parametrize('wd_list, total, sep, expected', [
    _Params(  # total=None (implicit 1)
        [(.1, 0)] * 3, total=None, sep=None, expected=(1, [0, .45, .9])),
    _Params(  # total larger than sum of widths
        [(3, 0), (1, 0), (2, 0)], total=10, sep=1, expected=(10, [0, 5, 8])),
    _Params(  # total smaller sum of widths: overlapping boxes
        [(3, 0), (1, 0), (2, 0)], total=5, sep=1, expected=(5, [0, 2.5, 3])),
])
def test_get_packed_offsets_expand(wd_list, total, sep, expected):
    result = _get_packed_offsets(wd_list, total, sep, mode='expand')
    assert result[0] == expected[0]
    nptest.assert_allclose(result[1], expected[1])


@pytest.mark.parametrize('wd_list, total, sep, expected', [
    _Params(  # total larger than required
        [(3, 0), (2, 0), (1, 0)], total=6, sep=None, expected=(6, [0, 2, 4])),
    _Params(  # total smaller sum of widths: overlapping boxes
        [(3, 0), (2, 0), (1, 0), (.5, 0)], total=2, sep=None,
        expected=(2, [0, 0.5, 1, 1.5])),
    _Params(  # total larger than required
        [(.5, 0), (1, 0), (.2, 0)], total=None, sep=1,
        expected=(6, [0, 2, 4])),
    # the case total=None, sep=None is tested separately below
])
def test_get_packed_offsets_equal(wd_list, total, sep, expected):
    result = _get_packed_offsets(wd_list, total, sep, mode='equal')
    assert result[0] == expected[0]
    nptest.assert_allclose(result[1], expected[1])


def test_get_packed_offsets_equal_total_none_sep_none():
    with pytest.raises(ValueError):
        _get_packed_offsets([(1, 0)] * 3, total=None, sep=None, mode='equal')
