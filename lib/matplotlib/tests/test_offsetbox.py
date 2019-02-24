import pytest
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.offsetbox import (
        AnchoredOffsetbox, DrawingArea, _get_packed_offsets)


@image_comparison(baseline_images=['offsetbox_clipping'], remove_text=True)
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


_loc_values = [['NW', 'N', 'NE', 'W', 'C', 'E', 'SW', 'S', 'SE'],
               ['northwest', 'north', 'northeast', 'west', 'center',
                'east', 'southwest', 'south', 'southeast'],
               ['upper left', 'upper center', 'upper right', 'center left',
                'center', 'right', 'lower left', 'lower center',
                'lower right']]


@pytest.mark.parametrize('locs', _loc_values)
def test_offsetbox_loc_codes(locs):

    codes = [2, 9, 1, 6, 10, 7, 3, 8, 4]

    fig1, axes1 = plt.subplots(3, 3)
    fig2, axes2 = plt.subplots(3, 3)

    for ax, loc in zip(axes1.flat, locs):
        da = DrawingArea(50, 50)
        anchored_box = AnchoredOffsetbox(loc=loc, child=da)
        ax.add_artist(anchored_box)

    for ax, loc in zip(axes2.flat, codes):
        da = DrawingArea(50, 50)
        anchored_box = AnchoredOffsetbox(loc="NW", child=da)
        ax.add_artist(anchored_box)
        anchored_box.set_loc(loc)

    for ax1, ax2 in zip(axes1.flat, axes2.flat):
        ab1, = ax1.findobj(match=AnchoredOffsetbox)
        ab2, = ax2.findobj(match=AnchoredOffsetbox)
        assert ab1.get_loc() == ab2.get_loc()
    fig1.canvas.draw()
    fig2.canvas.draw()


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
    _get_packed_offsets(wd_list, total, sep, mode=mode)
