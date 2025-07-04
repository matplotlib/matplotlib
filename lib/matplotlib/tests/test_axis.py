import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axis import XTick


def test_tick_labelcolor_array():
    # Smoke test that we can instantiate a Tick with labelcolor as array.
    ax = plt.axes()
    XTick(ax, 0, labelcolor=np.array([1, 0, 0, 1]))


def test_axis_not_in_layout():
    fig1, (ax1_left, ax1_right) = plt.subplots(ncols=2, layout='constrained')
    fig2, (ax2_left, ax2_right) = plt.subplots(ncols=2, layout='constrained')

    # 100 label overlapping the end of the axis
    ax1_left.set_xlim(0, 100)
    # 100 label not overlapping the end of the axis
    ax2_left.set_xlim(0, 120)

    for ax in ax1_left, ax2_left:
        ax.set_xticks([0, 100])
        ax.xaxis.set_in_layout(False)

    for fig in fig1, fig2:
        fig.draw_without_rendering()

    # Positions should not be affected by overlapping 100 label
    assert ax1_left.get_position().bounds == ax2_left.get_position().bounds
    assert ax1_right.get_position().bounds == ax2_right.get_position().bounds


def test_translate_tick_params_reverse():
    fig, ax = plt.subplots()
    kw = {'label1On': 'a', 'label2On': 'b', 'tick1On': 'c', 'tick2On': 'd'}
    assert (ax.xaxis._translate_tick_params(kw, reverse=True) ==
            {'labelbottom': 'a', 'labeltop': 'b', 'bottom': 'c', 'top': 'd'})
    assert (ax.yaxis._translate_tick_params(kw, reverse=True) ==
            {'labelleft': 'a', 'labelright': 'b', 'left': 'c', 'right': 'd'})


def test_get_tick_position_rcParams():
    """Test that get_tick_position() correctly picks up rcParams tick positions."""
    plt.rcParams.update({
        "xtick.top": 1, "xtick.labeltop": 1, "xtick.bottom": 0, "xtick.labelbottom": 0,
        "ytick.right": 1, "ytick.labelright": 1, "ytick.left": 0, "ytick.labelleft": 0,
    })
    ax = plt.figure().add_subplot()
    assert ax.xaxis.get_ticks_position() == "top"
    assert ax.yaxis.get_ticks_position() == "right"


def test_get_tick_position_tick_top_tick_right():
    """Test that get_tick_position() correctly picks up tick_top() / tick_right()."""
    ax = plt.figure().add_subplot()
    ax.xaxis.tick_top()
    ax.yaxis.tick_right()
    assert ax.xaxis.get_ticks_position() == "top"
    assert ax.yaxis.get_ticks_position() == "right"


def test_get_tick_position_tick_params():
    """Test that get_tick_position() correctly picks up tick_params()."""
    ax = plt.figure().add_subplot()
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False,
                   right=True, labelright=True, left=False, labelleft=False)
    assert ax.xaxis.get_ticks_position() == "top"
    assert ax.yaxis.get_ticks_position() == "right"


def test_grid_rcparams():
    """Tests that `grid.major/minor.*` overwrites `grid.*` in rcParams."""
    plt.rcParams.update({
        "axes.grid": True, "axes.grid.which": "both",
        "ytick.minor.visible": True, "xtick.minor.visible": True,
    })
    def_linewidth = plt.rcParams["grid.linewidth"]
    def_linestyle = plt.rcParams["grid.linestyle"]
    def_alpha = plt.rcParams["grid.alpha"]

    plt.rcParams.update({
        "grid.color": "gray", "grid.minor.color": "red",
        "grid.major.linestyle": ":", "grid.major.linewidth": 2,
        "grid.minor.alpha": 0.6,
    })
    _, ax = plt.subplots()
    ax.plot([0, 1])

    assert ax.xaxis.get_major_ticks()[0].gridline.get_color() == "gray"
    assert ax.xaxis.get_minor_ticks()[0].gridline.get_color() == "red"
    assert ax.xaxis.get_major_ticks()[0].gridline.get_linewidth() == 2
    assert ax.xaxis.get_minor_ticks()[0].gridline.get_linewidth() == def_linewidth
    assert ax.xaxis.get_major_ticks()[0].gridline.get_linestyle() == ":"
    assert ax.xaxis.get_minor_ticks()[0].gridline.get_linestyle() == def_linestyle
    assert ax.xaxis.get_major_ticks()[0].gridline.get_alpha() == def_alpha
    assert ax.xaxis.get_minor_ticks()[0].gridline.get_alpha() == 0.6
