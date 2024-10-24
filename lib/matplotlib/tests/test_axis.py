import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axis import XTick, XAxis


def test_tick_labelcolor_array():
    # Smoke test that we can instantiate a Tick with labelcolor as array.
    ax = plt.axes()
    XTick(ax, 0, labelcolor=np.array([1, 0, 0, 1]))


def test_axis_not_in_layout():
    fig1, (ax1_left, ax1_right) = plt.subplots(ncols=2, layout='constrained')
    fig2, (ax2_left, ax2_right) = plt.subplots(ncols=2, layout='constrained')

    # 100 label overlapping the end of the axis
    ax1_left.set_xlim([0, 100])
    # 100 label not overlapping the end of the axis
    ax2_left.set_xlim([0, 120])

    for ax in ax1_left, ax2_left:
        ax.set_xticks([0, 100])
        ax.xaxis.set_in_layout(False)

    for fig in fig1, fig2:
        fig.draw_without_rendering()

    # Positions should not be affected by overlapping 100 label
    assert ax1_left.get_position().bounds == ax2_left.get_position().bounds
    assert ax1_right.get_position().bounds == ax2_right.get_position().bounds


def test__translate_tick_params():
    fig, ax = plt.subplots()
    xaxis = XAxis(ax)
    kw = {'label1On': 'dummy_string_1', 'label2On': 'dummy_string_2',
          'tick1On': 'dummy_string_3', 'tick2On': 'dummy_string_4'}
    result = xaxis._translate_tick_params(kw, reverse=True)
    assert result['labelbottom'] == 'dummy_string_1'
    assert result['labeltop'] == 'dummy_string_2'
    assert result['bottom'] == 'dummy_string_3'
    assert result['top'] == 'dummy_string_4'
