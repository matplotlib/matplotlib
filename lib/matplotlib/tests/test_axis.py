import numpy as np
import pytest

import matplotlib.pyplot as plt
from matplotlib.axis import XTick


def test_tick_labelcolor_array():
    # Smoke test that we can instantiate a Tick with labelcolor as array.
    ax = plt.axes()
    XTick(ax, 0, labelcolor=np.array([1, 0, 0, 1]))


@pytest.mark.parametrize("which", ["major", "minor"])
@pytest.mark.parametrize(
    "axis, expected_basic",
    [
        (
            "x",
            {
                "bottom": True,
                "top": True,  # tests use classic.mplstyle with top = True
                "labelbottom": True,
                "labeltop": False,
                "gridOn": False,
            },
        ),
        (
            "y",
            {
                "left": True,
                "right": True,  # tests use classic.mplstyle with right = True
                "labelleft": True,
                "labelright": False,
                "gridOn": False,
            },
        ),
    ],
    ids=["xaxis", "yaxis"],
)
def test_axis_get_tick_params(axis, expected_basic, which):
    axis = getattr(plt.subplot(), axis + "axis")
    additional_kw = {"labelsize": 42}

    axis.set_tick_params(which=which, **additional_kw)
    expected = {**expected_basic, **additional_kw}
    assert expected == axis.get_tick_params(which=which)
    other = "minor" if which == "major" else "major"
    assert expected_basic == axis.get_tick_params(which=other)


def test_axis_set_tick_params_labelsize_labelcolor():
    # Tests fix for issue 4346
    axis_1 = plt.subplot()
    axis_1.yaxis.set_tick_params(labelsize=30, labelcolor='red',
                                 direction='out')

    # Expected values after setting the ticks
    assert axis_1.yaxis.majorTicks[0]._size == 4.0
    assert axis_1.yaxis.majorTicks[0].tick1line.get_color() == "k"
    assert axis_1.yaxis.majorTicks[0].label1.get_size() == 30.0
    assert axis_1.yaxis.majorTicks[0].label1.get_color() == "red"
