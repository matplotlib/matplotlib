from datetime import timedelta

import numpy as np
import pytest

from matplotlib.testing.decorators import check_figures_equal
import matplotlib.pyplot as plt


@pytest.mark.parametrize('plot_method', ['plot', 'scatter'])
@pytest.mark.parametrize('unit', ['seconds', 'minutes'])
@pytest.mark.parametrize(
    'ydata',
    [[timedelta(seconds=1), timedelta(seconds=2)],
     np.array([1, 2]) * np.timedelta64(1, 's')],
    ids=('Python timedelta', 'numpy timedelta')
)
@check_figures_equal(extensions=["png"])
def test_timedelta_plotting(fig_test, fig_ref, plot_method, ydata, unit):
    """
    Compare a plot of timedeltas to a manually constructed plot that
    should look identical.

    Exercises:
    - Different input types (builtin and numpy)
    - Different axis units
    """
    x = [3, 4]
    ax_test = fig_test.subplots()
    ax_test.yaxis.set_units(unit)
    getattr(ax_test, plot_method)(x, ydata)

    ax_ref = fig_ref.subplots()
    y = np.array([1, 2])
    if unit == 'minutes':
        y = y / 60
    getattr(ax_ref, plot_method)(x, y)
    ax_ref.set_ylabel(unit.capitalize())


@pytest.mark.parametrize(
    'xdata',
    [[np.timedelta64(1, 'h'), np.timedelta64(120, 'm')],
     [timedelta(hours=1), timedelta(minutes=120)]]
)
@check_figures_equal(extensions=["png"])
def test_different_units(fig_test, fig_ref, xdata):
    """
    Check that plotting objects with different time units
    works as expected.
    """
    y = [3, 4]
    ax_test = fig_test.subplots()
    ax_test.xaxis.set_units('hours')
    ax_test.scatter(xdata[0], y[0])
    ax_test.scatter(xdata[1], y[1])

    ax_ref = fig_ref.subplots()
    ax_ref.scatter([1, 2], y)


def test_invalid_unit():
    """
    Check that an error is raised when an invalid unit is set.
    """
    x = [3, 4]
    y = [timedelta(seconds=1), timedelta(seconds=2)]
    fig, ax = plt.subplots()
    ax.yaxis.set_units('invalid_unit')
    msg = r"'invalid_unit' is not a valid value for unit"
    with pytest.raises(ValueError, match=msg):
        ax.plot(x, y)
