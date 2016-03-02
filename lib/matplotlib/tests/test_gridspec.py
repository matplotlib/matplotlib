import matplotlib.gridspec as gridspec
from nose.tools import assert_raises, assert_equal


def test_equal():
    gs = gridspec.GridSpec(2, 1)
    assert_equal(gs[0, 0], gs[0, 0])
    assert_equal(gs[:, 0], gs[:, 0])


def test_width_ratios():
    """
    Addresses issue #5835.
    See at https://github.com/matplotlib/matplotlib/issues/5835.
    """
    assert_raises(ValueError, gridspec.GridSpec,
                  1, 1, width_ratios=[2, 1, 3])


def test_height_ratios():
    """
    Addresses issue #5835.
    See at https://github.com/matplotlib/matplotlib/issues/5835.
    """
    assert_raises(ValueError, gridspec.GridSpec,
                  1, 1, height_ratios=[2, 1, 3])
