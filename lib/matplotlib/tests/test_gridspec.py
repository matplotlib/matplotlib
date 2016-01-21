import matplotlib.gridspec as gridspec
from nose.tools import assert_raises, assert_equal


def test_equal():
    gs = gridspec.GridSpec(2, 1)
    assert_equal(gs[0, 0], gs[0, 0])
    assert_equal(gs[:, 0], gs[:, 0])


def test_ratios():
    assert_raises(ValueError, gridspec.GridSpec,
                  1, 2, width_ratios=[2, 1, 3], height_ratios=[3, 4])
