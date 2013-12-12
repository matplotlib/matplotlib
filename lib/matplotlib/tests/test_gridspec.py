import matplotlib.gridspec as gridspec
from nose.tools import assert_equal


def test_equal():
    gs = gridspec.GridSpec(2, 1)
    assert_equal(gs[0, 0], gs[0, 0])
    assert_equal(gs[:, 0], gs[:, 0])
