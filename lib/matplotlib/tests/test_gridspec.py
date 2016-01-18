import matplotlib.gridspec as gridspec
import numpy


def test_equal():
    gs = gridspec.GridSpec(2, 1)
    assert gs[0, 0] == gs[0, 0]
    assert gs[:, 0] == gs[:, 0]
