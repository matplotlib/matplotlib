from __future__ import print_function
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal
import numpy as np

import matplotlib.ticker as mticker


def test_MaxNLocator():
    loc = mticker.MaxNLocator(nbins=5)
    test_value = np.array([20., 40., 60., 80., 100.])
    assert_almost_equal(loc.tick_values(20, 100), test_value)

    test_value = np.array([0., 0.0002, 0.0004, 0.0006, 0.0008, 0.001])
    assert_almost_equal(loc.tick_values(0.001, 0.0001), test_value)

    test_value = np.array([-1.0e+15, -5.0e+14, 0e+00, 5e+14, 1.0e+15])
    assert_almost_equal(loc.tick_values(-1e15, 1e15), test_value)


def test_LinearLocator():
    loc = mticker.LinearLocator(numticks=3)
    test_value = np.array([-0.8, -0.3,  0.2])
    assert_almost_equal(loc.tick_values(-0.8, 0.2), test_value)


def test_MultipleLocator():
    loc = mticker.MultipleLocator(base=3.147)
    test_value = np.array([-9.441, -6.294, -3.147, 0., 3.147, 6.294,
                           9.441, 12.588])
    assert_almost_equal(loc.tick_values(-7, 10), test_value)


def test_LogLocator():
    loc = mticker.LogLocator(numticks=5)

    assert_raises(ValueError, loc.tick_values, 0, 1000)

    test_value = np.array([1.00000000e-05, 1.00000000e-03, 1.00000000e-01,
                           1.00000000e+01, 1.00000000e+03, 1.00000000e+05,
                           1.00000000e+07, 1.000000000e+09])
    assert_almost_equal(loc.tick_values(0.001, 1.1e5), test_value)

    loc = mticker.LogLocator(base=2)
    test_value = np.array([0.5, 1., 2., 4., 8., 16., 32., 64., 128., 256.])
    assert_almost_equal(loc.tick_values(1, 100), test_value)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
