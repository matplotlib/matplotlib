from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import nose.tools
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal
import numpy as np
import matplotlib

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
    test_value = np.array([-0.8, -0.3, 0.2])
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


def test_LogFormatterExponent():
    class FakeAxis(object):
        """Allow Formatter to be called without having a "full" plot set up."""
        def get_view_interval(self):
            return 1, 10

    i = np.arange(-3, 4, dtype=float)
    expected_result = ['-3', '-2', '-1', '0', '1', '2', '3']
    for base in [2, 5, 10, np.pi, np.e]:
        formatter = mticker.LogFormatterExponent(base=base)
        formatter.axis = FakeAxis()
        vals = base**i
        labels = [formatter(x, pos) for (x, pos) in zip(vals, i)]
        nose.tools.assert_equal(labels, expected_result)

    # Should be a blank string for non-integer powers if labelOnlyBase=True
    formatter = mticker.LogFormatterExponent(base=10, labelOnlyBase=True)
    formatter.axis = FakeAxis()
    nose.tools.assert_equal(formatter(10**0.1), '')

    # Otherwise, non-integer powers should be nicely formatted
    locs = np.array([0.1, 0.00001, np.pi, 0.2, -0.2, -0.00001])
    i = range(len(locs))
    expected_result = ['0.1', '1e-05', '3.14', '0.2', '-0.2', '-1e-05']
    for base in [2, 5, 10, np.pi, np.e]:
        formatter = mticker.LogFormatterExponent(base, labelOnlyBase=False)
        formatter.axis = FakeAxis()
        vals = base**locs
        labels = [formatter(x, pos) for (x, pos) in zip(vals, i)]
        nose.tools.assert_equal(labels, expected_result)


def test_use_offset():
    for use_offset in [True, False]:
        with matplotlib.rc_context({'axes.formatter.useoffset': use_offset}):
            tmp_form = mticker.ScalarFormatter()
            nose.tools.assert_equal(use_offset, tmp_form.get_useOffset())


def test_formatstrformatter():
    # test % style formatter
    tmp_form = mticker.FormatStrFormatter('%05d')
    nose.tools.assert_equal('00002', tmp_form(2))

    # test str.format() style formatter
    tmp_form = mticker.StrMethodFormatter('{x:05d}')
    nose.tools.assert_equal('00002', tmp_form(2))


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
