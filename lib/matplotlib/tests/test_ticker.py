from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
import nose.tools
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.testing.decorators import cleanup

import warnings


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


@cleanup
def test_AutoMinorLocator():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1.39)
    ax.minorticks_on()
    test_value = np.array([0.05, 0.1, 0.15, 0.25, 0.3, 0.35, 0.45,
                           0.5, 0.55, 0.65, 0.7, 0.75, 0.85, 0.9,
                           0.95, 1, 1.05, 1.1, 1.15, 1.25, 1.3, 1.35])
    assert_almost_equal(ax.xaxis.get_ticklocs(minor=True), test_value)


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


def test_LinearLocator_set_params():
    """
    Create linear locator with presets={}, numticks=2 and change it to
    something else. See if change was successful. Should not exception.
    """
    loc = mticker.LinearLocator(numticks=2)
    loc.set_params(numticks=8, presets={(0, 1): []})
    nose.tools.assert_equal(loc.numticks, 8)
    nose.tools.assert_equal(loc.presets, {(0, 1): []})


def test_LogLocator_set_params():
    """
    Create log locator with default value, base=10.0, subs=[1.0], numdecs=4,
    numticks=15 and change it to something else.
    See if change was successful.
    Should not exception.
    """
    loc = mticker.LogLocator()
    loc.set_params(numticks=8, numdecs=8, subs=[2.0], base=8)
    nose.tools.assert_equal(loc.numticks, 8)
    nose.tools.assert_equal(loc.numdecs, 8)
    nose.tools.assert_equal(loc.base, 8)
    nose.tools.assert_equal(loc.subs, [2.0])


def test_NullLocator_set_params():
    """
    Create null locator, and attempt to call set_params() on it.
    Should not exception, and should raise a warning.
    """
    loc = mticker.NullLocator()
    with warnings.catch_warnings(record=True) as w:
        loc.set_params()
        nose.tools.assert_equal(len(w), 1)


def test_MultipleLocator_set_params():
    """
    Create multiple locator with 0.7 base, and change it to something else.
    See if change was successful.
    Should not exception.
    """
    mult = mticker.MultipleLocator(base=0.7)
    mult.set_params(base=1.7)
    nose.tools.assert_equal(mult._base, 1.7)


def test_LogitLocator_set_params():
    """
    Create logit locator with default minor=False, and change it to something
    else. See if change was successful. Should not exception.
    """
    loc = mticker.LogitLocator()  # Defaults to false.
    loc.set_params(minor=True)
    nose.tools.assert_true(loc.minor)


def test_FixedLocator_set_params():
    """
    Create fixed locator with 5 nbins, and change it to something else.
    See if change was successful.
    Should not exception.
    """
    fixed = mticker.FixedLocator(range(0, 24), nbins=5)
    fixed.set_params(nbins=7)
    nose.tools.assert_equal(fixed.nbins, 7)


def test_IndexLocator_set_params():
    """
    Create index locator with 3 base, 4 offset. and change it to something
    else. See if change was successful.
    Should not exception.
    """
    index = mticker.IndexLocator(base=3, offset=4)
    index.set_params(base=7, offset=7)
    nose.tools.assert_equal(index._base, 7)
    nose.tools.assert_equal(index.offset, 7)


def test_SymmetricalLogLocator_set_params():
    """
    Create symmetrical log locator with default subs =[1.0] numticks = 15,
    and change it to something else.
    See if change was successful.
    Should not exception.
    """
    # since we only test for the params change. I will pass empty transform
    sym = mticker.SymmetricalLogLocator(None)
    sym.set_params(subs=[2.0], numticks=8)
    nose.tools.assert_equal(sym._subs, [2.0])
    nose.tools.assert_equal(sym.numticks, 8)


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
