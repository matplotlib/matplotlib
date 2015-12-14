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


def _logfe_helper(formatter, base, locs, i, expected_result):
    vals = base**locs
    labels = [formatter(x, pos) for (x, pos) in zip(vals, i)]
    nose.tools.assert_equal(labels, expected_result)


def test_LogFormatterExponent():
    class FakeAxis(object):
        """Allow Formatter to be called without having a "full" plot set up."""
        def __init__(self, vmin=1, vmax=10):
            self.vmin = vmin
            self.vmax = vmax

        def get_view_interval(self):
            return self.vmin, self.vmax

    i = np.arange(-3, 4, dtype=float)
    expected_result = ['-3', '-2', '-1', '0', '1', '2', '3']
    for base in [2, 5.0, 10.0, np.pi, np.e]:
        formatter = mticker.LogFormatterExponent(base=base)
        formatter.axis = FakeAxis(1, base**4)
        yield _logfe_helper, formatter, base, i, i, expected_result

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
        formatter.axis = FakeAxis(1, base**10)
        yield _logfe_helper, formatter, base, locs, i, expected_result

    expected_result = ['3', '5', '12', '42']
    locs = np.array([3, 5, 12, 42], dtype='float')
    for base in [2, 5.0, 10.0, np.pi, np.e]:
        formatter = mticker.LogFormatterExponent(base, labelOnlyBase=False)
        formatter.axis = FakeAxis(1, base**50)
        yield _logfe_helper, formatter, base, locs, i, expected_result


def _pprint_helper(value, domain, expected):
    fmt = mticker.LogFormatter()
    label = fmt.pprint_val(value, domain)
    nose.tools.assert_equal(label, expected)


def test_logformatter_pprint():
    test_cases = (
        (3.141592654e-05, 0.001, '3.142e-5'),
        (0.0003141592654, 0.001, '3.142e-4'),
        (0.003141592654, 0.001, '3.142e-3'),
        (0.03141592654, 0.001, '3.142e-2'),
        (0.3141592654, 0.001, '3.142e-1'),
        (3.141592654, 0.001, '3.142'),
        (31.41592654, 0.001, '3.142e1'),
        (314.1592654, 0.001, '3.142e2'),
        (3141.592654, 0.001, '3.142e3'),
        (31415.92654, 0.001, '3.142e4'),
        (314159.2654, 0.001, '3.142e5'),
        (1e-05, 0.001, '1e-5'),
        (0.0001, 0.001, '1e-4'),
        (0.001, 0.001, '1e-3'),
        (0.01, 0.001, '1e-2'),
        (0.1, 0.001, '1e-1'),
        (1, 0.001, '1'),
        (10, 0.001, '10'),
        (100, 0.001, '100'),
        (1000, 0.001, '1000'),
        (10000, 0.001, '1e4'),
        (100000, 0.001, '1e5'),
        (3.141592654e-05, 0.015, '0'),
        (0.0003141592654, 0.015, '0'),
        (0.003141592654, 0.015, '0.003'),
        (0.03141592654, 0.015, '0.031'),
        (0.3141592654, 0.015, '0.314'),
        (3.141592654, 0.015, '3.142'),
        (31.41592654, 0.015, '31.416'),
        (314.1592654, 0.015, '314.159'),
        (3141.592654, 0.015, '3141.593'),
        (31415.92654, 0.015, '31415.927'),
        (314159.2654, 0.015, '314159.265'),
        (1e-05, 0.015, '0'),
        (0.0001, 0.015, '0'),
        (0.001, 0.015, '0.001'),
        (0.01, 0.015, '0.01'),
        (0.1, 0.015, '0.1'),
        (1, 0.015, '1'),
        (10, 0.015, '10'),
        (100, 0.015, '100'),
        (1000, 0.015, '1000'),
        (10000, 0.015, '10000'),
        (100000, 0.015, '100000'),
        (3.141592654e-05, 0.5, '0'),
        (0.0003141592654, 0.5, '0'),
        (0.003141592654, 0.5, '0.003'),
        (0.03141592654, 0.5, '0.031'),
        (0.3141592654, 0.5, '0.314'),
        (3.141592654, 0.5, '3.142'),
        (31.41592654, 0.5, '31.416'),
        (314.1592654, 0.5, '314.159'),
        (3141.592654, 0.5, '3141.593'),
        (31415.92654, 0.5, '31415.927'),
        (314159.2654, 0.5, '314159.265'),
        (1e-05, 0.5, '0'),
        (0.0001, 0.5, '0'),
        (0.001, 0.5, '0.001'),
        (0.01, 0.5, '0.01'),
        (0.1, 0.5, '0.1'),
        (1, 0.5, '1'),
        (10, 0.5, '10'),
        (100, 0.5, '100'),
        (1000, 0.5, '1000'),
        (10000, 0.5, '10000'),
        (100000, 0.5, '100000'),
        (3.141592654e-05, 5, '0'),
        (0.0003141592654, 5, '0'),
        (0.003141592654, 5, '0'),
        (0.03141592654, 5, '0.03'),
        (0.3141592654, 5, '0.31'),
        (3.141592654, 5, '3.14'),
        (31.41592654, 5, '31.42'),
        (314.1592654, 5, '314.16'),
        (3141.592654, 5, '3141.59'),
        (31415.92654, 5, '31415.93'),
        (314159.2654, 5, '314159.27'),
        (1e-05, 5, '0'),
        (0.0001, 5, '0'),
        (0.001, 5, '0'),
        (0.01, 5, '0.01'),
        (0.1, 5, '0.1'),
        (1, 5, '1'),
        (10, 5, '10'),
        (100, 5, '100'),
        (1000, 5, '1000'),
        (10000, 5, '10000'),
        (100000, 5, '100000'),
        (3.141592654e-05, 100, '0'),
        (0.0003141592654, 100, '0'),
        (0.003141592654, 100, '0'),
        (0.03141592654, 100, '0'),
        (0.3141592654, 100, '0.3'),
        (3.141592654, 100, '3.1'),
        (31.41592654, 100, '31.4'),
        (314.1592654, 100, '314.2'),
        (3141.592654, 100, '3141.6'),
        (31415.92654, 100, '31415.9'),
        (314159.2654, 100, '314159.3'),
        (1e-05, 100, '0'),
        (0.0001, 100, '0'),
        (0.001, 100, '0'),
        (0.01, 100, '0'),
        (0.1, 100, '0.1'),
        (1, 100, '1'),
        (10, 100, '10'),
        (100, 100, '100'),
        (1000, 100, '1000'),
        (10000, 100, '10000'),
        (100000, 100, '100000'),
        (3.141592654e-05, 1000000.0, '3.1e-5'),
        (0.0003141592654, 1000000.0, '3.1e-4'),
        (0.003141592654, 1000000.0, '3.1e-3'),
        (0.03141592654, 1000000.0, '3.1e-2'),
        (0.3141592654, 1000000.0, '3.1e-1'),
        (3.141592654, 1000000.0, '3.1'),
        (31.41592654, 1000000.0, '3.1e1'),
        (314.1592654, 1000000.0, '3.1e2'),
        (3141.592654, 1000000.0, '3.1e3'),
        (31415.92654, 1000000.0, '3.1e4'),
        (314159.2654, 1000000.0, '3.1e5'),
        (1e-05, 1000000.0, '1e-5'),
        (0.0001, 1000000.0, '1e-4'),
        (0.001, 1000000.0, '1e-3'),
        (0.01, 1000000.0, '1e-2'),
        (0.1, 1000000.0, '1e-1'),
        (1, 1000000.0, '1'),
        (10, 1000000.0, '10'),
        (100, 1000000.0, '100'),
        (1000, 1000000.0, '1000'),
        (10000, 1000000.0, '1e4'),
        (100000, 1000000.0, '1e5')
    )

    for value, domain, expected in test_cases:
        yield _pprint_helper, value, domain, expected


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
