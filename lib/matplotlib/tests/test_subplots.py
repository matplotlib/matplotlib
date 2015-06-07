from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import warnings
from matplotlib.externals import six
from matplotlib.externals.six.moves import xrange

import numpy
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison, cleanup

from nose.tools import assert_raises


def check_shared(results, f, axs):
    """
    results is a 4 x 4 x 2 matrix of boolean values where
        if [i, j, 0] == True, X axis for subplots i and j should be shared
        if [i, j, 1] == False, Y axis for subplots i and j should not be shared
    """
    shared_str = ['x', 'y']
    shared = [axs[0]._shared_x_axes, axs[0]._shared_y_axes]
    #shared = {
    #        'x': a1._shared_x_axes,
    #        'y': a1._shared_y_axes,
    #        }
    tostr = lambda r: "not " if r else ""
    for i1 in xrange(len(axs)):
        for i2 in xrange(i1 + 1, len(axs)):
            for i3 in xrange(len(shared)):
                assert shared[i3].joined(axs[i1], axs[i2]) == \
                        results[i1, i2, i3], \
                        "axes %i and %i incorrectly %ssharing %s axis" % \
                        (i1, i2, tostr(results[i1, i2, i3]), shared_str[i3])


def check_visible(result, f, axs):
    tostr = lambda v: "invisible" if v else "visible"
    for (ax, vx, vy) in zip(axs, result['x'], result['y']):
        for l in ax.get_xticklabels():
            assert l.get_visible() == vx, \
                    "X axis was incorrectly %s" % (tostr(vx))
        for l in ax.get_yticklabels():
            assert l.get_visible() == vy, \
                    "Y axis was incorrectly %s" % (tostr(vy))


def test_shared():
    rdim = (4, 4, 2)
    share = {
            'all': numpy.ones(rdim[:2], dtype=bool),
            'none': numpy.zeros(rdim[:2], dtype=bool),
            'row': numpy.array([
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, True],
                [False, False, True, False]]),
            'col': numpy.array([
                [False, False, True, False],
                [False, False, False, True],
                [True, False, False, False],
                [False, True, False, False]]),
            }
    visible = {
            'x': {
                'all': [False, False, True, True],
                'col': [False, False, True, True],
                'row': [True] * 4,
                'none': [True] * 4,
                False: [True] * 4,
                True: [False, False, True, True],
                },
            'y': {
                'all': [True, False, True, False],
                'col': [True] * 4,
                'row': [True, False, True, False],
                'none': [True] * 4,
                False: [True] * 4,
                True: [True, False, True, False],
                },
            }
    share[False] = share['none']
    share[True] = share['all']

    # test default
    f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)
    axs = [a1, a2, a3, a4]
    check_shared(numpy.dstack((share['none'], share['none'])), \
            f, axs)
    plt.close(f)

    # test all option combinations
    ops = [False, True, 'all', 'none', 'row', 'col']
    for xo in ops:
        for yo in ops:
            f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, sharex=xo, sharey=yo)
            axs = [a1, a2, a3, a4]
            check_shared(numpy.dstack((share[xo], share[yo])), \
                    f, axs)
            check_visible(dict(x=visible['x'][xo], y=visible['y'][yo]), \
                    f, axs)
            plt.close(f)


def test_exceptions():
    # TODO should this test more options?
    assert_raises(ValueError, plt.subplots, 2, 2, sharex='blah')
    assert_raises(ValueError, plt.subplots, 2, 2, sharey='blah')
    # We filter warnings in this test which are genuine since
    # the pount of this test is to ensure that this raises.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                                message='.*sharex\ argument\ to\ subplots',
                                category=UserWarning)
        assert_raises(ValueError, plt.subplots, 2, 2, -1)
        # uncomment this for 1.5
        # assert_raises(ValueError, plt.subplots, 2, 2, 0)
        assert_raises(ValueError, plt.subplots, 2, 2, 5)


@image_comparison(baseline_images=['subplots_offset_text'], remove_text=False)
def test_subplots_offsettext():
    x = numpy.arange(0, 1e10, 1e9)
    y = numpy.arange(0, 100, 10)+1e4
    fig, axes = plt.subplots(2, 2, sharex='col', sharey='all')
    axes[0, 0].plot(x, x)
    axes[1, 0].plot(x, x)
    axes[0, 1].plot(y, x)
    axes[1, 1].plot(y, x)


@cleanup
def test_subplots():
    # things to test
    # - are axes actually shared?
    # - are tickmarks correctly hidden?
    test_shared()
    # - are exceptions thrown correctly
    test_exceptions()


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
