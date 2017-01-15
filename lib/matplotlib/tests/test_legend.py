from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import xrange
try:
    # mock in python 3.3+
    from unittest import mock
except ImportError:
    import mock
from nose.tools import assert_equal
import numpy as np

from matplotlib.testing.decorators import image_comparison, cleanup
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.transforms as mtrans
import matplotlib.collections as mcollections


@image_comparison(baseline_images=['legend_auto1'], remove_text=True)
def test_legend_auto1():
    'Test automatic legend placement'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(100)
    ax.plot(x, 50 - x, 'o', label='y=1')
    ax.plot(x, x - 50, 'o', label='y=-1')
    ax.legend(loc=0)


@image_comparison(baseline_images=['legend_auto2'], remove_text=True)
def test_legend_auto2():
    'Test automatic legend placement'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(100)
    b1 = ax.bar(x, x, color='m')
    b2 = ax.bar(x, x[::-1], color='g')
    ax.legend([b1[0], b2[0]], ['up', 'down'], loc=0)


@image_comparison(baseline_images=['legend_auto3'])
def test_legend_auto3():
    'Test automatic legend placement'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [0.9, 0.1, 0.1, 0.9, 0.9, 0.5]
    y = [0.95, 0.95, 0.05, 0.05, 0.5, 0.5]
    ax.plot(x, y, 'o-', label='line')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc=0)


@image_comparison(baseline_images=['legend_various_labels'], remove_text=True)
def test_various_labels():
    # tests all sorts of label types
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(list(xrange(4)), 'o', label=1)
    ax.plot(np.linspace(4, 4.1), 'o', label='D\xe9velopp\xe9s')
    ax.plot(list(xrange(4, 1, -1)), 'o', label='__nolegend__')
    ax.legend(numpoints=1, loc=0)


@image_comparison(baseline_images=['legend_labels_first'], extensions=['png'],
                  remove_text=True)
def test_labels_first():
    # test labels to left of markers
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(10), '-o', label=1)
    ax.plot(np.ones(10)*5, ':x', label="x")
    ax.plot(np.arange(20, 10, -1), 'd', label="diamond")
    ax.legend(loc=0, markerfirst=False)


@image_comparison(baseline_images=['rgba_alpha'],
                  extensions=['png'], remove_text=True)
def test_alpha_rgba():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(10), lw=5)
    leg = plt.legend(['Longlabel that will go away'], loc=10)
    leg.legendPatch.set_facecolor([1, 0, 0, 0.5])


@image_comparison(baseline_images=['rcparam_alpha'],
                  extensions=['png'], remove_text=True)
def test_alpha_rcparam():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(10), lw=5)
    with mpl.rc_context(rc={'legend.framealpha': .75}):
        leg = plt.legend(['Longlabel that will go away'], loc=10)
        # this alpha is going to be over-ridden by the rcparam whith
        # sets the alpha of the patch to be non-None which causes the alpha
        # value of the face color to be discarded.  This behavior may not be
        # ideal, but it is what it is and we should keep track of it changing
        leg.legendPatch.set_facecolor([1, 0, 0, 0.5])


@image_comparison(baseline_images=['fancy'], remove_text=True)
def test_fancy():
    # using subplot triggers some offsetbox functionality untested elsewhere
    plt.subplot(121)
    plt.scatter(list(xrange(10)), list(xrange(10, 0, -1)), label='XX\nXX')
    plt.plot([5] * 10, 'o--', label='XX')
    plt.errorbar(list(xrange(10)), list(xrange(10)), xerr=0.5,
                 yerr=0.5, label='XX')
    plt.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
               ncol=2, shadow=True, title="My legend", numpoints=1)


@image_comparison(baseline_images=['framealpha'], remove_text=True)
def test_framealpha():
    x = np.linspace(1, 100, 100)
    y = x
    plt.plot(x, y, label='mylabel', lw=10)
    plt.legend(framealpha=0.5)


@image_comparison(baseline_images=['scatter_rc3', 'scatter_rc1'],
                  remove_text=True)
def test_rc():
    # using subplot triggers some offsetbox functionality untested elsewhere
    fig = plt.figure()
    ax = plt.subplot(121)
    ax.scatter(list(xrange(10)), list(xrange(10, 0, -1)), label='three')
    ax.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
              title="My legend")

    mpl.rcParams['legend.scatterpoints'] = 1
    fig = plt.figure()
    ax = plt.subplot(121)
    ax.scatter(list(xrange(10)), list(xrange(10, 0, -1)), label='one')
    ax.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
              title="My legend")


@image_comparison(baseline_images=['legend_expand'], remove_text=True)
def test_legend_expand():
    'Test expand mode'
    legend_modes = [None, "expand"]
    fig, axes_list = plt.subplots(len(legend_modes), 1)
    x = np.arange(100)
    for ax, mode in zip(axes_list, legend_modes):
        ax.plot(x, 50 - x, 'o', label='y=1')
        l1 = ax.legend(loc=2, mode=mode)
        ax.add_artist(l1)
        ax.plot(x, x - 50, 'o', label='y=-1')
        l2 = ax.legend(loc=5, mode=mode)
        ax.add_artist(l2)
        ax.legend(loc=3, mode=mode, ncol=2)


@cleanup
def test_legend_remove():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lines = ax.plot(range(10))
    leg = fig.legend(lines, "test")
    leg.remove()
    assert_equal(fig.legends, [])
    leg = ax.legend("test")
    leg.remove()
    assert ax.get_legend() is None


class TestLegendFunction(object):
    # Tests the legend function on the Axes and pyplot.
    @cleanup
    def test_legend_handle_label(self):
        lines = plt.plot(range(10))
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend(lines, ['hello world'])
        Legend.assert_called_with(plt.gca(), lines, ['hello world'])

    @cleanup
    def test_legend_no_args(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend()
        Legend.assert_called_with(plt.gca(), lines, ['hello world'])

    @cleanup
    def test_legend_label_args(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend(['foobar'])
        Legend.assert_called_with(plt.gca(), lines, ['foobar'])

    @cleanup
    def test_legend_handler_map(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.axes.Axes.'
                        'get_legend_handles_labels') as handles_labels:
            handles_labels.return_value = lines, ['hello world']
            plt.legend(handler_map={'1': 2})
        handles_labels.assert_called_with({'1': 2})

    @cleanup
    def test_kwargs(self):
        fig, ax = plt.subplots(1, 1)
        th = np.linspace(0, 2*np.pi, 1024)
        lns, = ax.plot(th, np.sin(th), label='sin', lw=5)
        lnc, = ax.plot(th, np.cos(th), label='cos', lw=5)
        with mock.patch('matplotlib.legend.Legend') as Legend:
            ax.legend(handles=(lnc, lns), labels=('a', 'b'))
        Legend.assert_called_with(ax, (lnc, lns), ('a', 'b'))

    @cleanup
    def test_warn_args_kwargs(self):
        fig, ax = plt.subplots(1, 1)
        th = np.linspace(0, 2*np.pi, 1024)
        lns, = ax.plot(th, np.sin(th), label='sin', lw=5)
        lnc, = ax.plot(th, np.cos(th), label='cos', lw=5)
        with mock.patch('warnings.warn') as warn:
            ax.legend((lnc, lns), labels=('a', 'b'))

        warn.assert_called_with("You have mixed positional and keyword "
                                "arguments, some input will be "
                                "discarded.")


@image_comparison(baseline_images=['legend_stackplot'], extensions=['png'])
def test_legend_stackplot():
    '''test legend for PolyCollection using stackplot'''
    # related to #1341, #1943, and PR #3303
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.linspace(0, 10, 10)
    y1 = 1.0 * x
    y2 = 2.0 * x + 1
    y3 = 3.0 * x + 2
    ax.stackplot(x, y1, y2, y3, labels=['y1', 'y2', 'y3'])
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 70))
    ax.legend(loc=0)


@cleanup
def test_cross_figure_patch_legend():
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    brs = ax.bar(range(3), range(3))
    fig2.legend(brs, 'foo')


@cleanup
def test_nanscatter():
    fig, ax = plt.subplots()

    h = ax.scatter([np.nan], [np.nan], marker="o",
                   facecolor="r", edgecolor="r", s=3)

    ax.legend([h], ["scatter"])

    fig, ax = plt.subplots()
    for color in ['red', 'green', 'blue']:
        n = 750
        x, y = np.random.rand(2, n)
        scale = 200.0 * np.random.rand(n)
        ax.scatter(x, y, c=color, s=scale, label=color,
                   alpha=0.3, edgecolors='none')

    ax.legend()
    ax.grid(True)


@image_comparison(baseline_images=['not_covering_scatter'], extensions=['png'])
def test_not_covering_scatter():
    colors = ['b', 'g', 'r']

    for n in range(3):
        plt.scatter([n], [n], color=colors[n])

    plt.legend(['foo', 'foo', 'foo'], loc='best')
    plt.gca().set_xlim(-0.5, 2.2)
    plt.gca().set_ylim(-0.5, 2.2)


@image_comparison(baseline_images=['not_covering_scatter_transform'],
                  extensions=['png'])
def test_not_covering_scatter_transform():
    # Offsets point to top left, the default auto position
    offset = mtrans.Affine2D().translate(-20, 20)
    x = np.linspace(0, 30, 1000)
    plt.plot(x, x)

    plt.scatter([20], [10], transform=offset + plt.gca().transData)

    plt.legend(['foo', 'bar'], loc='best')


@cleanup
def test_linecollection_scaled_dashes():
    lines1 = [[(0, .5), (.5, 1)], [(.3, .6), (.2, .2)]]
    lines2 = [[[0.7, .2], [.8, .4]], [[.5, .7], [.6, .1]]]
    lines3 = [[[0.6, .2], [.8, .4]], [[.5, .7], [.1, .1]]]
    lc1 = mcollections.LineCollection(lines1, linestyles="--", lw=3)
    lc2 = mcollections.LineCollection(lines2, linestyles="-.")
    lc3 = mcollections.LineCollection(lines3, linestyles=":", lw=.5)

    fig, ax = plt.subplots()
    ax.add_collection(lc1)
    ax.add_collection(lc2)
    ax.add_collection(lc3)

    leg = ax.legend([lc1, lc2, lc3], ["line1", "line2", 'line 3'])
    h1, h2, h3 = leg.legendHandles

    for oh, lh in zip((lc1, lc2, lc3), (h1, h2, h3)):
        assert oh.get_linestyles()[0][1] == lh._dashSeq
        assert oh.get_linestyles()[0][0] == lh._dashOffset


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
