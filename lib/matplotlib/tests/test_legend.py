from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six.moves import xrange
try:
    # mock in python 3.3+
    from unittest import mock
except ImportError:
    import mock
from nose.tools import assert_equal
import numpy as np

from matplotlib.testing.decorators import image_comparison, cleanup
from matplotlib.cbook import MatplotlibDeprecationWarning
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches


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
    plt.errorbar(list(xrange(10)), list(xrange(10)), xerr=0.5, yerr=0.5, label='XX')
    plt.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
               ncol=2, shadow=True, title="My legend", numpoints=1)


@image_comparison(baseline_images=['framealpha'], remove_text=True)
def test_framealpha():
    x = np.linspace(1, 100, 100)
    y = x
    plt.plot(x, y, label='mylabel', lw=10)
    plt.legend(framealpha=0.5)


@image_comparison(baseline_images=['scatter_rc3', 'scatter_rc1'], remove_text=True)
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


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
