from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy.testing import assert_equal
from matplotlib import rcParams
from matplotlib.testing.decorators import image_comparison
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import warnings


def test_figure_label():
    # pyplot figure creation, selection and closing with figure label and
    # number
    plt.close('all')
    plt.figure('today')
    plt.figure(3)
    plt.figure('tomorrow')
    plt.figure()
    plt.figure(0)
    plt.figure(1)
    plt.figure(3)
    assert_equal(plt.get_fignums(), [0, 1, 3, 4, 5])
    assert_equal(plt.get_figlabels(), ['', 'today', '', 'tomorrow', ''])
    plt.close(10)
    plt.close()
    plt.close(5)
    plt.close('tomorrow')
    assert_equal(plt.get_fignums(), [0, 1])
    assert_equal(plt.get_figlabels(), ['', 'today'])


def test_fignum_exists():
    # pyplot figure creation, selection and closing with fignum_exists
    plt.figure('one')
    plt.figure(2)
    plt.figure('three')
    plt.figure()
    assert_equal(plt.fignum_exists('one'), True)
    assert_equal(plt.fignum_exists(2), True)
    assert_equal(plt.fignum_exists('three'), True)
    assert_equal(plt.fignum_exists(4), True)
    plt.close('one')
    plt.close(4)
    assert_equal(plt.fignum_exists('one'), False)
    assert_equal(plt.fignum_exists(4), False)


def test_clf_keyword():
    # test if existing figure is cleared with figure() and subplots()
    fig0 = plt.figure(num=1)
    fig0.suptitle("A fancy plot")
    assert_equal([t.get_text() for t in fig0.texts], ["A fancy plot"])

    fig1 = plt.figure(num=1, clear=False)
    fig1.text(0.5, 0.5, "Really fancy!")
    assert fig0 is fig1
    assert_equal([t.get_text() for t in fig1.texts],
                 ["A fancy plot", 'Really fancy!'])

    fig2, ax2 = plt.subplots(2, 1, num=1, clear=True)
    assert fig0 is fig2
    assert_equal([t.get_text() for t in fig2.texts], [])


@image_comparison(baseline_images=['figure_today'])
def test_figure():
    # named figure support
    fig = plt.figure('today')
    ax = fig.add_subplot(111)
    ax.set_title(fig.get_label())
    ax.plot(np.arange(5))
    # plot red line in a different figure.
    plt.figure('tomorrow')
    plt.plot([0, 1], [1, 0], 'r')
    # Return to the original; make sure the red line is not there.
    plt.figure('today')
    plt.close('tomorrow')


@image_comparison(baseline_images=['figure_legend'])
def test_figure_legend():
    fig, axes = plt.subplots(2)
    axes[0].plot([0, 1], [1, 0], label='x', color='g')
    axes[0].plot([0, 1], [0, 1], label='y', color='r')
    axes[0].plot([0, 1], [0.5, 0.5], label='y', color='k')

    axes[1].plot([0, 1], [1, 0], label='y', color='r')
    axes[1].plot([0, 1], [0, 1], label='z', color='b')
    fig.legend()


def test_gca():
    fig = plt.figure()

    ax1 = fig.add_axes([0, 0, 1, 1])
    assert fig.gca(projection='rectilinear') is ax1
    assert fig.gca() is ax1

    ax2 = fig.add_subplot(121, projection='polar')
    assert fig.gca() is ax2
    assert fig.gca(polar=True)is ax2

    ax3 = fig.add_subplot(122)
    assert fig.gca() is ax3

    # the final request for a polar axes will end up creating one
    # with a spec of 111.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        # Changing the projection will throw a warning
        assert fig.gca(polar=True) is not ax3
        assert len(w) == 1
    assert fig.gca(polar=True) is not ax2
    assert_equal(fig.gca().get_geometry(), (1, 1, 1))

    fig.sca(ax1)
    assert fig.gca(projection='rectilinear') is ax1
    assert fig.gca() is ax1


@image_comparison(baseline_images=['figure_suptitle'])
def test_suptitle():
    fig, _ = plt.subplots()
    fig.suptitle('hello', color='r')
    fig.suptitle('title', color='g', rotation='30')


def test_suptitle_fontproperties():
    from matplotlib.font_manager import FontProperties
    fig, ax = plt.subplots()
    fps = FontProperties(size='large', weight='bold')
    txt = fig.suptitle('fontprops title', fontproperties=fps)
    assert_equal(txt.get_fontsize(), fps.get_size_in_points())
    assert_equal(txt.get_weight(), fps.get_weight())


@image_comparison(baseline_images=['alpha_background'],
                  # only test png and svg. The PDF output appears correct,
                  # but Ghostscript does not preserve the background color.
                  extensions=['png', 'svg'],
                  savefig_kwarg={'facecolor': (0, 1, 0.4),
                                 'edgecolor': 'none'})
def test_alpha():
    # We want an image which has a background color and an
    # alpha of 0.4.
    fig = plt.figure(figsize=[2, 1])
    fig.set_facecolor((0, 1, 0.4))
    fig.patch.set_alpha(0.4)

    import matplotlib.patches as mpatches
    fig.patches.append(mpatches.CirclePolygon([20, 20],
                                              radius=15,
                                              alpha=0.6,
                                              facecolor='red'))


def test_too_many_figures():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for i in range(rcParams['figure.max_open_warning'] + 1):
            plt.figure()
        assert len(w) == 1


def test_iterability_axes_argument():

    # This is a regression test for matplotlib/matplotlib#3196. If one of the
    # arguments returned by _as_mpl_axes defines __getitem__ but is not
    # iterable, this would raise an execption. This is because we check
    # whether the arguments are iterable, and if so we try and convert them
    # to a tuple. However, the ``iterable`` function returns True if
    # __getitem__ is present, but some classes can define __getitem__ without
    # being iterable. The tuple conversion is now done in a try...except in
    # case it fails.

    class MyAxes(Axes):
        def __init__(self, *args, **kwargs):
            kwargs.pop('myclass', None)
            return Axes.__init__(self, *args, **kwargs)

    class MyClass(object):

        def __getitem__(self, item):
            if item != 'a':
                raise ValueError("item should be a")

        def _as_mpl_axes(self):
            return MyAxes, {'myclass': self}

    fig = plt.figure()
    fig.add_subplot(1, 1, 1, projection=MyClass())
    plt.close(fig)


def test_set_fig_size():
    fig = plt.figure()

    # check figwidth
    fig.set_figwidth(5)
    assert_equal(fig.get_figwidth(), 5)

    # check figheight
    fig.set_figheight(1)
    assert_equal(fig.get_figheight(), 1)

    # check using set_size_inches
    fig.set_size_inches(2, 4)
    assert_equal(fig.get_figwidth(), 2)
    assert_equal(fig.get_figheight(), 4)

    # check using tuple to first argument
    fig.set_size_inches((1, 3))
    assert_equal(fig.get_figwidth(), 1)
    assert_equal(fig.get_figheight(), 3)


def test_axes_remove():
    fig, axes = plt.subplots(2, 2)
    axes[-1, -1].remove()
    for ax in axes.ravel()[:-1]:
        assert ax in fig.axes
    assert axes[-1, -1] not in fig.axes
    assert_equal(len(fig.axes), 3)


def test_figaspect():
    w, h = plt.figaspect(np.float64(2) / np.float64(1))
    assert h / w == 2
    w, h = plt.figaspect(2)
    assert h / w == 2
    w, h = plt.figaspect(np.zeros((1, 2)))
    assert h / w == 0.5
    w, h = plt.figaspect(np.zeros((2, 2)))
    assert h / w == 1
