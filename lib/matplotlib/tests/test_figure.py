from datetime import datetime
from pathlib import Path
import platform

from matplotlib import rcParams
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator, FixedFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import numpy as np
import pytest


@image_comparison(['figure_align_labels'],
                  tol={'aarch64': 0.02}.get(platform.machine(), 0.0))
def test_align_labels():
    # Check the figure.align_labels() command
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(3, 3)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(np.arange(0, 1e6, 1000))
    ax.set_ylabel('Ylabel0 0')
    ax = fig.add_subplot(gs[0, -1])
    ax.plot(np.arange(0, 1e4, 100))

    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.set_ylabel('YLabel1 %d' % i)
        ax.set_xlabel('XLabel1 %d' % i)
        if i in [0, 2]:
            ax.xaxis.set_label_position("top")
            ax.xaxis.tick_top()
        if i == 0:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
        if i == 2:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.set_xlabel('XLabel2 %d' % (i))
        ax.set_ylabel('YLabel2 %d' % (i))

        if i == 2:
            ax.plot(np.arange(0, 1e4, 10))
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)

    fig.align_labels()


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
    assert plt.get_fignums() == [0, 1, 3, 4, 5]
    assert plt.get_figlabels() == ['', 'today', '', 'tomorrow', '']
    plt.close(10)
    plt.close()
    plt.close(5)
    plt.close('tomorrow')
    assert plt.get_fignums() == [0, 1]
    assert plt.get_figlabels() == ['', 'today']


def test_fignum_exists():
    # pyplot figure creation, selection and closing with fignum_exists
    plt.figure('one')
    plt.figure(2)
    plt.figure('three')
    plt.figure()
    assert plt.fignum_exists('one')
    assert plt.fignum_exists(2)
    assert plt.fignum_exists('three')
    assert plt.fignum_exists(4)
    plt.close('one')
    plt.close(4)
    assert not plt.fignum_exists('one')
    assert not plt.fignum_exists(4)


def test_clf_keyword():
    # test if existing figure is cleared with figure() and subplots()
    text1 = 'A fancy plot'
    text2 = 'Really fancy!'

    fig0 = plt.figure(num=1)
    fig0.suptitle(text1)
    assert [t.get_text() for t in fig0.texts] == [text1]

    fig1 = plt.figure(num=1, clear=False)
    fig1.text(0.5, 0.5, text2)
    assert fig0 is fig1
    assert [t.get_text() for t in fig1.texts] == [text1, text2]

    fig2, ax2 = plt.subplots(2, 1, num=1, clear=True)
    assert fig0 is fig2
    assert [t.get_text() for t in fig2.texts] == []


@image_comparison(['figure_today'])
def test_figure():
    # named figure support
    fig = plt.figure('today')
    ax = fig.add_subplot()
    ax.set_title(fig.get_label())
    ax.plot(np.arange(5))
    # plot red line in a different figure.
    plt.figure('tomorrow')
    plt.plot([0, 1], [1, 0], 'r')
    # Return to the original; make sure the red line is not there.
    plt.figure('today')
    plt.close('tomorrow')


@image_comparison(['figure_legend'])
def test_figure_legend():
    fig, axs = plt.subplots(2)
    axs[0].plot([0, 1], [1, 0], label='x', color='g')
    axs[0].plot([0, 1], [0, 1], label='y', color='r')
    axs[0].plot([0, 1], [0.5, 0.5], label='y', color='k')

    axs[1].plot([0, 1], [1, 0], label='_y', color='r')
    axs[1].plot([0, 1], [0, 1], label='z', color='b')
    fig.legend()


def test_gca():
    fig = plt.figure()

    ax1 = fig.add_axes([0, 0, 1, 1])
    assert fig.gca(projection='rectilinear') is ax1
    assert fig.gca() is ax1

    ax2 = fig.add_subplot(121, projection='polar')
    assert fig.gca() is ax2
    assert fig.gca(polar=True) is ax2

    ax3 = fig.add_subplot(122)
    assert fig.gca() is ax3

    # the final request for a polar axes will end up creating one
    # with a spec of 111.
    with pytest.warns(UserWarning):
        # Changing the projection will throw a warning
        assert fig.gca(polar=True) is not ax3
    assert fig.gca(polar=True) is not ax2
    assert fig.gca().get_geometry() == (1, 1, 1)

    fig.sca(ax1)
    assert fig.gca(projection='rectilinear') is ax1
    assert fig.gca() is ax1


def test_add_subplot_invalid():
    fig = plt.figure()
    with pytest.raises(ValueError):
        fig.add_subplot(2, 0, 1)
    with pytest.raises(ValueError):
        fig.add_subplot(0, 2, 1)
    with pytest.raises(ValueError):
        fig.add_subplot(2, 2, 0)
    with pytest.raises(ValueError):
        fig.add_subplot(2, 2, 5)


@image_comparison(['figure_suptitle'])
def test_suptitle():
    fig, _ = plt.subplots()
    fig.suptitle('hello', color='r')
    fig.suptitle('title', color='g', rotation='30')


def test_suptitle_fontproperties():
    from matplotlib.font_manager import FontProperties
    fig, ax = plt.subplots()
    fps = FontProperties(size='large', weight='bold')
    txt = fig.suptitle('fontprops title', fontproperties=fps)
    assert txt.get_fontsize() == fps.get_size_in_points()
    assert txt.get_weight() == fps.get_weight()


@image_comparison(['alpha_background'],
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
    with pytest.warns(RuntimeWarning):
        for i in range(rcParams['figure.max_open_warning'] + 1):
            plt.figure()


def test_iterability_axes_argument():

    # This is a regression test for matplotlib/matplotlib#3196. If one of the
    # arguments returned by _as_mpl_axes defines __getitem__ but is not
    # iterable, this would raise an exception. This is because we check
    # whether the arguments are iterable, and if so we try and convert them
    # to a tuple. However, the ``iterable`` function returns True if
    # __getitem__ is present, but some classes can define __getitem__ without
    # being iterable. The tuple conversion is now done in a try...except in
    # case it fails.

    class MyAxes(Axes):
        def __init__(self, *args, myclass=None, **kwargs):
            return Axes.__init__(self, *args, **kwargs)

    class MyClass:

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
    assert fig.get_figwidth() == 5

    # check figheight
    fig.set_figheight(1)
    assert fig.get_figheight() == 1

    # check using set_size_inches
    fig.set_size_inches(2, 4)
    assert fig.get_figwidth() == 2
    assert fig.get_figheight() == 4

    # check using tuple to first argument
    fig.set_size_inches((1, 3))
    assert fig.get_figwidth() == 1
    assert fig.get_figheight() == 3


def test_axes_remove():
    fig, axs = plt.subplots(2, 2)
    axs[-1, -1].remove()
    for ax in axs.ravel()[:-1]:
        assert ax in fig.axes
    assert axs[-1, -1] not in fig.axes
    assert len(fig.axes) == 3


def test_figaspect():
    w, h = plt.figaspect(np.float64(2) / np.float64(1))
    assert h / w == 2
    w, h = plt.figaspect(2)
    assert h / w == 2
    w, h = plt.figaspect(np.zeros((1, 2)))
    assert h / w == 0.5
    w, h = plt.figaspect(np.zeros((2, 2)))
    assert h / w == 1


@pytest.mark.parametrize('which', [None, 'both', 'major', 'minor'])
def test_autofmt_xdate(which):
    date = ['3 Jan 2013', '4 Jan 2013', '5 Jan 2013', '6 Jan 2013',
            '7 Jan 2013', '8 Jan 2013', '9 Jan 2013', '10 Jan 2013',
            '11 Jan 2013', '12 Jan 2013', '13 Jan 2013', '14 Jan 2013']

    time = ['16:44:00', '16:45:00', '16:46:00', '16:47:00', '16:48:00',
            '16:49:00', '16:51:00', '16:52:00', '16:53:00', '16:55:00',
            '16:56:00', '16:57:00']

    angle = 60
    minors = [1, 2, 3, 4, 5, 6, 7]

    x = mdates.datestr2num(date)
    y = mdates.datestr2num(time)

    fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.yaxis_date()
    ax.xaxis_date()

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_formatter(FixedFormatter(minors))

    fig.autofmt_xdate(0.2, angle, 'right', which)

    if which in ('both', 'major', None):
        for label in fig.axes[0].get_xticklabels(False, 'major'):
            assert int(label.get_rotation()) == angle

    if which in ('both', 'minor'):
        for label in fig.axes[0].get_xticklabels(True, 'minor'):
            assert int(label.get_rotation()) == angle


@pytest.mark.style('default')
def test_change_dpi():
    fig = plt.figure(figsize=(4, 4))
    fig.canvas.draw()
    assert fig.canvas.renderer.height == 400
    assert fig.canvas.renderer.width == 400
    fig.dpi = 50
    fig.canvas.draw()
    assert fig.canvas.renderer.height == 200
    assert fig.canvas.renderer.width == 200


@pytest.mark.parametrize('width, height', [
    (1, np.nan),
    (0, 1),
    (-1, 1),
    (np.inf, 1)
])
def test_invalid_figure_size(width, height):
    with pytest.raises(ValueError):
        plt.figure(figsize=(width, height))

    fig = plt.figure()
    with pytest.raises(ValueError):
        fig.set_size_inches(width, height)


def test_invalid_figure_add_axes():
    fig = plt.figure()
    with pytest.raises(ValueError):
        fig.add_axes((.1, .1, .5, np.nan))


def test_subplots_shareax_loglabels():
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=False)
    for ax in axs.flat:
        ax.plot([10, 20, 30], [10, 20, 30])

    ax.set_yscale("log")
    ax.set_xscale("log")

    for ax in axs[0, :]:
        assert 0 == len(ax.xaxis.get_ticklabels(which='both'))

    for ax in axs[1, :]:
        assert 0 < len(ax.xaxis.get_ticklabels(which='both'))

    for ax in axs[:, 1]:
        assert 0 == len(ax.yaxis.get_ticklabels(which='both'))

    for ax in axs[:, 0]:
        assert 0 < len(ax.yaxis.get_ticklabels(which='both'))


def test_savefig():
    fig = plt.figure()
    msg = r"savefig\(\) takes 2 positional arguments but 3 were given"
    with pytest.raises(TypeError, match=msg):
        fig.savefig("fname1.png", "fname2.png")


def test_figure_repr():
    fig = plt.figure(figsize=(10, 20), dpi=10)
    assert repr(fig) == "<Figure size 100x200 with 0 Axes>"


def test_warn_cl_plus_tl():
    fig, ax = plt.subplots(constrained_layout=True)
    with pytest.warns(UserWarning):
        # this should warn,
        fig.subplots_adjust(top=0.8)
    assert not(fig.get_constrained_layout())


@check_figures_equal(extensions=["png", "pdf"])
def test_add_artist(fig_test, fig_ref):
    fig_test.set_dpi(100)
    fig_ref.set_dpi(100)

    fig_test.subplots()
    l1 = plt.Line2D([.2, .7], [.7, .7], gid='l1')
    l2 = plt.Line2D([.2, .7], [.8, .8], gid='l2')
    r1 = plt.Circle((20, 20), 100, transform=None, gid='C1')
    r2 = plt.Circle((.7, .5), .05, gid='C2')
    r3 = plt.Circle((4.5, .8), .55, transform=fig_test.dpi_scale_trans,
                    facecolor='crimson', gid='C3')
    for a in [l1, l2, r1, r2, r3]:
        fig_test.add_artist(a)
    l2.remove()

    ax2 = fig_ref.subplots()
    l1 = plt.Line2D([.2, .7], [.7, .7], transform=fig_ref.transFigure,
                    gid='l1', zorder=21)
    r1 = plt.Circle((20, 20), 100, transform=None, clip_on=False, zorder=20,
                    gid='C1')
    r2 = plt.Circle((.7, .5), .05, transform=fig_ref.transFigure, gid='C2',
                    zorder=20)
    r3 = plt.Circle((4.5, .8), .55, transform=fig_ref.dpi_scale_trans,
                    facecolor='crimson', clip_on=False, zorder=20, gid='C3')
    for a in [l1, r1, r2, r3]:
        ax2.add_artist(a)


@pytest.mark.parametrize("fmt", ["png", "pdf", "ps", "eps", "svg"])
def test_fspath(fmt, tmpdir):
    out = Path(tmpdir, "test.{}".format(fmt))
    plt.savefig(out)
    with out.open("rb") as file:
        # All the supported formats include the format name (case-insensitive)
        # in the first 100 bytes.
        assert fmt.encode("ascii") in file.read(100).lower()


def test_tightbbox():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    t = ax.text(1., 0.5, 'This dangles over end')
    renderer = fig.canvas.get_renderer()
    x1Nom0 = 9.035  # inches
    assert abs(t.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    assert abs(fig.get_tightbbox(renderer).x1 - x1Nom0) < 0.05
    assert abs(fig.get_tightbbox(renderer).x0 - 0.679) < 0.05
    # now exclude t from the tight bbox so now the bbox is quite a bit
    # smaller
    t.set_in_layout(False)
    x1Nom = 7.333
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom * fig.dpi) < 2
    assert abs(fig.get_tightbbox(renderer).x1 - x1Nom) < 0.05

    t.set_in_layout(True)
    x1Nom = 7.333
    assert abs(ax.get_tightbbox(renderer).x1 - x1Nom0 * fig.dpi) < 2
    # test bbox_extra_artists method...
    assert abs(ax.get_tightbbox(renderer, bbox_extra_artists=[]).x1
               - x1Nom * fig.dpi) < 2


def test_axes_removal():
    # Check that units can set the formatter after an Axes removal
    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[1].remove()
    axs[0].plot([datetime(2000, 1, 1), datetime(2000, 2, 1)], [0, 1])
    assert isinstance(axs[0].xaxis.get_major_formatter(),
                      mdates.AutoDateFormatter)

    # Check that manually setting the formatter, then removing Axes keeps
    # the set formatter.
    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[1].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].remove()
    axs[0].plot([datetime(2000, 1, 1), datetime(2000, 2, 1)], [0, 1])
    assert isinstance(axs[0].xaxis.get_major_formatter(),
                      ScalarFormatter)


def test_removed_axis():
    # Simple smoke test to make sure removing a shared axis works
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].remove()
    fig.canvas.draw()


@check_figures_equal(extensions=["svg", "pdf", "eps", "png"])
def test_animated_with_canvas_change(fig_test, fig_ref):
    ax_ref = fig_ref.subplots()
    ax_ref.plot(range(5))

    ax_test = fig_test.subplots()
    ax_test.plot(range(5), animated=True)
