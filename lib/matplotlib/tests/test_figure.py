from nose.tools import assert_equal, assert_true
from matplotlib.testing.decorators import image_comparison, cleanup
import matplotlib.pyplot as plt


@cleanup
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


@image_comparison(baseline_images=['figure_today'])
def test_figure():
    # named figure support
    fig = plt.figure('today')
    ax = fig.add_subplot(111)
    ax.set_title(fig.get_label())
    ax.plot(range(5))
    # plot red line in a different figure.
    plt.figure('tomorrow')
    plt.plot([0, 1], [1, 0], 'r')
    # Return to the original; make sure the red line is not there.
    plt.figure('today')
    plt.close('tomorrow')


@cleanup
def test_gca():
    fig = plt.figure()

    ax1 = fig.add_axes([0, 0, 1, 1])
    assert_true(fig.gca(projection='rectilinear') is ax1)
    assert_true(fig.gca() is ax1)

    ax2 = fig.add_subplot(121, projection='polar')
    assert_true(fig.gca() is ax2)
    assert_true(fig.gca(polar=True)is ax2)

    ax3 = fig.add_subplot(122)
    assert_true(fig.gca() is ax3)

    # the final request for a polar axes will end up creating one
    # with a spec of 111.
    assert_true(fig.gca(polar=True) is not ax3)
    assert_true(fig.gca(polar=True) is not ax2)
    assert_equal(fig.gca().get_geometry(), (1, 1, 1))

    fig.sca(ax1)
    assert_true(fig.gca(projection='rectilinear') is ax1)
    assert_true(fig.gca() is ax1)


@image_comparison(baseline_images=['figure_suptitle'])
def test_suptitle():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle('hello', color='r')
    fig.suptitle('title', color='g', rotation='30')


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
