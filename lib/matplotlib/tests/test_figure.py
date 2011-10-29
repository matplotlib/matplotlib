import matplotlib
from nose.tools import assert_equal
from matplotlib.testing.decorators import image_comparison, knownfailureif, cleanup
import matplotlib.pyplot as plt


@cleanup
def test_figure_label():
    # pyplot figure creation, selection and closing with figure label and number
    plt.close('all')
    plt.figure('today')
    plt.figure(3)
    plt.figure('tomorow')
    plt.figure()
    plt.figure(0)
    plt.figure(1)
    plt.figure(3)
    assert_equal(plt.get_fignums(), [0, 1, 3, 4, 5])
    assert_equal(plt.get_figlabels(), ['', 'today', '', 'tomorow', ''])
    plt.close(10)
    plt.close()
    plt.close(5)
    plt.close('tomorow')
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
    plt.figure('tomorow')
    plt.plot([0, 1], [1,0], 'r')
    # Return to the original; make sure the red line is not there.
    plt.figure('today')
