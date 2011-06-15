import matplotlib
from nose.tools import assert_equal
from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.pyplot as plt


def test_figure_label():
    # pyplot figure creation and selection with figure label and number
    plt.close('all')
    plt.figure('today')
    plt.figure(3)
    plt.figure('tomorow')
    plt.figure()
    plt.figure(0)
    plt.figure(1)
    assert_equal(plt.get_fignums(), [0, 1, 3, 4, 5])
    assert_equal(plt.get_figlabels(), ['', 'today', '', 'tomorow', ''])

@image_comparison(baseline_images=['figure_today'])
def test_figure():
    # named figure support
    fig = plt.figure('today')
    ax = fig.add_subplot(111)
    ax.set_title(fig.get_label())
    ax.plot(range(5))
    fig = plt.figure('tomorow')
    plt.plot([0], 'or')
    fig = plt.figure('today')
    fig.savefig('figure_today')
