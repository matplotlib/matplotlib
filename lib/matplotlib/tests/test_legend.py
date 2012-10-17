import numpy as np

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


@image_comparison(baseline_images=['legend_auto1'], tol=1.5e-3, remove_text=True)
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


@image_comparison(baseline_images=['legend_various_labels'], remove_text=True)
def test_various_labels():
    # tests all sorts of label types
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(range(4), 'o', label=1)
    ax.plot(np.linspace(4, 4.1), 'o', label=u'D\xe9velopp\xe9s')
    ax.plot(range(4, 1, -1), 'o', label='__nolegend__')
    ax.legend(numpoints=1)


@image_comparison(baseline_images=['fancy'], remove_text=True)
def test_fancy():
    # using subplot triggers some offsetbox functionality untested elsewhere
    plt.subplot(121)
    plt.scatter(range(10), range(10, 0, -1), label='XX\nXX')
    plt.plot([5] * 10, 'o--', label='XX')
    plt.errorbar(range(10), range(10), xerr=0.5, yerr=0.5, label='XX')
    plt.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
               ncol=2, shadow=True, title="My legend", numpoints=1)
