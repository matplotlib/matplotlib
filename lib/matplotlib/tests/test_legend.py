import numpy as np

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    ax.plot(range(4), 'o', label=1)
    ax.plot(np.linspace(4, 4.1), 'o', label=u'D\xe9velopp\xe9s')
    ax.plot(range(4, 1, -1), 'o', label='__nolegend__')
    ax.legend(numpoints=1, loc=0)


@image_comparison(baseline_images=['fancy'], remove_text=True)
def test_fancy():
    # using subplot triggers some offsetbox functionality untested elsewhere
    plt.subplot(121)
    plt.scatter(range(10), range(10, 0, -1), label='XX\nXX')
    plt.plot([5] * 10, 'o--', label='XX')
    plt.errorbar(range(10), range(10), xerr=0.5, yerr=0.5, label='XX')
    plt.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
               ncol=2, shadow=True, title="My legend", numpoints=1)

@image_comparison(baseline_images=['framealpha'], remove_text=True)
def test_framealpha():
    x = np.linspace(1, 100, 100)
    y = x
    plt.plot(x, y, label='mylabel', lw=10)
    plt.legend(framealpha=0.5)

@image_comparison(baseline_images=['scatter_rc3','scatter_rc1'], remove_text=True)
def test_rc():
    # using subplot triggers some offsetbox functionality untested elsewhere
    fig = plt.figure()
    ax =  plt.subplot(121)
    ax.scatter(range(10), range(10, 0, -1), label='three')
    ax.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
                title="My legend")


    mpl.rcParams['legend.scatterpoints'] = 1
    fig = plt.figure()
    ax =  plt.subplot(121)
    ax.scatter(range(10), range(10, 0, -1), label='one')
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
