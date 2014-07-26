from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import warnings

import numpy as np

from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.pyplot as plt
from nose.tools import assert_raises
from numpy.testing import assert_array_equal


def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)


@image_comparison(baseline_images=['tight_layout1'])
def test_tight_layout1():
    'Test tight_layout for a single subplot'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    example_plot(ax, fontsize=24)
    plt.tight_layout()


@image_comparison(baseline_images=['tight_layout2'])
def test_tight_layout2():
    'Test tight_layout for mutiple subplots'
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    example_plot(ax1)
    example_plot(ax2)
    example_plot(ax3)
    example_plot(ax4)
    plt.tight_layout()


@image_comparison(baseline_images=['tight_layout3'])
def test_tight_layout3():
    'Test tight_layout for mutiple subplots'

    fig = plt.figure()

    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(122)

    example_plot(ax1)
    example_plot(ax2)
    example_plot(ax3)

    plt.tight_layout()


@image_comparison(baseline_images=['tight_layout4'],
                  freetype_version=('2.4.5', '2.4.9'))
def test_tight_layout4():
    'Test tight_layout for subplot2grid'

    fig = plt.figure()

    ax1 = plt.subplot2grid((3, 3), (0, 0))
    ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)

    example_plot(ax1)
    example_plot(ax2)
    example_plot(ax3)
    example_plot(ax4)

    plt.tight_layout()


@image_comparison(baseline_images=['tight_layout5'])
def test_tight_layout5():
    'Test tight_layout for image'

    fig = plt.figure()

    ax = plt.subplot(111)
    arr = np.arange(100).reshape((10, 10))
    ax.imshow(arr, interpolation="none")

    plt.tight_layout()


@image_comparison(baseline_images=['tight_layout6'])
def test_tight_layout6():
    'Test tight_layout for gridspec'

    # This raises warnings since tight layout cannot
    # do this fully automatically. But the test is
    # correct since the layout is manually edited
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig = plt.figure()

        import matplotlib.gridspec as gridspec

        gs1 = gridspec.GridSpec(2, 1)
        ax1 = fig.add_subplot(gs1[0])
        ax2 = fig.add_subplot(gs1[1])

        example_plot(ax1)
        example_plot(ax2)

        gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])

        gs2 = gridspec.GridSpec(3, 1)

        for ss in gs2:
            ax = fig.add_subplot(ss)
            example_plot(ax)
            ax.set_title("")
            ax.set_xlabel("")

        ax.set_xlabel("x-label", fontsize=12)

        gs2.tight_layout(fig, rect=[0.5, 0, 1, 1], h_pad=0.45)

        top = min(gs1.top, gs2.top)
        bottom = max(gs1.bottom, gs2.bottom)

        gs1.tight_layout(fig, rect=[None, 0 + (bottom-gs1.bottom),
                                    0.5, 1 - (gs1.top-top)])
        gs2.tight_layout(fig, rect=[0.5, 0 + (bottom-gs2.bottom),
                                    None, 1 - (gs2.top-top)],
                         h_pad=0.45)


@image_comparison(baseline_images=['tight_layout7'])
def test_tight_layout7():
    # tight layout with left and right titles
    fig = plt.figure()
    fontsize = 24
    ax = fig.add_subplot(111)
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Left Title', loc='left', fontsize=fontsize)
    ax.set_title('Right Title', loc='right', fontsize=fontsize)
    plt.tight_layout()

@image_comparison(baseline_images=['tight_layout8'])
def test_tight_layout8():
    'Test automatic use of tight_layout'
    fig = plt.figure()
    fig.set_tight_layout({'pad': .1})
    ax = fig.add_subplot(111)
    example_plot(ax, fontsize=24)
