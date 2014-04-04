from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import xrange

from nose.tools import assert_equal, assert_raises
import datetime

import numpy as np
from numpy import ma

import matplotlib
from matplotlib.testing.decorators import image_comparison, cleanup
import matplotlib.pyplot as plt


simple_dataset = [[0.9, 1.1, 0.9, 1.1, 1], [2.2, 2, 1.8, 1.8, 2, 2.2]]
other_dataset = [[3.3, 2.2, 1.1], [4.4, 8.8, 2.2]]


#test for positions is none
@image_comparison(baseline_images=['violinplot_simple_dist'])
def test_violinplot_positions_none():
    plt.violinplot(simple_dataset, positions=None)


#test for positions is invalid length
@cleanup
def test_violinplot_positions_invalid():
    assert_raises(ValueError, plt.violinplot, simple_dataset, positions=[5])


#test for positions is valid
@image_comparison(baseline_images=['violinplot_simple_pos'])
def test_violinplot_positions_valid():
    plt.violinplot(simple_dataset, positions=[1, 5])


#test for widths is valid
@image_comparison(baseline_images=['violinplot_simple_widths'])
def test_violinplot_widths_valid():
    plt.violinplot(simple_dataset, widths=[0.5, 1])


#test for widths is a scalar
@image_comparison(baseline_images=['violinplot_scalar_width'])
def test_violinplot_widths_scalar():
    plt.violinplot(simple_dataset, widths=1.5)


#test for widths is invalid length
@cleanup
def test_violinplot_widths_invalid():
    assert_raises(ValueError, plt.violinplot, simple_dataset, widths=[1])


#test for hold status true
@image_comparison(baseline_images=['violinplot_hold_true'])
def test_violinplot_hold_status_true():
    ax = plt.axes()
    ax.hold(True)
    ax.violinplot(simple_dataset)
    ax.violinplot(other_dataset)


#test for hold status false
@image_comparison(baseline_images=['violinplot_simple_dist'])
def test_violinplot_hold_status_false():
    ax = plt.axes()
    ax.hold(False)
    ax.violinplot(other_dataset)
    ax.violinplot(simple_dataset)


#test vert = true
@image_comparison(baseline_images=['violinplot_simple_dist'])
def test_violinplot_vert_true():
    plt.violinplot(simple_dataset, vert=True)


#test vert = false
@image_comparison(baseline_images=['violinplot_vert_false'])
def test_violinplot_vert_false():
    plt.violinplot(simple_dataset, vert=False)


#test show means = false
@image_comparison(baseline_images=['violinplot_simple_dist'])
def test_violinplot_means_false():
    plt.violinplot(simple_dataset, showmeans=False)


#test show means = true
@image_comparison(baseline_images=['violinplot_means'])
def test_violinplot_means_false():
    plt.violinplot(other_dataset, showmeans=True)


#test showmedians = false
@image_comparison(baseline_images=['violinplot_simple_dist'])
def test_violinplot_medians_false():
    plt.violinplot(simple_dataset, showmedians=False)


#test showmedians = true
@image_comparison(baseline_images=['violinplot_medians'])
def test_violinplot_medians_false():
    plt.violinplot(other_dataset, showmedians=True)


#test showextrema = true
@image_comparison(baseline_images=['violinplot_simple_dist'])
def test_violinplot_extrema_true():
    plt.violinplot(simple_dataset, showextrema=True)


#test showextrema = false
@image_comparison(baseline_images=['violinplot_noextrema'])
def test_violinplot_extrema_true():
    plt.violinplot(simple_dataset, showextrema=False)


#test empty dataset
@cleanup
def test_violinplot_nodata():
    artists = plt.violinplot([], showextrema=False)
    for k in artists:
        if k == 'bodies':
            assert len(artists[k]) == 0
        else:
            assert artists[k] is None


if __name__ == '__main__':
    import nose
    import sys

    args = ['-s', '--with-doctest']
    argv = sys.argv
    argv = argv[:1] + args + argv[1:]
    nose.runmodule(argv=argv, exit=False)
