from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt

@image_comparison(baseline_images=['spines_axes_positions'])
def test_spines_axes_positions():
    # SF bug 2852168
    fig = plt.figure()
    x = np.linspace(0,2*np.pi,100)
    y = 2*np.sin(x)
    ax = fig.add_subplot(1,1,1)
    ax.set_title('centered spines')
    ax.plot(x,y)
    ax.spines['right'].set_position(('axes',0.1))
    ax.yaxis.set_ticks_position('right')
    ax.spines['top'].set_position(('axes',0.25))
    ax.xaxis.set_ticks_position('top')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')

@image_comparison(baseline_images=['spines_data_positions'])
def test_spines_data_positions():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.spines['left'].set_position(('data', -1.5))
    ax.spines['top'].set_position(('data', 0.5))
    ax.spines['right'].set_position(('data', -0.5))
    ax.spines['bottom'].set_position('zero')
    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])

@image_comparison(baseline_images=['spines_capstyle'])
def test_spines_capstyle():
    # issue 2542
    plt.rc('axes', linewidth=20)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xticks([])
    ax.set_yticks([])
