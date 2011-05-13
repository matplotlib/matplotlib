import numpy as np

from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.pyplot as plt
from nose.tools import assert_raises
from numpy.testing import assert_array_equal

@image_comparison(baseline_images=['legend_auto1'], tol=1.5e-3)
def test_legend_auto1():
    'Test automatic legend placement'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(100)
    ax.plot(x, 50-x, 'o', label='y=1')
    ax.plot(x, x-50, 'o', label='y=-1')
    ax.legend(loc=0)

@image_comparison(baseline_images=['legend_auto2'])
def test_legend_auto2():
    'Test automatic legend placement'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(100)
    b1 = ax.bar(x, x, color='m')
    b2 = ax.bar(x, x[::-1], color='g')
    ax.legend([b1[0], b2[0]], ['up', 'down'], loc=0)

