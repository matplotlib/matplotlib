import numpy as np

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt


@image_comparison(baseline_images=['legend_auto1'], tol=1.5e-3, remove_text=True)
def test_legend_auto1():
    'Test automatic legend placement'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(100)
    ax.plot(x, 50-x, 'o', label='y=1')
    ax.plot(x, x-50, 'o', label='y=-1')
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
    x = np.arange(100)
    ax.plot(range(4), 'o', label=1)
    ax.plot(range(4, 1, -1), 'o', label='__nolegend__')
    ax.legend(numpoints=1)
