import numpy as np
from numpy import ma
import matplotlib
from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.pyplot as plt


@image_comparison(baseline_images=['scatter'])
def test_scatter_plot():
    ax = plt.axes()
    ax.scatter([3, 4, 2, 6], [2, 5, 2, 3], c=['r', 'y', 'b', 'lime'], s=[24, 15, 19, 29])

if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
