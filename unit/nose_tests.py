import nose, nose.tools as nt
import numpy.testing as nptest

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as maxes

def test_create_subplot_object():
    fig = plt.figure()
    ax = maxes.Subplot(fig, 1, 1, 1)
    fig.add_subplot(ax)

if __name__=='__main__':
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
    pass
