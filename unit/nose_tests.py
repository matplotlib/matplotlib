import numpy as np

import nose, nose.tools as nt
import numpy.testing as nptest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.axes as maxes

def test_create_subplot_object():
    fig = plt.figure()
    ax = maxes.Subplot(fig, 1, 1, 1)
    fig.add_subplot(ax)
    plt.close(fig)

def test_markevery():
    x, y = np.random.rand(2, 100)

    # check marker only plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'o')
    ax.plot(x, y, 'd', markevery=None)
    ax.plot(x, y, 's', markevery=10)
    ax.plot(x, y, '+', markevery=(5, 20))
    fig.canvas.draw()
    plt.close(fig)

    # check line/marker combos
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, '-o')
    ax.plot(x, y, '-d', markevery=None)
    ax.plot(x, y, '-s', markevery=10)
    ax.plot(x, y, '-+', markevery=(5, 20))
    fig.canvas.draw()
    plt.close(fig)

if __name__=='__main__':
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)

    plt.show()
