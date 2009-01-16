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
    ax.plot(x, y, 'o', label='default')
    ax.plot(x, y, 'd', markevery=None, label='mark all')
    ax.plot(x, y, 's', markevery=10, label='mark every 10')
    ax.plot(x, y, '+', markevery=(5, 20), label='mark every 5 starting at 10')
    ax.legend()
    fig.canvas.draw()
    plt.close(fig)

    # check line/marker combos
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, '-o', label='default')
    ax.plot(x, y, '-d', markevery=None, label='mark all')
    ax.plot(x, y, '-s', markevery=10, label='mark every 10')
    ax.plot(x, y, '-+', markevery=(5, 20), label='mark every 5 starting at 10')
    ax.legend()
    fig.canvas.draw()
    plt.close(fig)

def test_units_strings():
    # Make sure passing in sequences of strings doesn't cause the unit
    # conversion registry to recurse infinitely
    Id = ['50', '100', '150', '200', '250']
    pout = ['0', '7.4', '11.4', '14.2', '16.3']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Id, pout)
    fig.canvas.draw()
    plt.close(fig)

if __name__=='__main__':
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)

    plt.show()
