from __future__ import print_function
import os
import tempfile
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import cleanup
from matplotlib.testing.decorators import image_comparison


def draw_quiver(ax, **kw):
    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, 1),
                       np.arange(0, 2 * np.pi, 1))
    U = np.cos(X)
    V = np.sin(Y)

    Q = ax.quiver(U, V, **kw)
    return Q


@cleanup
def test_quiver_memory_leak():
    fig, ax = plt.subplots()

    Q = draw_quiver(ax)
    ttX = Q.X
    Q.remove()

    del Q

    assert sys.getrefcount(ttX) == 2


@cleanup
def test_quiver_key_memory_leak():
    fig, ax = plt.subplots()

    Q = draw_quiver(ax)

    qk = ax.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$',
                      labelpos='W',
                      fontproperties={'weight': 'bold'})
    assert sys.getrefcount(qk) == 3
    qk.remove()
    assert sys.getrefcount(qk) == 2


@image_comparison(baseline_images=['quiver_animated_test_image'],
                  extensions=['png'])
def test_quiver_animate():
    # Tests fix for #2616
    fig, ax = plt.subplots()

    Q = draw_quiver(ax, animated=True)

    qk = ax.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$',
                      labelpos='W',
                      fontproperties={'weight': 'bold'})


@image_comparison(baseline_images=['quiver_with_key_test_image'],
                  extensions=['png'])
def test_quiver_with_key():
    fig, ax = plt.subplots()
    ax.margins(0.1)

    Q = draw_quiver(ax)

    qk = ax.quiverkey(Q, 0.5, 0.95, 2,
                      r'$2\, \mathrm{m}\, \mathrm{s}^{-1}$',
                      coordinates='figure',
                      labelpos='W',
                      fontproperties={'weight': 'bold',
                                      'size': 'large'})


if __name__ == '__main__':
    import nose
    nose.runmodule()
