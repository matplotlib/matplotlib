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


@image_comparison(baseline_images=['quiver_single_test_image'],
                  extensions=['png'], remove_text=True)
def test_quiver_single():
    fig, ax = plt.subplots()
    ax.margins(0.1)

    ax.quiver([1], [1], [2], [2])


@cleanup
def test_quiver_copy():
    fig, ax = plt.subplots()
    uv = dict(u=np.array([1.1]), v=np.array([2.0]))
    q0 = ax.quiver([1], [1], uv['u'], uv['v'])
    uv['v'][0] = 0
    assert q0.V[0] == 2.0


@image_comparison(baseline_images=['quiver_key_pivot'],
                  extensions=['png'], remove_text=True)
def test_quiver_key_pivot():
    fig, ax = plt.subplots()

    u, v = np.mgrid[0:2*np.pi:10j, 0:2*np.pi:10j]

    q = ax.quiver(np.sin(u), np.cos(v))
    ax.set_xlim(-2, 11)
    ax.set_ylim(-2, 11)
    ax.quiverkey(q, 0.5, 1, 1, 'N', labelpos='N')
    ax.quiverkey(q, 1, 0.5, 1, 'E', labelpos='E')
    ax.quiverkey(q, 0.5, 0, 1, 'S', labelpos='S')
    ax.quiverkey(q, 0, 0.5, 1, 'W', labelpos='W')


@image_comparison(baseline_images=['barbs_test_image'],
                  extensions=['png'], remove_text=True)
def test_barbs():
    x = np.linspace(-5, 5, 5)
    X, Y = np.meshgrid(x, x)
    U, V = 12*X, 12*Y
    fig, ax = plt.subplots()
    ax.barbs(X, Y, U, V, np.sqrt(U*U + V*V), fill_empty=True, rounding=False,
             sizes=dict(emptybarb=0.25, spacing=0.2, height=0.3),
             cmap='viridis')

if __name__ == '__main__':
    import nose
    nose.runmodule()
