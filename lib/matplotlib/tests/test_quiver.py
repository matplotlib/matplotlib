from __future__ import print_function
import os
import tempfile
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import cleanup


WRITER_OUTPUT = dict(ffmpeg='mp4', ffmpeg_file='mp4',
                     mencoder='mp4', mencoder_file='mp4',
                     avconv='mp4', avconv_file='mp4',
                     imagemagick='gif', imagemagick_file='gif')


@cleanup
def test_quiver_memory_leak():
    fig, ax = plt.subplots()

    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .04),
                        np.arange(0, 2 * np.pi, .04))
    U = np.cos(X)
    V = np.sin(Y)

    Q = ax.quiver(U, V)
    ttX = Q.X
    Q.remove()

    del Q

    assert sys.getrefcount(ttX) == 2


@cleanup
def test_quiver_key_memory_leak():
    fig, ax = plt.subplots()

    X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .04),
                        np.arange(0, 2 * np.pi, .04))
    U = np.cos(X)
    V = np.sin(Y)

    Q = ax.quiver(U, V)

    qk = ax.quiverkey(Q, 0.5, 0.92, 2, r'$2 \frac{m}{s}$',
                   labelpos='W',
                   fontproperties={'weight': 'bold'})
    assert sys.getrefcount(qk) == 3
    qk.remove()
    assert sys.getrefcount(qk) == 2

if __name__ == '__main__':
    import nose
    nose.runmodule()
