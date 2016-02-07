from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import sys
import tempfile
import numpy as np
from nose import with_setup
from nose.tools import assert_false, assert_true
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.noseclasses import KnownFailureTest
from matplotlib.testing.decorators import cleanup
from matplotlib.testing.decorators import CleanupTest


WRITER_OUTPUT = dict(ffmpeg='mp4', ffmpeg_file='mp4',
                     mencoder='mp4', mencoder_file='mp4',
                     avconv='mp4', avconv_file='mp4',
                     imagemagick='gif', imagemagick_file='gif')


# Smoke test for saving animations.  In the future, we should probably
# design more sophisticated tests which compare resulting frames a-la
# matplotlib.testing.image_comparison
def test_save_animation_smoketest():
    for writer, extension in six.iteritems(WRITER_OUTPUT):
        yield check_save_animation, writer, extension


@cleanup
def check_save_animation(writer, extension='mp4'):
    try:
        # for ImageMagick the rcparams must be patched to account for
        # 'convert' being a built in MS tool, not the imagemagick
        # tool.
        writer._init_from_registry()
    except AttributeError:
        pass
    if not animation.writers.is_available(writer):
        raise KnownFailureTest("writer '%s' not available on this system"
                               % writer)
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i)
        line.set_data(x, y)
        return line,

    # Use NamedTemporaryFile: will be automatically deleted
    F = tempfile.NamedTemporaryFile(suffix='.' + extension)
    F.close()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=5)
    try:
        anim.save(F.name, fps=30, writer=writer, bitrate=500)
    except UnicodeDecodeError:
        raise KnownFailureTest("There can be errors in the numpy " +
                               "import stack, " +
                               "see issues #1891 and #2679")
    finally:
        try:
            os.remove(F.name)
        except Exception:
            pass


@cleanup
def test_no_length_frames():
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i)
        line.set_data(x, y)
        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=iter(range(5)))


def test_movie_writer_registry():
    ffmpeg_path = mpl.rcParams['animation.ffmpeg_path']
    # Not sure about the first state as there could be some writer
    # which set rcparams
    #assert_false(animation.writers._dirty)
    assert_true(len(animation.writers._registered) > 0)
    animation.writers.list()  # resets dirty state
    assert_false(animation.writers._dirty)
    mpl.rcParams['animation.ffmpeg_path'] = u"not_available_ever_xxxx"
    assert_true(animation.writers._dirty)
    animation.writers.list()  # resets
    assert_false(animation.writers._dirty)
    assert_false(animation.writers.is_available("ffmpeg"))
    # something which is guaranteed to be available in path
    # and exits immediately
    bin = u"true" if sys.platform != 'win32' else u"where"
    mpl.rcParams['animation.ffmpeg_path'] = bin
    assert_true(animation.writers._dirty)
    animation.writers.list()  # resets
    assert_false(animation.writers._dirty)
    assert_true(animation.writers.is_available("ffmpeg"))
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
