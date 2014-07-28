from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os
import tempfile
import sys
import numpy as np
from nose import with_setup
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


@with_setup(CleanupTest.setup_class, CleanupTest.teardown_class)
def check_save_animation(writer, extension='mp4'):
    if not animation.writers.is_available(writer):
        raise KnownFailureTest("writer '%s' not available on this system"
                               % writer)
    if 'mencoder' in writer:
        raise KnownFailureTest("mencoder is broken")

    ver = sys.version_info
    if ver[0] == 3 and ver[1] == 2:
        raise KnownFailureTest("animation saving broken on 3.2")

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

    # Use NamedTemporaryFile: will be automatically deleted
    F = tempfile.NamedTemporaryFile(suffix='.' + extension)
    F.close()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=5)
    try:
        anim.save(F.name, fps=30, writer=writer)
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


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
