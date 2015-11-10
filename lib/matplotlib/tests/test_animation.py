from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import os
import tempfile
import numpy as np
from numpy.testing import assert_equal
from nose import with_setup
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.noseclasses import KnownFailureTest
from matplotlib.testing.decorators import cleanup
from matplotlib.testing.decorators import CleanupTest


class NullMovieWriter(animation.AbstractMovieWriter):
    """
    A minimal MovieWriter.  It doesn't actually write anything.
    It just saves the arguments that were given to the setup() and
    grab_frame() methods as attributes, and counts how many times
    grab_frame() is called.

    This class doesn't have an __init__ method with the appropriate
    signature, and it doesn't define an isAvailable() method, so
    it cannot be added to the 'writers' registry.
    """

    def setup(self, fig, outfile, dpi, *args):
        self.fig = fig
        self.outfile = outfile
        self.dpi = dpi
        self.args = args
        self._count = 0

    def grab_frame(self, **savefig_kwargs):
        self.savefig_kwargs = savefig_kwargs
        self._count += 1

    def finish(self):
        pass


def test_null_movie_writer():
    # Test running an animation with NullMovieWriter.

    fig = plt.figure()

    def init():
        pass

    def animate(i):
        pass

    num_frames = 5
    filename = "unused.null"
    fps = 30
    dpi = 50
    savefig_kwargs = dict(foo=0)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=num_frames)
    writer = NullMovieWriter()
    anim.save(filename, fps=fps, dpi=dpi, writer=writer,
              savefig_kwargs=savefig_kwargs)

    assert_equal(writer.fig, fig)
    assert_equal(writer.outfile, filename)
    assert_equal(writer.dpi, dpi)
    assert_equal(writer.args, ())
    assert_equal(writer.savefig_kwargs, savefig_kwargs)
    assert_equal(writer._count, num_frames)


@animation.writers.register('null')
class RegisteredNullMovieWriter(NullMovieWriter):

    # To be able to add NullMovieWriter to the 'writers' registry,
    # we must define an __init__ method with a specific signature,
    # and we must define the class method isAvailable().
    # (These methods are not actually required to use an instance
    # of this class as the 'writer' argument of Animation.save().)

    def __init__(self, fps=None, codec=None, bitrate=None,
                 extra_args=None, metadata=None):
        pass

    @classmethod
    def isAvailable(self):
        return True


WRITER_OUTPUT = dict(ffmpeg='mp4', ffmpeg_file='mp4',
                     mencoder='mp4', mencoder_file='mp4',
                     avconv='mp4', avconv_file='mp4',
                     imagemagick='gif', imagemagick_file='gif',
                     null='null')


# Smoke test for saving animations.  In the future, we should probably
# design more sophisticated tests which compare resulting frames a-la
# matplotlib.testing.image_comparison
def test_save_animation_smoketest():
    for writer, extension in six.iteritems(WRITER_OUTPUT):
        yield check_save_animation, writer, extension


@cleanup
def check_save_animation(writer, extension='mp4'):
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


if __name__ == "__main__":
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
