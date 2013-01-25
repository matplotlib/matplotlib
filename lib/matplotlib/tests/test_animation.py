import os
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.noseclasses import KnownFailureTest
from matplotlib.testing.decorators import cleanup


WRITER_OUTPUT = dict(ffmpeg='mp4', ffmpeg_file='mp4',
                     mencoder='mp4', mencoder_file='mp4',
                     avconv='mp4', avconv_file='mp4',
                     imagemagick='gif', imagemagick_file='gif')



# Smoke test for saving animations.  In the future, we should probably
# design more sophisticated tests which compare resulting frames a-la
# matplotlib.testing.image_comparison
@cleanup
def test_save_animation_smoketest():
    for writer, extension in WRITER_OUTPUT.iteritems():
        yield check_save_animation, writer, extension


def check_save_animation(writer, extension='mp4'):
    if not animation.writers.is_available(writer):
        raise KnownFailureTest("writer '%s' not available on this system"
                               % writer)
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
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=5)
    anim.save(F.name, fps=30, writer=writer)


if __name__ == '__main__':
    import nose
    nose.runmodule()
