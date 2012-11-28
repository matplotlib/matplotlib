import os
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.noseclasses import KnownFailureTest


# Smoke test for saving animations.  In the future, we should probably
# design more sophisticated tests which compare resulting frames a-la
# matplotlib.testing.image_comparison
def test_save_animation_smoketest():
    writers = ['ffmpeg', 'ffmpeg_file',
               'mencoder', 'mencoder_file',
               'avconv', 'avconv_file',
               'imagemagick', 'imagemagick_file']

    for writer in writers:
        if writer.startswith('imagemagick'):
            extension = '.gif'
        else:
            extension = '.mp4'

        yield check_save_animation, writer, extension


def check_save_animation(writer, extension='.mp4'):
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

    fid, fname = tempfile.mkstemp(suffix=extension)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=5)
    anim.save(fname, fps=30, writer=writer)

    os.remove(fname)


if __name__ == '__main__':
    import nose
    nose.runmodule()
