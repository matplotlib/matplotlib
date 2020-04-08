import os
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation


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


def make_animation(**kwargs):
    fig, ax = plt.subplots()
    line, = ax.plot([])

    def init():
        pass

    def animate(i):
        line.set_data([0, 1], [0, i])
        return line,

    return animation.FuncAnimation(fig, animate, **kwargs)


def test_null_movie_writer():
    # Test running an animation with NullMovieWriter.

    num_frames = 5
    anim = make_animation(frames=num_frames)

    filename = "unused.null"
    dpi = 50
    savefig_kwargs = dict(foo=0)
    writer = NullMovieWriter()

    anim.save(filename, dpi=dpi, writer=writer,
              savefig_kwargs=savefig_kwargs)

    assert writer.fig == plt.figure(1)  # The figure used by make_animation.
    assert writer.outfile == filename
    assert writer.dpi == dpi
    assert writer.args == ()
    assert writer.savefig_kwargs == savefig_kwargs
    assert writer._count == num_frames


def test_movie_writer_dpi_default():
    class DummyMovieWriter(animation.MovieWriter):
        def _run(self):
            pass

    # Test setting up movie writer with figure.dpi default.
    fig = plt.figure()

    filename = "unused.null"
    fps = 5
    codec = "unused"
    bitrate = 1
    extra_args = ["unused"]

    writer = DummyMovieWriter(fps, codec, bitrate, extra_args)
    writer.setup(fig, filename)
    assert writer.dpi == fig.dpi


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
    def isAvailable(cls):
        return True


WRITER_OUTPUT = [
    ('ffmpeg', 'movie.mp4'),
    ('ffmpeg_file', 'movie.mp4'),
    ('avconv', 'movie.mp4'),
    ('avconv_file', 'movie.mp4'),
    ('imagemagick', 'movie.gif'),
    ('imagemagick_file', 'movie.gif'),
    ('pillow', 'movie.gif'),
    ('html', 'movie.html'),
    ('null', 'movie.null')
]
WRITER_OUTPUT += [
    (writer, Path(output)) for writer, output in WRITER_OUTPUT]


# Smoke test for saving animations.  In the future, we should probably
# design more sophisticated tests which compare resulting frames a-la
# matplotlib.testing.image_comparison
@pytest.mark.parametrize('writer, output', WRITER_OUTPUT)
def test_save_animation_smoketest(tmpdir, writer, output):
    if not animation.writers.is_available(writer):
        pytest.skip("writer '%s' not available on this system" % writer)
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)

    dpi = None
    codec = None
    if writer == 'ffmpeg':
        # Issue #8253
        fig.set_size_inches((10.85, 9.21))
        dpi = 100.
        codec = 'h264'

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = np.linspace(0, 10, 100)
        y = np.sin(x + i)
        line.set_data(x, y)
        return line,

    # Use temporary directory for the file-based writers, which produce a file
    # per frame with known names.
    with tmpdir.as_cwd():
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=5)
        anim.save(output, fps=30, writer=writer, bitrate=500, dpi=dpi,
                  codec=codec)


def test_no_length_frames():
    (make_animation(frames=iter(range(5)))
     .save('unused.null', writer=NullMovieWriter()))


def test_movie_writer_registry():
    assert len(animation.writers._registered) > 0
    mpl.rcParams['animation.ffmpeg_path'] = "not_available_ever_xxxx"
    assert not animation.writers.is_available("ffmpeg")
    # something guaranteed to be available in path and exits immediately
    bin = "true" if sys.platform != 'win32' else "where"
    mpl.rcParams['animation.ffmpeg_path'] = bin
    assert animation.writers.is_available("ffmpeg")


@pytest.mark.parametrize(
    "method_name",
    [pytest.param("to_html5_video", marks=pytest.mark.skipif(
        not animation.writers.is_available(mpl.rcParams["animation.writer"]),
        reason="animation writer not installed")),
     "to_jshtml"])
def test_embed_limit(method_name, caplog, tmpdir):
    caplog.set_level("WARNING")
    with tmpdir.as_cwd():
        with mpl.rc_context({"animation.embed_limit": 1e-6}):  # ~1 byte.
            getattr(make_animation(frames=1), method_name)()
    assert len(caplog.records) == 1
    record, = caplog.records
    assert (record.name == "matplotlib.animation"
            and record.levelname == "WARNING")


@pytest.mark.parametrize(
    "method_name",
    [pytest.param("to_html5_video", marks=pytest.mark.skipif(
        not animation.writers.is_available(mpl.rcParams["animation.writer"]),
        reason="animation writer not installed")),
     "to_jshtml"])
def test_cleanup_temporaries(method_name, tmpdir):
    with tmpdir.as_cwd():
        getattr(make_animation(frames=1), method_name)()
        assert list(Path(str(tmpdir)).iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX OS")
def test_failing_ffmpeg(tmpdir, monkeypatch):
    """
    Test that we correctly raise a CalledProcessError when ffmpeg fails.

    To do so, mock ffmpeg using a simple executable shell script that
    succeeds when called with no arguments (so that it gets registered by
    `isAvailable`), but fails otherwise, and add it to the $PATH.
    """
    with tmpdir.as_cwd():
        monkeypatch.setenv("PATH", ".:" + os.environ["PATH"])
        exe_path = Path(str(tmpdir), "ffmpeg")
        exe_path.write_text("#!/bin/sh\n"
                            "[[ $@ -eq 0 ]]\n")
        os.chmod(str(exe_path), 0o755)
        with pytest.raises(subprocess.CalledProcessError):
            make_animation().save("test.mpeg")


@pytest.mark.parametrize("cache_frame_data", [False, True])
def test_funcanimation_cache_frame_data(cache_frame_data):
    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    class Frame(dict):
        # this subclassing enables to use weakref.ref()
        pass

    def init():
        line.set_data([], [])
        return line,

    def animate(frame):
        line.set_data(frame['x'], frame['y'])
        return line,

    frames_generated = []

    def frames_generator():
        for _ in range(5):
            x = np.linspace(0, 10, 100)
            y = np.random.rand(100)

            frame = Frame(x=x, y=y)

            # collect weak references to frames
            # to validate their references later
            frames_generated.append(weakref.ref(frame))

            yield frame

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames_generator,
                                   cache_frame_data=cache_frame_data)

    writer = NullMovieWriter()
    anim.save('unused.null', writer=writer)
    assert len(frames_generated) == 5
    for f in frames_generated:
        # If cache_frame_data is True, then the weakref should be alive;
        # if cache_frame_data is False, then the weakref should be dead (None).
        assert (f() is None) != cache_frame_data
