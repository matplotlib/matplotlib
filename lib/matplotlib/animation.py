# TODO:
# * Loop Delay is broken on GTKAgg. This is because source_remove() is not
#   working as we want. PyGTK bug?
# * Documentation -- this will need a new section of the User's Guide.
#      Both for Animations and just timers.
#   - Also need to update http://www.scipy.org/Cookbook/Matplotlib/Animations
# * Blit
#   * Currently broken with Qt4 for widgets that don't start on screen
#   * Still a few edge cases that aren't working correctly
#   * Can this integrate better with existing matplotlib animation artist flag?
#     - If animated removes from default draw(), perhaps we could use this to
#       simplify initial draw.
# * Example
#   * Frameless animation - pure procedural with no loop
#   * Need example that uses something like inotify or subprocess
#   * Complex syncing examples
# * Movies
#   * Can blit be enabled for movies?
# * Need to consider event sources to allow clicking through multiple figures
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six.moves import xrange, zip

import os
import platform
import sys
import itertools
try:
    # python3
    from base64 import encodebytes
except ImportError:
    # python2
    from base64 import encodestring as encodebytes
import contextlib
import tempfile
import warnings
from matplotlib.cbook import iterable, is_string_like
from matplotlib.compat import subprocess
from matplotlib import verbose
from matplotlib import rcParams, rcParamsDefault, rc_context

# Process creation flag for subprocess to prevent it raising a terminal
# window. See for example:
# https://stackoverflow.com/questions/24130623/using-python-subprocess-popen-cant-prevent-exe-stopped-working-prompt
if platform.system() == 'Windows':
    CREATE_NO_WINDOW = 0x08000000
    subprocess_creation_flags = CREATE_NO_WINDOW
else:
    # Apparently None won't work here
    subprocess_creation_flags = 0

# Other potential writing methods:
# * http://pymedia.org/
# * libmng (produces swf) python wrappers: https://github.com/libming/libming
# * Wrap x264 API:

# (http://stackoverflow.com/questions/2940671/
# how-to-encode-series-of-images-into-h264-using-x264-api-c-c )


# A registry for available MovieWriter classes
class MovieWriterRegistry(object):
    def __init__(self):
        self.avail = dict()

    # Returns a decorator that can be used on classes to register them under
    # a name. As in:
    # @register('foo')
    # class Foo:
    #    pass
    def register(self, name):
        def wrapper(writerClass):
            if writerClass.isAvailable():
                self.avail[name] = writerClass
            return writerClass
        return wrapper

    def list(self):
        ''' Get a list of available MovieWriters.'''
        return list(self.avail.keys())

    def is_available(self, name):
        return name in self.avail

    def __getitem__(self, name):
        if not self.avail:
            raise RuntimeError("No MovieWriters available!")
        return self.avail[name]

writers = MovieWriterRegistry()


class MovieWriter(object):
    '''
    Base class for writing movies. Fundamentally, what a MovieWriter does
    is provide is a way to grab frames by calling grab_frame(). setup()
    is called to start the process and finish() is called afterwards.
    This class is set up to provide for writing movie frame data to a pipe.
    saving() is provided as a context manager to facilitate this process as::

      with moviewriter.saving('myfile.mp4'):
          # Iterate over frames
          moviewriter.grab_frame()

    The use of the context manager ensures that setup and cleanup are
    performed as necessary.

    frame_format: string
        The format used in writing frame data, defaults to 'rgba'
    '''

    # Specifies whether the size of all frames need to be identical
    # i.e. whether we can use savefig.bbox = 'tight'
    frame_size_can_vary = False

    def __init__(self, fps=5, codec=None, bitrate=None, extra_args=None,
                 metadata=None):
        '''
        Construct a new MovieWriter object.

        fps: int
            Framerate for movie.
        codec: string or None, optional
            The codec to use. If None (the default) the setting in the
            rcParam `animation.codec` is used.
        bitrate: int or None, optional
            The bitrate for the saved movie file, which is one way to control
            the output file size and quality. The default value is None,
            which uses the value stored in the rcParam `animation.bitrate`.
            A value of -1 implies that the bitrate should be determined
            automatically by the underlying utility.
        extra_args: list of strings or None
            A list of extra string arguments to be passed to the underlying
            movie utility. The default is None, which passes the additional
            arguments in the 'animation.extra_args' rcParam.
        metadata: dict of string:string or None
            A dictionary of keys and values for metadata to include in the
            output file. Some keys that may be of use include:
            title, artist, genre, subject, copyright, srcform, comment.
        '''
        self.fps = fps
        self.frame_format = 'rgba'

        if codec is None:
            self.codec = rcParams['animation.codec']
        else:
            self.codec = codec

        if bitrate is None:
            self.bitrate = rcParams['animation.bitrate']
        else:
            self.bitrate = bitrate

        if extra_args is None:
            self.extra_args = list(rcParams[self.args_key])
        else:
            self.extra_args = extra_args

        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

    @property
    def frame_size(self):
        'A tuple (width,height) in pixels of a movie frame.'
        width_inches, height_inches = self.fig.get_size_inches()
        return width_inches * self.dpi, height_inches * self.dpi

    def setup(self, fig, outfile, dpi, *args):
        '''
        Perform setup for writing the movie file.

        fig: `matplotlib.Figure` instance
            The figure object that contains the information for frames
        outfile: string
            The filename of the resulting movie file
        dpi: int
            The DPI (or resolution) for the file.  This controls the size
            in pixels of the resulting movie file.
        '''
        self.outfile = outfile
        self.fig = fig
        self.dpi = dpi

        # Run here so that grab_frame() can write the data to a pipe. This
        # eliminates the need for temp files.
        self._run()

    @contextlib.contextmanager
    def saving(self, *args):
        '''
        Context manager to facilitate writing the movie file.

        ``*args`` are any parameters that should be passed to `setup`.
        '''
        # This particular sequence is what contextlib.contextmanager wants
        self.setup(*args)
        yield
        self.finish()

    def _run(self):
        # Uses subprocess to call the program for assembling frames into a
        # movie file.  *args* returns the sequence of command line arguments
        # from a few configuration options.
        command = self._args()
        if verbose.ge('debug'):
            output = sys.stdout
        else:
            output = subprocess.PIPE
        verbose.report('MovieWriter.run: running command: %s' %
                       ' '.join(command))
        self._proc = subprocess.Popen(command, shell=False,
                                      stdout=output, stderr=output,
                                      stdin=subprocess.PIPE,
                                      creationflags=subprocess_creation_flags)

    def finish(self):
        'Finish any processing for writing the movie.'
        self.cleanup()

    def grab_frame(self, **savefig_kwargs):
        '''
        Grab the image information from the figure and save as a movie frame.
        All keyword arguments in savefig_kwargs are passed on to the 'savefig'
        command that saves the figure.
        '''
        verbose.report('MovieWriter.grab_frame: Grabbing frame.',
                       level='debug')
        try:
            # Tell the figure to save its data to the sink, using the
            # frame format and dpi.
            self.fig.savefig(self._frame_sink(), format=self.frame_format,
                             dpi=self.dpi, **savefig_kwargs)
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            verbose.report('MovieWriter -- Error '
                           'running proc:\n%s\n%s' % (out,
                                                      err), level='helpful')
            raise IOError('Error saving animation to file (cause: {0}) '
                          'Stdout: {1} StdError: {2}. It may help to re-run '
                          'with --verbose-debug.'.format(e, out, err))

    def _frame_sink(self):
        'Returns the place to which frames should be written.'
        return self._proc.stdin

    def _args(self):
        'Assemble list of utility-specific command-line arguments.'
        return NotImplementedError("args needs to be implemented by subclass.")

    def cleanup(self):
        'Clean-up and collect the process used to write the movie file.'
        out, err = self._proc.communicate()
        self._frame_sink().close()
        verbose.report('MovieWriter -- '
                       'Command stdout:\n%s' % out, level='debug')
        verbose.report('MovieWriter -- '
                       'Command stderr:\n%s' % err, level='debug')

    @classmethod
    def bin_path(cls):
        '''
        Returns the binary path to the commandline tool used by a specific
        subclass. This is a class method so that the tool can be looked for
        before making a particular MovieWriter subclass available.
        '''
        return rcParams[cls.exec_key]

    @classmethod
    def isAvailable(cls):
        '''
        Check to see if a MovieWriter subclass is actually available by
        running the commandline tool.
        '''
        if not cls.bin_path():
            return False
        try:
            p = subprocess.Popen(cls.bin_path(),
                             shell=False,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             creationflags=subprocess_creation_flags)
            p.communicate()
            return True
        except OSError:
            return False


class FileMovieWriter(MovieWriter):
    '`MovieWriter` subclass that handles writing to a file.'

    # In general, if frames are writen to files on disk, it's not important
    # that they all be identically sized
    frame_size_can_vary = True

    def __init__(self, *args, **kwargs):
        MovieWriter.__init__(self, *args, **kwargs)
        self.frame_format = rcParams['animation.frame_format']

    def setup(self, fig, outfile, dpi, frame_prefix='_tmp', clear_temp=True):
        '''
        Perform setup for writing the movie file.

        fig: `matplotlib.Figure` instance
            The figure object that contains the information for frames
        outfile: string
            The filename of the resulting movie file
        dpi: int
            The DPI (or resolution) for the file.  This controls the size
            in pixels of the resulting movie file.
        frame_prefix: string, optional
            The filename prefix to use for the temporary files. Defaults
            to '_tmp'
        clear_temp: bool
            Specifies whether the temporary files should be deleted after
            the movie is written. (Useful for debugging.) Defaults to True.
        '''
        self.fig = fig
        self.outfile = outfile
        self.dpi = dpi
        self.clear_temp = clear_temp
        self.temp_prefix = frame_prefix
        self._frame_counter = 0  # used for generating sequential file names
        self._temp_names = list()
        self.fname_format_str = '%s%%07d.%s'

    @property
    def frame_format(self):
        '''
        Format (png, jpeg, etc.) to use for saving the frames, which can be
        decided by the individual subclasses.
        '''
        return self._frame_format

    @frame_format.setter
    def frame_format(self, frame_format):
        if frame_format in self.supported_formats:
            self._frame_format = frame_format
        else:
            self._frame_format = self.supported_formats[0]

    def _base_temp_name(self):
        # Generates a template name (without number) given the frame format
        # for extension and the prefix.
        return self.fname_format_str % (self.temp_prefix, self.frame_format)

    def _frame_sink(self):
        # Creates a filename for saving using the basename and the current
        # counter.
        fname = self._base_temp_name() % self._frame_counter

        # Save the filename so we can delete it later if necessary
        self._temp_names.append(fname)
        verbose.report(
            'FileMovieWriter.frame_sink: saving frame %d to fname=%s' %
            (self._frame_counter, fname),
            level='debug')
        self._frame_counter += 1  # Ensures each created name is 'unique'

        # This file returned here will be closed once it's used by savefig()
        # because it will no longer be referenced and will be gc-ed.
        return open(fname, 'wb')

    def grab_frame(self, **savefig_kwargs):
        '''
        Grab the image information from the figure and save as a movie frame.
        All keyword arguments in savefig_kwargs are passed on to the 'savefig'
        command that saves the figure.
        '''
        # Overloaded to explicitly close temp file.
        verbose.report('MovieWriter.grab_frame: Grabbing frame.',
                       level='debug')
        try:
            # Tell the figure to save its data to the sink, using the
            # frame format and dpi.
            myframesink = self._frame_sink()
            self.fig.savefig(myframesink, format=self.frame_format,
                             dpi=self.dpi, **savefig_kwargs)
            myframesink.close()

        except RuntimeError:
            out, err = self._proc.communicate()
            verbose.report('MovieWriter -- Error '
                           'running proc:\n%s\n%s' % (out,
                                                      err), level='helpful')
            raise

    def finish(self):
        # Call run here now that all frame grabbing is done. All temp files
        # are available to be assembled.
        self._run()
        MovieWriter.finish(self)  # Will call clean-up

        # Check error code for creating file here, since we just run
        # the process here, rather than having an open pipe.
        if self._proc.returncode:
            raise RuntimeError('Error creating movie, return code: '
                               + str(self._proc.returncode)
                               + ' Try running with --verbose-debug')

    def cleanup(self):
        MovieWriter.cleanup(self)

        # Delete temporary files
        if self.clear_temp:
            verbose.report(
                'MovieWriter: clearing temporary fnames=%s' %
                str(self._temp_names),
                level='debug')
            for fname in self._temp_names:
                os.remove(fname)


# Base class of ffmpeg information. Has the config keys and the common set
# of arguments that controls the *output* side of things.
class FFMpegBase(object):
    exec_key = 'animation.ffmpeg_path'
    args_key = 'animation.ffmpeg_args'

    @property
    def output_args(self):
        args = ['-vcodec', self.codec]
        # For h264, the default format is yuv444p, which is not compatible
        # with quicktime (and others). Specifying yuv420p fixes playback on
        # iOS,as well as HTML5 video in firefox and safari (on both Win and
        # OSX). Also fixes internet explorer. This is as of 2015/10/29.
        if self.codec == 'h264' and '-pix_fmt' not in self.extra_args:
            args.extend(['-pix_fmt', 'yuv420p'])
        # The %dk adds 'k' as a suffix so that ffmpeg treats our bitrate as in
        # kbps
        if self.bitrate > 0:
            args.extend(['-b', '%dk' % self.bitrate])
        if self.extra_args:
            args.extend(self.extra_args)
        for k, v in six.iteritems(self.metadata):
            args.extend(['-metadata', '%s=%s' % (k, v)])

        return args + ['-y', self.outfile]


# Combine FFMpeg options with pipe-based writing
@writers.register('ffmpeg')
class FFMpegWriter(MovieWriter, FFMpegBase):
    def _args(self):
        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie using a pipe.
        args = [self.bin_path(), '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', '%dx%d' % self.frame_size, '-pix_fmt', self.frame_format,
                '-r', str(self.fps)]
        # Logging is quieted because subprocess.PIPE has limited buffer size.
        if not verbose.ge('debug'):
            args += ['-loglevel', 'quiet']
        args += ['-i', 'pipe:'] + self.output_args
        return args


# Combine FFMpeg options with temp file-based writing
@writers.register('ffmpeg_file')
class FFMpegFileWriter(FileMovieWriter, FFMpegBase):
    supported_formats = ['png', 'jpeg', 'ppm', 'tiff', 'sgi', 'bmp',
                         'pbm', 'raw', 'rgba']

    def _args(self):
        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie using a collection of temp images
        return [self.bin_path(), '-i', self._base_temp_name(),
                '-vframes', str(self._frame_counter),
                '-r', str(self.fps)] + self.output_args


# Base class of avconv information.  AVConv has identical arguments to
# FFMpeg
class AVConvBase(FFMpegBase):
    exec_key = 'animation.avconv_path'
    args_key = 'animation.avconv_args'


# Combine AVConv options with pipe-based writing
@writers.register('avconv')
class AVConvWriter(AVConvBase, FFMpegWriter):
    pass


# Combine AVConv options with file-based writing
@writers.register('avconv_file')
class AVConvFileWriter(AVConvBase, FFMpegFileWriter):
    pass


# Base class of mencoder information. Contains configuration key information
# as well as arguments for controlling *output*
class MencoderBase(object):
    exec_key = 'animation.mencoder_path'
    args_key = 'animation.mencoder_args'

    # Mencoder only allows certain keys, other ones cause the program
    # to fail.
    allowed_metadata = ['name', 'artist', 'genre', 'subject', 'copyright',
                        'srcform', 'comment']

    # Mencoder mandates using name, but 'title' works better with ffmpeg.
    # If we find it, just put it's value into name
    def _remap_metadata(self):
        if 'title' in self.metadata:
            self.metadata['name'] = self.metadata['title']

    @property
    def output_args(self):
        self._remap_metadata()
        lavcopts = {'vcodec': self.codec}
        if self.bitrate > 0:
            lavcopts.update(vbitrate=self.bitrate)
        args = ['-o', self.outfile, '-ovc', 'lavc', '-lavcopts',
                ':'.join(itertools.starmap('{0}={1}'.format,
                                           lavcopts.items()))]
        if self.extra_args:
            args.extend(self.extra_args)
        if self.metadata:
            args.extend(['-info', ':'.join('%s=%s' % (k, v)
                         for k, v in six.iteritems(self.metadata)
                         if k in self.allowed_metadata)])
        return args


# Combine Mencoder options with pipe-based writing
@writers.register('mencoder')
class MencoderWriter(MovieWriter, MencoderBase):
    def _args(self):
        # Returns the command line parameters for subprocess to use
        # mencoder to create a movie
        return [self.bin_path(), '-', '-demuxer', 'rawvideo', '-rawvideo',
                ('w=%i:h=%i:' % self.frame_size +
                'fps=%i:format=%s' % (self.fps,
                                      self.frame_format))] + self.output_args


# Combine Mencoder options with temp file-based writing
@writers.register('mencoder_file')
class MencoderFileWriter(FileMovieWriter, MencoderBase):
    supported_formats = ['png', 'jpeg', 'tga', 'sgi']

    def _args(self):
        # Returns the command line parameters for subprocess to use
        # mencoder to create a movie
        return [self.bin_path(),
                'mf://%s*.%s' % (self.temp_prefix, self.frame_format),
                '-frames', str(self._frame_counter), '-mf',
                'type=%s:fps=%d' % (self.frame_format,
                                    self.fps)] + self.output_args


# Base class for animated GIFs with convert utility
class ImageMagickBase(object):
    exec_key = 'animation.convert_path'
    args_key = 'animation.convert_args'

    @property
    def delay(self):
        return 100. / self.fps

    @property
    def output_args(self):
        return [self.outfile]

    @classmethod
    def _init_from_registry(cls):
        if sys.platform != 'win32' or rcParams[cls.exec_key] != 'convert':
            return
        from matplotlib.externals.six.moves import winreg
        for flag in (0, winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
            try:
                hkey = winreg.OpenKeyEx(winreg.HKEY_LOCAL_MACHINE,
                                        'Software\\Imagemagick\\Current',
                                        0, winreg.KEY_QUERY_VALUE | flag)
                binpath = winreg.QueryValueEx(hkey, 'BinPath')[0]
                winreg.CloseKey(hkey)
                binpath += '\\convert.exe'
                break
            except Exception:
                binpath = ''
        rcParams[cls.exec_key] = rcParamsDefault[cls.exec_key] = binpath


ImageMagickBase._init_from_registry()


@writers.register('imagemagick')
class ImageMagickWriter(MovieWriter, ImageMagickBase):
    def _args(self):
        return ([self.bin_path(),
                 '-size', '%ix%i' % self.frame_size, '-depth', '8',
                 '-delay', str(self.delay), '-loop', '0',
                 '%s:-' % self.frame_format]
                + self.output_args)


@writers.register('imagemagick_file')
class ImageMagickFileWriter(FileMovieWriter, ImageMagickBase):
    supported_formats = ['png', 'jpeg', 'ppm', 'tiff', 'sgi', 'bmp',
                         'pbm', 'raw', 'rgba']

    def _args(self):
        return ([self.bin_path(), '-delay', str(self.delay), '-loop', '0',
                 '%s*.%s' % (self.temp_prefix, self.frame_format)]
                + self.output_args)


class Animation(object):
    '''
    This class wraps the creation of an animation using matplotlib. It is
    only a base class which should be subclassed to provide needed behavior.

    *fig* is the figure object that is used to get draw, resize, and any
    other needed events.

    *event_source* is a class that can run a callback when desired events
    are generated, as well as be stopped and started. Examples include timers
    (see :class:`TimedAnimation`) and file system notifications.

    *blit* is a boolean that controls whether blitting is used to optimize
    drawing.
    '''
    def __init__(self, fig, event_source=None, blit=False):
        self._fig = fig
        # Disables blitting for backends that don't support it.  This
        # allows users to request it if available, but still have a
        # fallback that works if it is not.
        self._blit = blit and fig.canvas.supports_blit

        # These are the basics of the animation.  The frame sequence represents
        # information for each frame of the animation and depends on how the
        # drawing is handled by the subclasses. The event source fires events
        # that cause the frame sequence to be iterated.
        self.frame_seq = self.new_frame_seq()
        self.event_source = event_source

        # Instead of starting the event source now, we connect to the figure's
        # draw_event, so that we only start once the figure has been drawn.
        self._first_draw_id = fig.canvas.mpl_connect('draw_event', self._start)

        # Connect to the figure's close_event so that we don't continue to
        # fire events and try to draw to a deleted figure.
        self._close_id = self._fig.canvas.mpl_connect('close_event',
                                                      self._stop)
        if self._blit:
            self._setup_blit()

    def _start(self, *args):
        '''
        Starts interactive animation. Adds the draw frame command to the GUI
        handler, calls show to start the event loop.
        '''
        # First disconnect our draw event handler
        self._fig.canvas.mpl_disconnect(self._first_draw_id)
        self._first_draw_id = None  # So we can check on save

        # Now do any initial draw
        self._init_draw()

        # Add our callback for stepping the animation and
        # actually start the event_source.
        self.event_source.add_callback(self._step)
        self.event_source.start()

    def _stop(self, *args):
        # On stop we disconnect all of our events.
        if self._blit:
            self._fig.canvas.mpl_disconnect(self._resize_id)
        self._fig.canvas.mpl_disconnect(self._close_id)
        self.event_source.remove_callback(self._step)
        self.event_source = None

    def save(self, filename, writer=None, fps=None, dpi=None, codec=None,
             bitrate=None, extra_args=None, metadata=None, extra_anim=None,
             savefig_kwargs=None):
        '''
        Saves a movie file by drawing every frame.

        *filename* is the output filename, e.g., :file:`mymovie.mp4`

        *writer* is either an instance of :class:`MovieWriter` or a string
        key that identifies a class to use, such as 'ffmpeg' or 'mencoder'.
        If nothing is passed, the value of the rcparam `animation.writer` is
        used.

        *dpi* controls the dots per inch for the movie frames. This combined
        with the figure's size in inches controls the size of the movie.

        *savefig_kwargs* is a dictionary containing keyword arguments to be
        passed on to the 'savefig' command which is called repeatedly to save
        the individual frames. This can be used to set tight bounding boxes,
        for example.

        *extra_anim* is a list of additional `Animation` objects that should
        be included in the saved movie file. These need to be from the same
        `matplotlib.Figure` instance. Also, animation frames will just be
        simply combined, so there should be a 1:1 correspondence between
        the frames from the different animations.

        These remaining arguments are used to construct a :class:`MovieWriter`
        instance when necessary and are only considered valid if *writer* is
        not a :class:`MovieWriter` instance.

        *fps* is the frames per second in the movie. Defaults to None,
        which will use the animation's specified interval to set the frames
        per second.

        *codec* is the video codec to be used. Not all codecs are supported
        by a given :class:`MovieWriter`. If none is given, this defaults to the
        value specified by the rcparam `animation.codec`.

        *bitrate* specifies the amount of bits used per second in the
        compressed movie, in kilobits per second. A higher number means a
        higher quality movie, but at the cost of increased file size. If no
        value is given, this defaults to the value given by the rcparam
        `animation.bitrate`.

        *extra_args* is a list of extra string arguments to be passed to the
        underlying movie utility. The default is None, which passes the
        additional arguments in the 'animation.extra_args' rcParam.

        *metadata* is a dictionary of keys and values for metadata to include
        in the output file. Some keys that may be of use include:
        title, artist, genre, subject, copyright, srcform, comment.
        '''
        # If the writer is None, use the rc param to find the name of the one
        # to use
        if writer is None:
            writer = rcParams['animation.writer']
        elif (not is_string_like(writer) and
                any(arg is not None
                    for arg in (fps, codec, bitrate, extra_args, metadata))):
            raise RuntimeError('Passing in values for arguments for arguments '
                               'fps, codec, bitrate, extra_args, or metadata '
                               'is not supported when writer is an existing '
                               'MovieWriter instance. These should instead be '
                               'passed as arguments when creating the '
                               'MovieWriter instance.')

        if savefig_kwargs is None:
            savefig_kwargs = {}

        # Need to disconnect the first draw callback, since we'll be doing
        # draws. Otherwise, we'll end up starting the animation.
        if self._first_draw_id is not None:
            self._fig.canvas.mpl_disconnect(self._first_draw_id)
            reconnect_first_draw = True
        else:
            reconnect_first_draw = False

        if fps is None and hasattr(self, '_interval'):
            # Convert interval in ms to frames per second
            fps = 1000. / self._interval

        # Re-use the savefig DPI for ours if none is given
        if dpi is None:
            dpi = rcParams['savefig.dpi']
        if dpi == 'figure':
            dpi = self._fig.dpi

        if codec is None:
            codec = rcParams['animation.codec']

        if bitrate is None:
            bitrate = rcParams['animation.bitrate']

        all_anim = [self]
        if extra_anim is not None:
            all_anim.extend(anim
                            for anim
                            in extra_anim if anim._fig is self._fig)

        # If we have the name of a writer, instantiate an instance of the
        # registered class.
        if is_string_like(writer):
            if writer in writers.avail:
                writer = writers[writer](fps, codec, bitrate,
                                         extra_args=extra_args,
                                         metadata=metadata)
            else:
                warnings.warn("MovieWriter %s unavailable" % writer)

                try:
                    writer = writers[writers.list()[0]](fps, codec, bitrate,
                                                        extra_args=extra_args,
                                                        metadata=metadata)
                except IndexError:
                    raise ValueError("Cannot save animation: no writers are "
                                     "available. Please install mencoder or "
                                     "ffmpeg to save animations.")

        verbose.report('Animation.save using %s' % type(writer),
                       level='helpful')

        # FIXME: Using 'bbox_inches' doesn't currently work with
        # writers that pipe the data to the command because this
        # requires a fixed frame size (see Ryan May's reply in this
        # thread: [1]). Thus we drop the 'bbox_inches' argument if it
        # exists in savefig_kwargs.
        #
        # [1] (http://matplotlib.1069221.n5.nabble.com/
        # Animation-class-let-save-accept-kwargs-which-
        # are-passed-on-to-savefig-td39627.html)
        #
        if 'bbox_inches' in savefig_kwargs and not writer.frame_size_can_vary:
            warnings.warn("Warning: discarding the 'bbox_inches' argument in "
                          "'savefig_kwargs' as it not supported by "
                          "{0}).".format(writer.__class__.__name__))
            savefig_kwargs.pop('bbox_inches')

        # Create a new sequence of frames for saved data. This is different
        # from new_frame_seq() to give the ability to save 'live' generated
        # frame information to be saved later.
        # TODO: Right now, after closing the figure, saving a movie won't work
        # since GUI widgets are gone. Either need to remove extra code to
        # allow for this non-existent use case or find a way to make it work.
        with rc_context():
            # See above about bbox_inches savefig kwarg
            if (not writer.frame_size_can_vary and
                    rcParams['savefig.bbox'] == 'tight'):
                verbose.report("Disabling savefig.bbox = 'tight', as it is "
                               "not supported by "
                               "{0}.".format(writer.__class__.__name__),
                               level='helpful')
                rcParams['savefig.bbox'] = None
            with writer.saving(self._fig, filename, dpi):
                for anim in all_anim:
                    # Clear the initial frame
                    anim._init_draw()
                for data in zip(*[a.new_saved_frame_seq()
                                  for a in all_anim]):
                    for anim, d in zip(all_anim, data):
                        # TODO: See if turning off blit is really necessary
                        anim._draw_next_frame(d, blit=False)
                    writer.grab_frame(**savefig_kwargs)

        # Reconnect signal for first draw if necessary
        if reconnect_first_draw:
            self._first_draw_id = self._fig.canvas.mpl_connect('draw_event',
                                                               self._start)

    def _step(self, *args):
        '''
        Handler for getting events. By default, gets the next frame in the
        sequence and hands the data off to be drawn.
        '''
        # Returns True to indicate that the event source should continue to
        # call _step, until the frame sequence reaches the end of iteration,
        # at which point False will be returned.
        try:
            framedata = next(self.frame_seq)
            self._draw_next_frame(framedata, self._blit)
            return True
        except StopIteration:
            return False

    def new_frame_seq(self):
        'Creates a new sequence of frame information.'
        # Default implementation is just an iterator over self._framedata
        return iter(self._framedata)

    def new_saved_frame_seq(self):
        'Creates a new sequence of saved/cached frame information.'
        # Default is the same as the regular frame sequence
        return self.new_frame_seq()

    def _draw_next_frame(self, framedata, blit):
        # Breaks down the drawing of the next frame into steps of pre- and
        # post- draw, as well as the drawing of the frame itself.
        self._pre_draw(framedata, blit)
        self._draw_frame(framedata)
        self._post_draw(framedata, blit)

    def _init_draw(self):
        # Initial draw to clear the frame. Also used by the blitting code
        # when a clean base is required.
        pass

    def _pre_draw(self, framedata, blit):
        # Perform any cleaning or whatnot before the drawing of the frame.
        # This default implementation allows blit to clear the frame.
        if blit:
            self._blit_clear(self._drawn_artists, self._blit_cache)

    def _draw_frame(self, framedata):
        # Performs actual drawing of the frame.
        raise NotImplementedError('Needs to be implemented by subclasses to'
                                  ' actually make an animation.')

    def _post_draw(self, framedata, blit):
        # After the frame is rendered, this handles the actual flushing of
        # the draw, which can be a direct draw_idle() or make use of the
        # blitting.
        if blit and self._drawn_artists:
            self._blit_draw(self._drawn_artists, self._blit_cache)
        else:
            self._fig.canvas.draw_idle()

    # The rest of the code in this class is to facilitate easy blitting
    def _blit_draw(self, artists, bg_cache):
        # Handles blitted drawing, which renders only the artists given instead
        # of the entire figure.
        updated_ax = []
        for a in artists:
            # If we haven't cached the background for this axes object, do
            # so now. This might not always be reliable, but it's an attempt
            # to automate the process.
            if a.axes not in bg_cache:
                bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
            a.axes.draw_artist(a)
            updated_ax.append(a.axes)

        # After rendering all the needed artists, blit each axes individually.
        for ax in set(updated_ax):
            ax.figure.canvas.blit(ax.bbox)

    def _blit_clear(self, artists, bg_cache):
        # Get a list of the axes that need clearing from the artists that
        # have been drawn. Grab the appropriate saved background from the
        # cache and restore.
        axes = set(a.axes for a in artists)
        for a in axes:
            a.figure.canvas.restore_region(bg_cache[a])

    def _setup_blit(self):
        # Setting up the blit requires: a cache of the background for the
        # axes
        self._blit_cache = dict()
        self._drawn_artists = []
        self._resize_id = self._fig.canvas.mpl_connect('resize_event',
                                                       self._handle_resize)
        self._post_draw(None, self._blit)

    def _handle_resize(self, *args):
        # On resize, we need to disable the resize event handling so we don't
        # get too many events. Also stop the animation events, so that
        # we're paused. Reset the cache and re-init. Set up an event handler
        # to catch once the draw has actually taken place.
        self._fig.canvas.mpl_disconnect(self._resize_id)
        self.event_source.stop()
        self._blit_cache.clear()
        self._init_draw()
        self._resize_id = self._fig.canvas.mpl_connect('draw_event',
                                                       self._end_redraw)

    def _end_redraw(self, evt):
        # Now that the redraw has happened, do the post draw flushing and
        # blit handling. Then re-enable all of the original events.
        self._post_draw(None, self._blit)
        self.event_source.start()
        self._fig.canvas.mpl_disconnect(self._resize_id)
        self._resize_id = self._fig.canvas.mpl_connect('resize_event',
                                                       self._handle_resize)

    def to_html5_video(self):
        r'''Returns animation as an HTML5 video tag.

        This saves the animation as an h264 video, encoded in base64
        directly into the HTML5 video tag. This respects the rc parameters
        for the writer as well as the bitrate. This also makes use of the
        ``interval`` to control the speed, and uses the ``repeat``
        parameter to decide whether to loop.
        '''
        VIDEO_TAG = r'''<video {size} {options}>
  <source type="video/mp4" src="data:video/mp4;base64,{video}">
  Your browser does not support the video tag.
</video>'''
        # Cache the the rendering of the video as HTML
        if not hasattr(self, '_base64_video'):
            # First write the video to a tempfile. Set delete to False
            # so we can re-open to read binary data.
            with tempfile.NamedTemporaryFile(suffix='.m4v',
                                             delete=False) as f:
                # We create a writer manually so that we can get the
                # appropriate size for the tag
                Writer = writers[rcParams['animation.writer']]
                writer = Writer(codec='h264',
                                bitrate=rcParams['animation.bitrate'],
                                fps=1000. / self._interval)
                self.save(f.name, writer=writer)

            # Now open and base64 encode
            with open(f.name, 'rb') as video:
                vid64 = encodebytes(video.read())
                self._base64_video = vid64.decode('ascii')
                self._video_size = 'width="{0}" height="{1}"'.format(
                        *writer.frame_size)

            # Now we can remove
            os.remove(f.name)

        # Default HTML5 options are to autoplay and to display video controls
        options = ['controls', 'autoplay']

        # If we're set to repeat, make it loop
        if self.repeat:
            options.append('loop')
        return VIDEO_TAG.format(video=self._base64_video,
                                size=self._video_size,
                                options=' '.join(options))

    def _repr_html_(self):
        r'IPython display hook for rendering.'
        fmt = rcParams['animation.html']
        if fmt == 'html5':
            return self.to_html5_video()


class TimedAnimation(Animation):
    '''
    :class:`Animation` subclass that supports time-based animation, drawing
    a new frame every *interval* milliseconds.

    *repeat* controls whether the animation should repeat when the sequence
    of frames is completed.

    *repeat_delay* optionally adds a delay in milliseconds before repeating
    the animation.
    '''
    def __init__(self, fig, interval=200, repeat_delay=None, repeat=True,
                 event_source=None, *args, **kwargs):
        # Store the timing information
        self._interval = interval
        self._repeat_delay = repeat_delay
        self.repeat = repeat

        # If we're not given an event source, create a new timer. This permits
        # sharing timers between animation objects for syncing animations.
        if event_source is None:
            event_source = fig.canvas.new_timer()
            event_source.interval = self._interval

        Animation.__init__(self, fig, event_source=event_source,
                           *args, **kwargs)

    def _step(self, *args):
        '''
        Handler for getting events.
        '''
        # Extends the _step() method for the Animation class.  If
        # Animation._step signals that it reached the end and we want to
        # repeat, we refresh the frame sequence and return True. If
        # _repeat_delay is set, change the event_source's interval to our loop
        # delay and set the callback to one which will then set the interval
        # back.
        still_going = Animation._step(self, *args)
        if not still_going and self.repeat:
            self._init_draw()
            self.frame_seq = self.new_frame_seq()
            if self._repeat_delay:
                self.event_source.remove_callback(self._step)
                self.event_source.add_callback(self._loop_delay)
                self.event_source.interval = self._repeat_delay
                return True
            else:
                return Animation._step(self, *args)
        else:
            return still_going

    def _stop(self, *args):
        # If we stop in the middle of a loop delay (which is relatively likely
        # given the potential pause here, remove the loop_delay callback as
        # well.
        self.event_source.remove_callback(self._loop_delay)
        Animation._stop(self)

    def _loop_delay(self, *args):
        # Reset the interval and change callbacks after the delay.
        self.event_source.remove_callback(self._loop_delay)
        self.event_source.interval = self._interval
        self.event_source.add_callback(self._step)
        Animation._step(self)


class ArtistAnimation(TimedAnimation):
    '''
    Before calling this function, all plotting should have taken place
    and the relevant artists saved.

    frame_info is a list, with each list entry a collection of artists that
    represent what needs to be enabled on each frame. These will be disabled
    for other frames.
    '''
    def __init__(self, fig, artists, *args, **kwargs):
        # Internal list of artists drawn in the most recent frame.
        self._drawn_artists = []

        # Use the list of artists as the framedata, which will be iterated
        # over by the machinery.
        self._framedata = artists
        TimedAnimation.__init__(self, fig, *args, **kwargs)

    def _init_draw(self):
        # Make all the artists involved in *any* frame invisible
        figs = set()
        for f in self.new_frame_seq():
            for artist in f:
                artist.set_visible(False)
                artist.set_animated(self._blit)
                # Assemble a list of unique axes that need flushing
                if artist.axes.figure not in figs:
                    figs.add(artist.axes.figure)

        # Flush the needed axes
        for fig in figs:
            fig.canvas.draw_idle()

    def _pre_draw(self, framedata, blit):
        '''
        Clears artists from the last frame.
        '''
        if blit:
            # Let blit handle clearing
            self._blit_clear(self._drawn_artists, self._blit_cache)
        else:
            # Otherwise, make all the artists from the previous frame invisible
            for artist in self._drawn_artists:
                artist.set_visible(False)

    def _draw_frame(self, artists):
        # Save the artists that were passed in as framedata for the other
        # steps (esp. blitting) to use.
        self._drawn_artists = artists

        # Make all the artists from the current frame visible
        for artist in artists:
            artist.set_visible(True)


class FuncAnimation(TimedAnimation):
    '''
    Makes an animation by repeatedly calling a function *func*, passing in
    (optional) arguments in *fargs*.

    *frames* can be a generator, an iterable, or a number of frames.

    *init_func* is a function used to draw a clear frame. If not given, the
    results of drawing from the first item in the frames sequence will be
    used. This function will be called once before the first frame.

    If blit=True, *func* and *init_func* must return an iterable of
    artists to be re-drawn.

    *kwargs* include *repeat*, *repeat_delay*, and *interval*:
    *interval* draws a new frame every *interval* milliseconds.
    *repeat* controls whether the animation should repeat when the sequence
    of frames is completed.
    *repeat_delay* optionally adds a delay in milliseconds before repeating
    the animation.
    '''
    def __init__(self, fig, func, frames=None, init_func=None, fargs=None,
                 save_count=None, **kwargs):
        if fargs:
            self._args = fargs
        else:
            self._args = ()
        self._func = func

        # Amount of framedata to keep around for saving movies. This is only
        # used if we don't know how many frames there will be: in the case
        # of no generator or in the case of a callable.
        self.save_count = save_count

        # Set up a function that creates a new iterable when needed. If nothing
        # is passed in for frames, just use itertools.count, which will just
        # keep counting from 0. A callable passed in for frames is assumed to
        # be a generator. An iterable will be used as is, and anything else
        # will be treated as a number of frames.
        if frames is None:
            self._iter_gen = itertools.count
        elif six.callable(frames):
            self._iter_gen = frames
        elif iterable(frames):
            self._iter_gen = lambda: iter(frames)
            if hasattr(frames, '__len__'):
                self.save_count = len(frames)
        else:
            self._iter_gen = lambda: xrange(frames).__iter__()
            self.save_count = frames

        # If we're passed in and using the default, set it to 100.
        if self.save_count is None:
            self.save_count = 100

        self._init_func = init_func

        # Needs to be initialized so the draw functions work without checking
        self._save_seq = []

        TimedAnimation.__init__(self, fig, **kwargs)

        # Need to reset the saved seq, since right now it will contain data
        # for a single frame from init, which is not what we want.
        self._save_seq = []

    def new_frame_seq(self):
        # Use the generating function to generate a new frame sequence
        return self._iter_gen()

    def new_saved_frame_seq(self):
        # Generate an iterator for the sequence of saved data. If there are
        # no saved frames, generate a new frame sequence and take the first
        # save_count entries in it.
        if self._save_seq:
            # While iterating we are going to update _save_seq
            # so make a copy to safely iterate over
            self._old_saved_seq = list(self._save_seq)
            return iter(self._old_saved_seq)
        else:
            return itertools.islice(self.new_frame_seq(), self.save_count)

    def _init_draw(self):
        # Initialize the drawing either using the given init_func or by
        # calling the draw function with the first item of the frame sequence.
        # For blitting, the init_func should return a sequence of modified
        # artists.
        if self._init_func is None:
            self._draw_frame(next(self.new_frame_seq()))

        else:
            self._drawn_artists = self._init_func()
            if self._blit:
                if self._drawn_artists is None:
                    raise RuntimeError('The init_func must return a '
                                       'sequence of Artist objects.')
                for a in self._drawn_artists:
                    a.set_animated(self._blit)
        self._save_seq = []

    def _draw_frame(self, framedata):
        # Save the data for potential saving of movies.
        self._save_seq.append(framedata)

        # Make sure to respect save_count (keep only the last save_count
        # around)
        self._save_seq = self._save_seq[-self.save_count:]

        # Call the func with framedata and args. If blitting is desired,
        # func needs to return a sequence of any artists that were modified.
        self._drawn_artists = self._func(framedata, *self._args)
        if self._blit:
            if self._drawn_artists is None:
                    raise RuntimeError('The animation function must return a '
                                       'sequence of Artist objects.')
            for a in self._drawn_artists:
                a.set_animated(self._blit)
