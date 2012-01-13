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
#   * Library to make movies?
#   * Can blit be enabled for movies?
# * Need to consider event sources to allow clicking through multiple figures
import itertools
import contextlib
import subprocess
from matplotlib.cbook import iterable, is_string_like
from matplotlib import verbose
from matplotlib import rcParams

# Other potential writing methods:
# * ImageMagick convert: convert -set delay 3 -colorspace GRAY -colors 16 -dispose 1 -loop 0 -scale 50% *.png Output.gif
# * http://pymedia.org/
# * libmng (produces swf) python wrappers: https://github.com/libming/libming
# * Wrap x264 API: http://stackoverflow.com/questions/2940671/how-to-encode-series-of-images-into-h264-using-x264-api-c-c

#Needs:
# - Need comments, docstrings
# - Need to look at codecs
# - Is there a common way to add metadata?

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
        return self.avail.keys()

    def __getitem__(self, name):
        if not self.avail:
            raise RuntimeError("No MovieWriters available!")
        return self.avail[name]

writers = MovieWriterRegistry()

class MovieWriter(object):
    def __init__(self, fps=5, codec=None, bitrate=None, extra_args=None):
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

    @property
    def frame_size(self):
        width_inches, height_inches = self.fig.get_size_inches()
        return width_inches * self.dpi, height_inches * self.dpi

    def setup(self, fig, outfile, dpi, *args):
        self.outfile = outfile
        self.fig = fig
        self.dpi = dpi
        self._run()

    @contextlib.contextmanager
    def saving(self, *args):
        self.setup(*args)
        yield
        self.finish()

    def _run(self):
        # Uses subprocess to call the program for assembling frames into a
        # movie file.  *args* returns the sequence of command line arguments
        # from a few configuration options.
        command = self.args()
        verbose.report('MovieWriter.run: running command: %s'%' '.join(command))
        self._proc = subprocess.Popen(command, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.PIPE)

    def finish(self):
        self.cleanup()

    def grab_frame(self):
        verbose.report('MovieWriter.grab_frame: Grabbing frame.', level='debug')
        try:
            self.fig.savefig(self._frame_sink(), format=self.frame_format,
                dpi=self.dpi)
        except RuntimeError:
            out, err = self._proc.communicate()
            verbose.report('MovieWriter -- Error running proc:\n%s\n%s' % (out,
                err), level='helpful')
            raise

    def _frame_sink(self):
        return self._proc.stdin

    def args(self):
        return NotImplementedError("args needs to be implemented by subclass.")

    def cleanup(self):
        out,err = self._proc.communicate()
        verbose.report('MovieWriter -- Command stdout:\n%s' % out,
            level='debug')
        verbose.report('MovieWriter -- Command stderr:\n%s' % err,
            level='debug')

    @classmethod
    def bin_path(cls):
        return rcParams[cls.exec_key]

    @classmethod
    def isAvailable(cls):
        try:
            subprocess.Popen(cls.bin_path(), shell=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except OSError:
            return False


class FileMovieWriter(MovieWriter):
    def __init__(self, *args):
        MovieWriter.__init__(self, *args)
        self.frame_format = rcParams['animation.frame_format']

    def setup(self, fig, outfile, dpi, frame_prefix='_tmp', clear_temp=True):
        print fig, outfile, dpi, frame_prefix, clear_temp
        self.fig = fig
        self.outfile = outfile
        self.dpi = dpi
        self.clear_temp = clear_temp
        self.temp_prefix = frame_prefix
        self._frame_counter = 0
        self._temp_names = list()
        self.fname_format_str = '%s%%04d.%s'

    @property
    def frame_format(self):
        return self._frame_format

    @frame_format.setter
    def frame_format(self, frame_format):
        if frame_format in self.supported_formats:
            self._frame_format = frame_format
        else:
            self._frame_format = self.supported_formats[0]

    def _base_temp_name(self):
        return self.fname_format_str % (self.temp_prefix, self.frame_format)

    def _frame_sink(self):
        fname = self._base_temp_name() % self._frame_counter
        self._temp_names.append(fname)
        verbose.report(
            'FileMovieWriter.frame_sink: saving frame %d to fname=%s' % (self._frame_counter, fname),
            level='debug')
        self._frame_counter += 1

        # This file returned here will be closed once it's used by savefig()
        # because it will no longer be referenced and will be gc-ed.
        return open(fname, 'wb')

    def finish(self):
        #Delete temporary files
        self._run()
        MovieWriter.finish(self)

    def cleanup(self):
        MovieWriter.cleanup(self)
        if self.clear_temp:
            import os
            verbose.report(
                'MovieWriter: clearing temporary fnames=%s' % str(self._temp_names),
                level='debug')
            for fname in self._temp_names:
                os.remove(fname)


class FFMpegBase:
    exec_key = 'animation.ffmpeg_path'
    args_key = 'animation.ffmpeg_args'

    @property
    def output_args(self):
        # The %dk adds 'k' as a suffix so that ffmpeg treats our bitrate as in
        # kbps
        args = ['-vcodec', self.codec, '-b', '%dk' % self.bitrate]
        if self.extra_args:
            args.extend(self.extra_args)
        return args + ['-y', self.outfile]


@writers.register('ffmpeg')
class FFMpegWriter(MovieWriter, FFMpegBase):
    def args(self):
        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie
        return [self.bin_path(), '-f', 'rawvideo', '-vcodec', 'rawvideo',
             '-s', '%dx%d' % self.frame_size, '-pix_fmt', self.frame_format, 
             '-r', str(self.fps), '-i', 'pipe:'] + self.output_args


@writers.register('ffmpeg_file')
class FFMpegFileWriter(FileMovieWriter, FFMpegBase):
    supported_formats = ['png', 'jpeg', 'ppm', 'tiff', 'sgi', 'bmp', 'pbm', 'raw', 'rgba']
    def args(self):
        # Returns the command line parameters for subprocess to use
        # ffmpeg to create a movie
        return [self.bin_path(), '-r', str(self.fps), '-i',
            self._base_temp_name()] + self.output_args


class MencoderBase:
    exec_key = 'animation.mencoder_path'
    args_key = 'animation.mencoder_args'

    @property
    def output_args(self):
        args = ['-o', self.outfile, '-ovc', 'lavc', 'vcodec=%s' % self.codec,
            'vbitrate=%d' % self.bitrate]
        if self.extra_args:
            args.extend(self.extra_args)
        return args


@writers.register('mencoder')
class MencoderWriter(MovieWriter, MencoderBase):
    def args(self):
        # Returns the command line parameters for subprocess to use
        # mencoder to create a movie
        return [self.bin_path(), '-', '-demuxer', 'rawvideo', '-rawvideo',
            ('w=%i:h=%i:' % self.frame_size +
            'fps=%i:format=%s' % (self.fps, self.frame_format))] + self.output_args


@writers.register('mencoder_file')
class MencoderFileWriter(FileMovieWriter, MencoderBase):
    supported_formats = ['png', 'jpeg', 'tga', 'sgi']
    def args(self):
        # Returns the command line parameters for subprocess to use
        # mencoder to create a movie
        return [self.bin_path(),
            'mf://%s*.%s' % (self.temp_prefix, self.frame_format), '-mf',
            'type=%s:fps=%d' % (self.frame_format, self.fps)] + self.output_args


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
        self._blit = blit

        # These are the basics of the animation.  The frame sequence represents
        # information for each frame of the animation and depends on how the
        # drawing is handled by the subclasses. The event source fires events
        # that cause the frame sequence to be iterated.
        self.frame_seq = self.new_frame_seq()
        self.event_source = event_source

        # Clear the initial frame
        self._init_draw()

        # Instead of starting the event source now, we connect to the figure's
        # draw_event, so that we only start once the figure has been drawn.
        self._first_draw_id = fig.canvas.mpl_connect('draw_event', self._start)

        # Connect to the figure's close_event so that we don't continue to
        # fire events and try to draw to a deleted figure.
        self._close_id = self._fig.canvas.mpl_connect('close_event', self._stop)
        if blit:
            self._setup_blit()

    def _start(self, *args):
        '''
        Starts interactive animation. Adds the draw frame command to the GUI
        handler, calls show to start the event loop.
        '''
        # On start, we add our callback for stepping the animation and
        # actually start the event_source. We also disconnect _start
        # from the draw_events
        self.event_source.add_callback(self._step)
        self.event_source.start()
        self._fig.canvas.mpl_disconnect(self._first_draw_id)
        self._first_draw_id = None # So we can check on save

    def _stop(self, *args):
        # On stop we disconnect all of our events.
        if self._blit:
            self._fig.canvas.mpl_disconnect(self._resize_id)
        self._fig.canvas.mpl_disconnect(self._close_id)
        self.event_source.remove_callback(self._step)
        self.event_source = None

    def save(self, filename, writer=None, fps=None, dpi=None, codec=None,
            bitrate=None):
        '''
        Saves a movie file by drawing every frame.

        *filename* is the output filename, eg :file:`mymovie.mp4`

        *writer* is either an instance of :class:`MovieWriter` or a string
        key that identifies a class to use, such as 'ffmpeg' or 'mencoder'.
        If nothing is passed, the value of the rcparam `animation.writer` is
        used.

        *fps* is the frames per second in the movie. Defaults to None,
        which will use the animation's specified interval to set the frames
        per second.

        *dpi* controls the dots per inch for the movie frames. This combined
        with the figure's size in inches controls the size of the movie.

        *codec* is the video codec to be used. Not all codecs are supported
        by a given :class:`MovieWriter`. If none is given, this defaults to the
        value specified by the rcparam `animation.codec`.
        
        *bitrate* specifies the amount of bits used per second in the
        compressed movie, in kilobits per second. A higher number means a
        higher quality movie, but at the cost of increased file size. If no
        value is given, this defaults to the value given by the rcparam
        `animation.bitrate`.
        '''
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

        # If the writer is None, use the rc param to find the name of the one
        # to use
        if writer is None:
            writer = rcParams['animation.writer']

        # Re-use the savefig DPI for ours if none is given
        if dpi is None:
            dpi = rcParams['savefig.dpi']

        if codec is None:
            codec = rcParams['animation.codec']

        if bitrate is None:
            bitrate = rcParams['animation.bitrate']

        # If we have the name of a writer, instantiate an instance of the
        # registered class.
        if is_string_like(writer):
            if writer in writers.avail:
                writer = writers[writer](fps, codec, bitrate)
            else:
                import warnings
                warnings.warn("MovieWriter %s unavailable" % writer)
                writer = writers.list()[0]

        verbose.report('Animation.save using %s' % type(writer), level='helpful')
        # Create a new sequence of frames for saved data. This is different
        # from new_frame_seq() to give the ability to save 'live' generated
        # frame information to be saved later.
        # TODO: Right now, after closing the figure, saving a movie won't
        # work since GUI widgets are gone. Either need to remove extra code
        # to allow for this non-existant use case or find a way to make it work.
        with writer.saving(self._fig, filename, dpi):
            for data in self.new_saved_frame_seq():
                #TODO: Need to see if turning off blit is really necessary
                self._draw_next_frame(data, blit=False)
                writer.grab_frame()

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
        self._resize_id = self._fig.canvas.mpl_connect('draw_event', self._end_redraw)

    def _end_redraw(self, evt):
        # Now that the redraw has happened, do the post draw flushing and
        # blit handling. Then re-enable all of the original events.
        self._post_draw(None, self._blit)
        self.event_source.start()
        self._fig.canvas.mpl_disconnect(self._resize_id)
        self._resize_id = self._fig.canvas.mpl_connect('resize_event',
            self._handle_resize)


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

        Animation.__init__(self, fig, event_source=event_source, *args, **kwargs)

    def _step(self, *args):
        '''
        Handler for getting events.
        '''
        # Extends the _step() method for the Animation class.  If
        # Animation._step signals that it reached the end and we want to repeat,
        # we refresh the frame sequence and return True. If _repeat_delay is
        # set, change the event_source's interval to our loop delay and set the
        # callback to one which will then set the interval back.
        still_going = Animation._step(self, *args)
        if not still_going and self.repeat:
            if self._repeat_delay:
                self.event_source.remove_callback(self._step)
                self.event_source.add_callback(self._loop_delay)
                self.event_source.interval = self._repeat_delay
            self.frame_seq = self.new_frame_seq()
            return True
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
        axes = []
        for f in self.new_frame_seq():
            for artist in f:
                artist.set_visible(False)
                # Assemble a list of unique axes that need flushing
                if artist.axes not in axes:
                    axes.append(artist.axes)

        # Flush the needed axes
        for ax in axes:
            ax.figure.canvas.draw()

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
    used.
    '''
    def __init__(self, fig, func, frames=None ,init_func=None, fargs=None,
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
        elif callable(frames):
            self._iter_gen = frames
        elif iterable(frames):
            self._iter_gen = lambda: iter(frames)
            self.save_count = len(frames)
        else:
            self._iter_gen = lambda: iter(range(frames))
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
            return iter(self._save_seq)
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

    def _draw_frame(self, framedata):
        # Save the data for potential saving of movies.
        self._save_seq.append(framedata)

        # Make sure to respect save_count (keep only the last save_count around)
        self._save_seq = self._save_seq[-self.save_count:]

        # Call the func with framedata and args. If blitting is desired,
        # func needs to return a sequence of any artists that were modified.
        self._drawn_artists = self._func(framedata, *self._args)
