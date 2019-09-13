import matplotlib
from matplotlib import cbook, rcParams
from matplotlib._pylab_helpers import Gcf
from matplotlib.backends import _macosx
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    TimerBase)
from matplotlib.figure import Figure
from matplotlib.widgets import SubplotTool


########################################################################
#
# The following functions and classes are for pylab and implement
# window/figure managers, etc...
#
########################################################################


class TimerMac(_macosx.Timer, TimerBase):
    '''
    Subclass of :class:`backend_bases.TimerBase` that uses CoreFoundation
    run loops for timer events.

    Attributes
    ----------
    interval : int
        The time between timer events in milliseconds. Default is 1000 ms.
    single_shot : bool
        Boolean flag indicating whether this timer should operate as single
        shot (run once and then stop). Defaults to False.
    callbacks : list
        Stores list of (func, args) tuples that will be called upon timer
        events. This list can be manipulated directly, or the functions
        `add_callback` and `remove_callback` can be used.

    '''
    # completely implemented at the C-level (in _macosx.Timer)


class FigureCanvasMac(_macosx.FigureCanvas, FigureCanvasAgg):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Events such as button presses, mouse movements, and key presses
    are handled in the C code and the base class methods
    button_press_event, button_release_event, motion_notify_event,
    key_press_event, and key_release_event are called from there.

    Attributes
    ----------
    figure : `matplotlib.figure.Figure`
        A high-level Figure instance
    """

    required_interactive_framework = "macosx"

    def __init__(self, figure):
        FigureCanvasBase.__init__(self, figure)
        width, height = self.get_width_height()
        _macosx.FigureCanvas.__init__(self, width, height)
        self._device_scale = 1.0

    def _set_device_scale(self, value):
        if self._device_scale != value:
            self.figure.dpi = self.figure.dpi / self._device_scale * value
            self._device_scale = value

    def _draw(self):
        renderer = self.get_renderer(cleared=self.figure.stale)

        if self.figure.stale:
            self.figure.draw(renderer)

        return renderer

    def draw(self):
        # docstring inherited
        self.draw_idle()
        self.flush_events()

    # draw_idle is provided by _macosx.FigureCanvas

    @cbook.deprecated("3.2", alternative="draw_idle()")
    def invalidate(self):
        return self.draw_idle()

    def blit(self, bbox=None):
        self.draw_idle()

    def resize(self, width, height):
        dpi = self.figure.dpi
        width /= dpi
        height /= dpi
        self.figure.set_size_inches(width * self._device_scale,
                                    height * self._device_scale,
                                    forward=False)
        FigureCanvasBase.resize_event(self)
        self.draw_idle()

    def new_timer(self, *args, **kwargs):
        # docstring inherited
        return TimerMac(*args, **kwargs)


class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
    """
    Wrap everything up into a window for the pylab interface
    """
    def __init__(self, canvas, num):
        FigureManagerBase.__init__(self, canvas, num)
        title = "Figure %d" % num
        _macosx.FigureManager.__init__(self, canvas, title)
        if rcParams['toolbar'] == 'toolbar2':
            self.toolbar = NavigationToolbar2Mac(canvas)
        else:
            self.toolbar = None
        if self.toolbar is not None:
            self.toolbar.update()

        if matplotlib.is_interactive():
            self.show()
            self.canvas.draw_idle()

    def close(self):
        Gcf.destroy(self.num)


class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):

    def __init__(self, canvas):
        NavigationToolbar2.__init__(self, canvas)

    def _init_toolbar(self):
        _macosx.NavigationToolbar2.__init__(
            self, str(cbook._get_data_path('images')))

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))

    def release(self, event):
        self.canvas.remove_rubberband()

    def set_cursor(self, cursor):
        _macosx.set_cursor(cursor)

    def save_figure(self, *args):
        filename = _macosx.choose_save_file('Save the figure',
                                            self.canvas.get_default_filename())
        if filename is None:  # Cancel
            return
        self.canvas.figure.savefig(filename)

    def prepare_configure_subplots(self):
        toolfig = Figure(figsize=(6, 3))
        canvas = FigureCanvasMac(toolfig)
        toolfig.subplots_adjust(top=0.9)
        # Need to keep a reference to the tool.
        _tool = SubplotTool(self.canvas.figure, toolfig)
        return canvas

    def set_message(self, message):
        _macosx.NavigationToolbar2.set_message(self, message.encode('utf-8'))


########################################################################
#
# Now just provide the standard names that backend.__init__ is expecting
#
########################################################################

@_Backend.export
class _BackendMac(_Backend):
    FigureCanvas = FigureCanvasMac
    FigureManager = FigureManagerMac

    @staticmethod
    def trigger_manager_draw(manager):
        manager.canvas.draw_idle()

    @staticmethod
    def mainloop():
        _macosx.show()
