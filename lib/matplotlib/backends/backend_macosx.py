from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureManagerBase, FigureCanvasBase, \
    NavigationToolbar2, TimerBase
from matplotlib.backend_bases import ShowBase

from matplotlib.figure import Figure
from matplotlib import rcParams

from matplotlib.widgets import SubplotTool

import matplotlib
from matplotlib.backends import _macosx

from .backend_agg import RendererAgg, FigureCanvasAgg


class Show(ShowBase):
    def mainloop(self):
        _macosx.show()
show = Show()


########################################################################
#
# The following functions and classes are for pylab and implement
# window/figure managers, etc...
#
########################################################################

def draw_if_interactive():
    """
    For performance reasons, we don't want to redraw the figure after
    each draw command. Instead, we mark the figure as invalid, so that
    it will be redrawn as soon as the event loop resumes via PyOS_InputHook.
    This function should be called after each draw event, even if
    matplotlib is not running interactively.
    """
    if matplotlib.is_interactive():
        figManager =  Gcf.get_active()
        if figManager is not None:
            figManager.canvas.invalidate()


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    FigureClass = kwargs.pop('FigureClass', Figure)
    figure = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, figure)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasMac(figure)
    manager = FigureManagerMac(canvas, num)
    return manager


class TimerMac(_macosx.Timer, TimerBase):
    '''
    Subclass of :class:`backend_bases.TimerBase` that uses CoreFoundation
    run loops for timer events.

    Attributes:
    * interval: The time between timer events in milliseconds. Default
        is 1000 ms.
    * single_shot: Boolean flag indicating whether this timer should
        operate as single shot (run once and then stop). Defaults to False.
    * callbacks: Stores list of (func, args) tuples that will be called
        upon timer events. This list can be manipulated directly, or the
        functions add_callback and remove_callback can be used.
    '''
    # completely implemented at the C-level (in _macosx.Timer)


class FigureCanvasMac(_macosx.FigureCanvas, FigureCanvasAgg):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance

    Events such as button presses, mouse movements, and key presses
    are handled in the C code and the base class methods
    button_press_event, button_release_event, motion_notify_event,
    key_press_event, and key_release_event are called from there.
    """

    def __init__(self, figure):
        FigureCanvasBase.__init__(self, figure)
        width, height = self.get_width_height()
        _macosx.FigureCanvas.__init__(self, width, height)
        self._device_scale = 1.0

    def _set_device_scale(self, value):
        if self._device_scale != value:
            self.figure.dpi = self.figure.dpi / self._device_scale * value
            self._device_scale = value

    def get_renderer(self, cleared=False):
        l, b, w, h = self.figure.bbox.bounds
        key = w, h, self.figure.dpi
        try:
            self._lastKey, self._renderer
        except AttributeError:
            need_new_renderer = True
        else:
            need_new_renderer = (self._lastKey != key)

        if need_new_renderer:
            self._renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
        elif cleared:
            self._renderer.clear()

        return self._renderer

    def _draw(self):
        renderer = self.get_renderer()

        if not self.figure.stale:
            return renderer

        self.figure.draw(renderer)
        return renderer

    def draw(self):
        self.invalidate()

    def draw_idle(self, *args, **kwargs):
        self.invalidate()

    def blit(self, bbox):
        self.invalidate()

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
        """
        Creates a new backend-specific subclass of :class:`backend_bases.Timer`.
        This is useful for getting periodic events through the backend's native
        event loop. Implemented only for backends with GUIs.

        optional arguments:

        *interval*
          Timer interval in milliseconds
        *callbacks*
          Sequence of (func, args, kwargs) where func(*args, **kwargs) will
          be executed by the timer every *interval*.
        """
        return TimerMac(*args, **kwargs)


class FigureManagerMac(_macosx.FigureManager, FigureManagerBase):
    """
    Wrap everything up into a window for the pylab interface
    """
    def __init__(self, canvas, num):
        FigureManagerBase.__init__(self, canvas, num)
        title = "Figure %d" % num
        _macosx.FigureManager.__init__(self, canvas, title)
        if rcParams['toolbar']=='toolbar2':
            self.toolbar = NavigationToolbar2Mac(canvas)
        else:
            self.toolbar = None
        if self.toolbar is not None:
            self.toolbar.update()

        def notify_axes_change(fig):
            'this will be called whenever the current axes is changed'
            if self.toolbar != None: self.toolbar.update()
        self.canvas.figure.add_axobserver(notify_axes_change)

        if matplotlib.is_interactive():
            self.show()
            self.canvas.draw_idle()

    def close(self):
        Gcf.destroy(self.num)


class NavigationToolbar2Mac(_macosx.NavigationToolbar2, NavigationToolbar2):

    def __init__(self, canvas):
        NavigationToolbar2.__init__(self, canvas)

    def _init_toolbar(self):
        basedir = os.path.join(rcParams['datapath'], "images")
        _macosx.NavigationToolbar2.__init__(self, basedir)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))

    def release(self, event):
        self.canvas.remove_rubberband()

    def set_cursor(self, cursor):
        _macosx.set_cursor(cursor)

    def save_figure(self, *args):
        filename = _macosx.choose_save_file('Save the figure',
                                            self.canvas.get_default_filename())
        if filename is None: # Cancel
            return
        self.canvas.print_figure(filename)

    def prepare_configure_subplots(self):
        toolfig = Figure(figsize=(6,3))
        canvas = FigureCanvasMac(toolfig)
        toolfig.subplots_adjust(top=0.9)
        tool = SubplotTool(self.canvas.figure, toolfig)
        return canvas

    def set_message(self, message):
        _macosx.NavigationToolbar2.set_message(self, message.encode('utf-8'))

    def dynamic_update(self):
        self.canvas.draw_idle()

########################################################################
#
# Now just provide the standard names that backend.__init__ is expecting
#
########################################################################

FigureCanvas = FigureCanvasMac
FigureManager = FigureManagerMac
