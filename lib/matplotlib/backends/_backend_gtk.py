"""
Common code for GTK3 and GTK4 backends.
"""

import logging
import sys

import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    TimerBase, WindowBase, ExpandableBase)
from matplotlib.backend_tools import Cursors

import gi
# The GTK3/GTK4 backends will have already called `gi.require_version` to set
# the desired GTK.
from gi.repository import Gdk, Gio, GLib, Gtk


try:
    gi.require_foreign("cairo")
except ImportError as e:
    raise ImportError("Gtk-based backends require cairo") from e

_log = logging.getLogger(__name__)
_application = None  # Placeholder


def _shutdown_application(app):
    # The application might prematurely shut down if Ctrl-C'd out of IPython,
    # so close all windows.
    for win in app.get_windows():
        win.close()
    # The PyGObject wrapper incorrectly thinks that None is not allowed, or we
    # would call this:
    # Gio.Application.set_default(None)
    # Instead, we set this property and ignore default applications with it:
    app._created_by_matplotlib = True
    global _application
    _application = None


def _create_application():
    global _application

    if _application is None:
        app = Gio.Application.get_default()
        if app is None or getattr(app, '_created_by_matplotlib', False):
            # display_is_valid returns False only if on Linux and neither X11
            # nor Wayland display can be opened.
            if not mpl._c_internal_utils.display_is_valid():
                raise RuntimeError('Invalid DISPLAY variable')
            _application = Gtk.Application.new('org.matplotlib.Matplotlib3',
                                               Gio.ApplicationFlags.NON_UNIQUE)
            # The activate signal must be connected, but we don't care for
            # handling it, since we don't do any remote processing.
            _application.connect('activate', lambda *args, **kwargs: None)
            _application.connect('shutdown', _shutdown_application)
            _application.register()
            cbook._setup_new_guiapp()
        else:
            _application = app

    return _application


def mpl_to_gtk_cursor_name(mpl_cursor):
    return _api.check_getitem({
        Cursors.MOVE: "move",
        Cursors.HAND: "pointer",
        Cursors.POINTER: "default",
        Cursors.SELECT_REGION: "crosshair",
        Cursors.WAIT: "wait",
        Cursors.RESIZE_HORIZONTAL: "ew-resize",
        Cursors.RESIZE_VERTICAL: "ns-resize",
    }, cursor=mpl_cursor)


class TimerGTK(TimerBase):
    """Subclass of `.TimerBase` using GTK timer events."""

    def __init__(self, *args, **kwargs):
        self._timer = None
        super().__init__(*args, **kwargs)

    def _timer_start(self):
        # Need to stop it, otherwise we potentially leak a timer id that will
        # never be stopped.
        self._timer_stop()
        self._timer = GLib.timeout_add(self._interval, self._on_timer)

    def _timer_stop(self):
        if self._timer is not None:
            GLib.source_remove(self._timer)
            self._timer = None

    def _timer_set_interval(self):
        # Only stop and restart it if the timer has already been started.
        if self._timer is not None:
            self._timer_stop()
            self._timer_start()

    def _on_timer(self):
        super()._on_timer()

        # Gtk timeout_add() requires that the callback returns True if it
        # is to be called again.
        if self.callbacks and not self._single:
            return True
        else:
            self._timer = None
            return False


class _FigureCanvasGTK(FigureCanvasBase):
    _timer_cls = TimerGTK


_flow = [Gtk.Orientation.HORIZONTAL, Gtk.Orientation.VERTICAL]


class _WindowGTK(WindowBase, Gtk.Window):
    # Must be implemented in GTK3/GTK4 backends:
    # * _add_element - to add an widget to a container
    # * _setup_signals
    # * _get_self - a method to ensure that we have been fully initialised

    def __init__(self, title, **kwargs):
        super().__init__(title=title, **kwargs)

        self.set_window_title(title)

        self._layout = {}
        self._setup_box('_outer', Gtk.Orientation.VERTICAL, False, None)
        self._setup_box('north', Gtk.Orientation.VERTICAL, False, '_outer')
        self._setup_box('_middle', Gtk.Orientation.HORIZONTAL, True, '_outer')
        self._setup_box('south', Gtk.Orientation.VERTICAL, False, '_outer')

        self._setup_box('west', Gtk.Orientation.HORIZONTAL, False, '_middle')
        self._setup_box('center', Gtk.Orientation.VERTICAL, True, '_middle')
        self._setup_box('east', Gtk.Orientation.HORIZONTAL, False, '_middle')

        self.set_child(self._layout['_outer'])

        self._setup_signals()

    def _setup_box(self, name, orientation, grow, parent):
        self._layout[name] = Gtk.Box(orientation=orientation)
        if parent:
            self._add_element(self._layout[parent], self._layout[name], True, grow)
        self._layout[name].show()

    def add_element(self, element, place):
        element.show()

        # Get the flow of the element (the opposite of the container)
        flow_index = not _flow.index(self._layout[place].get_orientation())
        flow = _flow[flow_index]
        separator = Gtk.Separator(orientation=flow)
        separator.show()

        try:
            element.flow = element.flow_types[flow_index]
        except AttributeError:
            pass

        # Determine if this element should fill all the space given to it
        expand = isinstance(element, ExpandableBase)

        if place in ['north', 'west', 'center']:
            to_start = True
        elif place in ['south', 'east']:
            to_start = False
        else:
            raise KeyError('Unknown value for place, %s' % place)

        self._add_element(self._layout[place], element, to_start, expand)
        self._add_element(self._layout[place], separator, to_start, False)

        h = 0
        for e in [element, separator]:
            min_size, nat_size = e.get_preferred_size()
            h += nat_size.height

        return h

    def set_default_size(self, width, height):
        Gtk.Window.set_default_size(self, width, height)

    def show(self):
        # show the window
        Gtk.Window.show(self)
        if mpl.rcParams["figure.raise_window"]:
            if self._get_self():
                self.present()
            else:
                # If this is called by a callback early during init,
                # self.window (a GtkWindow) may not have an associated
                # low-level GdkWindow (on GTK3) or GdkSurface (on GTK4) yet,
                # and present() would crash.
                _api.warn_external("Cannot raise window yet to be setup")

    def destroy(self):
        Gtk.Window.destroy(self)

    def set_fullscreen(self, fullscreen):
        if fullscreen:
            self.fullscreen()
        else:
            self.unfullscreen()

    def get_window_title(self):
        return self.get_title()

    def set_window_title(self, title):
        self.set_title(title)

    def resize(self, width, height):
        Gtk.Window.resize(self, width, height)


class _FigureManagerGTK(FigureManagerBase):
    """
    Attributes
    ----------
    canvas : `FigureCanvas`
        The FigureCanvas instance
    num : int or str
        The Figure number
    toolbar : Gtk.Toolbar or Gtk.Box
        The toolbar
    vbox : Gtk.VBox
        The Gtk.VBox containing the canvas and toolbar
    window : Gtk.Window
        The Gtk.Window
    """

    def __init__(self, canvas, num):
        app = _create_application()
        self.window = self._window_class('Matplotlib Figure Manager')
        app.add_window(self.window)
        super().__init__(canvas, num)

        self.window.add_element(self.canvas, 'center')
        w, h = self.canvas.get_width_height()

        if self.toolbar:
            h += self.window.add_element(self.toolbar, 'south')  # put in ScrolledWindow in GTK4?

        self.window.set_default_size(w, h)

        self._destroying = False
        self.window.mpl_connect('window_destroy_event', lambda *args: Gcf.destroy(self))

        if mpl.is_interactive():
            self.window.show()
            self.canvas.draw_idle()

        self.canvas.grab_focus()

    def destroy(self, *args):
        if self._destroying:
            # Otherwise, this can be called twice when the user presses 'q',
            # which calls Gcf.destroy(self), then this destroy(), then triggers
            # Gcf.destroy(self) once again via
            # `connect("destroy", lambda *args: Gcf.destroy(self))`.
            return
        self._destroying = True
        self.window.destroy()
        self.canvas.destroy()

    @classmethod
    def start_main_loop(cls):
        global _application
        if _application is None:
            return

        try:
            _application.run()  # Quits when all added windows close.
        except KeyboardInterrupt:
            # Ensure all windows can process their close event from
            # _shutdown_application.
            context = GLib.MainContext.default()
            while context.pending():
                context.iteration(True)
            raise
        finally:
            # Running after quit is undefined, so create a new one next time.
            _application = None

    def show(self):
        # show the figure window
        self.window.show()
        self.canvas.draw()

    def full_screen_toggle(self):
        if self.window.is_fullscreen():
            self.window.unfullscreen()
        else:
            self.window.fullscreen()

    def get_window_title(self):
        return self.window.get_title()

    def set_window_title(self, title):
        self.window.set_title(title)

    def resize(self, width, height):
        width = int(width / self.canvas.device_pixel_ratio)
        height = int(height / self.canvas.device_pixel_ratio)
        if self.toolbar:
            min_size, nat_size = self.toolbar.get_preferred_size()
            height += nat_size.height
        canvas_size = self.canvas.get_allocation()
        if canvas_size.width == canvas_size.height == 1:
            # A canvas size of (1, 1) cannot exist in most cases, because
            # window decorations would prevent such a small window. This call
            # must be before the window has been mapped and widgets have been
            # sized, so just change the window's starting size.
            self.window.set_default_size(width, height)
        else:
            self.window.resize(width, height)


class _NavigationToolbar2GTK(NavigationToolbar2):
    # Must be implemented in GTK3/GTK4 backends:
    # * __init__
    # * save_figure

    def set_message(self, s):
        escaped = GLib.markup_escape_text(s)
        self.message.set_markup(f'<small>{escaped}</small>')

    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
        self.canvas._draw_rubberband(rect)

    def remove_rubberband(self):
        self.canvas._draw_rubberband(None)

    def _update_buttons_checked(self):
        for name, active in [("Pan", "PAN"), ("Zoom", "ZOOM")]:
            button = self._gtk_ids.get(name)
            if button:
                with button.handler_block(button._signal_handler):
                    button.set_active(self.mode.name == active)

    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()

    def set_history_buttons(self):
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
        if 'Back' in self._gtk_ids:
            self._gtk_ids['Back'].set_sensitive(can_backward)
        if 'Forward' in self._gtk_ids:
            self._gtk_ids['Forward'].set_sensitive(can_forward)


class RubberbandGTK(backend_tools.RubberbandBase):
    def draw_rubberband(self, x0, y0, x1, y1):
        _NavigationToolbar2GTK.draw_rubberband(
            self._make_classic_style_pseudo_toolbar(), None, x0, y0, x1, y1)

    def remove_rubberband(self):
        _NavigationToolbar2GTK.remove_rubberband(
            self._make_classic_style_pseudo_toolbar())


class ConfigureSubplotsGTK(backend_tools.ConfigureSubplotsBase):
    def trigger(self, *args):
        _NavigationToolbar2GTK.configure_subplots(self, None)


class _BackendGTK(_Backend):
    backend_version = "{}.{}.{}".format(
        Gtk.get_major_version(),
        Gtk.get_minor_version(),
        Gtk.get_micro_version(),
    )
    mainloop = _FigureManagerGTK.start_main_loop
