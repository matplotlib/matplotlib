import os
import functools

import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib._pylab_helpers import Gcf
from . import _macos
from .backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, NavigationToolbar2,
    CloseEvent, KeyEvent, LocationEvent, MouseEvent, ResizeEvent,
    MouseButton, TimerBase, _allow_interrupt, _Mode)


class TimerMac(_macos.Timer, TimerBase):
    """Subclass of `.TimerBase` using CFRunLoop timer events."""
    # completely implemented at the C-level (in _macos.Timer)


def _allow_interrupt_macos():
    """A context manager that allows terminating a plot by sending a SIGINT."""
    return _allow_interrupt(
        lambda rsock: _macos.wake_on_fd_write(rsock.fileno()), _macos.stop)


@functools.lru_cache
def _init_macos():
    data_path = cbook._get_data_path("images")
    _macos._init({"matplotlib": str(data_path / "matplotlib.pdf")})


class FigureCanvasMac(_macos.FigureCanvas, FigureCanvasBase):
    # docstring inherited

    required_interactive_framework = "macos"
    _timer_cls = TimerMac
    manager_class = _api.classproperty(lambda cls: FigureManagerMac)

    def __init__(self, figure):
        _init_macos()
        FigureCanvasBase.__init__(self, figure=figure)
        width, height = self.get_width_height()
        _macos.FigureCanvas.__init__(self, width, height)

    def draw(self):
        """Render the figure and send the buffer to the macOS CALayer."""
        super().draw()
        if not self._is_idle_drawing:
            self._request_display_layer(False)

    def _handle_display_layer(self, needs_draw):
        with self._idle_draw_cntx():
            if needs_draw:
                self.draw()
            self._update_layer_contents(self.get_renderer().buffer_rgba())

    def draw_idle(self):
        # docstring inherited
        self._request_display_layer(True)

    def blit(self, bbox=None):
        # docstring inherited
        super().blit(bbox)
        self._request_display_layer(False)

    def _handle_view_did_change_backing_properties(self, scale, width, height):
        # Size from macOS is physical pixels
        if self._set_device_pixel_ratio(scale):
            self._handle_resize(width, height)
            self.draw_idle()

    def _handle_resize(self, width, height):
        # Size from macOS is physical pixels
        scale = self.figure.dpi
        width /= scale
        height /= scale
        self.figure.set_size_inches(width, height, forward=False)
        ResizeEvent("resize_event", self)._process()
        self.draw_idle()

    def _mpl_button(self, button):
        """Converts from AppKit buttonNumber to a MouseButton"""
        mod_table = (
            MouseButton.LEFT,
            MouseButton.RIGHT,
            MouseButton.MIDDLE,
            MouseButton.BACK,
            MouseButton.FORWARD,
        )
        return mod_table[button] if 0 <= button < len(mod_table) else None

    def _mpl_buttons(self, buttons):
        """Converts from AppKit pressedMouseButtons to a MouseButton set"""
        return {self._mpl_button(i) for i in range(5) if buttons & (1 << i)}

    def _mpl_modifiers(self, modifiers):
        """Converts from AppKit modifierFlags to a list of strings"""
        mod_table = [
            ("ctrl", 1 << 18),
            ("alt", 1 << 19),
            ("shift", 1 << 17),
            ("cmd", 1 << 20),
        ]
        return [name for name, mask in mod_table if modifiers & mask]

    def _handle_key(self, is_press, key, x, y):
        event_name = "key_press_event" if is_press else "key_release_event"
        KeyEvent(event_name, self, key, x, y)._process()

    def _handle_mouse_entered(self, x, y, modifiers):
        LocationEvent("figure_enter_event", self, x, y,
                      modifiers=self._mpl_modifiers(modifiers))._process()

    def _handle_mouse_exited(self, x, y, modifiers):
        LocationEvent("figure_leave_event", self, x, y,
                      modifiers=self._mpl_modifiers(modifiers))._process()

    def _handle_mouse_down(self, x, y, button, modifiers, dblclick):
        button = self._mpl_button(button)
        if button is not None:
            MouseEvent("button_press_event", self, x, y, button, dblclick=dblclick,
                       modifiers=self._mpl_modifiers(modifiers))._process()

    def _handle_mouse_up(self, x, y, button, modifiers):
        button = self._mpl_button(button)
        if button is not None:
            MouseEvent("button_release_event", self, x, y, button,
                       modifiers=self._mpl_modifiers(modifiers))._process()

    def _handle_mouse_moved(self, x, y, buttons, modifiers):
        MouseEvent("motion_notify_event", self, x, y,
                   buttons=self._mpl_buttons(buttons),
                   modifiers=self._mpl_modifiers(modifiers))._process()

    def _handle_scroll_wheel(self, x, y, step, modifiers):
        MouseEvent("scroll_event", self, x, y, step=step,
                   modifiers=self._mpl_modifiers(modifiers))._process()

    def start_event_loop(self, timeout=0):
        # docstring inherited
        # Set up a SIGINT handler to allow terminating a plot via CTRL-C.
        with _allow_interrupt_macos():
            self._start_event_loop(timeout=timeout)  # Forward to ObjC implementation.


class FigureCanvasMacAgg(FigureCanvasAgg, FigureCanvasMac):
    pass


class NavigationToolbar2Mac(_macos.NavigationToolbar2, NavigationToolbar2):

    def __init__(self, canvas):
        _macos.NavigationToolbar2.__init__(self, canvas)
        data_path = cbook._get_data_path("images")
        for text, tooltip_text, image_name, callback in self.toolitems:
            if text is None:
                self.add_separator()
            else:
                image_path = str(data_path / image_name) + ".pdf"
                self.add_item(text, tooltip_text, image_path, callback)
        NavigationToolbar2.__init__(self, canvas)

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))

    def remove_rubberband(self):
        self.canvas.remove_rubberband()

    def _update_buttons_checked(self):
        mode_names = {_Mode.PAN: "pan", _Mode.ZOOM: "zoom"}
        self.update_selected_item(mode_names.get(self.mode, ""))

    def set_history_buttons(self):
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack) - 1
        self.update_history_items(can_backward, can_forward)

    def pan(self, *args):
        super().pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        super().zoom(*args)
        self._update_buttons_checked()

    def save_figure(self, *args):
        directory = os.path.expanduser(mpl.rcParams["savefig.directory"])
        filename = _macos.choose_save_file("Save the figure",
                                           directory,
                                           self.canvas.get_default_filename())
        if filename is None:  # Cancel
            return
        # Save dir for next time, unless empty str (which means use cwd).
        if mpl.rcParams["savefig.directory"]:
            mpl.rcParams["savefig.directory"] = os.path.dirname(filename)
        self.canvas.figure.savefig(filename)
        return filename


class FigureManagerMac(_macos.FigureManager, FigureManagerBase):
    _toolbar2_class = NavigationToolbar2Mac

    def __init__(self, canvas, num):
        self._shown = False
        _macos.FigureManager.__init__(self, canvas)
        FigureManagerBase.__init__(self, canvas, num)
        self._set_window_appearance(mpl.rcParams["macos.appearance"])
        self._set_window_mode(mpl.rcParams["macos.window_mode"])
        if self.toolbar is not None:
            self.toolbar.update()
        if mpl.is_interactive():
            self.show()
            self.canvas.draw_idle()

    def _handle_window_will_close(self):
        CloseEvent("close_event", self.canvas)._process()

    def _handle_window_should_close(self):
        Gcf.destroy(self)
        self.canvas.flush_events()

    def destroy(self):
        self._close_and_clear_window()
        super().destroy()

    @classmethod
    def start_main_loop(cls):
        # Set up a SIGINT handler to allow terminating a plot via CTRL-C.
        with _allow_interrupt_macos():
            _macos.show()

    def show(self):
        if self.canvas.figure.stale:
            self.canvas.draw_idle()
        if not self._shown:
            self._show()
            self._shown = True
        if mpl.rcParams["figure.raise_window"]:
            self._raise()


@_Backend.export
class _BackendMac(_Backend):
    FigureCanvas = FigureCanvasMacAgg
    FigureManager = FigureManagerMac
    mainloop = FigureManagerMac.start_main_loop
