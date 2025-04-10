from contextlib import contextmanager

import matplotlib as mpl
from matplotlib import _c_internal_utils
from matplotlib.backends.backend_tkagg import (
    _BackendTkAgg,
    FigureManagerTk as _FigureManagerTk,
)


@contextmanager
def _restore_foreground_window_at_end():
    foreground = _c_internal_utils.Win32_GetForegroundWindow()
    try:
        yield
    finally:
        if mpl.rcParams["tk.window_focus"]:
            _c_internal_utils.Win32_SetForegroundWindow(foreground)


class FigureManagerTk(_FigureManagerTk):
    _active_managers = None

    def show(self):
        with _restore_foreground_window_at_end():
            if not self._shown:
                self.window.protocol("WM_DELETE_WINDOW", self.destroy)
                self.window.deiconify()
                self.canvas._tkcanvas.focus_set()
            else:
                self.canvas.draw_idle()
            if mpl.rcParams["figure.raise_window"]:
                self.canvas.manager.window.attributes("-topmost", 1)
                self.canvas.manager.window.attributes("-topmost", 0)
            self._shown = True

    def destroy(self, *args):
        if self.canvas._idle_draw_id:
            self.canvas._tkcanvas.after_cancel(self.canvas._idle_draw_id)
        if self.canvas._event_loop_id:
            self.canvas._tkcanvas.after_cancel(self.canvas._event_loop_id)

        # NOTE: events need to be flushed before issuing destroy (GH #9956),
        # however, self.window.update() can break user code. This is the
        # safest way to achieve a complete draining of the event queue,
        # but it may require users to update() on their own to execute the
        # completion in obscure corner cases.
        def delayed_destroy():
            self.window.destroy()

            if self._owns_mainloop and not self._active_managers:
                self.window.quit()

        # "after idle after 0" avoids Tcl error/race (GH #19940)
        self.window.after_idle(self.window.after, 0, delayed_destroy)


@_BackendTkAgg.export
class _PatchedBackendTkAgg(_BackendTkAgg):
    @classmethod
    def mainloop(cls):
        managers = cls.FigureManager._active_managers
        if managers:
            first_manager = managers[0]
            manager_class = type(first_manager)
            if manager_class._owns_mainloop:
                return
            manager_class._owns_mainloop = True
            try:
                first_manager.window.mainloop()
            finally:
                manager_class._owns_mainloop = False

    FigureManager = FigureManagerTk


Backend = _PatchedBackendTkAgg
