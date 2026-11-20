from typing import Any

from matplotlib import backend_tools
from matplotlib.backend_bases import ShowBase, ToolContainerBase, _UntypedToolkitWidget
from matplotlib.figure import Figure

from ._backend_gtk import (
    _BackendGTK,
    _FigureCanvasGTK,
    _FigureManagerGTK,
    _NavigationToolbar2GTK,
    TimerGTK,
)

# GTK/GObject objects (Gtk.*, Gdk.*, etc.) have no type stubs → Any.
# GLib and Gtk are imported from gi.repository and re-exported by this module.
GLib: Any
Gtk: Any

# GTK4 uses the same TimerGTK implementation.
TimerGTK4 = TimerGTK

# The GTK widget classes below also inherit their native ``Gtk`` base at
# runtime.  ``gi`` has no type stubs, so ``_UntypedToolkitWidget`` stands in
# for the native base: downstream subclasses see the widget API (as ``Any``)
# exactly as they would from the runtime module.

class FigureCanvasGTK4(  # Gtk.DrawingArea
    _FigureCanvasGTK["NavigationToolbar2GTK4", "FigureManagerGTK4"], _UntypedToolkitWidget
):
    # GTK4 event-controller signal-handler methods (new; not base overrides):
    def scroll_event(self, controller: Any, dx: float, dy: float) -> bool: ...
    def button_press_event(
        self, controller: Any, n_press: int, x: float, y: float
    ) -> bool: ...
    def button_release_event(
        self, controller: Any, n_press: int, x: float, y: float
    ) -> bool: ...
    def key_press_event(
        self, controller: Any, keyval: int, keycode: int, state: Any
    ) -> bool: ...
    def key_release_event(
        self, controller: Any, keyval: int, keycode: int, state: Any
    ) -> bool: ...
    def motion_notify_event(self, controller: Any, x: float, y: float) -> None: ...
    def enter_notify_event(self, controller: Any, x: float, y: float) -> None: ...
    def leave_notify_event(self, controller: Any) -> None: ...
    def resize_event(self, area: Any, width: int, height: int) -> None: ...
    def on_draw_event(self, widget: Any, ctx: Any) -> bool: ...

class NavigationToolbar2GTK4(  # Gtk.Box
    _NavigationToolbar2GTK[FigureCanvasGTK4], _UntypedToolkitWidget
): ...

class ToolbarGTK4(ToolContainerBase, _UntypedToolkitWidget):  # Gtk.Box
    # Runtime names the fourth parameter ``image_file`` while the base uses
    # ``image``; the base only ever calls this positionally.
    def add_toolitem(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        name: str,
        group: str,
        position: int,
        image_file: str,
        description: str,
        toggle: bool,
    ) -> None: ...

class SaveFigureGTK4(backend_tools.SaveFigureBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any, **kwargs: Any
    ) -> None: ...

class HelpGTK4(backend_tools.ToolHelpBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any
    ) -> None: ...

class ToolCopyToClipboardGTK4(backend_tools.ToolCopyToClipboardBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any, **kwargs: Any
    ) -> None: ...

class FigureManagerGTK4(_FigureManagerGTK[FigureCanvasGTK4, NavigationToolbar2GTK4]):
    ...

class _BackendGTK4(_BackendGTK): ...

Toolbar = ToolbarGTK4

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasGTK4
FigureManager = FigureManagerGTK4

def new_figure_manager(
    num: int | str, *args: Any, **kwargs: Any
) -> FigureManagerGTK4: ...
def new_figure_manager_given_figure(
    num: int | str, figure: Figure
) -> FigureManagerGTK4: ...
def draw_if_interactive() -> None: ...
def show(*, block: bool | None = ...) -> None: ...

class Show(ShowBase): ...
