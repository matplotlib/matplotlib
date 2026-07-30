from typing import Any

from matplotlib import _api, backend_tools
from matplotlib.backend_bases import ShowBase, ToolContainerBase
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

class FigureCanvasGTK4(_FigureCanvasGTK["NavigationToolbar2GTK4", "FigureManagerGTK4"]):
    required_interactive_framework: str
    @_api.classproperty
    def supports_blit(cls) -> bool: ...
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

class NavigationToolbar2GTK4(_NavigationToolbar2GTK[FigureCanvasGTK4]): ...

class ToolbarGTK4(ToolContainerBase):
    # add_toolitem uses image_file as the positional image argument name.
    def add_toolitem(
        self,
        name: str,
        group: str,
        position: int,
        image_file: str,
        /,
        description: str,
        toggle: bool,
    ) -> None: ...

class SaveFigureGTK4(backend_tools.SaveFigureBase): ...
class HelpGTK4(backend_tools.ToolHelpBase): ...
class ToolCopyToClipboardGTK4(backend_tools.ToolCopyToClipboardBase): ...

class FigureManagerGTK4(_FigureManagerGTK[FigureCanvasGTK4]):
    _toolbar2_class: type[NavigationToolbar2GTK4]
    _toolmanager_toolbar_class: type[ToolbarGTK4]

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
