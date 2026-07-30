from typing import Any

from matplotlib import backend_tools
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

# GTK3 uses the same TimerGTK implementation.
TimerGTK3 = TimerGTK

class FigureCanvasGTK3(_FigureCanvasGTK["NavigationToolbar2GTK3", "FigureManagerGTK3"]):
    required_interactive_framework: str
    # GTK signal-handler methods (new; not base overrides):
    def scroll_event(self, widget: Any, event: Any) -> bool: ...
    def button_press_event(self, widget: Any, event: Any) -> bool: ...
    def button_release_event(self, widget: Any, event: Any) -> bool: ...
    def key_press_event(self, widget: Any, event: Any) -> bool: ...
    def key_release_event(self, widget: Any, event: Any) -> bool: ...
    def motion_notify_event(self, widget: Any, event: Any) -> bool: ...
    def enter_notify_event(self, widget: Any, event: Any) -> bool: ...
    def leave_notify_event(self, widget: Any, event: Any) -> bool: ...
    def size_allocate(self, widget: Any, allocation: Any) -> None: ...
    def configure_event(self, widget: Any, event: Any) -> bool: ...
    def on_draw_event(self, widget: Any, ctx: Any) -> bool: ...

class NavigationToolbar2GTK3(_NavigationToolbar2GTK[FigureCanvasGTK3]): ...

class ToolbarGTK3(ToolContainerBase):
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

class SaveFigureGTK3(backend_tools.SaveFigureBase): ...
class HelpGTK3(backend_tools.ToolHelpBase): ...
class ToolCopyToClipboardGTK3(backend_tools.ToolCopyToClipboardBase): ...

class FigureManagerGTK3(_FigureManagerGTK[FigureCanvasGTK3]):
    _toolbar2_class: type[NavigationToolbar2GTK3]
    _toolmanager_toolbar_class: type[ToolbarGTK3]

class _BackendGTK3(_BackendGTK): ...

Toolbar = ToolbarGTK3

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasGTK3
FigureManager = FigureManagerGTK3

def new_figure_manager(
    num: int | str, *args: Any, **kwargs: Any
) -> FigureManagerGTK3: ...
def new_figure_manager_given_figure(
    num: int | str, figure: Figure
) -> FigureManagerGTK3: ...
def draw_if_interactive() -> None: ...
def show(*, block: bool | None = ...) -> None: ...

class Show(ShowBase): ...
