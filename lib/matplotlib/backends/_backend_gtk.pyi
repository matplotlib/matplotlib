from typing import Any

from matplotlib import backend_tools
from matplotlib.backend_bases import (
    _Backend,
    CanvasT,
    FigureCanvasBase,
    FigureManagerBase,
    ManagerT,
    NavigationToolbar2,
    TimerBase,
    ToolbarT,
)

# GTK/GObject objects have no type stubs; they are typed as Any throughout.

def mpl_to_gtk_cursor_name(mpl_cursor: Any) -> str | None: ...

class TimerGTK(TimerBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

# _FigureCanvasGTK is generic over ToolbarT and ManagerT so that the
# version-specific subclasses (FigureCanvasGTK3, FigureCanvasGTK4) can fix
# them to concrete types without breaking the inheritance chain.
# TimerGTK is fixed here since all GTK versions share the same timer class.
class _FigureCanvasGTK(FigureCanvasBase[ToolbarT, ManagerT, TimerGTK]):
    _timer_cls: type[TimerGTK]

# Similarly generic so FigureManagerGTK3/4 can supply the concrete canvas type.
class _FigureManagerGTK(FigureManagerBase[CanvasT]):
    window: Any  # Gtk.Window
    vbox: Any  # Gtk.Box
    # destroy takes *args because it is connected as a GTK signal handler.
    def destroy(self, *args: Any) -> None: ...
    # resize uses width/height rather than the base's w/h.
    def resize(self, width: int, height: int) -> None: ...

# Similarly generic so NavigationToolbar2GTK3/4 can supply the canvas type.
class _NavigationToolbar2GTK(NavigationToolbar2[CanvasT]):
    message: Any  # Gtk.Label (or Gtk.StringVar equivalent)

class RubberbandGTK(backend_tools.RubberbandBase): ...
class ConfigureSubplotsGTK(backend_tools.ConfigureSubplotsBase): ...

class _BackendGTK(_Backend):
    backend_version: str
