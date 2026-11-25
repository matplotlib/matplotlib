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
class _FigureCanvasGTK(FigureCanvasBase[ToolbarT, ManagerT, TimerGTK]): ...

# Similarly generic so FigureManagerGTK3/4 can supply the concrete canvas and
# toolbar types.
class _FigureManagerGTK(FigureManagerBase[CanvasT, ToolbarT]):
    window: Any  # Gtk.Window
    vbox: Any  # Gtk.Box
    # destroy takes *args because it is connected as a GTK signal handler.
    def destroy(self, *args: Any) -> None: ...
    # Runtime names the parameters ``width, height`` while the base uses
    # ``w, h``; the base only ever calls this positionally.
    def resize(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, width: int, height: int
    ) -> None: ...

# Similarly generic so NavigationToolbar2GTK3/4 can supply the canvas type.
class _NavigationToolbar2GTK(NavigationToolbar2[CanvasT]):
    message: Any  # Gtk.Label

class RubberbandGTK(backend_tools.RubberbandBase):
    def draw_rubberband(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x0: float, y0: float, x1: float, y1: float
    ) -> None: ...

class ConfigureSubplotsGTK(backend_tools.ConfigureSubplotsBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any
    ) -> None: ...

class _BackendGTK(_Backend[Any]): ...
