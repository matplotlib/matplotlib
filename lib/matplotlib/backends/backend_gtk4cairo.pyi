from typing import Any

from matplotlib.backend_bases import ShowBase
from matplotlib.figure import Figure

from .backend_cairo import FigureCanvasCairo
from ._backend_gtk import TimerGTK
from .backend_gtk4 import (
    FigureCanvasGTK4,
    FigureManagerGTK4,
    NavigationToolbar2GTK4,
    _BackendGTK4,
)

class FigureCanvasGTK4Cairo(
    FigureCanvasCairo[NavigationToolbar2GTK4, FigureManagerGTK4, TimerGTK],
    FigureCanvasGTK4,
):
    def on_draw_event(self, widget: Any, ctx: Any) -> bool: ...

class _BackendGTK4Cairo(_BackendGTK4): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasGTK4Cairo
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
