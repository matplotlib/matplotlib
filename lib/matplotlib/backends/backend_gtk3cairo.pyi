from typing import Any

from matplotlib.backend_bases import ShowBase
from matplotlib.figure import Figure

from .backend_cairo import FigureCanvasCairo
from .backend_gtk3 import FigureCanvasGTK3, FigureManagerGTK3, _BackendGTK3

class FigureCanvasGTK3Cairo(FigureCanvasCairo, FigureCanvasGTK3):
    def on_draw_event(self, widget: Any, ctx: Any) -> bool: ...

class _BackendGTK3Cairo(_BackendGTK3): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasGTK3Cairo
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
