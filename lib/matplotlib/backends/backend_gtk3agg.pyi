from typing import Any

from matplotlib.backend_bases import ShowBase
from matplotlib.figure import Figure
from matplotlib.transforms import BboxBase

from .backend_agg import FigureCanvasAgg
from .backend_gtk3 import FigureCanvasGTK3, FigureManagerGTK3, _BackendGTK3

class FigureCanvasGTK3Agg(FigureCanvasAgg, FigureCanvasGTK3):
    def on_draw_event(self, widget: Any, ctx: Any) -> bool: ...
    def blit(self, bbox: BboxBase | None = ...) -> None: ...

class _BackendGTK3Agg(_BackendGTK3): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasGTK3Agg
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
