from typing import Any

from matplotlib.backend_bases import ShowBase
from matplotlib.figure import Figure
from matplotlib.transforms import BboxBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends._backend_tk import (
    FigureCanvasTk,
    FigureManagerTk as FigureManagerTk,
    NavigationToolbar2Tk as NavigationToolbar2Tk,
    _BackendTk,
)

class FigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def blit(self, bbox: BboxBase | None = ...) -> None: ...

class _BackendTkAgg(_BackendTk): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasTkAgg
FigureManager = FigureManagerTk

def new_figure_manager(
    num: int | str, *args: Any, **kwargs: Any
) -> FigureManagerTk: ...
def new_figure_manager_given_figure(
    num: int | str, figure: Figure
) -> FigureManagerTk: ...
def draw_if_interactive() -> None: ...
def show(*, block: bool | None = ...) -> None: ...

class Show(ShowBase): ...
