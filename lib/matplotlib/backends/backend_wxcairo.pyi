from typing import Any

from matplotlib.backend_bases import ShowBase
from matplotlib.figure import Figure

from .backend_cairo import FigureCanvasCairo
from .backend_wx import (
    _BackendWx,
    _FigureCanvasWxBase,
    FigureManagerWx,
    NavigationToolbar2Wx,
)

NavigationToolbar2WxCairo = NavigationToolbar2Wx

class FigureCanvasWxCairo(FigureCanvasCairo, _FigureCanvasWxBase):
    def draw(self, drawDC: object = ...) -> None: ...

class _BackendWxCairo(_BackendWx): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasWxCairo
FigureManager = FigureManagerWx

def new_figure_manager(
    num: int | str, *args: Any, **kwargs: Any
) -> FigureManagerWx: ...
def new_figure_manager_given_figure(
    num: int | str, figure: Figure
) -> FigureManagerWx: ...
def draw_if_interactive() -> None: ...
def show(*, block: bool | None = ...) -> None: ...

class Show(ShowBase): ...
