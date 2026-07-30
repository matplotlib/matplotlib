from typing import Any

from matplotlib.backend_bases import ShowBase
from matplotlib.figure import Figure
from matplotlib.transforms import BboxBase

from .backend_agg import FigureCanvasAgg
from .backend_wx import (
    _BackendWx,
    _FigureCanvasWxBase,
    FigureManagerWx,
    NavigationToolbar2Wx,
)

NavigationToolbar2WxAgg = NavigationToolbar2Wx

class FigureCanvasWxAgg(FigureCanvasAgg, _FigureCanvasWxBase):
    def draw(self, drawDC: object = ...) -> None: ...
    def blit(self, bbox: BboxBase | None = ...) -> None: ...

class _BackendWxAgg(_BackendWx): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasWxAgg
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
