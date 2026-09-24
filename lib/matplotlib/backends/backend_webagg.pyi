from typing import Any

from matplotlib.backend_bases import _Backend, ShowBase
from matplotlib.figure import Figure
from matplotlib.backends.backend_webagg_core import (
    FigureCanvasWebAggCore,
    FigureManagerWebAgg as _FigureManagerWebAggCore,
    NavigationToolbar2WebAgg,
    TimerAsyncio as TimerAsyncio,
    TimerTornado as TimerTornado,
)

# Re-export timer classes from core.
# TimerAsyncio and TimerTornado re-exported from core.

class FigureManagerWebAgg(_FigureManagerWebAggCore):
    _toolbar2_class: type[NavigationToolbar2WebAgg]
    @classmethod
    def pyplot_show(cls, *, block: bool | None = ...) -> None: ...

class FigureCanvasWebAgg(
    FigureCanvasWebAggCore[NavigationToolbar2WebAgg, FigureManagerWebAgg]
): ...

class WebAggApplication:
    # tornado.web.Application subclass; tornado has no stubs → Any.
    initialized: bool
    started: bool
    def __init__(self, url_prefix: str = ...) -> None: ...
    @classmethod
    def initialize(
        cls,
        url_prefix: str = ...,
        port: int | None = ...,
        address: str | None = ...,
    ) -> None: ...
    @classmethod
    def start(cls) -> None: ...

class _BackendWebAgg(_Backend[type[FigureCanvasWebAgg]]): ...

def ipython_inline_display(figure: Any) -> str: ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasWebAgg
FigureManager = FigureManagerWebAgg

def new_figure_manager(
    num: int | str, *args: Any, **kwargs: Any
) -> FigureManagerWebAgg: ...
def new_figure_manager_given_figure(
    num: int | str, figure: Figure
) -> FigureManagerWebAgg: ...
def draw_if_interactive() -> None: ...
def show(*, block: bool | None = ...) -> None: ...

class Show(ShowBase): ...
