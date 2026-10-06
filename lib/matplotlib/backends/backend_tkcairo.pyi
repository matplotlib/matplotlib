from typing import Any

from matplotlib.backend_bases import ShowBase
from matplotlib.figure import Figure
from matplotlib.backends.backend_cairo import FigureCanvasCairo
from matplotlib.backends._backend_tk import (
    FigureCanvasTk,
    FigureManagerTk,
    NavigationToolbar2Tk,
    TimerTk,
    _BackendTk,
)

class FigureCanvasTkCairo(
    FigureCanvasCairo[NavigationToolbar2Tk, FigureManagerTk, TimerTk], FigureCanvasTk
): ...
class _BackendTkCairo(_BackendTk): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: float  # ``tk.TkVersion``
FigureCanvas = FigureCanvasTkCairo
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
