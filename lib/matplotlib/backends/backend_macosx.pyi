from typing import Any

from matplotlib.backend_bases import (
    _Backend,
    Event,
    FigureManagerBase,
    NavigationToolbar2,
    ShowBase,
    TimerBase,
)
from matplotlib.figure import Figure
from matplotlib.transforms import BboxBase
from matplotlib.backends._macosx import (
    FigureCanvas as _FigureCanvasMacOSX,
    FigureManager as _FigureManagerMacOSX,
    NavigationToolbar2 as _NavigationToolbar2MacOSX,
    Timer as _TimerMacOSX,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg

class TimerMac(_TimerMacOSX, TimerBase): ...

class FigureCanvasMac(
    FigureCanvasAgg[NavigationToolbar2Mac, FigureManagerMac, TimerMac],
    _FigureCanvasMacOSX,
):
    required_interactive_framework: str
    _timer_cls: type[TimerMac]
    def draw(self) -> None: ...
    def draw_idle(self) -> None: ...
    def blit(self, bbox: BboxBase | None = ...) -> None: ...
    def resize(self, width: float, height: float) -> None: ...

class NavigationToolbar2Mac(
    _NavigationToolbar2MacOSX, NavigationToolbar2[FigureCanvasMac]
):
    def draw_rubberband(
        self, event: Event, x0: float, y0: float, x1: float, y1: float
    ) -> None: ...
    def remove_rubberband(self) -> None: ...
    def save_figure(self, *args: object) -> str | None: ...

class FigureManagerMac(_FigureManagerMacOSX, FigureManagerBase[FigureCanvasMac]):
    _toolbar2_class: type[NavigationToolbar2Mac]
    def destroy(self) -> None: ...
    @classmethod
    def start_main_loop(cls) -> None: ...
    def show(self) -> None: ...

class _BackendMac(_Backend[type[FigureCanvasMac]]): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasMac
FigureManager = FigureManagerMac

def new_figure_manager(
    num: int | str, *args: Any, **kwargs: Any
) -> FigureManagerMac: ...
def new_figure_manager_given_figure(
    num: int | str, figure: Figure
) -> FigureManagerMac: ...
def draw_if_interactive() -> None: ...
def show(*, block: bool | None = ...) -> None: ...

class Show(ShowBase): ...
