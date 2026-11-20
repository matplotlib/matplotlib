from typing import Any

from matplotlib.backends.backend_qt import (
    SPECIAL_KEYS as SPECIAL_KEYS,
    cursord as cursord,
    _create_qApp as _create_qApp,
    _BackendQT as _BackendQT,
    TimerQT as TimerQT,
    MainWindow as MainWindow,
    FigureCanvasQT as FigureCanvasQT,
    FigureManagerQT as FigureManagerQT,
    ToolbarQt as ToolbarQt,
    NavigationToolbar2QT as NavigationToolbar2QT,
    SubplotToolQt as SubplotToolQt,
    SaveFigureQt as SaveFigureQt,
    ConfigureSubplotsQt as ConfigureSubplotsQt,
    RubberbandQt as RubberbandQt,
    HelpQt as HelpQt,
    ToolCopyToClipboardQT as ToolCopyToClipboardQT,
    FigureCanvasBase as FigureCanvasBase,
    FigureManagerBase as FigureManagerBase,
    MouseButton as MouseButton,
    NavigationToolbar2 as NavigationToolbar2,
    ShowBase as ShowBase,
    TimerBase as TimerBase,
    ToolContainerBase as ToolContainerBase,
    figureoptions as figureoptions,
    Gcf as Gcf,
)
from matplotlib.figure import Figure

qApp: Any

class _BackendQT5(_BackendQT): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasQT
FigureManager = FigureManagerQT

def new_figure_manager(
    num: int | str, *args: Any, **kwargs: Any
) -> FigureManagerQT: ...
def new_figure_manager_given_figure(
    num: int | str, figure: Figure
) -> FigureManagerQT: ...
def draw_if_interactive() -> None: ...
def show(*, block: bool | None = ...) -> None: ...

class Show(ShowBase): ...
