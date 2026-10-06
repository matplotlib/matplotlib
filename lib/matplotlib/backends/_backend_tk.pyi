import tkinter as tk
from typing import Any

from matplotlib import backend_tools
from matplotlib.backend_bases import (
    _Backend,
    FigureCanvasBase,
    FigureManagerBase,
    NavigationToolbar2,
    ShowBase,
    TimerBase,
    ToolContainerBase,
)
from matplotlib.figure import Figure

# Tkinter objects are typed as Any where they cross the matplotlib API, as the
# stdlib tkinter stubs are not consistently available across all members used.
cursord: dict[Any, str]
_blit_args: dict[str, Any]

def blit(
    photoimage: Any,
    aggimage: Any,
    offsets: tuple[int, int, int, int],
    bbox: Any = ...,
) -> None: ...

class TimerTk(TimerBase):
    def __init__(self, parent: tk.Misc, *args: Any, **kwargs: Any) -> None: ...

class FigureCanvasTk(FigureCanvasBase[NavigationToolbar2Tk, FigureManagerTk, TimerTk]):
    # Backend-authoring API: the ``tk.Canvas`` hosting the figure image and the
    # ``tk.PhotoImage`` it displays.  Subclasses bind events and draw on these.
    _tkcanvas: tk.Canvas
    _tkphoto: tk.PhotoImage
    def __init__(self, figure: Figure | None = ..., master: Any = ...) -> None: ...
    def resize(self, event: Any) -> None: ...
    def get_tk_widget(self) -> Any: ...
    def motion_notify_event(self, event: Any) -> None: ...
    def enter_notify_event(self, event: Any) -> None: ...
    def leave_notify_event(self, event: Any) -> None: ...
    def button_press_event(self, event: Any, dblclick: bool = ...) -> None: ...
    def button_dblclick_event(self, event: Any) -> None: ...
    def button_release_event(self, event: Any) -> None: ...
    def scroll_event(self, event: Any) -> None: ...
    def scroll_event_windows(self, event: Any) -> None: ...
    def key_press(self, event: Any) -> None: ...
    def key_release(self, event: Any) -> None: ...
    def new_timer(self, *args: Any, **kwargs: Any) -> TimerTk: ...
    def start_event_loop(self, timeout: float = ...) -> None: ...

class FigureManagerTk(FigureManagerBase[FigureCanvasTk, NavigationToolbar2Tk]):
    window: Any
    def __init__(self, canvas: FigureCanvasTk, num: int | str, window: Any) -> None: ...
    # Runtime names the parameters ``width, height`` while the base uses
    # ``w, h``; the base only ever calls this positionally.
    def resize(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, width: int, height: int
    ) -> None: ...
    def destroy(self, *args: Any) -> None: ...

def add_tooltip(widget: Any, text: str) -> None: ...

class NavigationToolbar2Tk(NavigationToolbar2[FigureCanvasTk], tk.Frame):
    window: Any
    # ``message`` is a ``tk.StringVar`` at runtime (unlike the ``str`` on the
    # base class), so it is typed as ``Any`` to avoid a spurious conflict.
    message: Any
    def __init__(
        self,
        canvas: FigureCanvasTk,
        window: Any = ...,
        *,
        pack_toolbar: bool = ...,
    ) -> None: ...

class RubberbandTk(backend_tools.RubberbandBase):
    def draw_rubberband(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x0: float, y0: float, x1: float, y1: float
    ) -> None: ...

class ToolbarTk(ToolContainerBase, tk.Frame):
    def __init__(self, toolmanager: Any, window: Any = ...) -> None: ...
    # Runtime names the fourth parameter ``image_file`` while the base uses
    # ``image``; the base only ever calls this positionally.
    def add_toolitem(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        name: str,
        group: str,
        position: int,
        image_file: str,
        description: str,
        toggle: bool,
    ) -> None: ...

class SaveFigureTk(backend_tools.SaveFigureBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any
    ) -> None: ...

class ConfigureSubplotsTk(backend_tools.ConfigureSubplotsBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any
    ) -> None: ...

class HelpTk(backend_tools.ToolHelpBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any
    ) -> None: ...

Toolbar = ToolbarTk

class _BackendTk(_Backend[type[FigureCanvasTk]]): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: float  # ``tk.TkVersion``
FigureCanvas = FigureCanvasTk
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
