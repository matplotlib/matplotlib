from typing import Any

from matplotlib import backend_tools
from matplotlib.backend_bases import (
    _Backend,
    FigureCanvasBase,
    FigureManagerBase,
    GraphicsContextBase,
    NavigationToolbar2,
    RendererBase,
    ShowBase,
    TimerBase,
    ToolContainerBase,
    _UntypedToolkitWidget,
)
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

# wx/wxPython objects have no type stubs → Any.

PIXELS_PER_INCH: int

# The wx classes below also inherit their native ``wx`` base at runtime.
# wxPython has no type stubs, so ``_UntypedToolkitWidget`` stands in for the
# native base: downstream subclasses see the widget API (as ``Any``) exactly
# as they would from the runtime module.

class TimerWx(TimerBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class RendererWx(RendererBase[GraphicsContextWx]):
    """DEPRECATED renderer (wx backend is deprecated; use wxagg)."""

    fontweights: dict[Any, Any]
    fontangles: dict[Any, Any]
    fontnames: dict[Any, Any]

    def __init__(self, bitmap: Any, dpi: float) -> None: ...  # bitmap: wx.Bitmap
    def handle_clip_rectangle(self, gc: Any) -> None: ...
    def get_wx_font(self, s: str, prop: FontProperties) -> Any: ...  # returns wx.Font

class GraphicsContextWx(GraphicsContextBase):
    def __init__(
        self,
        bitmap: Any,
        renderer: RendererWx,  # bitmap: wx.Bitmap
    ) -> None: ...

class _FigureCanvasWxBase(  # wx.Panel
    FigureCanvasBase["NavigationToolbar2Wx", "FigureManagerWx", TimerWx], _UntypedToolkitWidget
):
    keyvald: dict[Any, str]
    def __init__(
        self,
        parent: Any,  # wx.Window
        id: Any,  # wx.WindowID
        figure: Figure | None = ...,
    ) -> None: ...
    def stop_event_loop(
        self,
        event: Any = ...,  # wx.Event
    ) -> None: ...
    def gui_repaint(self, drawDC: object = ...) -> None: ...

class FigureCanvasWx(_FigureCanvasWxBase):
    def draw(self, drawDC: object = ...) -> None: ...
    def print_bmp(self, filename: str, **kwargs: Any) -> None: ...
    def print_jpeg(self, filename: str, **kwargs: Any) -> None: ...
    def print_jpg(self, filename: str, **kwargs: Any) -> None: ...
    def print_pcx(self, filename: str, **kwargs: Any) -> None: ...
    def print_png(self, filename: str, **kwargs: Any) -> None: ...
    def print_tiff(self, filename: str, **kwargs: Any) -> None: ...
    def print_tif(self, filename: str, **kwargs: Any) -> None: ...
    def print_xpm(self, filename: str, **kwargs: Any) -> None: ...

class FigureFrameWx(_UntypedToolkitWidget):  # wx.Frame
    def __init__(
        self,
        num: int | str,
        fig: Figure,
        *,
        canvas_class: type[FigureCanvasWx],
    ) -> None: ...

class FigureManagerWx(FigureManagerBase[_FigureCanvasWxBase, NavigationToolbar2Wx]):
    frame: Any  # wx.Frame
    window: Any  # wx.Frame (alias of frame)

    def __init__(
        self,
        canvas: _FigureCanvasWxBase,
        num: int | str,
        frame: Any,  # wx.Frame
    ) -> None: ...
    def destroy(self, *args: Any) -> None: ...
    @classmethod
    def start_main_loop(cls) -> None: ...
    @classmethod
    def create_with_canvas(
        cls,
        canvas_class: type[_FigureCanvasWxBase],
        figure: Figure,
        num: int | str,
    ) -> FigureManagerWx: ...

class NavigationToolbar2Wx(NavigationToolbar2[_FigureCanvasWxBase], _UntypedToolkitWidget):  # wx.ToolBar
    def __init__(
        self,
        canvas: _FigureCanvasWxBase,
        coordinates: bool = ...,
        *,
        style: Any = ...,  # wx.TB_* flag
    ) -> None: ...

class ToolbarWx(ToolContainerBase, _UntypedToolkitWidget):  # wx.ToolBar
    def __init__(
        self,
        toolmanager: Any,  # ToolManager; wx mix-in makes base __init__ incompatible
        parent: Any = ...,  # wx.Window
        style: Any = ...,  # wx.TB_* flag
    ) -> None: ...
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

class ConfigureSubplotsWx(backend_tools.ConfigureSubplotsBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any
    ) -> None: ...

class SaveFigureWx(backend_tools.SaveFigureBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any
    ) -> None: ...

class RubberbandWx(backend_tools.RubberbandBase):
    def draw_rubberband(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, x0: float, y0: float, x1: float, y1: float
    ) -> None: ...

class HelpWx(backend_tools.ToolHelpBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any
    ) -> None: ...

class ToolCopyToClipboardWx(backend_tools.ToolCopyToClipboardBase):
    def trigger(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, *args: Any, **kwargs: Any
    ) -> None: ...
class _BackendWx(_Backend[type[_FigureCanvasWxBase]]): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasWx
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
