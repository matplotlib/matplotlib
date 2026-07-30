from typing import Any, BinaryIO

from matplotlib.backend_bases import (
    _Backend,
    FigureCanvasBase,
    FigureManagerBase,
    GraphicsContextBase,
    ManagerT,
    RendererBase,
    ShowBase,
    ToolbarT,
    TimerT,
)
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox

def get_hinting_flag() -> int: ...

class RendererAgg(RendererBase[GraphicsContextBase]):
    dpi: float
    width: int
    height: int
    mathtext_parser: Any
    bbox: Bbox
    def __init__(self, width: int, height: int, dpi: float) -> None: ...
    def draw_mathtext(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
    ) -> None: ...
    def buffer_rgba(self) -> memoryview: ...
    def tostring_argb(self) -> bytes: ...
    def clear(self) -> None: ...
    def restore_region(
        self,
        region: Any,
        bbox: Bbox | None = ...,
        xy: tuple[float, float] | None = ...,
    ) -> None: ...
    def stop_filter(self, post_processing: Any) -> None: ...

class FigureCanvasAgg(FigureCanvasBase[ToolbarT, ManagerT, TimerT]):
    renderer: RendererAgg
    def copy_from_bbox(self, bbox: Bbox) -> Any: ...
    def restore_region(
        self,
        region: Any,
        bbox: Bbox | None = ...,
        xy: tuple[float, float] | None = ...,
    ) -> None: ...
    def get_renderer(self) -> RendererAgg: ...
    def tostring_argb(self) -> bytes: ...
    def buffer_rgba(self) -> memoryview: ...
    def print_raw(
        self, filename_or_obj: str | BinaryIO, *, metadata: dict[str, Any] | None = ...
    ) -> None: ...
    print_rgba = print_raw
    def print_png(
        self,
        filename_or_obj: str | BinaryIO,
        *,
        metadata: dict[str, Any] | None = ...,
        pil_kwargs: dict[str, Any] | None = ...,
    ) -> None: ...
    def print_to_buffer(self) -> tuple[bytes, tuple[int, int]]: ...
    def print_gif(
        self,
        filename_or_obj: str | BinaryIO,
        *,
        metadata: dict[str, Any] | None = ...,
        pil_kwargs: dict[str, Any] | None = ...,
    ) -> None: ...
    def print_jpg(
        self,
        filename_or_obj: str | BinaryIO,
        *,
        metadata: dict[str, Any] | None = ...,
        pil_kwargs: dict[str, Any] | None = ...,
    ) -> None: ...
    print_jpeg = print_jpg
    def print_tif(
        self,
        filename_or_obj: str | BinaryIO,
        *,
        metadata: dict[str, Any] | None = ...,
        pil_kwargs: dict[str, Any] | None = ...,
    ) -> None: ...
    print_tiff = print_tif
    def print_webp(
        self,
        filename_or_obj: str | BinaryIO,
        *,
        metadata: dict[str, Any] | None = ...,
        pil_kwargs: dict[str, Any] | None = ...,
    ) -> None: ...
    def print_avif(
        self,
        filename_or_obj: str | BinaryIO,
        *,
        metadata: dict[str, Any] | None = ...,
        pil_kwargs: dict[str, Any] | None = ...,
    ) -> None: ...

class _BackendAgg(_Backend[type[FigureCanvasAgg]]): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasAgg
FigureManager = FigureManagerBase

def new_figure_manager(
    num: int | str, *args: Any, **kwargs: Any
) -> FigureManagerBase: ...
def new_figure_manager_given_figure(
    num: int | str, figure: Figure
) -> FigureManagerBase: ...
def draw_if_interactive() -> None: ...
def show(*, block: bool | None = ...) -> None: ...

class Show(ShowBase): ...
