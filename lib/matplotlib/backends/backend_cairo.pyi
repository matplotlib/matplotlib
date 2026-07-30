from typing import Any, BinaryIO

from matplotlib.backend_bases import (
    _Backend,
    FigureCanvasBase,
    FigureManagerBase,
    GraphicsContextBase,
    NavigationToolbar2,
    RendererBase,
    ShowBase,
)
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.transforms import Bbox, Transform

from matplotlib.typing import ColorType

# Re-exported for use by GUI cairo backends; ``Any`` because the ``cairo``
# module has no type stubs available.
cairo: Any

class RendererCairo(RendererBase[GraphicsContextCairo]):
    dpi: float
    gc: GraphicsContextCairo
    text_ctx: Any
    mathtext_parser: Any
    def __init__(self, dpi: float) -> None: ...
    def set_context(self, ctx: Any) -> None: ...
    # draw_markers renames the 5th param from the base's `trans` to `transform`.
    def draw_markers(
        self,
        gc: GraphicsContextCairo,
        marker_path: Path,
        marker_trans: Transform,
        path: Path,
        transform: Transform,
        rgbFace: ColorType | None = ...,
    ) -> None: ...

class GraphicsContextCairo(GraphicsContextBase):
    renderer: RendererCairo
    ctx: Any
    def __init__(self, renderer: RendererCairo) -> None: ...
    def set_ctx_from_surface(self, surface: Any) -> None: ...
    def set_context(self, ctx: Any) -> None: ...

class _CairoRegion:
    def __init__(self, slices: Any, data: Any) -> None: ...

class FigureCanvasCairo(FigureCanvasBase[NavigationToolbar2, FigureManagerBase]):
    @property
    def _renderer(self) -> RendererCairo: ...
    def get_renderer(self) -> RendererCairo: ...
    def copy_from_bbox(self, bbox: Bbox) -> _CairoRegion: ...
    def restore_region(self, region: _CairoRegion) -> None: ...
    def print_png(self, fobj: str | BinaryIO) -> None: ...
    def print_rgba(self, fobj: str | BinaryIO) -> None: ...
    print_raw = print_rgba
    def print_pdf(self, fobj: str | BinaryIO, *, orientation: str = ...) -> None: ...
    def print_ps(self, fobj: str | BinaryIO, *, orientation: str = ...) -> None: ...
    def print_svg(self, fobj: str | BinaryIO, *, orientation: str = ...) -> None: ...
    def print_svgz(self, fobj: str | BinaryIO, *, orientation: str = ...) -> None: ...

class _BackendCairo(_Backend[type[FigureCanvasCairo]]): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasCairo
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
