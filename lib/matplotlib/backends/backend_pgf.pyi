import os
from collections.abc import Callable
from typing import IO, Any, Self

from matplotlib.backend_bases import (
    _Backend,
    FigureCanvasBase,
    FigureManagerBase,
    NavigationToolbar2,
    GraphicsContextBase,
    RendererBase,
    ShowBase,
)
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

def _metadata_to_str(key: str, value: Any) -> str: ...
def make_pdf_to_png_converter() -> Callable[..., Any]: ...

class LatexError(Exception):
    latex_output: str
    def __init__(self, message: str, latex_output: str = ...) -> None: ...
    def __str__(self) -> str: ...

class LatexManager:
    @classmethod
    def _get_cached_or_new(cls) -> LatexManager: ...
    def get_width_height_descent(
        self, text: str, prop: FontProperties
    ) -> tuple[float, float, float]: ...

class RendererPgf(RendererBase[GraphicsContextBase]):
    dpi: float
    figure: Figure
    image_counter: int
    def __init__(self, figure: Figure, fh: IO[Any]) -> None: ...

class FigureCanvasPgf(FigureCanvasBase[NavigationToolbar2, FigureManagerBase]):
    filetypes: dict[str, str]
    @classmethod
    def get_default_filetype(cls) -> str: ...
    def get_renderer(self) -> RendererPgf: ...
    def print_pgf(
        self, fname_or_fh: str | os.PathLike | IO[Any], **kwargs: Any
    ) -> None: ...
    def print_pdf(
        self,
        fname_or_fh: str | os.PathLike | IO[Any],
        *,
        metadata: dict[str, Any] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def print_png(
        self, fname_or_fh: str | os.PathLike | IO[Any], **kwargs: Any
    ) -> None: ...

FigureManagerPgf = FigureManagerBase

class _BackendPgf(_Backend[type[FigureCanvasPgf]]): ...

class PdfPages:
    def __init__(
        self,
        filename: str | os.PathLike,
        *,
        metadata: dict[str, Any] | None = ...,
    ) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None: ...
    def close(self) -> None: ...
    def savefig(self, figure: Figure | int | None = ..., **kwargs: Any) -> None: ...
    def get_pagecount(self) -> int: ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasPgf
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
