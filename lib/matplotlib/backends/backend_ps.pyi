import os
from io import StringIO
from typing import IO, Any

from matplotlib.backend_bases import (
    _Backend,
    FigureCanvasBase,
    FigureManagerBase,
    NavigationToolbar2,
    GraphicsContextBase,
    ShowBase,
)
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from numpy.typing import ArrayLike

from . import _backend_pdf_ps

# Mapping of paper-size name to (width, height) in inches.
papersize: dict[str, tuple[float, float]]

def gs_distill(
    tmpfile: str,
    eps: bool = ...,
    ptype: str = ...,
    bbox: tuple[float, float, float, float] | None = ...,
    rotated: bool = ...,
) -> None: ...
def xpdf_distill(
    tmpfile: str,
    eps: bool = ...,
    ptype: str = ...,
    bbox: tuple[float, float, float, float] | None = ...,
    rotated: bool = ...,
) -> None: ...
def pstoeps(
    tmpfile: str,
    bbox: tuple[float, float, float, float] | None = ...,
    rotated: bool = ...,
) -> None: ...

class RendererPS(_backend_pdf_ps.RendererPDFPSBase[GraphicsContextBase]):
    def __init__(
        self, width: float, height: float, pswriter: StringIO, imagedpi: float = ...
    ) -> None: ...
    def set_color(self, r: float, g: float, b: float, store: bool = ...) -> None: ...
    def set_linewidth(self, linewidth: float, store: bool = ...) -> None: ...
    def set_linejoin(self, linejoin: int, store: bool = ...) -> None: ...
    def set_linecap(self, linecap: int, store: bool = ...) -> None: ...
    def set_linedash(
        self, offset: float, seq: ArrayLike | None, store: bool = ...
    ) -> None: ...
    def set_font(self, fontname: str, fontsize: float, store: bool = ...) -> None: ...
    def create_hatch(self, hatch: str, linewidth: float) -> None: ...
    def draw_mathtext(
        self,
        gc: GraphicsContextBase,
        x: float,
        y: float,
        s: str,
        prop: FontProperties,
        angle: float,
    ) -> None: ...

class FigureCanvasPS(FigureCanvasBase[NavigationToolbar2, FigureManagerBase]):
    fixed_dpi: int
    filetypes: dict[str, str]
    @classmethod
    def get_default_filetype(cls) -> str: ...
    def print_ps(
        self,
        outfile: str | os.PathLike | IO[Any],
        *,
        metadata: dict[str, Any] | None = ...,
        papertype: str | None = ...,
        orientation: str = ...,
        bbox_inches_restore: Any | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def print_eps(
        self,
        outfile: str | os.PathLike | IO[Any],
        *,
        metadata: dict[str, Any] | None = ...,
        papertype: str | None = ...,
        orientation: str = ...,
        bbox_inches_restore: Any | None = ...,
        **kwargs: Any,
    ) -> None: ...

FigureManagerPS = FigureManagerBase

class _BackendPS(_Backend[type[FigureCanvasPS]]): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasPS
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
