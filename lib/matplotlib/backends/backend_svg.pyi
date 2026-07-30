import os
from typing import IO, Any

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

svgProlog: str

class _XMLWriter:
    def __init__(self, file: IO[Any]) -> None: ...
    def start(
        self, tag: str, attrib: dict[str, Any] | None = ..., **extra: Any
    ) -> int: ...
    def comment(self, comment: str) -> None: ...
    def data(self, text: str) -> None: ...
    def end(self, tag: str | None = ..., indent: bool = ...) -> None: ...
    def close(self, id: int) -> None: ...
    def element(
        self,
        tag: str,
        text: str | None = ...,
        attrib: dict[str, Any] | None = ...,
        **extra: Any,
    ) -> None: ...
    def flush(self) -> None: ...

class XMLWriter(_XMLWriter): ...

class RendererSVG(RendererBase[GraphicsContextBase]):
    width: float
    height: float
    image_dpi: float
    basename: str
    def __init__(
        self,
        width: float,
        height: float,
        svgwriter: IO[Any],
        basename: str | None = ...,
        image_dpi: float = ...,
        *,
        metadata: dict[str, Any] | None = ...,
    ) -> None: ...
    def finalize(self) -> None: ...

class FigureCanvasSVG(FigureCanvasBase[NavigationToolbar2, FigureManagerBase]):
    filetypes: dict[str, str]
    fixed_dpi: int
    @classmethod
    def get_default_filetype(cls) -> str: ...
    def print_svg(
        self,
        filename: str | os.PathLike | IO[Any],
        *,
        bbox_inches_restore: Any | None = ...,
        metadata: dict[str, Any] | None = ...,
    ) -> None: ...
    def print_svgz(
        self, filename: str | os.PathLike | IO[Any], **kwargs: Any
    ) -> None: ...

FigureManagerSVG = FigureManagerBase

class _BackendSVG(_Backend[type[FigureCanvasSVG]]): ...

# Names injected into the module namespace by ``@_Backend.export``.
backend_version: str
FigureCanvas = FigureCanvasSVG
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
