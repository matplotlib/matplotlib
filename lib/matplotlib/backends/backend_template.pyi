from typing import Any

from matplotlib.backend_bases import (
    FigureCanvasBase,
    FigureManagerBase,
    NavigationToolbar2,
    GraphicsContextBase,
    RendererBase,
)

class RendererTemplate(RendererBase[GraphicsContextBase]):
    dpi: float
    def __init__(self, dpi: float) -> None: ...

class GraphicsContextTemplate(GraphicsContextBase): ...
class FigureManagerTemplate(FigureManagerBase[FigureCanvasTemplate]): ...

class FigureCanvasTemplate(FigureCanvasBase[NavigationToolbar2, FigureManagerBase]):
    filetypes: dict[str, str]
    @classmethod
    def get_default_filetype(cls) -> str: ...
    def print_foo(self, filename: str, **kwargs: Any) -> None: ...

FigureCanvas = FigureCanvasTemplate
FigureManager = FigureManagerTemplate
