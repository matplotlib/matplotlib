from typing import Any

from numpy.typing import ArrayLike

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
    # The template omits the base's optional ``transform`` parameter; see
    # RISKY_RUNTIME_CHANGES_FOR_TYPING_VIBES.md.
    def draw_image(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self, gc: GraphicsContextBase, x: float, y: float, im: ArrayLike
    ) -> None: ...

class GraphicsContextTemplate(GraphicsContextBase): ...
class FigureManagerTemplate(
    FigureManagerBase[FigureCanvasTemplate, NavigationToolbar2[Any]]
): ...

class FigureCanvasTemplate(
    FigureCanvasBase[NavigationToolbar2[Any], FigureManagerTemplate]
):
    filetypes: dict[str, str]
    @classmethod
    def get_default_filetype(cls) -> str: ...
    def print_foo(self, filename: str, **kwargs: Any) -> None: ...

FigureCanvas = FigureCanvasTemplate
FigureManager = FigureManagerTemplate
