from typing import Any

from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure

class MixedModeRenderer:
    """Proxy renderer that switches between vector and raster drawing."""

    dpi: float
    figure: Figure

    def __init__(
        self,
        figure: Figure,
        width: float,
        height: float,
        dpi: float,
        vector_renderer: RendererBase,
        raster_renderer_class: type[RendererBase] | None = ...,
        bbox_inches_restore: Any | None = ...,
    ) -> None: ...
    def __getattr__(self, attr: str) -> Any: ...
    def start_rasterizing(self) -> None: ...
    def stop_rasterizing(self) -> None: ...
