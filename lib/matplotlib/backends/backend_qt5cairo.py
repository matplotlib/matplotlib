from .backend_qtcairo import _BackendQTCairo
from .backend_qtcairo import (  # noqa: F401 # pylint: disable=W0611
    FigureCanvasQTCairo, FigureCanvasCairo, FigureCanvasQT)


@_BackendQTCairo.export
class _BackendQT5Cairo(_BackendQTCairo):
    pass
