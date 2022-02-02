from .backend_qtcairo import (
    _BackendQTCairo, FigureCanvasQTCairo, FigureCanvasCairo, FigureCanvasQT)


@_BackendQTCairo.export
class _BackendQT5Cairo(_BackendQTCairo):
    pass
