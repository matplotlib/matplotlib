from .backend_qt5cairo import _BackendQT5Cairo, FigureCanvasQTCairo


@_BackendQT5Cairo.export
class _BackendQT4Cairo(_BackendQT5Cairo):
    class FigureCanvas(FigureCanvasQTCairo):
        required_interactive_framework = "qt4"
