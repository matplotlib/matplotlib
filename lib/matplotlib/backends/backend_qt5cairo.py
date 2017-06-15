from . import backend_cairo  # Keep the RendererCairo class swappable.
from .backend_qt5 import QtCore, QtGui, _BackendQT5, FigureCanvasQT


class FigureCanvasQTCairo(FigureCanvasQT):
    def __init__(self, figure):
        super(FigureCanvasQTCairo, self).__init__(figure=figure)
        self._renderer = backend_cairo.RendererCairo(self.figure.dpi)

    def paintEvent(self, event):
        width = self.width()
        height = self.height()
        surface = backend_cairo.cairo.ImageSurface(
            backend_cairo.cairo.FORMAT_ARGB32, width, height)
        self._renderer.set_ctx_from_surface(surface)
        # This should be really done by set_ctx_from_surface...
        self._renderer.set_width_height(width, height)
        self.figure.draw(self._renderer)
        qimage = QtGui.QImage(surface.get_data(), width, height,
                              QtGui.QImage.Format_ARGB32_Premultiplied)
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, qimage)
        self._draw_rect_callback(painter)
        painter.end()


@_BackendQT5.export
class _BackendQT5Cairo(_BackendQT5):
    FigureCanvas = FigureCanvasQTCairo
