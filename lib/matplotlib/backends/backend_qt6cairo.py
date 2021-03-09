from .backend_cairo import cairo, FigureCanvasCairo, RendererCairo
from .backend_qt6 import QtCore, QtGui, _BackendQT6, FigureCanvasQT
from .qt_compat import QT_API, _setDevicePixelRatio


class FigureCanvasQTCairo(FigureCanvasQT, FigureCanvasCairo):
    def __init__(self, figure=None):
        super().__init__(figure=figure)
        self._renderer = RendererCairo(self.figure.dpi)
        self._renderer.set_width_height(-1, -1)  # Invalid values.

    def draw(self):
        if hasattr(self._renderer.gc, "ctx"):
            self.figure.draw(self._renderer)
        super().draw()

    def paintEvent(self, event):
        dpi_ratio = self._dpi_ratio
        width = int(dpi_ratio * self.width())
        height = int(dpi_ratio * self.height())
        if (width, height) != self._renderer.get_canvas_width_height():
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
            self._renderer.set_ctx_from_surface(surface)
            self._renderer.set_width_height(width, height)
            self.figure.draw(self._renderer)
        buf = self._renderer.gc.ctx.get_target().get_data()
        qimage = QtGui.QImage(buf, width, height,
                              QtGui.QImage.Format_ARGB32_Premultiplied)

        _setDevicePixelRatio(qimage, dpi_ratio)
        painter = QtGui.QPainter(self)
        painter.eraseRect(event.rect())
        painter.drawImage(0, 0, qimage)
        self._draw_rect_callback(painter)
        painter.end()


@_BackendQT6.export
class _BackendQT6Cairo(_BackendQT6):
    FigureCanvas = FigureCanvasQTCairo
