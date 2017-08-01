from . import backend_cairo  # Keep the RendererCairo class swappable.
from .backend_qt5 import QtCore, QtGui, _BackendQT5, FigureCanvasQT
from .qt_compat import QT_API


class FigureCanvasQTCairo(FigureCanvasQT):
    def __init__(self, figure):
        super(FigureCanvasQTCairo, self).__init__(figure=figure)
        self._renderer = backend_cairo.RendererCairo(self.figure.dpi)

    def paintEvent(self, event):
        self._update_dpi()
        width = self.width()
        height = self.height()
        surface = backend_cairo.cairo.ImageSurface(
            backend_cairo.cairo.FORMAT_ARGB32, width, height)
        self._renderer.set_ctx_from_surface(surface)
        self.figure.draw(self._renderer)
        buf = surface.get_data()
        qimage = QtGui.QImage(buf, width, height,
                              QtGui.QImage.Format_ARGB32_Premultiplied)
        # Adjust the buf reference count to work around a memory leak bug in
        # QImage under PySide on Python 3.
        if QT_API == 'PySide' and six.PY3:
            ctypes.c_long.from_address(id(buf)).value = 1
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, qimage)
        self._draw_rect_callback(painter)
        painter.end()


@_BackendQT5.export
class _BackendQT5Cairo(_BackendQT5):
    FigureCanvas = FigureCanvasQTCairo
