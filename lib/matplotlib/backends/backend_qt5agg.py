"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import ctypes

from matplotlib import cbook
from matplotlib.transforms import Bbox

from .backend_agg import FigureCanvasAgg
from .backend_qt5 import (
    QtCore, QtGui, QtWidgets, _BackendQT5, FigureCanvasQT, FigureManagerQT,
    NavigationToolbar2QT, backend_version)
from .qt_compat import QT_API


class FigureCanvasQTAgg(FigureCanvasAgg, FigureCanvasQT):

    def __init__(self, figure):
        super(FigureCanvasQTAgg, self).__init__(figure=figure)
        self._bbox_queue = []

    @property
    @cbook.deprecated("2.1")
    def blitbox(self):
        return self._bbox_queue

    def paintEvent(self, e):
        """Copy the image from the Agg canvas to the qt.drawable.

        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        # if there is a pending draw, run it now as we need the updated render
        # to paint the widget
        if self._agg_draw_pending:
            self.__draw_idle_agg()
        # As described in __init__ above, we need to be careful in cases with
        # mixed resolution displays if dpi_ratio is changing between painting
        # events.
        if self._dpi_ratio != self._dpi_ratio_prev:
            # We need to update the figure DPI
            self._update_figure_dpi()
            self._dpi_ratio_prev = self._dpi_ratio
            # The easiest way to resize the canvas is to emit a resizeEvent
            # since we implement all the logic for resizing the canvas for
            # that event.
            event = QtGui.QResizeEvent(self.size(), self.size())
            # We use self.resizeEvent here instead of QApplication.postEvent
            # since the latter doesn't guarantee that the event will be emitted
            # straight away, and this causes visual delays in the changes.
            self.resizeEvent(event)
            # resizeEvent triggers a paintEvent itself, so we exit this one.
            return

        # if the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called
        if not hasattr(self, 'renderer'):
            return

        painter = QtGui.QPainter(self)

        if self._bbox_queue:
            bbox_queue = self._bbox_queue
        else:
            painter.eraseRect(self.rect())
            bbox_queue = [
                Bbox([[0, 0], [self.renderer.width, self.renderer.height]])]
        self._bbox_queue = []
        for bbox in bbox_queue:
            l, b, r, t = map(int, bbox.extents)
            w = r - l
            h = t - b
            reg = self.copy_from_bbox(bbox)
            buf = reg.to_string_argb()
            qimage = QtGui.QImage(buf, w, h, QtGui.QImage.Format_ARGB32)
            # Adjust the buf reference count to work around a memory leak bug
            # in QImage under PySide on Python 3.
            if QT_API == 'PySide' and six.PY3:
                ctypes.c_long.from_address(id(buf)).value = 1
            if hasattr(qimage, 'setDevicePixelRatio'):
                # Not available on Qt4 or some older Qt5.
                qimage.setDevicePixelRatio(self._dpi_ratio)
            origin = QtCore.QPoint(l, self.renderer.height - t)
            painter.drawImage(origin / self._dpi_ratio, qimage)

        self._draw_rect_callback(painter)

        painter.end()

    def draw(self):
        """Draw the figure with Agg, and queue a request for a Qt draw.
        """
        # The Agg draw is done here; delaying causes problems with code that
        # uses the result of the draw() to update plot elements.
        if self._agg_is_drawing:
            return

        self._agg_is_drawing = True
        try:
            super(FigureCanvasQTAgg, self).draw()
        finally:
            self._agg_is_drawing = False
        self.update()

    def blit(self, bbox=None):
        """Blit the region in bbox.
        """
        # If bbox is None, blit the entire canvas. Otherwise
        # blit only the area defined by the bbox.
        if bbox is None and self.figure:
            bbox = self.figure.bbox

        self._bbox_queue.append(bbox)

        # repaint uses logical pixels, not physical pixels like the renderer.
        l, b, w, h = [pt / self._dpi_ratio for pt in bbox.bounds]
        t = b + h
        self.repaint(l, self.renderer.height / self._dpi_ratio - t, w, h)

    def print_figure(self, *args, **kwargs):
        super(FigureCanvasQTAgg, self).print_figure(*args, **kwargs)
        self.draw()


@_BackendQT5.export
class _BackendQT5Agg(_BackendQT5):
    FigureCanvas = FigureCanvasQTAgg
