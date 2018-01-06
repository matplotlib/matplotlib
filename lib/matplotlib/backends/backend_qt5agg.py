"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import ctypes
import traceback

from matplotlib import cbook
from matplotlib.transforms import Bbox

from .backend_agg import FigureCanvasAgg
from .backend_qt5 import (
    QtCore, QtGui, QtWidgets, _BackendQT5, FigureCanvasQT, FigureManagerQT,
    NavigationToolbar2QT, backend_version)
from .qt_compat import QT_API


class FigureCanvasQTAggBase(FigureCanvasAgg):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Attributes
    ----------
    figure : `matplotlib.figure.Figure`
        A high-level Figure instance

    """

    def __init__(self, figure):
        super(FigureCanvasQTAggBase, self).__init__(figure=figure)
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        self._agg_draw_pending = False
        self._agg_is_drawing = False
        self._bbox_queue = []
        self._drawRect = None

    def drawRectangle(self, rect):
        if rect is not None:
            self._drawRect = [pt / self._dpi_ratio for pt in rect]
        else:
            self._drawRect = None
        self.update()

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
            if hasattr(qimage, 'setDevicePixelRatio'):
                # Not available on Qt4 or some older Qt5.
                qimage.setDevicePixelRatio(self._dpi_ratio)
            origin = QtCore.QPoint(l, self.renderer.height - t)
            painter.drawImage(origin / self._dpi_ratio, qimage)
            # Adjust the buf reference count to work around a memory
            # leak bug in QImage under PySide on Python 3.
            if QT_API == 'PySide' and six.PY3:
                ctypes.c_long.from_address(id(buf)).value = 1

        # draw the zoom rectangle to the QPainter
        if self._drawRect is not None:
            pen = QtGui.QPen(QtCore.Qt.black, 1 / self._dpi_ratio,
                             QtCore.Qt.DotLine)
            painter.setPen(pen)
            x, y, w, h = self._drawRect
            painter.drawRect(x, y, w, h)

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
            super(FigureCanvasQTAggBase, self).draw()
        finally:
            self._agg_is_drawing = False
        self.update()

    def draw_idle(self):
        """Queue redraw of the Agg buffer and request Qt paintEvent.
        """
        # The Agg draw needs to be handled by the same thread matplotlib
        # modifies the scene graph from. Post Agg draw request to the
        # current event loop in order to ensure thread affinity and to
        # accumulate multiple draw requests from event handling.
        # TODO: queued signal connection might be safer than singleShot
        if not (self._agg_draw_pending or self._agg_is_drawing):
            self._agg_draw_pending = True
            QtCore.QTimer.singleShot(0, self.__draw_idle_agg)

    def __draw_idle_agg(self, *args):
        # if nothing to do, bail
        if not self._agg_draw_pending:
            return
        # we have now tried this function at least once, do not run
        # again until re-armed.  Doing this here rather than after
        # protects against recursive calls triggered through self.draw
        # The recursive call is via `repaintEvent`
        self._agg_draw_pending = False
        # if negative size, bail
        if self.height() < 0 or self.width() < 0:
            return
        try:
            # actually do the drawing
            self.draw()
        except Exception:
            # Uncaught exceptions are fatal for PyQt5, so catch them instead.
            traceback.print_exc()

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
        super(FigureCanvasQTAggBase, self).print_figure(*args, **kwargs)
        self.draw()


class FigureCanvasQTAgg(FigureCanvasQTAggBase, FigureCanvasQT):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc.

    Modified to import from Qt5 backend for new-style mouse events.

    Attributes
    ----------
    figure : `matplotlib.figure.Figure`
        A high-level Figure instance

    """


@_BackendQT5.export
class _BackendQT5Agg(_BackendQT5):
    FigureCanvas = FigureCanvasQTAgg
