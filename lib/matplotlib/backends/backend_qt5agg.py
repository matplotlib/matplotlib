"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import ctypes
import sys
import traceback

from matplotlib.figure import Figure

from .backend_agg import FigureCanvasAgg
from .backend_qt5 import QtCore
from .backend_qt5 import QtGui
from .backend_qt5 import FigureManagerQT
from .backend_qt5 import NavigationToolbar2QT
##### Modified Qt5 backend import
from .backend_qt5 import FigureCanvasQT
##### not used
from .backend_qt5 import show
from .backend_qt5 import draw_if_interactive
from .backend_qt5 import backend_version
######
from .qt_compat import QT_API

DEBUG = False

_decref = ctypes.pythonapi.Py_DecRef
_decref.argtypes = [ctypes.py_object]
_decref.restype = None


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if DEBUG:
        print('backend_qt5agg.new_figure_manager')
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasQTAgg(figure)
    return FigureManagerQT(canvas, num)


class FigureCanvasQTAggBase(object):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

        figure - A Figure instance
    """

    def __init__(self, figure):
        super(FigureCanvasQTAggBase, self).__init__(figure=figure)
        self._agg_draw_pending = False

    def drawRectangle(self, rect):
        self._drawRect = rect
        self.update()

    def paintEvent(self, e):
        """
        Copy the image from the Agg canvas to the qt.drawable.
        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        # if the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called
        if not hasattr(self, 'renderer'):
            return

        if DEBUG:
            print('FigureCanvasQtAgg.paintEvent: ', self,
                  self.get_width_height())

        if self.blitbox is None:
            # matplotlib is in rgba byte order.  QImage wants to put the bytes
            # into argb format and is in a 4 byte unsigned int.  Little endian
            # system is LSB first and expects the bytes in reverse order
            # (bgra).
            if QtCore.QSysInfo.ByteOrder == QtCore.QSysInfo.LittleEndian:
                stringBuffer = self.renderer._renderer.tostring_bgra()
            else:
                stringBuffer = self.renderer._renderer.tostring_argb()

            refcnt = sys.getrefcount(stringBuffer)

            # convert the Agg rendered image -> qImage
            qImage = QtGui.QImage(stringBuffer, self.renderer.width,
                                  self.renderer.height,
                                  QtGui.QImage.Format_ARGB32)
            # get the rectangle for the image
            rect = qImage.rect()
            p = QtGui.QPainter(self)
            # reset the image area of the canvas to be the back-ground color
            p.eraseRect(rect)
            # draw the rendered image on to the canvas
            p.drawPixmap(QtCore.QPoint(0, 0), QtGui.QPixmap.fromImage(qImage))

            # draw the zoom rectangle to the QPainter
            if self._drawRect is not None:
                p.setPen(QtGui.QPen(QtCore.Qt.black, 1, QtCore.Qt.DotLine))
                x, y, w, h = self._drawRect
                p.drawRect(x, y, w, h)
            p.end()

            # This works around a bug in PySide 1.1.2 on Python 3.x,
            # where the reference count of stringBuffer is incremented
            # but never decremented by QImage.
            # TODO: revert PR #1323 once the issue is fixed in PySide.
            del qImage
            if refcnt != sys.getrefcount(stringBuffer):
                _decref(stringBuffer)
        else:
            bbox = self.blitbox
            l, b, r, t = bbox.extents
            w = int(r) - int(l)
            h = int(t) - int(b)
            t = int(b) + h
            reg = self.copy_from_bbox(bbox)
            stringBuffer = reg.to_string_argb()
            qImage = QtGui.QImage(stringBuffer, w, h,
                                  QtGui.QImage.Format_ARGB32)
            # Adjust the stringBuffer reference count to work around a memory
            # leak bug in QImage() under PySide on Python 3.x
            if QT_API == 'PySide' and six.PY3:
                ctypes.c_long.from_address(id(stringBuffer)).value = 1

            pixmap = QtGui.QPixmap.fromImage(qImage)
            p = QtGui.QPainter(self)
            p.drawPixmap(QtCore.QPoint(l, self.renderer.height-t), pixmap)

            # draw the zoom rectangle to the QPainter
            if self._drawRect is not None:
                p.setPen(QtGui.QPen(QtCore.Qt.black, 1, QtCore.Qt.DotLine))
                x, y, w, h = self._drawRect
                p.drawRect(x, y, w, h)
            p.end()
            self.blitbox = None

    def draw(self):
        """
        Draw the figure with Agg, and queue a request for a Qt draw.
        """
        # The Agg draw is done here; delaying causes problems with code that
        # uses the result of the draw() to update plot elements.
        FigureCanvasAgg.draw(self)
        self.update()

    def draw_idle(self):
        """
        Queue redraw of the Agg buffer and request Qt paintEvent.
        """
        # The Agg draw needs to be handled by the same thread matplotlib
        # modifies the scene graph from. Post Agg draw request to the
        # current event loop in order to ensure thread affinity and to
        # accumulate multiple draw requests from event handling.
        # TODO: queued signal connection might be safer than singleShot
        if not self._agg_draw_pending:
            self._agg_draw_pending = True
            QtCore.QTimer.singleShot(0, self.__draw_idle_agg)

    def __draw_idle_agg(self, *args):
        if self.height() < 0 or self.width() < 0:
            self._agg_draw_pending = False
            return
        try:
            FigureCanvasAgg.draw(self)
            self.update()
        except Exception:
            # Uncaught exceptions are fatal for PyQt5, so catch them instead.
            traceback.print_exc()
        finally:
            self._agg_draw_pending = False

    def blit(self, bbox=None):
        """
        Blit the region in bbox
        """
        # If bbox is None, blit the entire canvas. Otherwise
        # blit only the area defined by the bbox.
        if bbox is None and self.figure:
            bbox = self.figure.bbox

        self.blitbox = bbox
        l, b, w, h = bbox.bounds
        t = b + h
        self.repaint(l, self.renderer.height-t, w, h)

    def print_figure(self, *args, **kwargs):
        FigureCanvasAgg.print_figure(self, *args, **kwargs)
        self.draw()


class FigureCanvasQTAgg(FigureCanvasQTAggBase,
                        FigureCanvasQT, FigureCanvasAgg):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc.

    Modified to import from Qt5 backend for new-style mouse events.

    Public attribute

      figure - A Figure instance
    """

    def __init__(self, figure):
        if DEBUG:
            print('FigureCanvasQtAgg: ', figure)
        super(FigureCanvasQTAgg, self).__init__(figure=figure)
        self._drawRect = None
        self.blitbox = None
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)


FigureCanvas = FigureCanvasQTAgg
FigureManager = FigureManagerQT
