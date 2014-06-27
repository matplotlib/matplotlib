"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os  # not used
import sys
import ctypes
import warnings

import matplotlib
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


from matplotlib.cbook import mplDeprecation

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

    def drawRectangle(self, rect):
        self._drawRect = rect
        self.repaint()

    def paintEvent(self, e):
        """
        Copy the image from the Agg canvas to the qt.drawable.
        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """

        # FigureCanvasQT.paintEvent(self, e)
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
            pixmap = QtGui.QPixmap.fromImage(qImage)
            p = QtGui.QPainter(self)
            p.drawPixmap(QtCore.QPoint(l, self.renderer.height-t), pixmap)
            p.end()
            self.blitbox = None
        self._drawRect = None

    def draw(self):
        """
        Draw the figure with Agg, and queue a request
        for a Qt draw.
        """
        # The Agg draw is done here; delaying it until the paintEvent
        # causes problems with code that uses the result of the
        # draw() to update plot elements.
        FigureCanvasAgg.draw(self)
        self._priv_update()

    def blit(self, bbox=None):
        """
        Blit the region in bbox
        """
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
        FigureCanvasQT.__init__(self, figure)
        FigureCanvasAgg.__init__(self, figure)
        self._drawRect = None
        self.blitbox = None
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        # it has been reported that Qt is semi-broken in a windows
        # environment.  If `self.draw()` uses `update` to trigger a
        # system-level window repaint (as is explicitly advised in the
        # Qt documentation) the figure responds very slowly to mouse
        # input.  The work around is to directly use `repaint`
        # (against the advice of the Qt documentation).  The
        # difference between `update` and repaint is that `update`
        # schedules a `repaint` for the next time the system is idle,
        # where as `repaint` repaints the window immediately.  The
        # risk is if `self.draw` gets called with in another `repaint`
        # method there will be an infinite recursion.  Thus, we only
        # expose windows users to this risk.
        if sys.platform.startswith('win'):
            self._priv_update = self.repaint
        else:
            self._priv_update = self.update


class NavigationToolbar2QTAgg(NavigationToolbar2QT):
    def __init__(*args, **kwargs):
        warnings.warn('This class has been deprecated in 1.4 ' +
                      'as it has no additional functionality over ' +
                      '`NavigationToolbar2QT`.  Please change your code to '
                      'use `NavigationToolbar2QT` instead',
                    mplDeprecation)
        NavigationToolbar2QT.__init__(*args, **kwargs)


FigureCanvas = FigureCanvasQTAgg
FigureManager = FigureManagerQT
