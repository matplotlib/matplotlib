"""
Render to qt from agg
"""
from __future__ import division

import os, sys

import matplotlib
from matplotlib.figure import Figure

from backend_agg import FigureCanvasAgg
from backend_qt4 import QtCore, QtGui, FigureManagerQT, FigureCanvasQT,\
     show, draw_if_interactive, backend_version, \
     NavigationToolbar2QT

DEBUG = False


def new_figure_manager( num, *args, **kwargs ):
    """
    Create a new figure manager instance
    """
    if DEBUG: print 'backend_qtagg.new_figure_manager'
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass( *args, **kwargs )
    canvas = FigureCanvasQTAgg( thisFig )
    return FigureManagerQT( canvas, num )

class NavigationToolbar2QTAgg(NavigationToolbar2QT):
    def _get_canvas(self, fig):
        return FigureCanvasQTAgg(fig)

class FigureManagerQTAgg(FigureManagerQT):
    def _get_toolbar(self, canvas, parent):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar']=='classic':
            print "Classic toolbar is not supported"
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            toolbar = NavigationToolbar2QTAgg(canvas, parent)
        else:
            toolbar = None
        return toolbar

class FigureCanvasQTAgg( FigureCanvasQT, FigureCanvasAgg ):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance
   """

    def __init__( self, figure ):
        if DEBUG: print 'FigureCanvasQtAgg: ', figure
        FigureCanvasQT.__init__( self, figure )
        FigureCanvasAgg.__init__( self, figure )
        self.drawRect = False
        self.rect = []
        self.blitbox = None
        self.replot = True
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def drawRectangle( self, rect ):
        self.rect = rect
        self.drawRect = True
        self.repaint( )

    def paintEvent( self, e ):
        """
        Draw to the Agg backend and then copy the image to the qt.drawable.
        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """

        #FigureCanvasQT.paintEvent( self, e )
        if DEBUG: print 'FigureCanvasQtAgg.paintEvent: ', self, \
           self.get_width_height()

        if self.replot:
            FigureCanvasAgg.draw(self)
            self.replot = False

        if self.blitbox is None:
            # matplotlib is in rgba byte order.  QImage wants to put the bytes
            # into argb format and is in a 4 byte unsigned int.  Little endian
            # system is LSB first and expects the bytes in reverse order
            # (bgra).
            if QtCore.QSysInfo.ByteOrder == QtCore.QSysInfo.LittleEndian:
                stringBuffer = self.renderer._renderer.tostring_bgra()
            else:
                stringBuffer = self.renderer._renderer.tostring_argb()

            qImage = QtGui.QImage(stringBuffer, self.renderer.width,
                                  self.renderer.height,
                                  QtGui.QImage.Format_ARGB32)
            p = QtGui.QPainter(self)
            p.drawPixmap(QtCore.QPoint(0, 0), QtGui.QPixmap.fromImage(qImage))

            # draw the zoom rectangle to the QPainter
            if self.drawRect:
                p.setPen( QtGui.QPen( QtCore.Qt.black, 1, QtCore.Qt.DotLine ) )
                p.drawRect( self.rect[0], self.rect[1], self.rect[2], self.rect[3] )
            p.end()
        else:
            bbox = self.blitbox
            l, b, r, t = bbox.extents
            w = int(r) - int(l)
            h = int(t) - int(b)
            t = int(b) + h
            reg = self.copy_from_bbox(bbox)
            stringBuffer = reg.to_string_argb()
            qImage = QtGui.QImage(stringBuffer, w, h, QtGui.QImage.Format_ARGB32)
            pixmap = QtGui.QPixmap.fromImage(qImage)
            p = QtGui.QPainter( self )
            p.drawPixmap(QtCore.QPoint(l, self.renderer.height-t), pixmap)
            p.end()
            self.blitbox = None
        self.drawRect = False

    def draw( self ):
        """
        Draw the figure when xwindows is ready for the update
        """

        if DEBUG: print "FigureCanvasQtAgg.draw", self
        self.replot = True
        self.update()

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
