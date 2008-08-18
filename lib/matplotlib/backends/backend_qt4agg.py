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
        self.replot = True
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def resizeEvent( self, e ):
        FigureCanvasQT.resizeEvent( self, e )

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

        # only replot data when needed
        if type(self.replot) is bool: # might be a bbox for blitting
            if self.replot:
                FigureCanvasAgg.draw(self)

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
        # we are blitting here
        else:
            bbox = self.replot
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
        self.replot = False
        self.drawRect = False

    def draw( self ):
        """
        Draw the figure when xwindows is ready for the update
        """

        if DEBUG: print "FigureCanvasQtAgg.draw", self
        self.replot = True
        FigureCanvasAgg.draw(self)
        self.update()
        # Added following line to improve realtime pan/zoom on windows:
        QtGui.qApp.processEvents()

    def blit(self, bbox=None):
        """
        Blit the region in bbox
        """

        self.replot = bbox
        l, b, w, h = bbox.bounds
        t = b + h
        self.update(l, self.renderer.height-t, w, h)

    def print_figure(self, *args, **kwargs):
        FigureCanvasAgg.print_figure(self, *args, **kwargs)
        self.draw()
