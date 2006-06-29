"""
Render to qt from agg
"""
from __future__ import division

import os, sys
from matplotlib import verbose
from matplotlib.cbook import enumerate
from matplotlib.figure import Figure

from backend_agg import FigureCanvasAgg
from backend_qt4 import QtCore, QtGui, FigureManagerQT, FigureCanvasQT,\
     show, draw_if_interactive, backend_version

DEBUG = False


def new_figure_manager( num, *args, **kwargs ):
    """
    Create a new figure manager instance
    """
    if DEBUG: print 'backend_qtagg.new_figure_manager'
    print args, kwargs
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass( *args, **kwargs )
    canvas = FigureCanvasQTAgg( thisFig )
    return FigureManagerQT( canvas, num )

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
        self.pixmap = QtGui.QPixmap()
     
    def resizeEvent( self, e ):
        FigureCanvasQT.resizeEvent( self, e )
        w = e.size().width()
        h = e.size().height()
        if DEBUG: print "FigureCanvasQtAgg.resizeEvent(", w, ",", h, ")"
        dpival = self.figure.dpi.get()
        winch = w/dpival
        hinch = h/dpival
        self.figure.set_figsize_inches( winch, hinch )
        self.draw()
        
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
        
        FigureCanvasQT.paintEvent( self, e )
        if DEBUG: print 'FigureCanvasQtAgg.paintEvent: ', \
           self.get_width_height()

        p = QtGui.QPainter( self )
        FigureCanvasAgg.draw( self )

        # only replot data when needed
        if ( self.replot ):
            stringBuffer = str( self.buffer_rgba(0,0) )
            

            # matplotlib is in rgba byte order.
            # qImage wants to put the bytes into argb format and
            # is in a 4 byte unsigned int.  little endian system is LSB first
            # and expects the bytes in reverse order (bgra).
            if ( QtCore.QSysInfo.ByteOrder == QtCore.QSysInfo.LittleEndian ):
                stringBuffer = self.renderer._renderer.tostring_bgra()
            else:
                stringBuffer = self.renderer._renderer.tostring_argb()
            qImage = QtGui.QImage( stringBuffer, self.renderer.width,
                                   self.renderer.height,
                                   QtGui.QImage.Format_ARGB32)
            self.pixmap = self.pixmap.fromImage( qImage )
        p.drawPixmap( QtCore.QPoint( 0, 0 ), self.pixmap )

        # draw the zoom rectangle to the QPainter
        if ( self.drawRect ):
            p.setPen( QtGui.QPen( QtCore.Qt.black, 1, QtCore.Qt.DotLine ) )
            p.drawRect( self.rect[0], self.rect[1], self.rect[2], self.rect[3] )
           
        p.end()
        self.replot = False
        self.drawRect = False

    def draw( self ):
        """
        Draw the figure when xwindows is ready for the update
        """
        
        if DEBUG: print "FigureCanvasQtAgg.draw"
        self.replot = True
        self.repaint( )

    def print_figure( self, filename, dpi=150, facecolor='w', edgecolor='w',
                      orientation='portrait', **kwargs ):
        if DEBUG: print 'FigureCanvasQTAgg.print_figure'
        agg = self.switch_backends( FigureCanvasAgg )
        agg.print_figure( filename, dpi, facecolor, edgecolor, orientation,
                          **kwargs )

        
