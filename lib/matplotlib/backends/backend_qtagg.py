"""
Render to qt from agg
"""
from __future__ import division

import os, sys
from matplotlib import verbose
from matplotlib.cbook import enumerate
from matplotlib.figure import Figure

from backend_agg import FigureCanvasAgg
from backend_qt import qt, FigureManagerQT, FigureCanvasQT,\
     show, draw_if_interactive, backend_version

DEBUG = False


def new_figure_manager( num, *args, **kwargs ):
    """
    Create a new figure manager instance
    """
    if DEBUG: print 'backend_qtagg.new_figure_manager'
    thisFig = Figure( *args, **kwargs )
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
        self.pixmap = qt.QPixmap()
     
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
        # False in repaint does not clear the image before repainting
        self.repaint( False )

    def paintEvent( self, e ):
        """
        Draw to the Agg backend and then copy the image to the qt.drawable.
        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        
        FigureCanvasQT.paintEvent( self, e )
        if DEBUG: print 'FigureCanvasQtAgg.paintEvent: ', \
           self.get_width_height()

        p = qt.QPainter( self )
        FigureCanvasAgg.draw( self )

        # only replot data when needed
        if ( self.replot ):
            stringBuffer = str( self.buffer_rgba(0,0) )
            

            # matplotlib is in rgba byte order.
            # qImage wants to put the bytes into argb format and
            # is in a 4 byte unsigned int.  little endian system is LSB first
            # and expects the bytes in reverse order (bgra).
            if ( qt.QImage.systemByteOrder() == qt.QImage.LittleEndian ):
                stringBuffer = self.renderer._renderer.tostring_bgra()
            else:
                stringBuffer = self.renderer._renderer.tostring_argb()
               
            qImage = qt.QImage( stringBuffer, self.renderer.width,
                                self.renderer.height, 32, None, 0,
                                qt.QImage.IgnoreEndian )
                
            self.pixmap.convertFromImage( qImage, qt.QPixmap.Color )
    
        p.drawPixmap( qt.QPoint( 0, 0 ), self.pixmap )

        # draw the zoom rectangle to the QPainter
        if ( self.drawRect ):
            p.setPen( qt.QPen( qt.Qt.black, 1, qt.Qt.DotLine ) )
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
        self.repaint( False )

    def print_figure( self, filename, dpi=150, facecolor='w', edgecolor='w',
                      orientation='portrait' ):
        if DEBUG: print 'FigureCanvasQTAgg.print_figure'
        agg = self.switch_backends( FigureCanvasAgg )
        agg.print_figure( filename, dpi, facecolor, edgecolor, orientation )

        
