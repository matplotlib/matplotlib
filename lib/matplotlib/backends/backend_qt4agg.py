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
            print "Classic toolbar is not yet supported"
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
        self.pixmap = QtGui.QPixmap()
     
    def resizeEvent( self, e ):
        FigureCanvasQT.resizeEvent( self, e )
        w = e.size().width()
        h = e.size().height()
        if DEBUG: print "FigureCanvasQtAgg.resizeEvent(", w, ",", h, ")"
        dpival = self.figure.dpi.get()
        winch = w/dpival
        hinch = h/dpival
        self.figure.set_size_inches( winch, hinch )
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
        
        #FigureCanvasQT.paintEvent( self, e )
        if DEBUG: print 'FigureCanvasQtAgg.paintEvent: ', self, \
           self.get_width_height()

        p = QtGui.QPainter( self )

        # only replot data when needed
        if type(self.replot) is bool: # might be a bbox for blitting
            if ( self.replot ):
                #stringBuffer = str( self.buffer_rgba(0,0) )
                FigureCanvasAgg.draw( self )
    
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
                
        # we are blitting here
        else:
            bbox = self.replot
            w, h = int(bbox.width()), int(bbox.height())
            l, t = bbox.ll().x().get(), bbox.ur().y().get()
            reg = self.copy_from_bbox(bbox)
            stringBuffer = reg.to_string()
            qImage = QtGui.QImage(stringBuffer, w, h, QtGui.QImage.Format_ARGB32)
            self.pixmap = self.pixmap.fromImage( qImage )
            p.drawPixmap(QtCore.QPoint(l, self.renderer.height-t), self.pixmap)
           
        p.end()
        self.replot = False
        self.drawRect = False

    def draw( self ):
        """
        Draw the figure when xwindows is ready for the update
        """
        
        if DEBUG: print "FigureCanvasQtAgg.draw", self
        self.replot = True
        self.update( )
        
    def blit(self, bbox=None):
        """
        Blit the region in bbox
        """
        
        self.replot = bbox
        w, h = int(bbox.width()), int(bbox.height())
        l, t = bbox.ll().x().get(), bbox.ur().y().get()
        self.update(l, self.renderer.height-t, w, h)

    def print_figure( self, filename, dpi=None, facecolor='w', edgecolor='w',
                      orientation='portrait', **kwargs ):
        if DEBUG: print 'FigureCanvasQTAgg.print_figure'
        if dpi is None: dpi = matplotlib.rcParams['savefig.dpi']
        agg = self.switch_backends( FigureCanvasAgg )
        agg.print_figure( filename, dpi, facecolor, edgecolor, orientation,
                          **kwargs )
        self.figure.set_canvas(self)
