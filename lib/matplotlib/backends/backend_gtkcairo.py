"""
GTK+ Matplotlib interface using Cairo (not GDK) drawing operations.
Author: Steve Chaplin
"""
from __future__ import division

import os
import sys
def function_name(): return sys._getframe(1).f_code.co_name

from matplotlib import verbose
from matplotlib.cbook import enumerate, True, False
from matplotlib.figure import Figure
from backend_cairo import FigureCanvasCairo, RendererCairo
from backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK,\
     show, draw_if_interactive,\
     error_msg, NavigationToolbar, PIXELS_PER_INCH, backend_version

try:
    import cairo
    import cairo.gtk
    # version > x, check - later
except:
    verbose.report_error('PyCairo is required to run the Matplotlib Cairo backend')
    raise SystemExit()
backend_version = '0.1.23' # cairo does not report version, yet


Debug = False


# ref gtk+/gtk/gtkwidget.h
def GTK_WIDGET_DRAWABLE(w): flags = w.flags(); return flags & gtk.VISIBLE !=0 and flags & gtk.MAPPED != 0


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if Debug: print 'backend_gtkcairo.%s()' % function_name()
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasGTKCairo(thisFig)
    return FigureManagerGTK(canvas, num)


class FigureCanvasGTKCairo(FigureCanvasGTK, FigureCanvasCairo):
    """Use all of the FigureCanvasGTK functionality.
    Override expose_event() and print_figure() to use Cairo rather than GDK.

    """
    def expose_event(self, widget, event):
        if Debug: print 'backend_gtkcairo.%s()' % function_name()

        if GTK_WIDGET_DRAWABLE(self) and self._new_pixmap:
            width, height = self.allocation.width, self.allocation.height

            self._pixmap = gtk.gdk.Pixmap (self.window, width, height)
            # cant (yet) use lines below - when width,height shrinks renderer still draws to the old width, height
            
            #if width > self._pixmap_width or height > self._pixmap_height:
            #    if Debug: print 'backend_gtkcairo.%s: new pixmap allocated' % function_name()
            #    self._pixmap = gtk.gdk.Pixmap (self.window, width, height)
            #    self._pixmap_width, self._pixmap_height = width, height

            # create in __init__() once and save?
            # - but when pixmap changes (resized larger) render must update width, height and redirect all gc to new pixmap!
            renderer = RendererCairoGTK(self._pixmap, width, height, self.figure.dpi)
            self.figure.draw(renderer) # matplotlib draw command
            
            # Test drawing primitives here:
            #renderer.draw_rectangle (gc, (1.0,1.0,1.0), 0, 0, width, height) # white screen
            #renderer.draw_lines (gc, [0,width,width/2,0], [0,height,0,height])

            #renderer.draw_line (gc, 0,0, width,height)


            self.window.set_back_pixmap (self._pixmap, False)
            self.window.clear()  # draws the pixmap onto the window bg
            self._new_pixmap = False

        return True


    def print_figure(self, filename, dpi=150, facecolor='w', edgecolor='w',
                     orientation='portrait'):
        if Debug: print 'backend_gtkcairo.%s()' % function_name()
        print "print_figure() Not implemented yet"
        
        # delete the renderer to prevent improper blitting after print
        #cairo = self.switch_backends(FigureCanvasCairo)
        #cairo.print_figure(filename, dpi, facecolor, edgecolor, orientation)


class RendererCairoGTK(RendererCairo):
    """Override RendererCairo to direct drawing to the gtk.Pixmap
    """
    def __init__(self, pixmap, width, height, dpi):
        RendererCairo.__init__(self, width, height, dpi)
        self._pixmap = pixmap

    def new_gc(self):
        gc = RendererCairo.new_gc(self)
        # tell cairo to draw to off-screen pixmap
        cairo.gtk.set_target_drawable (gc.ctx, self._pixmap)
        return gc
