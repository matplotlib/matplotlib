"""
Render to gtk from agg
"""
from __future__ import division

import os, sys
from matplotlib import verbose
from matplotlib.cbook import enumerate, True, False
from matplotlib.figure import Figure

from backend_agg import FigureCanvasAgg
from backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK,\
     show, draw_if_interactive,\
     error_msg, NavigationToolbar, PIXELS_PER_INCH, backend_version

from _gtkagg import agg_to_gtk_drawable


DEBUG = 0


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if DEBUG: print 'backend_gtkagg.new_figure_manager'
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasGTKAgg(thisFig)
    return FigureManagerGTK(canvas, num)

class FigureCanvasGTKAgg(FigureCanvasGTK, FigureCanvasAgg):

    def draw(self):
        """
        Draw to the Agg backend and then copy the image to the
        gtk.gdk.drawable.

        """
        if DEBUG: print 'FigureCanvasGTKAgg.draw'  
            
        FigureCanvasAgg.draw(self)
        if self.window is None: 
            return
        else: 
            self.blit()
        

    def blit(self):
        if self.window is None: return        
        agg_to_gtk_drawable(self.window, self.renderer._renderer)

    def print_figure(self, filename, dpi=150,
                     facecolor='w', edgecolor='w',
                     orientation='portrait'):
        if DEBUG: print 'FigureCanvasGTKAgg.print_figure'
        # delete the renderer to prevent improper blitting after print

        root, ext = os.path.splitext(filename)       
        ext = ext.lower()[1:]
        if ext == 'jpg':
            FigureCanvasGTK.print_figure(self, filename, dpi, facecolor,
                                         edgecolor, orientation)
            
        else:
            agg = self.switch_backends(FigureCanvasAgg)
            agg.print_figure(filename, dpi, facecolor, edgecolor, orientation)



    def configure_event(self, widget, event=None):
        if DEBUG: print 'FigureCanvasGTKAgg.configure_event'
        if widget.window is None: return 
        try: del self.renderer
        except AttributeError: pass
        w,h = widget.window.get_size()
        if w==1 or h==1: return # empty fig

        # compute desired figure size in inches
        dpival = self.figure.dpi.get()
        winch = w/dpival
        hinch = h/dpival
        
        self.figure.set_figsize_inches(winch, hinch)
        
        return gtk.TRUE
    
    def expose_event(self, widget, event):
        if DEBUG: print 'FigureCanvasGTKAgg.expose_event'        
        if widget.window is None: return 

        def callback(w):
            if hasattr(self, 'renderer'):  self.blit()
            else: self.draw()
            self._idleID=0
            return gtk.FALSE

        if self._idleID==0:
            self._idleID = gtk.idle_add(callback, self)


        return gtk.TRUE
