"""
Render to gtk from agg
"""
from __future__ import division

import os
from matplotlib.figure import Figure

from backend_agg import FigureCanvasAgg
from backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK,\
     show, draw_if_interactive,\
     error_msg_gtk, NavigationToolbar, PIXELS_PER_INCH, backend_version

from _gtkagg import agg_to_gtk_drawable


DEBUG = False


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if DEBUG: print 'backend_gtkagg.new_figure_manager'
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasGTKAgg(thisFig)
    return FigureManagerGTK(canvas, num)


class FigureCanvasGTKAgg(FigureCanvasGTK, FigureCanvasAgg):

    def configure_event(self, widget, event=None):
        if DEBUG: print 'FigureCanvasGTKAgg.configure_event'
        if widget.window is None:
            return 
        try:
            del self.renderer
        except AttributeError:
            pass
        w,h = widget.window.get_size()
        if w==1 or h==1: return # empty fig

        # compute desired figure size in inches
        dpival = self.figure.dpi.get()
        winch = w/dpival
        hinch = h/dpival
        self.figure.set_figsize_inches(winch, hinch)
        self._draw_pixmap = True
        
        if DEBUG: print 'FigureCanvasGTKAgg.configure_event end'        
        return True
    

    def _render_figure(self, width, height):
        """Render the figure to a gdk.Pixmap, used by expose_event().
        """
        if DEBUG: print 'FigureCanvasGTKAgg._render_figure'
        create_pixmap = False
        if width > self._pixmap_width:
            # increase the pixmap in 10%+ (rather than 1 pixel) steps
            self._pixmap_width  = max (int (self._pixmap_width  * 1.1), width)
            create_pixmap = True

        if height > self._pixmap_height:
            self._pixmap_height = max (int (self._pixmap_height * 1.1), height)
            create_pixmap = True

        if create_pixmap:
            if DEBUG: print 'FigureCanvasGTK._render_figure new pixmap'
            self._pixmap = gtk.gdk.Pixmap (self.window, self._pixmap_width,
                                           self._pixmap_height)

        FigureCanvasAgg.draw(self)
        agg_to_gtk_drawable(self._pixmap, self.renderer._renderer)


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
            try: agg.print_figure(filename, dpi, facecolor, edgecolor, orientation)
            except IOError, msg:
                error_msg_gtk('Failed to save\nError message: %s'%(msg,), self)

        self.figure.set_canvas(self)
