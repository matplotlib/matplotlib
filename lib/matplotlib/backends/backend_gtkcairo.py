"""
GTK+ Matplotlib interface using Cairo (not GDK) drawing operations.
Author: Steve Chaplin
"""
from __future__ import division

import os
import sys
def _fn_name(): return sys._getframe(1).f_code.co_name

from matplotlib import verbose
from matplotlib.cbook import True, False
from matplotlib.figure import Figure
from backend_cairo import RendererCairo
from backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK, show,    \
     draw_if_interactive, error_msg, NavigationToolbar, IMAGE_FORMAT, \
     IMAGE_FORMAT_DEFAULT

import cairo
import cairo.gtk

backend_version = 'PyGTK(%s),PyCairo(%s)' % ("%d.%d.%d" % gtk.pygtk_version, 'unknown')
#backend_version = 'PyGTK(%d.%d.%d),PyCairo(%s)' % (gtk.pygtk_version, cairo.version)

DEBUG = False


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if DEBUG: print 'backend_gtkcairo.%s()' % _fn_name()
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasGTKCairo(thisFig)
    return FigureManagerGTK(canvas, num)


class FigureCanvasGTKCairo(FigureCanvasGTK):
    """Override GTK._render_to_pixmap() to use Cairo rather than GDK renderer
    """
    def _render_to_pixmap(self, width, height):
        create_pixmap = False
        if width > self._pixmap_width:
            # increase the pixmap in 10%+ (rather than 1 pixel) steps
            self._pixmap_width  = max (int (self._pixmap_width  * 1.1), width)
            create_pixmap = True

        if height > self._pixmap_height:
            self._pixmap_height = max (int (self._pixmap_height * 1.1), height)
            create_pixmap = True

        if create_pixmap:
            if DEBUG: print 'backend_gtk.%s: new pixmap' % _fn_name()
            self._pixmap = gtk.gdk.Pixmap (self.window, self._pixmap_width,
                                           self._pixmap_height)
            self._surface = cairo.gtk.surface_create_for_drawable (self._pixmap)
                    
        matrix = cairo.Matrix ()
        self._renderer = RendererCairo (self._surface, matrix, width, height, self.figure.dpi)
        self._renderer._set_width_height (width, height)
        self.figure.draw (self._renderer)
