"""
GTK+ Matplotlib interface using Cairo (not GDK) drawing operations.
Author: Steve Chaplin
"""
from backend_gtk import *
from backend_cairo import RendererCairo

import cairo
import cairo.gtk

backend_version = 'PyGTK(%d.%d.%d),PyCairo(%d.%d.%d)' % (gtk.pygtk_version + cairo.version_info)
#backend_version = 'PyGTK(%d.%d.%d)' % gtk.pygtk_version

DEBUG = False


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if DEBUG: print 'backend_gtkcairo.%s()' % fn_name()
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasGTKCairo(thisFig)
    return FigureManagerGTK(canvas, num)


class FigureCanvasGTKCairo(FigureCanvasGTK):
    def _renderer_init(self):
        """Override to use Cairo rather than GDK renderer"""
        if DEBUG: print 'backend_gtkcairo.%s()' % fn_name()
        matrix = cairo.Matrix ()
        self._renderer = RendererCairo (matrix, self.figure.dpi)
