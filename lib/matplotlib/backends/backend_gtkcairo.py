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
from backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK, show, \
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
    def _renderer_init(self):
        """Override to use Cairo rather than GDK renderer"""
        if DEBUG: print 'backend_gtkcairo.%s()' % _fn_name()
        matrix = cairo.Matrix ()
        self._renderer = RendererCairo (matrix, self.figure.dpi)
