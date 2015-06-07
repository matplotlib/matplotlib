"""
GTK+ Matplotlib interface using cairo (not GDK) drawing operations.
Author: Steve Chaplin
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six

import gtk
if gtk.pygtk_version < (2,7,0):
    import cairo.gtk

from matplotlib.backends import backend_cairo
from matplotlib.backends.backend_gtk import *

backend_version = 'PyGTK(%d.%d.%d) ' % gtk.pygtk_version + \
                  'Pycairo(%s)' % backend_cairo.backend_version


_debug = False
#_debug = True


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if _debug: print('backend_gtkcairo.%s()' % fn_name())
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasGTKCairo(figure)
    return FigureManagerGTK(canvas, num)


class RendererGTKCairo (backend_cairo.RendererCairo):
    if gtk.pygtk_version >= (2,7,0):
        def set_pixmap (self, pixmap):
            self.gc.ctx = pixmap.cairo_create()
    else:
        def set_pixmap (self, pixmap):
            self.gc.ctx = cairo.gtk.gdk_cairo_create (pixmap)


class FigureCanvasGTKCairo(backend_cairo.FigureCanvasCairo, FigureCanvasGTK):
    filetypes = FigureCanvasGTK.filetypes.copy()
    filetypes.update(backend_cairo.FigureCanvasCairo.filetypes)

    def _renderer_init(self):
        """Override to use cairo (rather than GDK) renderer"""
        if _debug: print('%s.%s()' % (self.__class__.__name__, _fn_name()))
        self._renderer = RendererGTKCairo (self.figure.dpi)


class FigureManagerGTKCairo(FigureManagerGTK):
    def _get_toolbar(self, canvas):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar']=='toolbar2':
            toolbar = NavigationToolbar2GTKCairo (canvas, self.window)
        else:
            toolbar = None
        return toolbar


class NavigationToolbar2Cairo(NavigationToolbar2GTK):
    def _get_canvas(self, fig):
        return FigureCanvasGTKCairo(fig)


FigureCanvas = FigureCanvasGTKCairo
FigureManager = FigureManagerGTKCairo
