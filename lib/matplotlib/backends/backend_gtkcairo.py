"""
GTK+ Matplotlib interface using Cairo (not GDK) drawing operations.
Author: Steve Chaplin
"""
from backend_gtk import *
from backend_cairo import RendererCairo

import cairo
import cairo.gtk

backend_version = 'PyGTK(%d.%d.%d),Pycairo(%d.%d.%d)' % (gtk.pygtk_version + cairo.version_info)


_debug = False
#_debug = True


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if _debug: print 'backend_gtkcairo.%s()' % fn_name()
    thisFig = Figure(*args, **kwargs)
    canvas = FigureCanvasGTKCairo(thisFig)
    return FigureManagerGTK(canvas, num)




class FigureCanvasGTKCairo(FigureCanvasGTK):
    def _renderer_init(self):
        """Override to use Cairo rather than GDK renderer"""
        if _debug: print '%s.%s()' % (self.__class__.__name__, _fn_name())
        self._renderer = RendererCairo (self.figure.dpi)

class NavigationToolbar2Cairo(NavigationToolbar2GTK):
    def _get_canvas(self, fig):
        return FigureCanvasGTKCairo(fig)
    

class FigureManagerGTKAgg(FigureManagerGTK):
    def _get_toolbar(self, canvas):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar']=='classic':
            toolbar = NavigationToolbar (canvas, self.window)
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            toolbar = NavigationToolbar2GTKCairo (canvas, self.window)
        else:
            toolbar = None
        return toolbar

