"""
Render to gtk from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK,\
     show, draw_if_interactive,\
     error_msg_gtk, PIXELS_PER_INCH, backend_version, \
     NavigationToolbar2GTK
from matplotlib.backends._gtkagg import agg_to_gtk_drawable


DEBUG = False

class NavigationToolbar2GTKAgg(NavigationToolbar2GTK):
    def _get_canvas(self, fig):
        return FigureCanvasGTKAgg(fig)


class FigureManagerGTKAgg(FigureManagerGTK):
    def _get_toolbar(self, canvas):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar']=='toolbar2':
            toolbar = NavigationToolbar2GTKAgg (canvas, self.window)
        else:
            toolbar = None
        return toolbar


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if DEBUG: print('backend_gtkagg.new_figure_manager')
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasGTKAgg(figure)
    return FigureManagerGTKAgg(canvas, num)
    if DEBUG: print('backend_gtkagg.new_figure_manager done')


class FigureCanvasGTKAgg(FigureCanvasGTK, FigureCanvasAgg):
    filetypes = FigureCanvasGTK.filetypes.copy()
    filetypes.update(FigureCanvasAgg.filetypes)

    def configure_event(self, widget, event=None):

        if DEBUG: print('FigureCanvasGTKAgg.configure_event')
        if widget.window is None:
            return
        try:
            del self.renderer
        except AttributeError:
            pass
        w,h = widget.window.get_size()
        if w==1 or h==1: return # empty fig

        # compute desired figure size in inches
        dpival = self.figure.dpi
        winch = w/dpival
        hinch = h/dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        self._need_redraw = True
        self.resize_event()
        if DEBUG: print('FigureCanvasGTKAgg.configure_event end')
        return True

    def _render_figure(self, pixmap, width, height):
        if DEBUG: print('FigureCanvasGTKAgg.render_figure')
        FigureCanvasAgg.draw(self)
        if DEBUG: print('FigureCanvasGTKAgg.render_figure pixmap', pixmap)
        #agg_to_gtk_drawable(pixmap, self.renderer._renderer, None)

        buf = self.buffer_rgba()
        ren = self.get_renderer()
        w = int(ren.width)
        h = int(ren.height)

        pixbuf = gtk.gdk.pixbuf_new_from_data(
            buf, gtk.gdk.COLORSPACE_RGB,  True, 8, w, h, w*4)
        pixmap.draw_pixbuf(pixmap.new_gc(), pixbuf, 0, 0, 0, 0, w, h,
                           gtk.gdk.RGB_DITHER_NONE, 0, 0)
        if DEBUG: print('FigureCanvasGTKAgg.render_figure done')

    def blit(self, bbox=None):
        if DEBUG: print('FigureCanvasGTKAgg.blit', self._pixmap)
        agg_to_gtk_drawable(self._pixmap, self.renderer._renderer, bbox)

        x, y, w, h = self.allocation

        self.window.draw_drawable (self.style.fg_gc[self.state], self._pixmap,
                                   0, 0, 0, 0, w, h)
        if DEBUG: print('FigureCanvasGTKAgg.done')

    def print_png(self, filename, *args, **kwargs):
        # Do this so we can save the resolution of figure in the PNG file
        agg = self.switch_backends(FigureCanvasAgg)
        return agg.print_png(filename, *args, **kwargs)


"""\
Traceback (most recent call last):
  File "/home/titan/johnh/local/lib/python2.3/site-packages/matplotlib/backends/backend_gtk.py", line 304, in expose_event
    self._render_figure(self._pixmap, w, h)
  File "/home/titan/johnh/local/lib/python2.3/site-packages/matplotlib/backends/backend_gtkagg.py", line 77, in _render_figure
    pixbuf = gtk.gdk.pixbuf_new_from_data(
ValueError: data length (3156672) is less then required by the other parameters (3160608)
"""

FigureCanvas = FigureCanvasGTKAgg
FigureManager = FigureManagerGTKAgg
