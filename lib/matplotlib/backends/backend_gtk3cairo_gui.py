from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from . import backend_gtk3_gui
from . import backend_cairo


class RendererGTK3Cairo(backend_cairo.RendererCairo):
    def set_context(self, ctx):
        self.gc.ctx = ctx


class FigureCanvasGTK3CairoGui(backend_gtk3_gui.FigureCanvasGTK3Gui,
                            backend_cairo.FigureCanvasCairo):
    def __init__(self, figure):
        backend_gtk3_gui.FigureCanvasGTK3Gui.__init__(self, figure)

    def _renderer_init(self):
        """use cairo renderer"""
        self._renderer = RendererGTK3Cairo(self.figure.dpi)

    def _render_figure(self, width, height):
        self._renderer.set_width_height(width, height)
        self.figure.draw(self._renderer)

    def on_draw_event(self, widget, ctx):
        """ GtkDrawable draw event, like expose_event in GTK 2.X
        """
        # the _need_redraw flag doesnt work. it sometimes prevents
        # the rendering and leaving the canvas blank
        #if self._need_redraw:
        self._renderer.set_context(ctx)
        allocation = self.get_allocation()
        w, h = allocation.width, allocation.height
        self._render_figure(w, h)
        #self._need_redraw = False

        return False  # finish event propagation?

FigureManagerGTK3CairoGui = backend_gtk3_gui.FigureManagerGTK3Gui

FigureCanvas = FigureCanvasGTK3CairoGui
FigureManager = FigureManagerGTK3CairoGui
