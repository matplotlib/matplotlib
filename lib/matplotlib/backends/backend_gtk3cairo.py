from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from . import backend_cairo, backend_gtk3
from .backend_cairo import cairo, HAS_CAIRO_CFFI
from .backend_gtk3 import _BackendGTK3
from matplotlib.backend_bases import cursors
from matplotlib.figure import Figure


class RendererGTK3Cairo(backend_cairo.RendererCairo):
    def set_context(self, ctx):
        if HAS_CAIRO_CFFI:
            ctx = cairo.Context._from_pointer(
                cairo.ffi.cast(
                    'cairo_t **',
                    id(ctx) + object.__basicsize__)[0],
                incref=True)

        self.gc.ctx = ctx


class FigureCanvasGTK3Cairo(backend_gtk3.FigureCanvasGTK3,
                            backend_cairo.FigureCanvasCairo):

    def _renderer_init(self):
        """use cairo renderer"""
        self._renderer = RendererGTK3Cairo(self.figure.dpi)

    def _render_figure(self, width, height):
        self._renderer.set_width_height(width, height)
        self.figure.draw(self._renderer)

    def on_draw_event(self, widget, ctx):
        """ GtkDrawable draw event, like expose_event in GTK 2.X
        """
        toolbar = self.toolbar
        # if toolbar:
        #     toolbar.set_cursor(cursors.WAIT)
        self._renderer.set_context(ctx)
        allocation = self.get_allocation()
        x, y, w, h = allocation.x, allocation.y, allocation.width, allocation.height
        self._render_figure(w, h)
        # if toolbar:
        #     toolbar.set_cursor(toolbar._lastCursor)
        return False  # finish event propagation?


class FigureManagerGTK3Cairo(backend_gtk3.FigureManagerGTK3):
    pass


@_BackendGTK3.export
class _BackendGTK3Cairo(_BackendGTK3):
    FigureCanvas = FigureCanvasGTK3Cairo
    FigureManager = FigureManagerGTK3Cairo
