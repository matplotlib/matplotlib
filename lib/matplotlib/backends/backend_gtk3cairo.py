from . import backend_cairo, backend_gtk3
from .backend_gtk3 import Gtk, _BackendGTK3
from matplotlib import cbook
from matplotlib.backend_bases import cursors


class RendererGTK3Cairo(backend_cairo.RendererCairo):
    def set_context(self, ctx):
        self.gc.ctx = backend_cairo._to_context(ctx)


class FigureCanvasGTK3Cairo(backend_gtk3.FigureCanvasGTK3,
                            backend_cairo.FigureCanvasCairo):

    def _renderer_init(self):
        """Use cairo renderer."""
        self._renderer = RendererGTK3Cairo(self.figure.dpi)

    def _render_figure(self, width, height):
        self._renderer.set_width_height(width, height)
        self.figure.draw(self._renderer)

    def on_draw_event(self, widget, ctx):
        """GtkDrawable draw event."""
        # toolbar = self.toolbar
        # if toolbar:
        #     toolbar.set_cursor(cursors.WAIT)
        self._renderer.set_context(ctx)
        allocation = self.get_allocation()
        Gtk.render_background(
            self.get_style_context(), ctx,
            allocation.x, allocation.y, allocation.width, allocation.height)
        self._render_figure(allocation.width, allocation.height)
        # if toolbar:
        #     toolbar.set_cursor(toolbar._lastCursor)
        return False  # finish event propagation?


@cbook.deprecated("3.1", alternative="backend_gtk3.FigureManagerGTK3")
class FigureManagerGTK3Cairo(backend_gtk3.FigureManagerGTK3):
    pass


@_BackendGTK3.export
class _BackendGTK3Cairo(_BackendGTK3):
    FigureCanvas = FigureCanvasGTK3Cairo
