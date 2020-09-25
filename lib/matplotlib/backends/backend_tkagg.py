from .backend_agg import FigureCanvasAgg
from ._backend_tk import (
    _BackendTk, FigureCanvasTk, FigureManagerTk, NavigationToolbar2Tk, blit)


class FigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def draw(self):
        super().draw()
        self.blit()

    def blit(self, bbox=None):
        blit(self._tkcanvas, self._tkphoto,
             self.renderer._renderer, (0, 1, 2, 3), bbox=bbox)


@_BackendTk.export
class _BackendTkAgg(_BackendTk):
    FigureCanvas = FigureCanvasTkAgg
