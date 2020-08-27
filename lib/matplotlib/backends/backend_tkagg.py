from .backend_agg import FigureCanvasAgg
from ._backend_tk import (
    _BackendTk, FigureCanvasTk, FigureManagerTk, NavigationToolbar2Tk)


class FigureCanvasTkAgg(FigureCanvasAgg, FigureCanvasTk):
    def draw(self):
        super().draw()
        self.blit()


@_BackendTk.export
class _BackendTkAgg(_BackendTk):
    FigureCanvas = FigureCanvasTkAgg
