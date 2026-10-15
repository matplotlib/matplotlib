import numpy as np

from .. import cbook
from . import _backend_tk
from .backend_cairo import cairo, FigureCanvasCairo
from ._backend_tk import _BackendTk, FigureCanvasTk


class FigureCanvasTkCairo(FigureCanvasCairo, FigureCanvasTk):
    def draw(self):
        width, height = self.get_width_height(physical=True)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self._renderer.set_context(cairo.Context(surface))
        self._renderer.dpi = self.figure.dpi
        self.figure.draw(self._renderer)
        premult_argb = np.reshape(surface.get_data(), (height, width, 4))
        unmult_rgba = cbook._premultiplied_argb32_to_unmultiplied_rgba8888(premult_argb)
        _backend_tk.blit(self._tkphoto, unmult_rgba, (0, 1, 2, 3))


@_BackendTk.export
class _BackendTkCairo(_BackendTk):
    FigureCanvas = FigureCanvasTkCairo
