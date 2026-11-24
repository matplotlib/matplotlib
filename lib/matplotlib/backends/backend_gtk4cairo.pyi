from gi.repository import Gtk

from .backend_cairo import FigureCanvasCairo
from .backend_gtk4 import FigureCanvasGTK4

class FigureCanvasGTK4Cairo(FigureCanvasCairo, FigureCanvasGTK4):
    def on_snapshot_event(self, snapshot: Gtk.Snapshot) -> None: ...
