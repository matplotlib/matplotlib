from . import backend_agg, backend_gtk4
from .backend_gtk4 import GLib, Gdk, _BackendGTK4


class FigureCanvasGTK4Agg(backend_agg.FigureCanvasAgg,
                          backend_gtk4.FigureCanvasGTK4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._texture = None

    def on_snapshot_event(self, snapshot):
        if self._idle_draw_id:
            GLib.source_remove(self._idle_draw_id)
            self._idle_draw_id = 0
            self.draw()

            view = self.get_renderer().buffer_rgba()
            buf = GLib.Bytes.new(bytes(view))
            self._texture = Gdk.MemoryTexture.new(view.shape[1], view.shape[0],
                                                  Gdk.MemoryFormat.R8G8B8A8,
                                                  buf, view.strides[0])

        okay, rect = self.compute_bounds(self)
        if not okay:  # Bounds were invalid for some reason.
            return
        snapshot.append_texture(self._texture, rect)


@_BackendGTK4.export
class _BackendGTK4Agg(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Agg
