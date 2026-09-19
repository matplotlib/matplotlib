from math import ceil, floor

from . import backend_agg, backend_gtk4
from .backend_gtk4 import GLib, Gdk, Graphene, _BackendGTK4, _USE_SCALED_TEXTURE

if _USE_SCALED_TEXTURE:
    from .backend_gtk4 import Gsk


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

        width = self.get_width()
        height = self.get_height()
        # Yes, Graphene.Rect really does have this strange initialization API.
        area = Graphene.Rect()
        Graphene.Rect.init(
            area,
            # Snap the texture to a physical pixel so it is not blurred.
            1 - ceil(self.device_pixel_ratio) / self.device_pixel_ratio,
            1 - ceil(self.device_pixel_ratio) / self.device_pixel_ratio,
            ceil(width * self.device_pixel_ratio),
            ceil(height * self.device_pixel_ratio))

        snapshot.save()
        snapshot.scale(1 / self.device_pixel_ratio, 1 / self.device_pixel_ratio)

        if (_USE_SCALED_TEXTURE and
                self._texture.get_height() == floor(area.size.height)):
            snapshot.append_scaled_texture(self._texture, Gsk.ScalingFilter.NEAREST,
                                           area)
        else:
            snapshot.append_texture(self._texture, area)

        snapshot.restore()


@_BackendGTK4.export
class _BackendGTK4Agg(_BackendGTK4):
    FigureCanvas = FigureCanvasGTK4Agg
