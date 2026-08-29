import numpy as np

from .. import cbook, transforms
from . import backend_agg, backend_gtk3
from .backend_gtk3 import GLib, Gtk, _BackendGTK3

import cairo  # Presence of cairo is already checked by _backend_gtk.


class FigureCanvasGTK3Agg(backend_agg.FigureCanvasAgg,
                          backend_gtk3.FigureCanvasGTK3):
    def __init__(self, figure):
        super().__init__(figure=figure)
        self._bbox_queue = []

    def on_draw_event(self, widget, ctx):
        if self._idle_draw_id:
            GLib.source_remove(self._idle_draw_id)
            self._idle_draw_id = 0
            self.draw()

        scale = self.device_pixel_ratio
        allocation = self.get_allocation()
        w = allocation.width * scale
        h = allocation.height * scale

        if not len(self._bbox_queue):
            Gtk.render_background(
                self.get_style_context(), ctx,
                allocation.x, allocation.y,
                allocation.width, allocation.height)
            bbox_queue = [transforms.Bbox([[0, 0], [w, h]])]
        else:
            bbox_queue = self._bbox_queue

        for bbox in bbox_queue:
            region = self.copy_from_bbox(bbox)
            # The region is rounded out to whole pixels; take its extents
            # (in buffer coordinates) rather than rounding the bbox again.
            x, y, x2, y2 = region.get_extents()
            width = x2 - x
            height = y2 - y

            buf = cbook._unmultiplied_rgba8888_to_premultiplied_argb32(
                np.asarray(region))
            image = cairo.ImageSurface.create_for_data(
                buf.ravel().data, cairo.FORMAT_ARGB32, width, height)
            image.set_device_scale(scale, scale)
            ctx.set_source_surface(image, x / scale, y / scale)
            ctx.paint()

        if len(self._bbox_queue):
            self._bbox_queue = []

        return False

    def blit(self, bbox=None):
        # If bbox is None, blit the entire canvas to gtk. Otherwise
        # blit only the area defined by the bbox.
        if bbox is None:
            bbox = self.figure.bbox

        allocation = self.get_allocation()
        x0, y0, x1, y1 = bbox._pixel_bounds(scale=self.device_pixel_ratio)

        self._bbox_queue.append(bbox)
        self.queue_draw_area(x0, allocation.height - y1, x1 - x0, y1 - y0)


@_BackendGTK3.export
class _BackendGTK3Agg(_BackendGTK3):
    FigureCanvas = FigureCanvasGTK3Agg
