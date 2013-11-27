from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import cairo
import numpy as np
import warnings

from . import backend_agg
from . import backend_gtk3_gui
from matplotlib import transforms

if six.PY3:
    warnings.warn("The Gtk3Agg backend is not known to work on Python 3.x.")


class FigureCanvasGTK3AggGui(backend_gtk3_gui.FigureCanvasGTK3Gui,
                          backend_agg.FigureCanvasAgg):
    def __init__(self, figure):
        backend_gtk3_gui.FigureCanvasGTK3Gui.__init__(self, figure)
        self._bbox_queue = []

    def _renderer_init(self):
        pass

    def _render_figure(self, width, height):
        backend_agg.FigureCanvasAgg.draw(self)

    def on_draw_event(self, widget, ctx):
        """ GtkDrawable draw event, like expose_event in GTK 2.X
        """
        allocation = self.get_allocation()
        w, h = allocation.width, allocation.height

        if not len(self._bbox_queue):
            if self._need_redraw:
                self._render_figure(w, h)
                bbox_queue = [transforms.Bbox([[0, 0], [w, h]])]
            else:
                return
        else:
            bbox_queue = self._bbox_queue

        for bbox in bbox_queue:
            area = self.copy_from_bbox(bbox)
            buf = np.fromstring(area.to_string_argb(), dtype='uint8')

            x = int(bbox.x0)
            y = h - int(bbox.y1)
            width = int(bbox.x1) - int(bbox.x0)
            height = int(bbox.y1) - int(bbox.y0)

            image = cairo.ImageSurface.create_for_data(
                buf, cairo.FORMAT_ARGB32, width, height)
            ctx.set_source_surface(image, x, y)
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
        w, h = allocation.width, allocation.height
        x = int(bbox.x0)
        y = h - int(bbox.y1)
        width = int(bbox.x1) - int(bbox.x0)
        height = int(bbox.y1) - int(bbox.y0)

        self._bbox_queue.append(bbox)
        self.queue_draw_area(x, y, width, height)

    def print_png(self, filename, *args, **kwargs):
        # Do this so we can save the resolution of figure in the PNG file
        agg = self.switch_backends(backend_agg.FigureCanvasAgg)
        return agg.print_png(filename, *args, **kwargs)


class FigureManagerGTK3AggGui(backend_gtk3_gui.FigureManagerGTK3Gui):
    pass


FigureCanvas = FigureCanvasGTK3AggGui
FigureManager = FigureManagerGTK3AggGui
