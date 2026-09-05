"""
Render to qt from agg.
"""


from matplotlib.transforms import Bbox
from matplotlib.backend_bases import DrawEvent

from .qt_compat import QT_API, QtCore, QtGui
from .backend_agg import FigureCanvasAgg, RendererAgg
from .backend_qt import _BackendQT, FigureCanvasQT
from .backend_qt import (  # noqa: F401 # pylint: disable=W0611
    FigureManagerQT, NavigationToolbar2QT)


class FigureCanvasQTAgg(FigureCanvasAgg, FigureCanvasQT):

    def __init__(self, figure=None):
        super().__init__(figure=figure)
        self._layer_renderers = {}
        self._renderer_key = None

    def draw(self):
        """
        Render the figure using the per-layer caching optimization.
        """
        fig = self.figure
        w, h = self.get_width_height(physical=True)
        dpi = fig.dpi

        # Run layout engine once before drawing
        if fig.axes and fig.get_layout_engine() is not None:
            try:
                fig.get_layout_engine().execute(fig)
            except ValueError:
                pass

        key = (w, h, dpi)
        is_resized = getattr(self, '_renderer_key', None) != key
        if is_resized:
            self._renderer_key = key

        for layer_name in fig._children_by_layer:
            stale = fig._stale_layers.get(layer_name, True)

            # Re-render if: layer is stale OR canvas was resized
            if stale or is_resized:
                layer_renderer = RendererAgg(w, h, dpi)
                fig._draw_layer(layer_renderer, layer_name)

                self._layer_renderers[layer_name] = layer_renderer

        fig.stale = False

        # Fire the draw event
        base_renderer = self._layer_renderers.get("patch")
        DrawEvent("draw_event", self, base_renderer)._process()

        self.update()

    def paintEvent(self, event):
        """
        Copy the image from the Agg canvas to the qt.drawable.

        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        self._draw_idle()  # Only does something if a draw is pending.

        # If the layers haven't been rendered yet, give up and wait for draw()
        if not self._layer_renderers:
            return

        painter = QtGui.QPainter(self)
        try:
            # See documentation of QRect: bottom() and right() are off
            # by 1, so use left() + width() and top() + height().
            rect = event.rect()
            # scale rect dimensions using the screen dpi ratio to get
            # correct values for the Figure coordinates (rather than
            # QT5's coords)
            width = rect.width() * self.device_pixel_ratio
            height = rect.height() * self.device_pixel_ratio
            left, top = self.mouseEventCoords(rect.topLeft())
            # shift the "top" by the height of the image to get the
            # correct corner for our coordinate system
            bottom = top - height
            # same with the right side of the image
            right = left + width
            # create a buffer using the image bounding box
            bbox = Bbox([[left, bottom], [right, top]])

            painter.eraseRect(rect)  # clear the widget canvas
            origin = QtCore.QPoint(rect.left(), rect.top())

            for layer_name in self.figure._children_by_layer:
                if layer_name in self._layer_renderers:
                    layer_renderer = self._layer_renderers[layer_name]

                    buf = memoryview(layer_renderer.copy_from_bbox(bbox))

                    if QT_API == "PyQt6":
                        from PyQt6 import sip
                        ptr = int(sip.voidptr(buf))
                    else:
                        ptr = buf

                    qimage = QtGui.QImage(ptr, buf.shape[1], buf.shape[0],
                                          QtGui.QImage.Format.Format_RGBA8888)
                    qimage.setDevicePixelRatio(self.device_pixel_ratio)

                    # Qt's QPainter natively handles alpha blending!
                    painter.drawImage(origin, qimage)

            self._draw_rect_callback(painter)
        finally:
            painter.end()

    def print_figure(self, *args, **kwargs):
        super().print_figure(*args, **kwargs)
        # In some cases, Qt will itself trigger a paint event after closing the file
        # save dialog. When that happens, we need to be sure that the internal canvas is
        # re-drawn. However, if the user is using an automatically-chosen Qt backend but
        # saving with a different backend (such as pgf), we do not want to trigger a
        # full draw in Qt, so just set the flag for next time.
        self._draw_pending = True


@_BackendQT.export
class _BackendQTAgg(_BackendQT):
    FigureCanvas = FigureCanvasQTAgg
