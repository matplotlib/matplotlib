"""
Render to qt from agg.
"""


from matplotlib.transforms import Bbox

from .qt_compat import QT_API, QtCore, QtGui
from .backend_agg import FigureCanvasAgg
from .backend_qt import _BackendQT, FigureCanvasQT
from .backend_qt import (  # noqa: F401 # pylint: disable=W0611
    FigureManagerQT, NavigationToolbar2QT)


class FigureCanvasQTAgg(FigureCanvasAgg, FigureCanvasQT):
    supports_overlay = True

    def paintEvent(self, event):
        """
        Copy the image from the Agg canvas to the qt.drawable.

        In Qt, all drawing should be done inside of here when a widget is
        shown onscreen.
        """
        self._draw_idle()  # Only does something if a draw is pending.

        # If the canvas does not have a renderer, then give up and wait for
        # FigureCanvasAgg.draw(self) to be called.
        if not hasattr(self, 'renderer'):
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
            buf = memoryview(self.copy_from_bbox(bbox))

            if QT_API == "PyQt6":
                from PyQt6 import sip
                ptr = int(sip.voidptr(buf))
            else:
                ptr = buf

            painter.eraseRect(rect)  # clear the widget canvas
            qimage = QtGui.QImage(ptr, buf.shape[1], buf.shape[0],
                                  QtGui.QImage.Format.Format_RGBA8888)
            qimage.setDevicePixelRatio(self.device_pixel_ratio)
            # set origin using original QT coordinates
            origin = QtCore.QPoint(rect.left(), rect.top())
            painter.drawImage(origin, qimage)
            if hasattr(self, '_overlay_qimage'):
                painter.drawImage(origin, self._overlay_qimage)
            self._draw_rect_callback(painter)
        finally:
            painter.end()

    def print_figure(self, *args, **kwargs):
        super().print_figure(*args, **kwargs)
        # In some cases, Qt will itself trigger a paint event after closing the file
        # save dialog. When that happens, we need to be sure that the internal canvas is
        # re-drawn. However, if the user is using an automatically-chosen Qt backend but
        # saving with a different backend (such as pgf), we do not want to trigger a
        self._draw_pending = True

    def draw(self):
        """
        Perform a full redraw of the main figure, and also update the overlay.
        """
        super().draw()
        self.draw_overlay()

    def draw_overlay(self):
        """
        Draw only the overlay artists into a separate QImage buffer,
        then trigger a Qt repaint to composite it on screen.
        """
        # Find all artists with in_overlay=True that are not animated
        overlay_artists = self.figure.findobj(
            lambda x: (getattr(x, 'get_in_overlay', lambda: False)()
                       and not x.get_animated())
        )

        # Sort artists by zorder to ensure proper rendering stacking
        overlay_artists.sort(key=lambda a: a.get_zorder())

        if not overlay_artists:
            if hasattr(self, '_overlay_qimage'):
                del self._overlay_qimage
                if hasattr(self, '_overlay_buf'):
                    del self._overlay_buf
            self.update()
            return

        from matplotlib.backends.backend_agg import RendererAgg
        w, h = self.figure.bbox.size
        renderer = RendererAgg(w, h, self.figure.dpi)

        for artist in overlay_artists:
            if hasattr(artist, 'get_visible') and not artist.get_visible():
                continue
            artist.draw(renderer)

        buf = memoryview(renderer.buffer_rgba())
        if QT_API == "PyQt6":
            from PyQt6 import sip
            ptr = int(sip.voidptr(buf))
        else:
            ptr = buf

        # Clear old image to prevent memory leak
        if hasattr(self, '_overlay_qimage'):
            del self._overlay_qimage

        # Keep a reference to the buffer so it outlives the method
        self._overlay_buf = buf

        self._overlay_qimage = QtGui.QImage(
            ptr, buf.shape[1], buf.shape[0],
            QtGui.QImage.Format.Format_RGBA8888
        )
        self._overlay_qimage.setDevicePixelRatio(self.device_pixel_ratio)
        self.update()


@_BackendQT.export
class _BackendQTAgg(_BackendQT):
    FigureCanvas = FigureCanvasQTAgg
