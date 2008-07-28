from __future__ import division
"""

 backend_wxagg.py

 A wxPython backend for Agg.  This uses the GUI widgets written by
 Jeremy O'Donoghue (jeremy@o-donoghue.com) and the Agg backend by John
 Hunter (jdhunter@ace.bsd.uchicago.edu)

 Copyright (C) 2003-5 Jeremy O'Donoghue, John Hunter, Illinois Institute of
 Technology


 License: This work is licensed under the matplotlib license( PSF
 compatible). A copy should be included with this source code.

"""

import wx
import matplotlib
from matplotlib.figure import Figure

from backend_agg import FigureCanvasAgg
import backend_wx
from backend_wx import FigureManager, FigureManagerWx, FigureCanvasWx, \
    FigureFrameWx, DEBUG_MSG, NavigationToolbar2Wx, error_msg_wx, \
    draw_if_interactive, show, Toolbar, backend_version


class FigureFrameWxAgg(FigureFrameWx):
    def get_canvas(self, fig):
        return FigureCanvasWxAgg(self, -1, fig)

    def _get_toolbar(self, statbar):
        if matplotlib.rcParams['toolbar']=='classic':
            toolbar = NavigationToolbarWx(self.canvas, True)
        elif matplotlib.rcParams['toolbar']=='toolbar2':
            toolbar = NavigationToolbar2WxAgg(self.canvas)
            toolbar.set_status_bar(statbar)
        else:
            toolbar = None
        return toolbar

class FigureCanvasWxAgg(FigureCanvasAgg, FigureCanvasWx):
    """
    The FigureCanvas contains the figure and does event handling.

    In the wxPython backend, it is derived from wxPanel, and (usually)
    lives inside a frame instantiated by a FigureManagerWx. The parent
    window probably implements a wxSizer to control the displayed
    control size - but we give a hint as to our preferred minimum
    size.
    """

    def draw(self, drawDC=None):
        """
        Render the figure using agg.
        """
        DEBUG_MSG("draw()", 1, self)
        FigureCanvasAgg.draw(self)

        self.bitmap = _convert_agg_to_wx_bitmap(self.get_renderer(), None)
        self._isDrawn = True
        self.gui_repaint(drawDC=drawDC)

    def blit(self, bbox=None):
        """
        Transfer the region of the agg buffer defined by bbox to the display.
        If bbox is None, the entire buffer is transferred.
        """
        if bbox is None:
            self.bitmap = _convert_agg_to_wx_bitmap(self.get_renderer(), None)
            self.gui_repaint()
            return

        l, b, w, h = bbox.bounds
        r = l + w
        t = b + h
        x = int(l)
        y = int(self.bitmap.GetHeight() - t)

        srcBmp = _convert_agg_to_wx_bitmap(self.get_renderer(), None)
        srcDC = wx.MemoryDC()
        srcDC.SelectObject(srcBmp)

        destDC = wx.MemoryDC()
        destDC.SelectObject(self.bitmap)

        destDC.BeginDrawing()
        destDC.Blit(x, y, int(w), int(h), srcDC, x, y)
        destDC.EndDrawing()

        destDC.SelectObject(wx.NullBitmap)
        srcDC.SelectObject(wx.NullBitmap)
        self.gui_repaint()

    filetypes = FigureCanvasAgg.filetypes

    def print_figure(self, filename, *args, **kwargs):
        # Use pure Agg renderer to draw
        FigureCanvasAgg.print_figure(self, filename, *args, **kwargs)
        # Restore the current view; this is needed because the
        # artist contains methods rely on particular attributes
        # of the rendered figure for determining things like
        # bounding boxes.
        if self._isDrawn:
            self.draw()

class NavigationToolbar2WxAgg(NavigationToolbar2Wx):
    def get_canvas(self, frame, fig):
        return FigureCanvasWxAgg(frame, -1, fig)


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    # in order to expose the Figure constructor to the pylab
    # interface we need to create the figure here
    DEBUG_MSG("new_figure_manager()", 3, None)
    backend_wx._create_wx_app()

    FigureClass = kwargs.pop('FigureClass', Figure)
    fig = FigureClass(*args, **kwargs)
    frame = FigureFrameWxAgg(num, fig)
    figmgr = frame.get_figure_manager()
    if matplotlib.is_interactive():
        figmgr.frame.Show()
    return figmgr


#
# agg/wxPython image conversion functions (wxPython <= 2.6)
#

def _py_convert_agg_to_wx_image(agg, bbox):
    """
    Convert the region of the agg buffer bounded by bbox to a wx.Image.  If
    bbox is None, the entire buffer is converted.

    Note: agg must be a backend_agg.RendererAgg instance.
    """
    image = wx.EmptyImage(int(agg.width), int(agg.height))
    image.SetData(agg.tostring_rgb())

    if bbox is None:
        # agg => rgb -> image
        return image
    else:
        # agg => rgb -> image => bitmap => clipped bitmap => image
        return wx.ImageFromBitmap(_clipped_image_as_bitmap(image, bbox))


def _py_convert_agg_to_wx_bitmap(agg, bbox):
    """
    Convert the region of the agg buffer bounded by bbox to a wx.Bitmap.  If
    bbox is None, the entire buffer is converted.

    Note: agg must be a backend_agg.RendererAgg instance.
    """
    if bbox is None:
        # agg => rgb -> image => bitmap
        return wx.BitmapFromImage(_py_convert_agg_to_wx_image(agg, None))
    else:
        # agg => rgb -> image => bitmap => clipped bitmap
        return _clipped_image_as_bitmap(
            _py_convert_agg_to_wx_image(agg, None),
            bbox)


def _clipped_image_as_bitmap(image, bbox):
    """
    Convert the region of a wx.Image bounded by bbox to a wx.Bitmap.
    """
    l, b, width, height = bbox.get_bounds()
    r = l + width
    t = b + height

    srcBmp = wx.BitmapFromImage(image)
    srcDC = wx.MemoryDC()
    srcDC.SelectObject(srcBmp)

    destBmp = wx.EmptyBitmap(int(width), int(height))
    destDC = wx.MemoryDC()
    destDC.SelectObject(destBmp)

    destDC.BeginDrawing()
    x = int(l)
    y = int(image.GetHeight() - t)
    destDC.Blit(0, 0, int(width), int(height), srcDC, x, y)
    destDC.EndDrawing()

    srcDC.SelectObject(wx.NullBitmap)
    destDC.SelectObject(wx.NullBitmap)

    return destBmp


#
# agg/wxPython image conversion functions (wxPython >= 2.8)
#

def _py_WX28_convert_agg_to_wx_image(agg, bbox):
    """
    Convert the region of the agg buffer bounded by bbox to a wx.Image.  If
    bbox is None, the entire buffer is converted.

    Note: agg must be a backend_agg.RendererAgg instance.
    """
    if bbox is None:
        # agg => rgb -> image
        image = wx.EmptyImage(int(agg.width), int(agg.height))
        image.SetData(agg.tostring_rgb())
        return image
    else:
        # agg => rgba buffer -> bitmap => clipped bitmap => image
        return wx.ImageFromBitmap(_WX28_clipped_agg_as_bitmap(agg, bbox))


def _py_WX28_convert_agg_to_wx_bitmap(agg, bbox):
    """
    Convert the region of the agg buffer bounded by bbox to a wx.Bitmap.  If
    bbox is None, the entire buffer is converted.

    Note: agg must be a backend_agg.RendererAgg instance.
    """
    if bbox is None:
        # agg => rgba buffer -> bitmap
        return wx.BitmapFromBufferRGBA(int(agg.width), int(agg.height),
            agg.buffer_rgba(0, 0))
    else:
        # agg => rgba buffer -> bitmap => clipped bitmap
        return _WX28_clipped_agg_as_bitmap(agg, bbox)


def _WX28_clipped_agg_as_bitmap(agg, bbox):
    """
    Convert the region of a the agg buffer bounded by bbox to a wx.Bitmap.

    Note: agg must be a backend_agg.RendererAgg instance.
    """
    l, b, width, height = bbox.get_bounds()
    r = l + width
    t = b + height

    srcBmp = wx.BitmapFromBufferRGBA(int(agg.width), int(agg.height),
        agg.buffer_rgba(0, 0))
    srcDC = wx.MemoryDC()
    srcDC.SelectObject(srcBmp)

    destBmp = wx.EmptyBitmap(int(width), int(height))
    destDC = wx.MemoryDC()
    destDC.SelectObject(destBmp)

    destDC.BeginDrawing()
    x = int(l)
    y = int(int(agg.height) - t)
    destDC.Blit(0, 0, int(width), int(height), srcDC, x, y)
    destDC.EndDrawing()

    srcDC.SelectObject(wx.NullBitmap)
    destDC.SelectObject(wx.NullBitmap)

    return destBmp


def _use_accelerator(state):
    """
    Enable or disable the WXAgg accelerator, if it is present and is also
    compatible with whatever version of wxPython is in use.
    """
    global _convert_agg_to_wx_image
    global _convert_agg_to_wx_bitmap

    if getattr(wx, '__version__', '0.0')[0:3] < '2.8':
        # wxPython < 2.8, so use the C++ accelerator or the Python routines
        if state and _wxagg is not None:
            _convert_agg_to_wx_image  = _wxagg.convert_agg_to_wx_image
            _convert_agg_to_wx_bitmap = _wxagg.convert_agg_to_wx_bitmap
        else:
            _convert_agg_to_wx_image  = _py_convert_agg_to_wx_image
            _convert_agg_to_wx_bitmap = _py_convert_agg_to_wx_bitmap
    else:
        # wxPython >= 2.8, so use the accelerated Python routines
        _convert_agg_to_wx_image  = _py_WX28_convert_agg_to_wx_image
        _convert_agg_to_wx_bitmap = _py_WX28_convert_agg_to_wx_bitmap


# try to load the WXAgg accelerator
try:
    import _wxagg
except ImportError:
    _wxagg = None

# if it's present, use it
_use_accelerator(True)
