"""
A wx API adapter to hide differences between wxPython classic and phoenix.

It is assumed that the user code is selecting what version it wants to use,
here we just ensure that it meets the minimum required by matplotlib.

For an example see embedding_in_wx2.py
"""
import wx

from .. import cbook
from .backend_wx import RendererWx


cbook.warn_deprecated("3.0", "{} is deprecated.".format(__name__))

backend_version = wx.VERSION_STRING
is_phoenix = 'phoenix' in wx.PlatformInfo

fontweights = RendererWx.fontweights
fontangles = RendererWx.fontangles
fontnames = RendererWx.fontnames

dashd_wx = {'solid': wx.PENSTYLE_SOLID,
            'dashed': wx.PENSTYLE_SHORT_DASH,
            'dashdot': wx.PENSTYLE_DOT_DASH,
            'dotted': wx.PENSTYLE_DOT}

# functions changes
BitmapFromBuffer = wx.Bitmap.FromBufferRGBA
EmptyBitmap = wx.Bitmap
EmptyImage = wx.Image
Cursor = wx.Cursor
EventLoop = wx.GUIEventLoop
NamedColour = wx.Colour
StockCursor = wx.Cursor
