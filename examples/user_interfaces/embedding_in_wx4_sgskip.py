"""
==================
Embedding in wx #4
==================

An example of how to use wxagg in a wx application with a custom toolbar.
"""

from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)
from matplotlib.figure import Figure

import numpy as np

import wx


class MyNavigationToolbar(NavigationToolbar):
    """Extend the default wx toolbar with your own event handlers."""

    def __init__(self, canvas):
        super().__init__(canvas)
        # We use a stock wx bitmap, but you could also use your own image file.
        bmp = wx.ArtProvider.GetBitmap(wx.ART_CROSS_MARK, wx.ART_TOOLBAR)
        tool = self.AddTool(wx.ID_ANY, 'Click me', bmp,
                            'Activate custom control')
        self.Bind(wx.EVT_TOOL, self._on_custom, id=tool.GetId())

    def _on_custom(self, event):
        # add some text to the axes in a random location in axes coords with a
        # random color
        ax = self.canvas.figure.axes[0]
        x, y = np.random.rand(2)  # generate a random location
        rgb = np.random.rand(3)  # generate a random color
        ax.text(x, y, 'You clicked me', transform=ax.transAxes, color=rgb)
        self.canvas.draw()
        event.Skip()


class CanvasFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, -1, 'CanvasFrame', size=(550, 350))

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.axes = self.figure.add_subplot()
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)

        self.axes.plot(t, s)

        self.canvas = FigureCanvas(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.TOP | wx.LEFT | wx.EXPAND)

        self.toolbar = MyNavigationToolbar(self.canvas)
        self.toolbar.Realize()
        # By adding toolbar in sizer, we are able to put it at the bottom
        # of the frame - so appearance is closer to GTK version.
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)

        # update the axes menu on the toolbar
        self.toolbar.update()
        self.SetSizer(self.sizer)
        self.Fit()


class App(wx.App):
    def OnInit(self):
        """Create the main window and insert the custom frame."""
        frame = CanvasFrame()
        frame.Show(True)
        return True


if __name__ == "__main__":
    app = App()
    app.MainLoop()
