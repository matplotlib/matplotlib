"""
================
Embedding In Wx4
================

An example of how to use wx or wxagg in an application with a custom toolbar.
"""

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.backends.backend_wx import _load_bitmap
from matplotlib.figure import Figure

import numpy as np

import wx


class MyNavigationToolbar(NavigationToolbar):
    """
    Extend the default wx toolbar with your own event handlers
    """
    ON_CUSTOM = wx.NewId()

    def __init__(self, canvas, cankill):
        NavigationToolbar.__init__(self, canvas)

        # for simplicity I'm going to reuse a bitmap from wx, you'll
        # probably want to add your own.
        if 'phoenix' in wx.PlatformInfo:
            self.AddTool(self.ON_CUSTOM, 'Click me',
                         _load_bitmap('back.png'),
                         'Activate custom contol')
            self.Bind(wx.EVT_TOOL, self._on_custom, id=self.ON_CUSTOM)
        else:
            self.AddSimpleTool(self.ON_CUSTOM, _load_bitmap('back.png'),
                               'Click me', 'Activate custom contol')
            self.Bind(wx.EVT_TOOL, self._on_custom, id=self.ON_CUSTOM)

    def _on_custom(self, evt):
        # add some text to the axes in a random location in axes (0,1)
        # coords) with a random color

        # get the axes
        ax = self.canvas.figure.axes[0]

        # generate a random location can color
        x, y = np.random.rand(2)
        rgb = np.random.rand(3)

        # add the text and draw
        ax.text(x, y, 'You clicked me',
                transform=ax.transAxes,
                color=rgb)
        self.canvas.draw()
        evt.Skip()


class CanvasFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1,
                          'CanvasFrame', size=(550, 350))

        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)

        self.axes.plot(t, s)

        self.canvas = FigureCanvas(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.TOP | wx.LEFT | wx.EXPAND)

        self.toolbar = MyNavigationToolbar(self.canvas, True)
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
        'Create the main window and insert the custom frame'
        frame = CanvasFrame()
        frame.Show(True)

        return True

app = App(0)
app.MainLoop()
