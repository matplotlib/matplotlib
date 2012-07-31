#!/usr/bin/env python
"""
An example of how to use wx or wxagg in an application with the new
toolbar - comment out the setA_toolbar line for no toolbar
"""

# Used to guarantee to use at least Wx2.8
import wxversion
wxversion.ensureMinimal('2.8')

from numpy import arange, sin, pi

import matplotlib

# uncomment the following to use wx rather than wxagg
#matplotlib.use('WX')
#from matplotlib.backends.backend_wx import FigureCanvasWx as FigureCanvas

# comment out the following to use wx rather than wxagg
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from matplotlib.figure import Figure

import wx

class CanvasFrame(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self,None,-1,
                         'CanvasFrame',size=(550,350))

        self.SetBackgroundColour(wx.NamedColour("WHITE"))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)

        self.axes.plot(t,s)
        self.canvas = FigureCanvas(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        self.add_toolbar()  # comment this out for no toolbar


    def add_toolbar(self):
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        if wx.Platform == '__WXMAC__':
            # Mac platform (OSX 10.3, MacPython) does not seem to cope with
            # having a toolbar in a sizer. This work-around gets the buttons
            # back, but at the expense of having the toolbar at the top
            self.SetToolBar(self.toolbar)
        else:
            # On Windows platform, default window size is incorrect, so set
            # toolbar width to figure width.
            tw, th = self.toolbar.GetSizeTuple()
            fw, fh = self.canvas.GetSizeTuple()
            # By adding toolbar in sizer, we are able to put it at the bottom
            # of the frame - so appearance is closer to GTK version.
            # As noted above, doesn't work for Mac.
            self.toolbar.SetSize(wx.Size(fw, th))
            self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()


    def OnPaint(self, event):
        self.canvas.draw()

class App(wx.App):

    def OnInit(self):
        'Create the main window and insert the custom frame'
        frame = CanvasFrame()
        frame.Show(True)

        return True

app = App(0)
app.MainLoop()
