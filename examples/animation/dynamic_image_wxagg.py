#!/usr/bin/env python
"""
Copyright (C) 2003-2004 Jeremy O'Donoghue and others

License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
http://www.python.org/psf/license.html

"""
import sys, time, os, gc

import matplotlib
matplotlib.use('WXAgg')

from matplotlib import rcParams
import matplotlib.cm as cm

from matplotlib.backends.backend_wxagg import Toolbar, FigureCanvasWxAgg

from matplotlib.figure import Figure
import numpy as npy
import wx


TIMER_ID = wx.NewId()

class PlotFigure(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, -1, "Test embedded wxFigure")

        self.fig = Figure((5,4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = Toolbar(self.canvas)
        self.toolbar.Realize()

        # On Windows, default frame size behaviour is incorrect
        # you don't need this under Linux
        tw, th = self.toolbar.GetSizeTuple()
        fw, fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(wx.Size(fw, th))

        # Initialise the timer - wxPython requires this to be connected to
        # the receiving event handler
        self.t = wx.Timer(self, TIMER_ID)
        self.t.Start(10)

        # Create a figure manager to manage things

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT|wx.TOP|wx.GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()
        self.Bind(wx.EVT_TIMER, self.onTimer, id=TIMER_ID)
        self.Bind(wx.EVT_CLOSE, self.onClose)

    def init_plot_data(self):
        # jdh you can add a subplot directly from the fig rather than
        # the fig manager
        a = self.fig.add_subplot(111)
        self.x = npy.arange(120.0)*2*npy.pi/120.0
        self.x.resize((100,120))
        self.y = npy.arange(100.0)*2*npy.pi/100.0
        self.y.resize((120,100))
        self.y = npy.transpose(self.y)
        z = npy.sin(self.x) + npy.cos(self.y)
        self.im = a.imshow( z, cmap=cm.jet)#, interpolation='nearest')

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

    def onTimer(self, evt):
        self.x += npy.pi/15
        self.y += npy.pi/20
        z = npy.sin(self.x) + npy.cos(self.y)
        self.im.set_array(z)
        self.canvas.draw()
        #self.canvas.gui_repaint()  # jdh wxagg_draw calls this already

    def onClose(self, evt):
        self.t.Stop()
        evt.Skip()

    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass

if __name__ == '__main__':
    app = wx.PySimpleApp()
    frame = PlotFigure()
    frame.init_plot_data()

    frame.Show()
    app.MainLoop()
