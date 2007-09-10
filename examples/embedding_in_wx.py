#!/usr/bin/env python
# embedding_in_wx.py
#

"""
Copyright (C) Jeremy O'Donoghue, 2003

License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
http://www.python.org/psf/license.html

This is a sample showing how to embed a matplotlib figure in a wxPanel.

The example implements the full navigation toolbar, so you can automatically
inherit standard matplotlib features such as the ability to zoom, pan and
save figures in the supported formats.

There are a few small complexities worth noting in the example:

1) By default, a wxFrame can contain a toolbar (added with SetToolBar())
   but this is at the top of the frame. Matplotlib default is to put the
   controls at the bottom of the frame, so you have to manage the toolbar
   yourself. I have done this by putting the figure and toolbar into a
   sizer, but this means that you need to override GetToolBar for your
   wxFrame so that the figure manager can find the toolbar.

2) I have implemented a figure manager to look after the plots and axes.
   If you don't want a toolbar, it is simpler to add the figure directly
   and not worry. However, the figure manager looks after clipping of the
   figure contents, so you will need it if you want to navigate

3) There is a bug in the way in which my copy of wxPython calculates
   toolbar width on Win32, so there is a tricky line to ensure that the
   width of the toolbat is the same as the width of the figure.

4) Depending on the parameters you pass to the sizer, you can make the
   figure resizable or not.
"""

import matplotlib
matplotlib.use('WX')
from matplotlib.backends.backend_wx import Toolbar, FigureCanvasWx,\
     FigureManager

from matplotlib.figure import Figure
from matplotlib.axes import Subplot
import  numpy
from wx import *



class PlotFigure(Frame):
    def __init__(self):
        Frame.__init__(self, None, -1, "Test embedded wxFigure")

        self.fig = Figure((9,8), 75)
        self.canvas = FigureCanvasWx(self, -1, self.fig)
        self.toolbar = Toolbar(self.canvas)
        self.toolbar.Realize()

        # On Windows, default frame size behaviour is incorrect
        # you don't need this under Linux
        tw, th = self.toolbar.GetSizeTuple()
        fw, fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(Size(fw, th))

        # Create a figure manager to manage things
        self.figmgr = FigureManager(self.canvas, 1, self)
        # Now put all into a sizer
        sizer = BoxSizer(VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, LEFT|TOP|GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, GROW)
        self.SetSizer(sizer)
        self.Fit()

    def plot_data(self):
        # Use ths line if using a toolbar
        a = self.fig.add_subplot(111)

        # Or this one if there is no toolbar
        #a = Subplot(self.fig, 111)

        t = numpy.arange(0.0,3.0,0.01)
        s = numpy.sin(2*numpy.pi*t)
        c = numpy.cos(2*numpy.pi*t)
        a.plot(t,s)
        a.plot(t,c)
        self.toolbar.update()

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

if __name__ == '__main__':
    app = PySimpleApp(0)
    frame = PlotFigure()
    frame.plot_data()
    frame.Show()
    app.MainLoop()
