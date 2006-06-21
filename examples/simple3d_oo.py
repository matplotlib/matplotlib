#!/usr/bin/env python

import matplotlib
matplotlib.use('WXAgg')
matplotlib.rcParams['numerix'] = 'numarray'

from wxPython.wx import *
import matplotlib.axes3d
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, FigureManager, NavigationToolbar2WxAgg

class PlotFigure(wxFrame):
    def __init__(self):
        wxFrame.__init__(self, None, -1, "Test embedded wxFigure")
        
        self.fig = Figure((9,8), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = NavigationToolbar2WxAgg(self.canvas)
        self.toolbar.Realize()
        
        self.figmgr = FigureManager(self.canvas, 1, self)
        tw, th = self.toolbar.GetSizeTuple()
        fw, fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(wxSize(fw, th))
        sizer = wxBoxSizer(wxVERTICAL)
        
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wxLEFT|wxTOP|wxGROW)
        sizer.Add(self.toolbar, 0, wxGROW)
        self.SetSizer(sizer)
        self.Fit()
        
        self.plot3d()
        
    def plot3d(self):
        ax3d = matplotlib.axes3d.Axes3D(self.fig)
        plt = self.fig.axes.append(ax3d)
        
if __name__ == '__main__':
    app = wxPySimpleApp(0)
    frame = PlotFigure()
    frame.Show()
    app.MainLoop()
