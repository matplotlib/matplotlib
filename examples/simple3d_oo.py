#!/usr/bin/env python

import matplotlib
matplotlib.use('WXAgg')
matplotlib.rcParams['numerix'] = 'numarray'

from wxPython.wx import *
import matplotlib.axes3d
from matplotlib import numerix as nx
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
        # sample taken from http://www.scipy.org/Cookbook/Matplotlib/mplot3D
        ax3d = matplotlib.axes3d.Axes3D(self.fig)
        plt = self.fig.axes.append(ax3d)
        
        delta = nx.pi / 100.0
        u = nx.arange(0, 2*nx.pi, delta)
        v = nx.arange(0, nx.pi, delta)
        
        x=10*nx.outerproduct(nx.cos(u),nx.sin(v))
        y=10*nx.outerproduct(nx.sin(u),nx.sin(v))
        z=10*nx.outerproduct(nx.ones(nx.size(u)),nx.cos(v))
        
        ax3d.plot_wireframe(x,y,z)
        ax3d.plot_surface(x+10,y,z)
        
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        
if __name__ == '__main__':
    app = wxPySimpleApp(0)
    frame = PlotFigure()
    frame.Show()
    app.MainLoop()
