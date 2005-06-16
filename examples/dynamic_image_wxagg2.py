#!/usr/bin/env python
"""
Copyright (C) 2003-2005 Jeremy O'Donoghue and others
 
License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
http://www.python.org/psf/license.html

"""
import sys, time, os, gc

import matplotlib
matplotlib.use('WXAgg')

# jdh: you need to control Numeric vs numarray with numerix, otherwise
# matplotlib may be using numeric under the hood and while you are
# using numarray and this isn't efficient.  Also, if you use
# numerix=numarray, it is important to compile matplotlib for numarray
# by setting NUMERIX = 'numarray' in setup.py before building
from matplotlib import rcParams
rcParams['numerix'] = 'numarray'


# jdh: you can import cm directly, you don't need to go via
# pylab
import matplotlib.cm as cm

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

# jdh: you don't need a figure manager in the GUI - this class was
# designed for the pylab interface

from matplotlib.figure import Figure
import matplotlib.numerix as numerix
from wxPython.wx import *


TIMER_ID = wxNewId()

class PlotFigure(wxFrame):

    def __init__(self):
        wxFrame.__init__(self, None, -1, "Test embedded wxFigure")

        self.fig = Figure((5,4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()

        # On Windows, default frame size behaviour is incorrect
        # you don't need this under Linux
        tw, th = self.toolbar.GetSizeTuple()
        fw, fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(wxSize(fw, th))

        # Create a figure manager to manage things

        # Now put all into a sizer
        sizer = wxBoxSizer(wxVERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wxLEFT|wxTOP|wxGROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wxGROW)
        self.SetSizer(sizer)
        self.Fit()
        EVT_TIMER(self, TIMER_ID, self.onTimer)

    def init_plot_data(self):
        # jdh you can add a subplot directly from the fig rather than
        # the fig manager
        a = self.fig.add_axes([0.075,0.1,0.75,0.85])
        cax = self.fig.add_axes([0.85,0.1,0.075,0.85])
        self.x = numerix.arange(120.0)*2*numerix.pi/120.0
        self.x.resize((100,120))
        self.y = numerix.arange(100.0)*2*numerix.pi/100.0
        self.y.resize((120,100))
        self.y = numerix.transpose(self.y)
        z = numerix.sin(self.x) + numerix.cos(self.y)
        self.im = a.imshow( z, cmap=cm.jet)#, interpolation='nearest')
        self.fig.colorbar(self.im,cax=cax,orientation='vertical')

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an 
        # unmanaged toolbar in your frame
        return self.toolbar
		
    def onTimer(self, evt):
        self.x += numerix.pi/15
        self.y += numerix.pi/20
        z = numerix.sin(self.x) + numerix.cos(self.y)
        self.im.set_array(z)
        self.canvas.draw()
        #self.canvas.gui_repaint()  # jdh wxagg_draw calls this already
        
    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass
        
if __name__ == '__main__':
    app = wxPySimpleApp()
    frame = PlotFigure()
    frame.init_plot_data()
    
    # Initialise the timer - wxPython requires this to be connected to
    # the receiving event handler
    t = wxTimer(frame, TIMER_ID)
    t.Start(200)
    
    frame.Show()
    app.MainLoop()

