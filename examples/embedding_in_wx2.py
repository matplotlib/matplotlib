#!/usr/bin/env python
"""
An example of how to use wx or wxagg in an application w/o the toolbar
"""

from matplotlib.numerix import arange, sin, pi

import matplotlib

# uncomment the following to use wx rather than wxagg
#matplotlib.use('WX')
#from matplotlib.backends.backend_wx import FigureCanvasWx as FigureCanvas

# comment out the following to use wx rather than wxagg
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from matplotlib.figure import Figure

from wxPython.wx import *

class CanvasFrame(wxFrame):
    
    def __init__(self):
        wxFrame.__init__(self,None,-1,
                         'CanvasFrame',size=(550,350))

        self.SetBackgroundColour(wxNamedColor("WHITE"))

        self.figure = Figure(figsize=(5,4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)
        
        self.axes.plot(t,s)

        self.canvas = FigureCanvas(self, -1, self.figure)

        self.sizer = wxBoxSizer(wxVERTICAL)
        self.sizer.Add(self.canvas, 1, wxTOP | wxLEFT | wxEXPAND)
        # Capture the paint message        
        EVT_PAINT(self, self.OnPaint)        

    def OnPaint(self, event):
        self.canvas.draw()
        
class App(wxApp):
    
    def OnInit(self):
        'Create the main window and insert the custom frame'
        frame = CanvasFrame()
        frame.Show(true)

        return true

app = App(0)
app.MainLoop()
