#!/usr/bin/env python
"""
Show how to have wx draw a cursor over an axes that moves with the
mouse and reports the data coords
"""

from matplotlib.numerix import arange, sin, pi

import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from matplotlib.figure import Figure

from wxPython.wx import *

class CanvasFrame(wxFrame):
    
    def __init__(self):
        wxFrame.__init__(self,None,-1,
                         'CanvasFrame',size=(550,350))

        self.SetBackgroundColour(wxNamedColor("WHITE"))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)
        
        self.axes.plot(t,s)
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Price ($)')                
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.sizer = wxBoxSizer(wxVERTICAL)
        self.sizer.Add(self.canvas, 1, wxLEFT | wxTOP | wxGROW)
        self.SetSizer(self.sizer)
        self.Fit()

        self.statusBar =         wxStatusBar(self, -1)
        self.statusBar.SetFieldsCount(1)
        self.SetStatusBar(self.statusBar)

        self.add_toolbar()  # comment this out for no toolbar


    def mouse_move(self, event):
        self.draw_cursor(event)

    def add_toolbar(self):
        self.toolbar = NavigationToolbar2Wx(self.canvas)
        self.toolbar.Realize()
        tw, th = self.toolbar.GetSizeTuple()
        fw, fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(wxSize(fw, th))
        self.sizer.Add(self.toolbar, 0, wxLEFT | wxEXPAND)
        # update the axes menu on the toolbar
        self.toolbar.update()  

        
    def OnPaint(self, event):
        try: del self.lastInfo
        except AttributeError: pass
        self.canvas.draw()

    def draw_cursor(self, event):
        'event is a MplEvent.  Draw a cursor over the axes'

        if event.inaxes is None:
            try: lastline1, lastline2, lastax, lastdc = self.lastInfo
            except AttributeError: pass
            else:
                lastdc.DrawLine(*lastline1) # erase old
                lastdc.DrawLine(*lastline2) # erase old
                del self.lastInfo
            return
        canvas = self.canvas
        figheight = canvas.figure.bbox.height()
        ax = event.inaxes
        left,bottom,width,height = ax.bbox.get_bounds()
        bottom = figheight-bottom
        top = bottom - height
        right = left + width
        x, y = event.x, event.y
        y = figheight-y

        dc = wxClientDC(canvas)
        dc.SetLogicalFunction(wxXOR)            
        wbrush = wxBrush(wxColour(255,255,255), wxTRANSPARENT)
        wpen = wxPen(wxColour(200, 200, 200), 1, wxSOLID)
        dc.SetBrush(wbrush)
        dc.SetPen(wpen)
            
        dc.ResetBoundingBox()
        dc.BeginDrawing()

        x, y, left, right, bottom, top = [int(val) for val in x, y, left, right, bottom, top]

        try: lastline1, lastline2, lastax, lastdc = self.lastInfo
        except AttributeError: pass
        else:
            lastdc.DrawLine(*lastline1) # erase old
            lastdc.DrawLine(*lastline2) # erase old            

        line1 = (x, bottom, x, top)
        line2 = (left, y, right, y)
        self.lastInfo = line1, line2, ax, dc
        dc.DrawLine(*line1) # draw new
        dc.DrawLine(*line2) # draw new        
        dc.EndDrawing()

        time, price = event.xdata, event.ydata
        self.statusBar.SetStatusText("Time=%f  Price=%f"% (time, price), 0)


class App(wxApp):
    
    def OnInit(self):
        'Create the main window and insert the custom frame'
        frame = CanvasFrame()
        frame.Show(true)

        return true

app = App(0)
app.MainLoop()

