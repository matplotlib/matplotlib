"""
Copyright (C) Jeremy O'Donoghue, 2003
 
License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
http://www.python.org/psf/license.html

This is a sample showing how to embed a matplotlib figure in a wxPanel,
and update the contents whenever a timer event occurs. It is inspired
by the GTK script dynamic_demo.py, by John Hunter (should be supplied with
this file) but I have assumed that you may wish to embed a figure inside
your own arbitrary frame, which makes the code slightly more complicated.

It goes without saying that you can update the display on any event, not
just a timer...

Should you require a toolbar and navigation, inspire yourself from
embedding_in_wx.py, which provides these features.

Modification History:
$Log$
Revision 1.1  2003/12/30 17:22:09  jodonoghue
First version of dynamic_demo for backend_wx

"""

import matplotlib
matplotlib.use('WX')

from matplotlib.backends import Figure, Toolbar, FigureManager
from matplotlib.axes import Subplot
import Numeric as numpy

from matplotlib.matlab import *
from wxPython.wx import *

TIMER_ID = wxNewId()

class PlotFigure(wxFrame):
    def __init__(self):
        wxFrame.__init__(self, None, -1, "Test embedded wxFigure")
        self.fig = Figure(self, -1, (5,4), 75)
        self.toolbar = Toolbar(self.fig)
        self.toolbar.Realize()

		# On Windows, default frame size behaviour is incorrect
        # you don't need this under Linux
        tw, th = self.toolbar.GetSizeTuple()
        fw, fh = self.fig.GetSizeTuple()
        self.toolbar.SetSize(wxSize(fw, th))
        
        # Create a figure manager to manage things
        self.figmgr = FigureManager(self.fig, 1, self)
        
        # Now put all into a sizer
        sizer = wxBoxSizer(wxVERTICAL)
		# This way of adding to sizer prevents resizing
        #sizer.Add(self.fig, 0, wxLEFT|wxTOP)
		
		# This way of adding to sizer allows resizing
        sizer.Add(self.fig, 1, wxLEFT|wxTOP|wxGROW)
		
		# Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wxGROW)
        self.SetSizer(sizer)
        self.Fit()
        EVT_TIMER(self, TIMER_ID, self.onTimer)
        
    def init_plot_data(self):
        a = self.figmgr.add_subplot(111)
        self.ind = numpy.arange(60)
        tmp = []
        for i in range(60):
            tmp.append(numpy.sin((self.ind+i)*numpy.pi/15))
        self.X = numpy.array(tmp)
        self.lines = a.plot(self.X[:,0],'o')
        self.count = 0

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an 
        # unmanaged toolbar in your frame
        return self.toolbar
		
    def onTimer(self, evt):
        self.count += 1
        if self.count > 99: self.count = 0
        self.lines[0].set_data(self.ind, self.X[:,self.count])
        self.fig.draw()
        self.fig.gui_repaint()
        
if __name__ == '__main__':
    app = wxPySimpleApp()
    frame = PlotFigure()
    frame.init_plot_data()
    
    # Initialise the timer - wxPython requires this to be connected to the
    # receiving event handler
    t = wxTimer(frame, TIMER_ID)
    t.Start(500)
    
    frame.Show()
    app.MainLoop()
