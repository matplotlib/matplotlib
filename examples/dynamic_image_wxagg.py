#!/usr/bin/env python
"""
Copyright (C) 2003-2004 Jeremy O'Donoghue, Andrew Straw and others
 
License: This work is licensed under the PSF. A copy should be included
with this source code, and is also available at
http://www.python.org/psf/license.html

Modification History:
$Log$
Revision 1.1  2004/07/10 05:54:54  astraw
First version of dynamic_image_wxagg, modified from dynamic_demo_wx by Jeremy
O'Donoghue.

Revision 1.4  2004/05/03 12:12:26  jdh2358
added bang header to examples

Revision 1.3  2004/03/08 22:17:20  jdh2358

* Fixed embedding_in_wx and dynamic_demo_wx examples

* Ported build to darwin

* Tk:

  removed default figman=None from nav toolbar since it needs the
  figman

  fixed close bug

  small changes to aid darwin build

Revision 1.2  2004/02/26 20:22:58  jaytmiller
Added the "numerix" Numeric/numarray selector module enabling matplotlib
to work with either numarray or Numeric.  See matplotlib.numerix.__doc__.

Revision 1.1  2003/12/30 17:22:09  jodonoghue
First version of dynamic_demo for backend_wx
"""

import matplotlib
from matplotlib.matlab import cm
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import Toolbar, FigureCanvasWxAgg,\
     FigureManager

from matplotlib.figure import Figure
from matplotlib.axes import Subplot
import numarray # segfaults w/ Numeric 23.1 ADS
from wxPython.wx import *


TIMER_ID = wxNewId()

class PlotFigure(wxFrame):

    def __init__(self):
        wxFrame.__init__(self, None, -1, "Test embedded wxFigure")

        self.fig = Figure((5,4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = Toolbar(self.canvas)
        self.toolbar.Realize()

        # On Windows, default frame size behaviour is incorrect
        # you don't need this under Linux
        tw, th = self.toolbar.GetSizeTuple()
        fw, fh = self.canvas.GetSizeTuple()
        self.toolbar.SetSize(wxSize(fw, th))

        # Create a figure manager to manage things
        self.figmgr = FigureManager(self.canvas, 1, self)
        # Now put all into a sizer
        sizer = wxBoxSizer(wxVERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wxLEFT|wxTOP|wxGROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wxGROW)
        self.SetSizer(sizer)
        self.Fit()
        EVT_TIMER(self, TIMER_ID, self.onTimer)
#        EVT_ERASE_BACKGROUND( self, self.onEraseBackground)
        
    def init_plot_data(self):
        a = self.figmgr.add_subplot(111)
        self.x = numarray.arange(120.0)*2*numarray.pi/120.0
        self.x.resize((100,120))
        self.y = numarray.arange(100.0)*2*numarray.pi/100.0
        self.y.resize((120,100))
        self.y = numarray.transpose(self.y)
        z = numarray.sin(self.x) + numarray.cos(self.y)
        self.im = a.imshow( z, cmap=cm.jet)#, interpolation='nearest')

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar
		
    def onTimer(self, evt):
        self.x += numarray.pi/15
        self.y += numarray.pi/20
        z = numarray.sin(self.x) + numarray.cos(self.y)
        self.im.set_array(z)
        self.canvas.draw()
        self.canvas.gui_repaint()
        
    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers
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
