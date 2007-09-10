#!/usr/bin/env python
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
Revision 1.7  2005/06/15 20:24:56  jdh2358
syncing for 82

Revision 1.6  2004/10/26 18:08:13  astraw
Converted to use new NavigationToolbar2 (from old Toolbar).

Revision 1.5  2004/06/26 06:37:20  astraw
Trivial bugfix to eliminate IndexError

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
matplotlib.use('WX')
from matplotlib.backends.backend_wx import FigureCanvasWx,\
     FigureManager, NavigationToolbar2Wx

from matplotlib.figure import Figure
from matplotlib.axes import Subplot
import  numpy
from wx import *


TIMER_ID = NewId()

class PlotFigure(Frame):

    def __init__(self):
        Frame.__init__(self, None, -1, "Test embedded wxFigure")

        self.fig = Figure((5,4), 75)
        self.canvas = FigureCanvasWx(self, -1, self.fig)
        self.toolbar = NavigationToolbar2Wx(self.canvas)
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
        EVT_TIMER(self, TIMER_ID, self.onTimer)

    def init_plot_data(self):
        a = self.fig.add_subplot(111)
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
        if self.count >= 60: self.count = 0
        self.lines[0].set_data(self.ind, self.X[:,self.count])
        self.canvas.draw()
        self.canvas.gui_repaint()

if __name__ == '__main__':
    app = PySimpleApp()
    frame = PlotFigure()
    frame.init_plot_data()

    # Initialise the timer - wxPython requires this to be connected to the
    # receivicng event handler
    t = Timer(frame, TIMER_ID)
    t.Start(100)

    frame.Show()
    app.MainLoop()
