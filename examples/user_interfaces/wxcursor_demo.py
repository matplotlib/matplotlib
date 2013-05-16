"""
Example to draw a cursor and report the data coords in wx
"""
# Used to guarantee to use at least Wx2.8
import wxversion
#wxversion.ensureMinimal('2.8')
#wxversion.select('2.8')
#wxversion.select('2.9.5') # 2.9.x classic
wxversion.select('2.9.6-msw-phoenix') # 2.9.x phoenix

import matplotlib
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure
from numpy import arange, sin, pi

import wx
print wx.VERSION_STRING

class CanvasFrame(wx.Frame):

    def __init__(self, ):
        wx.Frame.__init__(self,None,-1,
                         'CanvasFrame',size=(550,350))
        if 'phoenix' in wx.PlatformInfo:
            self.SetBackgroundColour(wx.Colour("WHITE"))
        else:
            self.SetBackgroundColour(wx.NamedColour("WHITE"))

        self.figure = Figure()
        self.axes = self.figure.add_subplot(111)
        t = arange(0.0,3.0,0.01)
        s = sin(2*pi*t)

        self.axes.plot(t,s)
        self.axes.set_xlabel('t')
        self.axes.set_ylabel('sin(t)')
        self.figure_canvas = FigureCanvas(self, -1, self.figure)

        # Note that event is a MplEvent
        self.figure_canvas.mpl_connect('motion_notify_event', self.UpdateStatusBar)
        self.figure_canvas.Bind(wx.EVT_ENTER_WINDOW, self.ChangeCursor)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.figure_canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        self.statusBar = wx.StatusBar(self, -1)
        self.statusBar.SetFieldsCount(1)
        self.SetStatusBar(self.statusBar)

        self.toolbar = NavigationToolbar2Wx(self.figure_canvas)
        self.sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        self.toolbar.Show()
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        
    def OnPaint(self, event):
        self.figure_canvas.draw()
        event.Skip()
        
    def ChangeCursor(self, event):
        if 'phoenix' in wx.PlatformInfo:
            self.figure_canvas.SetCursor(wx.Cursor(wx.CURSOR_BULLSEYE))
        else:
            self.figure_canvas.SetCursor(wx.StockCursor(wx.CURSOR_BULLSEYE))

    def UpdateStatusBar(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.statusBar.SetStatusText(( "x= " + str(x) +
                                           "  y=" +str(y) ),
                                           0)

class App(wx.App):

    def OnInit(self):
        'Create the main window and insert the custom frame'
        frame = CanvasFrame()
        self.SetTopWindow(frame)
        frame.Show(True)
        return True

if __name__=='__main__':
    app = App(0)
    app.MainLoop()
