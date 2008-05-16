#!/usr/bin/env python
# printing_in_wx.py
#
"""
This examples demonstrates Printing (ie, to a printer) with a
matplotlib figure using a wx Frame.  This borrows the data from
embedding_in_wx.py, but with several changes:
  Menus for
     Save           export figure (png,eps,bmp) to file
     Copy           copy bitmap of figure to the system clipboard
     Print Setup    setup size of figure for printing
     Print Preview  preview printer page
     Print          send figure to a system printer
     Exit           end application

     where 'figure' means an image of the matplotlib canvas

  In addition, "Ctrl-C" is bound to Copy-figure-to-clipboard


This is a very simple use of matplotlib, and mostly focused on
demonstrating the interaction between wxPython and the matplotlib
figure canvas.

Matt Newville <newville@cars.uchicago.edu>
last modified: 12-Nov-2004
license:  use it any way you want
"""

import wx
import os
import matplotlib

# either WX or WXAgg can be used here.
# matplotlib.use('WX')
# from matplotlib.backends.backend_wxagg import FigureCanvasWx as FigCanvas

matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas

from matplotlib.figure import Figure
import  numpy

class PlotFrame(wx.Frame):
    help_msg="""  Menus for
     Save           export figure (png,eps,bmp) to file
     Copy           copy bitmap of figure to the system clipboard
     Print Setup    setup size of figure for printing
     Print Preview  preview printer page
     Print          send figure to a system printer
     Exit           end application

     where 'figure' means an image of the matplotlib canvas

  In addition, "Ctrl-C" is bound to copy-figure-to-clipboard
"""

    start_msg  = """        Use Menus to test printing
        or Ctrl-C to copy plot image to clipboard  """

    about_msg =  """        printing_in_wx version 0.1  12-Nov-2004
        Matt Newville <newville@cars.uchicago.edu>"""

    def __init__(self):
        wx.Frame.__init__(self, None, -1, "Test Printing with WX Backend")
        self.fig   = Figure((5.0,3.0), 100)
        self.canvas= FigCanvas(self, -1, self.fig)
        self.axes  = self.fig.add_axes([0.15,0.15,0.75,0.75])


        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.LEFT|wx.TOP|wx.GROW)
        sizer.Add(wx.StaticText(self,-1, self.start_msg), 0,
                  wx.ALIGN_LEFT|wx.TOP)

        self.canvas.Bind(wx.EVT_KEY_DOWN, self.onKeyEvent)

        self.SetSizer(sizer)
        self.Fit()
        self.Build_Menus()
        self.Plot_Data()

    def Build_Menus(self):
        """ build menus """
        MENU_EXIT  = wx.NewId()
        MENU_SAVE  = wx.NewId()
        MENU_PRINT = wx.NewId()
        MENU_PSETUP= wx.NewId()
        MENU_PREVIEW=wx.NewId()
        MENU_CLIPB  =wx.NewId()
        MENU_HELP   =wx.NewId()

        menuBar = wx.MenuBar()

        f0 = wx.Menu()
        f0.Append(MENU_SAVE,   "&Export",   "Save Image of Plot")
        f0.AppendSeparator()
        f0.Append(MENU_PSETUP, "Page Setup...",    "Printer Setup")
        f0.Append(MENU_PREVIEW,"Print Preview...", "Print Preview")
        f0.Append(MENU_PRINT,  "&Print",           "Print Plot")
        f0.AppendSeparator()
        f0.Append(MENU_EXIT,   "E&xit", "Exit")
        menuBar.Append(f0,     "&File");

        f1 = wx.Menu()
        f1.Append(MENU_HELP, "Quick Reference",  "Quick Reference")

        menuBar.Append(f1, "&Help");

        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, self.onPrint,        id=MENU_PRINT)
        self.Bind(wx.EVT_MENU, self.onPrinterSetup, id=MENU_PSETUP)
        self.Bind(wx.EVT_MENU, self.onPrinterPreview, id=MENU_PREVIEW)
        self.Bind(wx.EVT_MENU, self.onClipboard,    id=MENU_CLIPB)
        self.Bind(wx.EVT_MENU, self.onExport,       id=MENU_SAVE)
        self.Bind(wx.EVT_MENU, self.onExit ,        id=MENU_EXIT)
        self.Bind(wx.EVT_MENU, self.onHelp,         id=MENU_HELP)

    # the printer / clipboard methods are implemented
    # in backend_wx, and so are very simple to use.
    def onPrinterSetup(self,event=None):
        self.canvas.Printer_Setup(event=event)

    def onPrinterPreview(self,event=None):
        self.canvas.Printer_Preview(event=event)

    def onPrint(self,event=None):
        self.canvas.Printer_Print(event=event)

    def onClipboard(self,event=None):
        self.canvas.Copy_to_Clipboard(event=event)


    def onKeyEvent(self,event=None):
        """ capture , act upon keystroke events"""
        if event == None: return
        key = event.KeyCode()
        if (key < wx.WXK_SPACE or  key > 255):  return

        if (event.ControlDown() and chr(key)=='C'): # Ctrl-C
            self.onClipboard(event=event)

    def onHelp(self, event=None):
        dlg = wx.MessageDialog(self, self.help_msg,
                               "Quick Reference",
                               wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

    def onExport(self,event=None):
        """ save figure image to file"""
        file_choices = "PNG (*.png)|*.png|" \
                       "PS (*.ps)|*.ps|" \
                       "EPS (*.eps)|*.eps|" \
                       "BMP (*.bmp)|*.bmp"

        thisdir  = os.getcwd()

        dlg = wx.FileDialog(self, message='Save Plot Figure as...',
                            defaultDir = thisdir, defaultFile='plot.png',
                            wildcard=file_choices, style=wx.SAVE)

        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            self.canvas.print_figure(path,dpi=300)
            if (path.find(thisdir) ==  0):
                path = path[len(thisdir)+1:]
            print 'Saved plot to %s' % path

    def onExit(self,event=None):
        self.Destroy()

    def Plot_Data(self):
        t = numpy.arange(0.0,5.0,0.01)
        s = numpy.sin(2.0*numpy.pi*t)
        c = numpy.cos(0.4*numpy.pi*t)
        self.axes.plot(t,s)
        self.axes.plot(t,c)


if __name__ == '__main__':
    app = wx.PySimpleApp()
    fig = PlotFrame()
    fig.Show(True)
    app.MainLoop()
