from __future__ import division 
"""

 backend_wx.py

 A wxPython backend for Agg.  This uses the GUI widgets written by
 Jeremy O'Donoghue (jeremy@o-donoghue.com) and the Agg backend by John
 Hunter (jdhunter@ace.bsd.uchicago.edu)

 Copyright (C) Jeremy O'Donoghue & John Hunter, 2003-4
 
 License: This work is licensed under the matplotlib license( PSF
 compatible). A copy should be included with this source code.

"""

import sys, os, os.path, math, StringIO
from backend_agg import FigureCanvasAgg

from backend_wx import FigureManager
from backend_wx import FigureManagerWx, FigureCanvasWx, FigureFrameWx, \
     DEBUG_MSG
from backend_wx import error_msg_wx, draw_if_interactive, show, Toolbar, \
     backend_version
from matplotlib.figure import Figure
from matplotlib import rcParams
import matplotlib
from wxPython.wx import *

class FigureFrameWxAgg(FigureFrameWx):
    def get_canvas(self, fig):
        return FigureCanvasWxAgg(self, -1, fig)

              
class FigureCanvasWxAgg(FigureCanvasWx,FigureCanvasAgg):
    """
    The FigureCanvas contains the figure and does event handling.
    
    In the wxPython backend, it is derived from wxPanel, and (usually)
    lives inside a frame instantiated by a FigureManagerWx. The parent
    window probably implements a wxSizer to control the displayed
    control size - but we give a hint as to our preferred minimum
    size.
    """


    def draw(self):
        """
        Render the figure using agg
        """
        DEBUG_MSG("draw()", 1, self)
        FigureCanvasAgg.draw(self)
        s = self.tostring_rgb()  
        w = int(self.renderer.width)
        h = int(self.renderer.height)
        image = wxEmptyImage(w,h)
        image.SetData(s)
        self.bitmap = image.ConvertToBitmap()
        self.gui_repaint()
        

    def print_figure(self, filename, dpi=150, facecolor='w',
                     edgecolor='w', orientation='portrait'):

        """
        Render the figure to hardcopy
        """
        agg = self.switch_backends(FigureCanvasAgg)
        agg.print_figure(filename, dpi, facecolor, edgecolor, orientation)
        self.figure.set_canvas(self)

    def _get_imagesave_wildcards(self):
        'return the wildcard string for the filesave dialog'
        return "PS (*.ps)|*.ps|"     \
               "EPS (*.eps)|*.eps|"  \
               "SVG (*.svg)|*.svg|"  \
               "BMP (*.bmp)|*.bmp|"  \
               "PNG (*.png)|*.png"  \

    

def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    # in order to expose the Figure constructor to the pylab
    # interface we need to create the figure here
    DEBUG_MSG("new_figure_manager()", 3, None)

    fig = Figure(*args, **kwargs)
    frame = FigureFrameWxAgg(num, fig)
    figmgr = frame.get_figure_manager()
    if matplotlib.is_interactive():
        figmgr.canvas.realize()
        figmgr.frame.Show() 
    return figmgr


