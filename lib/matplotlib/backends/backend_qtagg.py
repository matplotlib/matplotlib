"""
backend_qtagg.py version 0.0.1
http://www.tjora.no/python/matplotlib/qtaggbackend

This is a not very tested matplotlib qt-backend that uses the
Agg-backend for rendering etc.

Tested on Windows with 
  python 2.3.4
  matplotlib 0.70.1
  PyQt-3.11-040518-Python23

It does not work in IPython interactive mode, and will therefore raise
an Exception if that is tried. I suppose that IPython must be changed
to support a Qt main loop in a thread for this to work ok. 

Problems:
    Should FigureManagerQtAgg.destroy do anything?
    
REQUIREMENTS

  PyQt of some unknown version. Does not require anything but a basic Qt,
  that is no Enterprise version etc. (Uses QWidget, and not QCanvas)

Copyright (C) Sigve Tjora, 2005
 
 License: This work is licensed under the matplotlib license( PSF
 compatible). A copy should be included with this source code.  

"""

from __future__ import division

import sys
from matplotlib import verbose
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import RendererBase, GraphicsContextBase,\
     FigureManagerBase, FigureCanvasBase, error_msg, NavigationToolbar2

from matplotlib.cbook import enumerate
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import matplotlib 
from backend_agg import RendererAgg, FigureCanvasAgg
import qt
import os.path

def error_msg_qtagg(msg, *args):
    """
    Signal an error condition.
    - in a GUI backend, popup a error dialog.
    - in a non-GUI backend delete this function and use
    'from matplotlib.backend_bases import error_msg'
    """
    makeSureWeHaveAQApp()
    qt.QMessageBox.warning(None, "Matplotlib", msg, qt.QMessageBox.Ok)
    raise SystemExit
    
       
        
########################################################################
#    
# The following functions and classes are for pylab and implement
# window/figure managers, etc...
#
########################################################################

def draw_if_interactive():
    """
    For image backends - is not required
    For GUI backends - this should be overriden if drawing should be done in
    interactive python mode
    """
    if matplotlib.is_interactive():
        raise NotImplementedError("Interactive drawing not supported on Qt-backend.")
        #figManager =  Gcf.get_active()
        #if figManager != None:
            #figManager.canvas.draw()

def show(mainloop = True):
    """
    For image backends - is not required
    For GUI backends - show() is usually the last line of a pylab script and
    tells the backend that it is time to draw.  In interactive mode, this may
    be a do nothing func.  See the GTK backend for an example of how to handle
    interactive versus batch mode
    """
    for manager in Gcf.get_all_fig_managers():
        # do something to display the GUI
        manager.window.show()
    if mainloop:
        qt.qApp.connect(qt.qApp, qt.SIGNAL("lastWindowClosed()")
                    , qt.qApp
                    , qt.SLOT("quit()")
                    )
        qt.qApp.exec_loop()

def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    makeSureWeHaveAQApp()
    thisFig = Figure(*args, **kwargs)
    window = qt.QMainWindow()
    canvas = FigureCanvasQtAgg(thisFig, window)
    if matplotlib.rcParams['toolbar']=='toolbar2':
        toolbar = NavigationToolbar2QtAgg(canvas, window)
    else:
        toolbar = None
    manager = FigureManagerQtAgg(canvas, num, window, toolbar)
    window.resize(qt.QSize(600,483).expandedTo(window.minimumSizeHint()))
    window.clearWState(qt.Qt.WState_Polished)
    return manager


class FigureCanvasQtAgg(qt.QWidget, FigureCanvasAgg):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance

    Note GUI templates will want to connect events for button presses,
    mouse movements and key presses to functions that call the base
    class methods button_press_event, button_release_event,
    motion_notify_event, key_press_event, and key_release_event.  See,
    eg backend_gtk.py, backend_wx.py and backend_tkagg.py
    """
    def __init__(self, figure, window):
        qt.QWidget.__init__(self, window)
        FigureCanvasAgg.__init__(self, figure)
        self.buffer = qt.QPixmap()

        #Coordinates of the select rectangle if it is needed.
        self.rect = None

    def draw(self):
        """
        Draw the figure using the renderer
        """
        FigureCanvasAgg.draw(self)
        self.stringBuffer = str(self.buffer_rgba())
        self.qimage = qt.QImage(self.stringBuffer, 
                            self.renderer.width, 
                            self.renderer.height,
                            32,
                            None, 
                            0, 
                            qt.QImage.IgnoreEndian)
        self.update()
        
    def paintEvent(self, ev):
        self.buffer.convertFromImage(self.qimage, qt.Qt.OrderedAlphaDither)
        if self.rect:
            self.p = qt.QPainter()
            self.p.pen().setStyle(qt.QPen.DotLine)
            self.p.begin(self.buffer)
            x0,y0, x1,y1 = self.rect
            self.p.drawRect(x0,y0, x1-x0, y1-y0)
            self.p.flush()
            self.p.end()
        # blit the pixmap to the widget
        qt.bitBlt(self, 0, 0, self.buffer)

    def resizeEvent(self, ev):
        width, height = ev.size().width(), ev.size().height() 

        # compute desired figure size in inches
        dpival = self.figure.dpi.get()
        winch = width/dpival
        hinch = height/dpival
        self.figure.set_figsize_inches(winch, hinch)

        self.draw()
        
    #Events to pass on to matplotlib
    def keyPressEvent(self, ev):
        self.keyPressEvent(ev.text())
    def keyReleaseEvent(self, ev): 
        self.keyReleaseEvent(ev.text())
    def mouseMoveEvent(self, ev):
        x, y = self.getMatplotlibCoord(ev)
        self.motion_notify_event(x, y)        
    def mousePressEvent(self, ev):
        x, y = self.getMatplotlibCoord(ev)
        self.button_press_event(x, y, self.get_matlab_button(ev))
    def mouseReleaseEvent(self, ev): 
        x, y = self.getMatplotlibCoord(ev)
        self.button_release_event(x, y, self.get_matlab_button(ev))
    def get_matlab_button(self, ev):
        b = ev.button()
        if b == qt.QMouseEvent.NoButton:
            return None
        elif b == qt.QMouseEvent.LeftButton:
            return 1
        elif b == qt.QMouseEvent.MidButton:
            return 2
        elif b == qt.QMouseEvent.RightButton:
            return 3
        else:
            return None
    def getMatplotlibCoord(self, ev):
        x = ev.x()
        y = self.height() - ev.y()
        return x,y
    def print_figure(self, filename, dpi=150,
                     facecolor='w', edgecolor='w',
                     orientation='portrait'):

        agg = self.switch_backends(FigureCanvasAgg)
        agg.print_figure(filename, dpi, facecolor, edgecolor, orientation)
                    
class NavigationToolbar2QtAgg(NavigationToolbar2, qt.QToolBar):
    """
    Public attriubutes

      canvas   - the FigureCanvas (qt.QWidget)
      window   - the qt.QMainWindow
    """
    def __init__(self, canvas, window):
        qt.QToolBar.__init__(self, qt.QString(""), window, qt.Qt.DockTop)
        self.canvas = canvas
        self.window = window
        NavigationToolbar2.__init__(self, canvas)
        
    def _init_toolbar(self):
        self.window.statusBar()

        basedir = matplotlib.rcParams['datapath']

        self.imageHome = qt.QPixmap(os.path.join(basedir, "home.png"))
        self.imageBack = qt.QPixmap(os.path.join(basedir, "back.png"))
        self.imageForward = qt.QPixmap(os.path.join(basedir, "forward.png"))
        self.imageMove = qt.QPixmap(os.path.join(basedir, "move.png"))
        self.imageSave = qt.QPixmap(os.path.join(basedir, "filesave.png"))
        self.imageZoom = qt.QPixmap(os.path.join(basedir, "zoom_to_rect.png"))

        self.homeAction = qt.QAction(self,"homeAction")
        self.homeAction.setIconSet(qt.QIconSet(self.imageHome))
        self.backAction = qt.QAction(self,"backAction")
        self.backAction.setIconSet(qt.QIconSet(self.imageBack))
        self.forwardAction = qt.QAction(self,"forwardAction")
        self.forwardAction.setIconSet(qt.QIconSet(self.imageForward))
        self.moveAction = qt.QAction(self,"moveAction")
        self.moveAction.setIconSet(qt.QIconSet(self.imageMove))
        self.zoomAction = qt.QAction(self,"zoomAction")
        self.zoomAction.setIconSet(qt.QIconSet(self.imageZoom))
        self.fileSaveAction = qt.QAction(self,"fileSaveAction")
        self.fileSaveAction.setIconSet(qt.QIconSet(self.imageSave))

        self.homeAction.addTo(self)
        self.backAction.addTo(self)
        self.forwardAction.addTo(self)
        self.moveAction.addTo(self)
        self.zoomAction.addTo(self)
        self.fileSaveAction.addTo(self)

        self.connect(self.fileSaveAction,qt.SIGNAL("activated()"),self.save_figure)
        self.connect(self.backAction,qt.SIGNAL("activated()"),self.back)
        self.connect(self.forwardAction,qt.SIGNAL("activated()"),self.forward)
        self.connect(self.homeAction,qt.SIGNAL("activated()"),self.home)
        self.connect(self.moveAction,qt.SIGNAL("activated()"),self.pan)
        self.connect(self.zoomAction,qt.SIGNAL("activated()"),self.zoom)

        self.setCaption("Navigation toolbar")
        self.homeAction.setText("Home")
        self.homeAction.setMenuText("Home")
        self.backAction.setText("Back")
        self.backAction.setMenuText("Back")
        self.forwardAction.setText("Forward")
        self.forwardAction.setMenuText("Forward")
        self.moveAction.setText("Move / pan")
        self.moveAction.setMenuText("Move / pan")
        self.zoomAction.setText("Zoom")
        self.zoomAction.setMenuText("Zoom")
        self.fileSaveAction.setText("Save figure")
        self.fileSaveAction.setMenuText("Save figure")

    def set_message(self, s):
        self.window.statusBar().message(s)
        
    def save_figure(self):
        s = qt.QFileDialog.getSaveFileName("",
                                        "Images (*.png *.xpm *.jpg)",
                                        self.window,
                                        "save file dialog"
                                        "Choose a filename to save figure under" )
        if s:
            self.canvas.print_figure(str(s))
            
    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height()
        y0 =  height-y0
        y1 =  height-y1
        self.canvas.rect = (x0, y0, x1, y1)
        self.canvas.update()

    def release(self, event):
        self.canvas.rect = None


class FigureManagerQtAgg(FigureManagerBase):
    """
    Wrap everything up into a window for the pylab interface

    For non interactive backends, the base class does all the work
    """
    def __init__(self, canvas, num, window, toolbar):
        FigureManagerBase.__init__(self, canvas, num)
        self.window = window
        self.window.setCaption("Figure %d" % num)
        self.window.setCentralWidget(self.canvas)
        self.toolbar = toolbar
        
#    def destroy(self, *args):
#        print "destroy figure manager"
#        self.window.hide()
        
########################################################################
#    
# Now just provide the standard names that backend.__init__ is expecting
# 
########################################################################

def makeSureWeHaveAQApp():
    """This have to be seperated, so if the classes are used in
    in a program with another QApplication instance, we use that 
    one instead.
    """
    global myQtApp
    try:
        #Detect if a qApp exists.
        n = qt.qApp.name()
    except RuntimeError:
        myQtApp = qt.QApplication([])

FigureManager = FigureManagerQtAgg
error_msg = error_msg_qtagg

