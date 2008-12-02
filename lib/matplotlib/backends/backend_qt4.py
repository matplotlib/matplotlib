from __future__ import division
import math
import os
import sys

import matplotlib
from matplotlib import verbose
from matplotlib.cbook import is_string_like, onetrue
from matplotlib.backend_bases import RendererBase, GraphicsContextBase, \
     FigureManagerBase, FigureCanvasBase, NavigationToolbar2, IdleEvent, cursors
from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure
from matplotlib.mathtext import MathTextParser
from matplotlib.widgets import SubplotTool

try:
    from PyQt4 import QtCore, QtGui, Qt
except ImportError:
    raise ImportError("Qt4 backend requires that PyQt4 is installed.")

backend_version = "0.9.1"
def fn_name(): return sys._getframe(1).f_code.co_name

DEBUG = False

cursord = {
    cursors.MOVE          : QtCore.Qt.SizeAllCursor,
    cursors.HAND          : QtCore.Qt.PointingHandCursor,
    cursors.POINTER       : QtCore.Qt.ArrowCursor,
    cursors.SELECT_REGION : QtCore.Qt.CrossCursor,
    }

def draw_if_interactive():
    """
    Is called after every pylab drawing command
    """
    if matplotlib.is_interactive():
        figManager =  Gcf.get_active()
        if figManager != None:
            figManager.canvas.draw()

def _create_qApp():
    """
    Only one qApp can exist at a time, so check before creating one.
    """
    if QtGui.QApplication.startingUp():
        if DEBUG: print "Starting up QApplication"
        global qApp
        qApp = QtGui.QApplication( [" "] )
        QtCore.QObject.connect( qApp, QtCore.SIGNAL( "lastWindowClosed()" ),
                            qApp, QtCore.SLOT( "quit()" ) )
        #remember that matplotlib created the qApp - will be used by show()
        _create_qApp.qAppCreatedHere = True

_create_qApp.qAppCreatedHere = False

def show():
    """
    Show all the figures and enter the qt main loop
    This should be the last line of your script
    """
    for manager in Gcf.get_all_fig_managers():
        manager.window.show()

    if DEBUG: print 'Inside show'

    figManager =  Gcf.get_active()
    if figManager != None:
        figManager.canvas.draw()

    if _create_qApp.qAppCreatedHere:
        QtGui.qApp.exec_()


def new_figure_manager( num, *args, **kwargs ):
    """
    Create a new figure manager instance
    """
    thisFig = Figure( *args, **kwargs )
    canvas = FigureCanvasQT( thisFig )
    manager = FigureManagerQT( canvas, num )
    return manager


class FigureCanvasQT( QtGui.QWidget, FigureCanvasBase ):
    keyvald = { QtCore.Qt.Key_Control : 'control',
                QtCore.Qt.Key_Shift : 'shift',
                QtCore.Qt.Key_Alt : 'alt',
               }
    # left 1, middle 2, right 3
    buttond = {1:1, 2:3, 4:2}
    def __init__( self, figure ):
        if DEBUG: print 'FigureCanvasQt: ', figure
        _create_qApp()

        QtGui.QWidget.__init__( self )
        FigureCanvasBase.__init__( self, figure )
        self.figure = figure
        self.setMouseTracking( True )
        # hide until we can test and fix
        #self.startTimer(backend_IdleEvent.milliseconds)
        w,h = self.get_width_height()
        self.resize( w, h )

    def __timerEvent(self, event):
        # hide until we can test and fix
        self.mpl_idle_event(event)

    def enterEvent(self, event):
        FigureCanvasBase.enter_notify_event(self, event)

    def leaveEvent(self, event):
        FigureCanvasBase.leave_notify_event(self, event)

    def mousePressEvent( self, event ):
        x = event.pos().x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.pos().y()
        button = self.buttond[event.button()]
        FigureCanvasBase.button_press_event( self, x, y, button )
        if DEBUG: print 'button pressed:', event.button()

    def mouseMoveEvent( self, event ):
        x = event.x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y()
        FigureCanvasBase.motion_notify_event( self, x, y )
        #if DEBUG: print 'mouse move'

    def mouseReleaseEvent( self, event ):
        x = event.x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y()
        button = self.buttond[event.button()]
        FigureCanvasBase.button_release_event( self, x, y, button )
        if DEBUG: print 'button released'

    def keyPressEvent( self, event ):
        key = self._get_key( event )
        FigureCanvasBase.key_press_event( self, key )
        if DEBUG: print 'key press', key

    def keyReleaseEvent( self, event ):
        key = self._get_key(event)
        FigureCanvasBase.key_release_event( self, key )
        if DEBUG: print 'key release', key

    def resizeEvent( self, event ):
        if DEBUG: print 'resize (%d x %d)' % (event.size().width(), event.size().height())
        QtGui.QWidget.resizeEvent( self, event )
        w = event.size().width()
        h = event.size().height()
        if DEBUG: print "FigureCanvasQtAgg.resizeEvent(", w, ",", h, ")"
        dpival = self.figure.dpi
        winch = w/dpival
        hinch = h/dpival
        self.figure.set_size_inches( winch, hinch )
        self.draw()

    def resize( self, w, h ):
        # Pass through to Qt to resize the widget.
        QtGui.QWidget.resize( self, w, h )

        # Resize the figure by converting pixels to inches.
        pixelPerInch = self.figure.dpi
        wInch = w / pixelPerInch
        hInch = h / pixelPerInch
        self.figure.set_size_inches( wInch, hInch )

        # Redraw everything.
        self.draw()

    def sizeHint( self ):
        w, h = self.get_width_height()
        return QtCore.QSize( w, h )

    def minumumSizeHint( self ):
        return QtCore.QSize( 10, 10 )

    def _get_key( self, event ):
        if event.key() < 256:
            key = str(event.text())
        elif event.key() in self.keyvald:
            key = self.keyvald[ event.key() ]
        else:
            key = None

        return key

    def flush_events(self):
        Qt.qApp.processEvents()

    def start_event_loop(self,timeout):
        FigureCanvasBase.start_event_loop_default(self,timeout)
    start_event_loop.__doc__=FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)
    stop_event_loop.__doc__=FigureCanvasBase.stop_event_loop_default.__doc__

class FigureManagerQT( FigureManagerBase ):
    """
    Public attributes

    canvas      : The FigureCanvas instance
    num         : The Figure number
    toolbar     : The qt.QToolBar
    window      : The qt.QMainWindow
    """

    def __init__( self, canvas, num ):
        if DEBUG: print 'FigureManagerQT.%s' % fn_name()
        FigureManagerBase.__init__( self, canvas, num )
        self.canvas = canvas
        self.window = QtGui.QMainWindow()
        self.window.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.window.setWindowTitle("Figure %d" % num)
        image = os.path.join( matplotlib.rcParams['datapath'],'images','matplotlib.png' )
        self.window.setWindowIcon(QtGui.QIcon( image ))

        # Give the keyboard focus to the figure instead of the manager
        self.canvas.setFocusPolicy( QtCore.Qt.ClickFocus )
        self.canvas.setFocus()

        QtCore.QObject.connect( self.window, QtCore.SIGNAL( 'destroyed()' ),
                            self._widgetclosed )
        self.window._destroying = False

        self.toolbar = self._get_toolbar(self.canvas, self.window)
        self.window.addToolBar(self.toolbar)
        QtCore.QObject.connect(self.toolbar, QtCore.SIGNAL("message"),
                self.window.statusBar().showMessage)

        self.window.setCentralWidget(self.canvas)

        if matplotlib.is_interactive():
            self.window.show()

        # attach a show method to the figure for pylab ease of use
        self.canvas.figure.show = lambda *args: self.window.show()

        def notify_axes_change( fig ):
           # This will be called whenever the current axes is changed
           if self.toolbar != None: self.toolbar.update()
        self.canvas.figure.add_axobserver( notify_axes_change )

    def _widgetclosed( self ):
        if self.window._destroying: return
        self.window._destroying = True
        Gcf.destroy(self.num)

    def _get_toolbar(self, canvas, parent):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar'] == 'classic':
            print "Classic toolbar is not supported"
        elif matplotlib.rcParams['toolbar'] == 'toolbar2':
            toolbar = NavigationToolbar2QT(canvas, parent, False)
        else:
            toolbar = None
        return toolbar

    def resize(self, width, height):
        'set the canvas size in pixels'
        self.window.resize(width, height)

    def destroy( self, *args ):
        if self.window._destroying: return
        self.window._destroying = True
        QtCore.QObject.disconnect( self.window, QtCore.SIGNAL( 'destroyed()' ),
                                   self._widgetclosed )
        if self.toolbar: self.toolbar.destroy()
        if DEBUG: print "destroy figure manager"
        self.window.close()

    def set_window_title(self, title):
        self.window.setWindowTitle(title)

class NavigationToolbar2QT( NavigationToolbar2, QtGui.QToolBar ):
    def __init__(self, canvas, parent, coordinates=True):
        """ coordinates: should we show the coordinates on the right? """
        self.canvas = canvas
        self.coordinates = coordinates
        QtGui.QToolBar.__init__( self, parent )
        NavigationToolbar2.__init__( self, canvas )

    def _icon(self, name):
        return QtGui.QIcon(os.path.join(self.basedir, name))

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams[ 'datapath' ],'images')

        a = self.addAction(self._icon('home.svg'), 'Home', self.home)
        a.setToolTip('Reset original view')
        a = self.addAction(self._icon('back.svg'), 'Back', self.back)
        a.setToolTip('Back to previous view')
        a = self.addAction(self._icon('forward.svg'), 'Forward', self.forward)
        a.setToolTip('Forward to next view')
        self.addSeparator()
        a = self.addAction(self._icon('move.svg'), 'Pan', self.pan)
        a.setToolTip('Pan axes with left mouse, zoom with right')
        a = self.addAction(self._icon('zoom_to_rect.svg'), 'Zoom', self.zoom)
        a.setToolTip('Zoom to rectangle')
        self.addSeparator()
        a = self.addAction(self._icon('subplots.png'), 'Subplots',
                self.configure_subplots)
        a.setToolTip('Configure subplots')
        a = self.addAction(self._icon('filesave.svg'), 'Save',
                self.save_figure)
        a.setToolTip('Save the figure')

        self.buttons = {}

        # Add the x,y location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = QtGui.QLabel( "", self )
            self.locLabel.setAlignment(
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignTop )
            self.locLabel.setSizePolicy(
                QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding,
                                  QtGui.QSizePolicy.Ignored))
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)

        # reference holder for subplots_adjust window
        self.adj_window = None

    def dynamic_update( self ):
        self.canvas.draw()

    def set_message( self, s ):
        self.emit(QtCore.SIGNAL("message"), s)
        if self.coordinates:
            self.locLabel.setText(s.replace(', ', '\n'))

    def set_cursor( self, cursor ):
        if DEBUG: print 'Set cursor' , cursor
        QtGui.QApplication.restoreOverrideCursor()
        QtGui.QApplication.setOverrideCursor( QtGui.QCursor( cursord[cursor] ) )

    def draw_rubberband( self, event, x0, y0, x1, y1 ):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0

        w = abs(x1 - x0)
        h = abs(y1 - y0)

        rect = [ int(val)for val in min(x0,x1), min(y0, y1), w, h ]
        self.canvas.drawRectangle( rect )

    def configure_subplots(self):
        self.adj_window = QtGui.QMainWindow()
        win = self.adj_window
        win.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        win.setWindowTitle("Subplot Configuration Tool")
        image = os.path.join( matplotlib.rcParams['datapath'],'images','matplotlib.png' )
        win.setWindowIcon(QtGui.QIcon( image ))

        tool = SubplotToolQt(self.canvas.figure, win)
        win.setCentralWidget(tool)
        win.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)

        win.show()

    def _get_canvas(self, fig):
        return FigureCanvasQT(fig)

    def save_figure( self ):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = filetypes.items()
        sorted_filetypes.sort()
        default_filetype = self.canvas.get_default_filetype()

        start = "image." + default_filetype
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)

        fname = QtGui.QFileDialog.getSaveFileName(
            self, "Choose a filename to save to", start, filters, selectedFilter)
        if fname:
            try:
                self.canvas.print_figure( unicode(fname) )
            except Exception, e:
                QtGui.QMessageBox.critical(
                    self, "Error saving file", str(e),
                    QtGui.QMessageBox.Ok, QtGui.QMessageBox.NoButton)



class SubplotToolQt( SubplotTool, QtGui.QWidget ):
    def __init__(self, targetfig, parent):
        QtGui.QWidget.__init__(self, None)

        self.targetfig = targetfig
        self.parent = parent

        self.sliderleft = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sliderbottom = QtGui.QSlider(QtCore.Qt.Vertical)
        self.sliderright = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.slidertop = QtGui.QSlider(QtCore.Qt.Vertical)
        self.sliderwspace = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sliderhspace = QtGui.QSlider(QtCore.Qt.Vertical)

        # constraints
        QtCore.QObject.connect( self.sliderleft,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.sliderright.setMinimum )
        QtCore.QObject.connect( self.sliderright,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.sliderleft.setMaximum )
        QtCore.QObject.connect( self.sliderbottom,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.slidertop.setMinimum )
        QtCore.QObject.connect( self.slidertop,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.sliderbottom.setMaximum )

        sliders = (self.sliderleft, self.sliderbottom, self.sliderright,
                   self.slidertop, self.sliderwspace, self.sliderhspace, )
        adjustments = ('left:', 'bottom:', 'right:', 'top:', 'wspace:', 'hspace:')

        for slider, adjustment in zip(sliders, adjustments):
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setSingleStep(5)

        layout = QtGui.QGridLayout()

        leftlabel = QtGui.QLabel('left')
        layout.addWidget(leftlabel, 2, 0)
        layout.addWidget(self.sliderleft, 2, 1)

        toplabel = QtGui.QLabel('top')
        layout.addWidget(toplabel, 0, 2)
        layout.addWidget(self.slidertop, 1, 2)
        layout.setAlignment(self.slidertop, QtCore.Qt.AlignHCenter)

        bottomlabel = QtGui.QLabel('bottom')
        layout.addWidget(QtGui.QLabel('bottom'), 4, 2)
        layout.addWidget(self.sliderbottom, 3, 2)
        layout.setAlignment(self.sliderbottom, QtCore.Qt.AlignHCenter)

        rightlabel = QtGui.QLabel('right')
        layout.addWidget(rightlabel, 2, 4)
        layout.addWidget(self.sliderright, 2, 3)

        hspacelabel = QtGui.QLabel('hspace')
        layout.addWidget(hspacelabel, 0, 6)
        layout.setAlignment(hspacelabel, QtCore.Qt.AlignHCenter)
        layout.addWidget(self.sliderhspace, 1, 6)
        layout.setAlignment(self.sliderhspace, QtCore.Qt.AlignHCenter)

        wspacelabel = QtGui.QLabel('wspace')
        layout.addWidget(wspacelabel, 4, 6)
        layout.setAlignment(wspacelabel, QtCore.Qt.AlignHCenter)
        layout.addWidget(self.sliderwspace, 3, 6)
        layout.setAlignment(self.sliderwspace, QtCore.Qt.AlignBottom)

        layout.setRowStretch(1,1)
        layout.setRowStretch(3,1)
        layout.setColumnStretch(1,1)
        layout.setColumnStretch(3,1)
        layout.setColumnStretch(6,1)

        self.setLayout(layout)

        self.sliderleft.setSliderPosition(int(targetfig.subplotpars.left*1000))
        self.sliderbottom.setSliderPosition(\
                                    int(targetfig.subplotpars.bottom*1000))
        self.sliderright.setSliderPosition(\
                                    int(targetfig.subplotpars.right*1000))
        self.slidertop.setSliderPosition(int(targetfig.subplotpars.top*1000))
        self.sliderwspace.setSliderPosition(\
                                    int(targetfig.subplotpars.wspace*1000))
        self.sliderhspace.setSliderPosition(\
                                    int(targetfig.subplotpars.hspace*1000))

        QtCore.QObject.connect( self.sliderleft,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.funcleft )
        QtCore.QObject.connect( self.sliderbottom,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.funcbottom )
        QtCore.QObject.connect( self.sliderright,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.funcright )
        QtCore.QObject.connect( self.slidertop,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.functop )
        QtCore.QObject.connect( self.sliderwspace,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.funcwspace )
        QtCore.QObject.connect( self.sliderhspace,
                                QtCore.SIGNAL( "valueChanged(int)" ),
                                self.funchspace )

    def funcleft(self, val):
        if val == self.sliderright.value():
            val -= 1
        self.targetfig.subplots_adjust(left=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def funcright(self, val):
        if val == self.sliderleft.value():
            val += 1
        self.targetfig.subplots_adjust(right=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def funcbottom(self, val):
        if val == self.slidertop.value():
            val -= 1
        self.targetfig.subplots_adjust(bottom=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def functop(self, val):
        if val == self.sliderbottom.value():
            val += 1
        self.targetfig.subplots_adjust(top=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def funcwspace(self, val):
        self.targetfig.subplots_adjust(wspace=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()

    def funchspace(self, val):
        self.targetfig.subplots_adjust(hspace=val/1000.)
        if self.drawon: self.targetfig.canvas.draw()


def error_msg_qt( msg, parent=None ):
    if not is_string_like( msg ):
        msg = ','.join( map( str,msg ) )

    QtGui.QMessageBox.warning( None, "Matplotlib", msg, QtGui.QMessageBox.Ok )

def exception_handler( type, value, tb ):
    """Handle uncaught exceptions
    It does not catch SystemExit
    """
    msg = ''
    # get the filename attribute if available (for IOError)
    if hasattr(value, 'filename') and value.filename != None:
        msg = value.filename + ': '
    if hasattr(value, 'strerror') and value.strerror != None:
        msg += value.strerror
    else:
        msg += str(value)

    if len( msg ) : error_msg_qt( msg )


FigureManager = FigureManagerQT
