from __future__ import division, print_function
import math
import os
import re
import signal
import sys

import matplotlib
from matplotlib import verbose
from matplotlib.cbook import is_string_like, onetrue
from matplotlib.backend_bases import RendererBase, GraphicsContextBase, \
     FigureManagerBase, FigureCanvasBase, NavigationToolbar2, IdleEvent, \
     cursors, TimerBase
from matplotlib.backend_bases import ShowBase

from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure
from matplotlib.mathtext import MathTextParser
from matplotlib.widgets import SubplotTool
try:
    import matplotlib.backends.qt4_editor.figureoptions as figureoptions
except ImportError:
    figureoptions = None

from qt4_compat import QtCore, QtGui, _getSaveFileName, __version__

backend_version = __version__
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
            figManager.canvas.draw_idle()

def _create_qApp():
    """
    Only one qApp can exist at a time, so check before creating one.
    """
    if QtGui.QApplication.startingUp():
        if DEBUG: print("Starting up QApplication")
        global qApp
        app = QtGui.QApplication.instance()
        if app is None:
          
            # check for DISPLAY env variable on X11 build of Qt
            if hasattr(QtGui, "QX11Info"):
                display = os.environ.get('DISPLAY')
                if display is None or not re.search(':\d', display):
                    raise RuntimeError('Invalid DISPLAY variable')
        
            qApp = QtGui.QApplication( [" "] )
            QtCore.QObject.connect( qApp, QtCore.SIGNAL( "lastWindowClosed()" ),
                                qApp, QtCore.SLOT( "quit()" ) )
        else:
            qApp = app

class Show(ShowBase):
    def mainloop(self):
        # allow KeyboardInterrupt exceptions to close the plot window.
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        QtGui.qApp.exec_()
show = Show()


def new_figure_manager( num, *args, **kwargs ):
    """
    Create a new figure manager instance
    """
    thisFig = Figure(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasQT(figure)
    manager = FigureManagerQT(canvas, num)
    return manager


class TimerQT(TimerBase):
    '''
    Subclass of :class:`backend_bases.TimerBase` that uses Qt4 timer events.

    Attributes:
    * interval: The time between timer events in milliseconds. Default
        is 1000 ms.
    * single_shot: Boolean flag indicating whether this timer should
        operate as single shot (run once and then stop). Defaults to False.
    * callbacks: Stores list of (func, args) tuples that will be called
        upon timer events. This list can be manipulated directly, or the
        functions add_callback and remove_callback can be used.
    '''
    def __init__(self, *args, **kwargs):
        TimerBase.__init__(self, *args, **kwargs)

        # Create a new timer and connect the timeout() signal to the
        # _on_timer method.
        self._timer = QtCore.QTimer()
        QtCore.QObject.connect(self._timer, QtCore.SIGNAL('timeout()'),
            self._on_timer)
        self._timer_set_interval()

    def __del__(self):
        # Probably not necessary in practice, but is good behavior to disconnect
        try:
            TimerBase.__del__(self)
            QtCore.QObject.disconnect(self._timer,
                    QtCore.SIGNAL('timeout()'), self._on_timer)
        except RuntimeError:
            # Timer C++ object already deleted
            pass

    def _timer_set_single_shot(self):
        self._timer.setSingleShot(self._single)

    def _timer_set_interval(self):
        self._timer.setInterval(self._interval)

    def _timer_start(self):
        self._timer.start()

    def _timer_stop(self):
        self._timer.stop()


class FigureCanvasQT( QtGui.QWidget, FigureCanvasBase ):
    keyvald = { QtCore.Qt.Key_Control : 'control',
                QtCore.Qt.Key_Shift : 'shift',
                QtCore.Qt.Key_Alt : 'alt',
                QtCore.Qt.Key_Meta : 'super',
                QtCore.Qt.Key_Return : 'enter',
                QtCore.Qt.Key_Left : 'left',
                QtCore.Qt.Key_Up : 'up',
                QtCore.Qt.Key_Right : 'right',
                QtCore.Qt.Key_Down : 'down',
                QtCore.Qt.Key_Escape : 'escape',
                QtCore.Qt.Key_F1 : 'f1',
                QtCore.Qt.Key_F2 : 'f2',
                QtCore.Qt.Key_F3 : 'f3',
                QtCore.Qt.Key_F4 : 'f4',
                QtCore.Qt.Key_F5 : 'f5',
                QtCore.Qt.Key_F6 : 'f6',
                QtCore.Qt.Key_F7 : 'f7',
                QtCore.Qt.Key_F8 : 'f8',
                QtCore.Qt.Key_F9 : 'f9',
                QtCore.Qt.Key_F10 : 'f10',
                QtCore.Qt.Key_F11 : 'f11',
                QtCore.Qt.Key_F12 : 'f12',
                QtCore.Qt.Key_Home : 'home',
                QtCore.Qt.Key_End : 'end',
                QtCore.Qt.Key_PageUp : 'pageup',
                QtCore.Qt.Key_PageDown : 'pagedown',
               }

    # define the modifier keys which are to be collected on keyboard events.
    # format is: [(modifier_flag, modifier_name, equivalent_key)
    _modifier_keys = [
                      (QtCore.Qt.MetaModifier, 'super', QtCore.Qt.Key_Meta),
                      (QtCore.Qt.AltModifier, 'alt', QtCore.Qt.Key_Alt),
                      (QtCore.Qt.ControlModifier, 'ctrl', QtCore.Qt.Key_Control)
                      ]

    _ctrl_modifier = QtCore.Qt.ControlModifier

    if sys.platform == 'darwin':
        # in OSX, the control and super (aka cmd/apple) keys are switched, so
        # switch them back.
        keyvald.update({
                        QtCore.Qt.Key_Control : 'super', # cmd/apple key
                        QtCore.Qt.Key_Meta : 'control',
                        })

        _modifier_keys = [
                          (QtCore.Qt.ControlModifier, 'super', QtCore.Qt.Key_Control),
                          (QtCore.Qt.AltModifier, 'alt', QtCore.Qt.Key_Alt),
                          (QtCore.Qt.MetaModifier, 'ctrl', QtCore.Qt.Key_Meta),
                         ]

        _ctrl_modifier = QtCore.Qt.MetaModifier

    # map Qt button codes to MouseEvent's ones:
    buttond = {QtCore.Qt.LeftButton  : 1,
               QtCore.Qt.MidButton   : 2,
               QtCore.Qt.RightButton : 3,
               # QtCore.Qt.XButton1 : None,
               # QtCore.Qt.XButton2 : None,
               }

    def __init__( self, figure ):
        if DEBUG: print('FigureCanvasQt: ', figure)
        _create_qApp()

        QtGui.QWidget.__init__( self )
        FigureCanvasBase.__init__( self, figure )
        self.figure = figure
        self.setMouseTracking( True )
        self._idle = True
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
        QtGui.QApplication.restoreOverrideCursor()
        FigureCanvasBase.leave_notify_event(self, event)

    def mousePressEvent( self, event ):
        x = event.pos().x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.pos().y()
        button = self.buttond.get(event.button())
        if button is not None:
            FigureCanvasBase.button_press_event( self, x, y, button )
        if DEBUG: print('button pressed:', event.button())

    def mouseDoubleClickEvent(self, event):
        x = event.pos().x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.pos().y()
        button = self.buttond.get(event.button())
        if button is not None:
            FigureCanvasBase.button_press_event( self, x, y, button, dblclick=True )
        if DEBUG: print ('button doubleclicked:', event.button())

    def mouseMoveEvent( self, event ):
        x = event.x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y()
        FigureCanvasBase.motion_notify_event( self, x, y )
        #if DEBUG: print('mouse move')

    def mouseReleaseEvent( self, event ):
        x = event.x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y()
        button = self.buttond.get(event.button())
        if button is not None:
            FigureCanvasBase.button_release_event( self, x, y, button )
        if DEBUG: print('button released')

    def wheelEvent( self, event ):
        x = event.x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y()
        # from QWheelEvent::delta doc
        steps = event.delta()/120
        if (event.orientation() == QtCore.Qt.Vertical):
            FigureCanvasBase.scroll_event( self, x, y, steps)
            if DEBUG: print('scroll event : delta = %i, steps = %i ' % (event.delta(),steps))

    def keyPressEvent( self, event ):
        key = self._get_key( event )
        if key is None:
            return
        FigureCanvasBase.key_press_event( self, key )
        if DEBUG: print('key press', key)

    def keyReleaseEvent( self, event ):
        key = self._get_key(event)
        if key is None:
            return
        FigureCanvasBase.key_release_event( self, key )
        if DEBUG: print('key release', key)

    def resizeEvent(self, event):
        w = event.size().width()
        h = event.size().height()
        if DEBUG:
            print('resize (%d x %d)' % (w, h))
            print("FigureCanvasQt.resizeEvent(%d, %d)" % (w, h))
        dpival = self.figure.dpi
        winch = w/dpival
        hinch = h/dpival
        self.figure.set_size_inches( winch, hinch )
        FigureCanvasBase.resize_event(self)
        self.draw()
        self.update()
        QtGui.QWidget.resizeEvent(self, event)

    def sizeHint( self ):
        w, h = self.get_width_height()
        return QtCore.QSize( w, h )

    def minumumSizeHint( self ):
        return QtCore.QSize( 10, 10 )

    def _get_key( self, event ):
        if event.isAutoRepeat():
            return None

        if event.key() < 256:
            key = unicode(event.text())
            # if the control key is being pressed, we don't get the correct
            # characters, so interpret them directly from the event.key().
            # Unfortunately, this means that we cannot handle key's case
            # since event.key() is not case sensitive, whereas event.text() is,
            # Finally, since it is not possible to get the CapsLock state
            # we cannot accurately compute the case of a pressed key when
            # ctrl+shift+p is pressed.
            if int(event.modifiers()) & self._ctrl_modifier:
                # we always get an uppercase character
                key = chr(event.key())
                # if shift is not being pressed, lowercase it (as mentioned,
                # this does not take into account the CapsLock state)
                if not int(event.modifiers()) & QtCore.Qt.ShiftModifier:
                    key = key.lower()

        else:
            key = self.keyvald.get(event.key())

        if key is not None:
            # prepend the ctrl, alt, super keys if appropriate (sorted in that order)
            for modifier, prefix, Qt_key in self._modifier_keys:
                if event.key() != Qt_key and int(event.modifiers()) & modifier == modifier:
                    key = u'{0}+{1}'.format(prefix, key)

        return key

    def new_timer(self, *args, **kwargs):
        """
        Creates a new backend-specific subclass of :class:`backend_bases.Timer`.
        This is useful for getting periodic events through the backend's native
        event loop. Implemented only for backends with GUIs.

        optional arguments:

        *interval*
          Timer interval in milliseconds
        *callbacks*
          Sequence of (func, args, kwargs) where func(*args, **kwargs) will
          be executed by the timer every *interval*.
        """
        return TimerQT(*args, **kwargs)

    def flush_events(self):
        QtGui.qApp.processEvents()

    def start_event_loop(self,timeout):
        FigureCanvasBase.start_event_loop_default(self,timeout)
    start_event_loop.__doc__=FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)
    stop_event_loop.__doc__=FigureCanvasBase.stop_event_loop_default.__doc__

    def draw_idle(self):
        'update drawing area only if idle'
        d = self._idle
        self._idle = False
        def idle_draw(*args):
            self.draw()
            self._idle = True
        if d: QtCore.QTimer.singleShot(0, idle_draw)


class MainWindow(QtGui.QMainWindow):
    def closeEvent(self, event):
        self.emit(QtCore.SIGNAL('closing()'))
        QtGui.QMainWindow.closeEvent(self, event)


class FigureManagerQT(FigureManagerBase):
    """
    Public attributes

    canvas      : The FigureCanvas instance
    num         : The Figure number
    toolbar     : The qt.QToolBar
    window      : The qt.QMainWindow
    """

    def __init__(self, canvas, num):
        if DEBUG:
            print('FigureManagerQT.%s' % fn_name())
        FigureManagerBase.__init__(self, canvas, num)
        self.canvas = canvas
        self.window = MainWindow()
        self.window.connect(self.window, QtCore.SIGNAL('closing()'),
                            canvas.close_event)
        self.window.connect(self.window, QtCore.SIGNAL('closing()'),
                            self._widgetclosed)

        self.window.setWindowTitle("Figure %d" % num)
        image = os.path.join(matplotlib.rcParams['datapath'], 'images', 'matplotlib.png')
        self.window.setWindowIcon(QtGui.QIcon(image))

        # Give the keyboard focus to the figure instead of the
        # manager; StrongFocus accepts both tab and click to focus and
        # will enable the canvas to process event w/o clicking.
        # ClickFocus only takes the focus is the window has been
        # clicked
        # on. http://qt-project.org/doc/qt-4.8/qt.html#FocusPolicy-enum or
        # http://doc.qt.digia.com/qt/qt.html#FocusPolicy-enum
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.setFocus()

        self.window._destroying = False

        self.toolbar = self._get_toolbar(self.canvas, self.window)
        if self.toolbar is not None:
            self.window.addToolBar(self.toolbar)
            QtCore.QObject.connect(self.toolbar, QtCore.SIGNAL("message"),
                                   self._show_message)
            tbs_height = self.toolbar.sizeHint().height()
        else:
            tbs_height = 0

        # resize the main window so it will display the canvas with the
        # requested size:
        cs = canvas.sizeHint()
        sbs = self.window.statusBar().sizeHint()
        self._status_and_tool_height = tbs_height + sbs.height()
        height = cs.height() + self._status_and_tool_height
        self.window.resize(cs.width(), height)

        self.window.setCentralWidget(self.canvas)

        if matplotlib.is_interactive():
            self.window.show()

        def notify_axes_change(fig):
            # This will be called whenever the current axes is changed
            if self.toolbar is not None:
                self.toolbar.update()
        self.canvas.figure.add_axobserver(notify_axes_change)

    @QtCore.Slot()
    def _show_message(self, s):
        # Fixes a PySide segfault.
        self.window.statusBar().showMessage(s)

    def full_screen_toggle(self):
        if self.window.isFullScreen():
            self.window.showNormal()
        else:
            self.window.showFullScreen()

    def _widgetclosed(self):
        if self.window._destroying:
            return
        self.window._destroying = True
        try:
            Gcf.destroy(self.num)
        except AttributeError:
            pass
            # It seems that when the python session is killed,
            # Gcf can get destroyed before the Gcf.destroy
            # line is run, leading to a useless AttributeError.

    def _get_toolbar(self, canvas, parent):
        # must be inited after the window, drawingArea and figure
        # attrs are set
        if matplotlib.rcParams['toolbar'] == 'classic':
            print("Classic toolbar is not supported")
        elif matplotlib.rcParams['toolbar'] == 'toolbar2':
            toolbar = NavigationToolbar2QT(canvas, parent, False)
        else:
            toolbar = None
        return toolbar

    def resize(self, width, height):
        'set the canvas size in pixels'
        self.window.resize(width, height + self._status_and_tool_height)

    def show(self):
        self.window.show()

    def destroy(self, *args):
        # check for qApp first, as PySide deletes it in its atexit handler
        if QtGui.QApplication.instance() is None:
            return
        if self.window._destroying:
            return
        self.window._destroying = True
        QtCore.QObject.disconnect(self.window, QtCore.SIGNAL('destroyed()'),
                                  self._widgetclosed)
        if self.toolbar:
                self.toolbar.destroy()
        if DEBUG:
                print("destroy figure manager")
        self.window.close()

    def get_window_title(self):
        return str(self.window.windowTitle())

    def set_window_title(self, title):
        self.window.setWindowTitle(title)

class NavigationToolbar2QT( NavigationToolbar2, QtGui.QToolBar ):
    def __init__(self, canvas, parent, coordinates=True):
        """ coordinates: should we show the coordinates on the right? """
        self.canvas = canvas
        self.coordinates = coordinates
        self._actions = {}
        """A mapping of toolitem method names to their QActions"""

        QtGui.QToolBar.__init__( self, parent )
        NavigationToolbar2.__init__( self, canvas )

    def _icon(self, name):
        return QtGui.QIcon(os.path.join(self.basedir, name))

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams[ 'datapath' ],'images')

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.addSeparator()
            else:
                a = self.addAction(self._icon(image_file + '.png'),
                                         text, getattr(self, callback))
                self._actions[callback] = a
                if callback in ['zoom', 'pan']:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)

        if figureoptions is not None:
            a = self.addAction(self._icon("qt4_editor_options.png"),
                               'Customize', self.edit_parameters)
            a.setToolTip('Edit curves line and axes parameters')

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

    if figureoptions is not None:
        def edit_parameters(self):
            allaxes = self.canvas.figure.get_axes()
            if len(allaxes) == 1:
                axes = allaxes[0]
            else:
                titles = []
                for axes in allaxes:
                    title = axes.get_title()
                    ylabel = axes.get_ylabel()
                    if title:
                        fmt = "%(title)s"
                        if ylabel:
                            fmt += ": %(ylabel)s"
                        fmt += " (%(axes_repr)s)"
                    elif ylabel:
                        fmt = "%(axes_repr)s (%(ylabel)s)"
                    else:
                        fmt = "%(axes_repr)s"
                    titles.append(fmt % dict(title = title,
                                         ylabel = ylabel,
                                         axes_repr = repr(axes)))
                item, ok = QtGui.QInputDialog.getItem(self, 'Customize',
                                                      'Select axes:', titles,
                                                      0, False)
                if ok:
                    axes = allaxes[titles.index(unicode(item))]
                else:
                    return

            figureoptions.figure_edit(axes, self)

    def _update_buttons_checked(self):
        #sync button checkstates to match active mode
        self._actions['pan'].setChecked(self._active == 'PAN')
        self._actions['zoom'].setChecked(self._active == 'ZOOM')

    def pan(self, *args):
        super(NavigationToolbar2QT, self).pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        super(NavigationToolbar2QT, self).zoom(*args)
        self._update_buttons_checked()

    def dynamic_update( self ):
        self.canvas.draw()

    def set_message( self, s ):
        self.emit(QtCore.SIGNAL("message"), s)
        if self.coordinates:
            self.locLabel.setText(s.replace(', ', '\n'))

    def set_cursor( self, cursor ):
        if DEBUG: print('Set cursor' , cursor)
        self.canvas.setCursor(cursord[cursor])

    def draw_rubberband( self, event, x0, y0, x1, y1 ):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0

        w = abs(x1 - x0)
        h = abs(y1 - y0)

        rect = [ int(val)for val in (min(x0,x1), min(y0, y1), w, h) ]
        self.canvas.drawRectangle( rect )

    def configure_subplots(self):
        self.adj_window = QtGui.QMainWindow()
        win = self.adj_window

        win.setWindowTitle("Subplot Configuration Tool")
        image = os.path.join( matplotlib.rcParams['datapath'],'images','matplotlib.png' )
        win.setWindowIcon(QtGui.QIcon( image ))

        tool = SubplotToolQt(self.canvas.figure, win)
        win.setCentralWidget(tool)
        win.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)

        win.show()

    def _get_canvas(self, fig):
        return FigureCanvasQT(fig)

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = filetypes.items()
        sorted_filetypes.sort()
        default_filetype = self.canvas.get_default_filetype()

        startpath = matplotlib.rcParams.get('savefig.directory', '')
        startpath = os.path.expanduser(startpath)
        start = os.path.join(startpath, self.canvas.get_default_filename())
        filters = []
        selectedFilter = None
        for name, exts in sorted_filetypes:
            exts_list = " ".join(['*.%s' % ext for ext in exts])
            filter = '%s (%s)' % (name, exts_list)
            if default_filetype in exts:
                selectedFilter = filter
            filters.append(filter)
        filters = ';;'.join(filters)
        fname = _getSaveFileName(self, "Choose a filename to save to",
                                        start, filters, selectedFilter)
        if fname:
            if startpath == '':
                # explicitly missing key or empty str signals to use cwd
                matplotlib.rcParams['savefig.directory'] = startpath
            else:
                # save dir for next time
                matplotlib.rcParams['savefig.directory'] = os.path.dirname(unicode(fname))
            try:
                self.canvas.print_figure( unicode(fname) )
            except Exception as e:
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
