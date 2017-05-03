from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

import os
import re
import signal
import sys
from six import unichr

import matplotlib

from matplotlib.cbook import is_string_like
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backend_bases import NavigationToolbar2

from matplotlib.backend_bases import cursors
from matplotlib.backend_bases import TimerBase
from matplotlib.backend_bases import ShowBase

from matplotlib._pylab_helpers import Gcf
from matplotlib.figure import Figure

from matplotlib.widgets import SubplotTool
import matplotlib.backends.qt_editor.figureoptions as figureoptions

from .qt_compat import (QtCore, QtGui, QtWidgets, _getSaveFileName,
                        __version__, is_pyqt5)
from matplotlib.backends.qt_editor.formsubplottool import UiSubplotTool

backend_version = __version__

# SPECIAL_KEYS are keys that do *not* return their unicode name
# instead they have manually specified names
SPECIAL_KEYS = {QtCore.Qt.Key_Control: 'control',
                QtCore.Qt.Key_Shift: 'shift',
                QtCore.Qt.Key_Alt: 'alt',
                QtCore.Qt.Key_Meta: 'super',
                QtCore.Qt.Key_Return: 'enter',
                QtCore.Qt.Key_Left: 'left',
                QtCore.Qt.Key_Up: 'up',
                QtCore.Qt.Key_Right: 'right',
                QtCore.Qt.Key_Down: 'down',
                QtCore.Qt.Key_Escape: 'escape',
                QtCore.Qt.Key_F1: 'f1',
                QtCore.Qt.Key_F2: 'f2',
                QtCore.Qt.Key_F3: 'f3',
                QtCore.Qt.Key_F4: 'f4',
                QtCore.Qt.Key_F5: 'f5',
                QtCore.Qt.Key_F6: 'f6',
                QtCore.Qt.Key_F7: 'f7',
                QtCore.Qt.Key_F8: 'f8',
                QtCore.Qt.Key_F9: 'f9',
                QtCore.Qt.Key_F10: 'f10',
                QtCore.Qt.Key_F11: 'f11',
                QtCore.Qt.Key_F12: 'f12',
                QtCore.Qt.Key_Home: 'home',
                QtCore.Qt.Key_End: 'end',
                QtCore.Qt.Key_PageUp: 'pageup',
                QtCore.Qt.Key_PageDown: 'pagedown',
                QtCore.Qt.Key_Tab: 'tab',
                QtCore.Qt.Key_Backspace: 'backspace',
                QtCore.Qt.Key_Enter: 'enter',
                QtCore.Qt.Key_Insert: 'insert',
                QtCore.Qt.Key_Delete: 'delete',
                QtCore.Qt.Key_Pause: 'pause',
                QtCore.Qt.Key_SysReq: 'sysreq',
                QtCore.Qt.Key_Clear: 'clear', }

# define which modifier keys are collected on keyboard events.
# elements are (mpl names, Modifier Flag, Qt Key) tuples
SUPER = 0
ALT = 1
CTRL = 2
SHIFT = 3
MODIFIER_KEYS = [('super', QtCore.Qt.MetaModifier, QtCore.Qt.Key_Meta),
                 ('alt', QtCore.Qt.AltModifier, QtCore.Qt.Key_Alt),
                 ('ctrl', QtCore.Qt.ControlModifier, QtCore.Qt.Key_Control),
                 ('shift', QtCore.Qt.ShiftModifier, QtCore.Qt.Key_Shift),
                 ]

if sys.platform == 'darwin':
    # in OSX, the control and super (aka cmd/apple) keys are switched, so
    # switch them back.
    SPECIAL_KEYS.update({QtCore.Qt.Key_Control: 'super',  # cmd/apple key
                         QtCore.Qt.Key_Meta: 'control',
                         })
    MODIFIER_KEYS[0] = ('super', QtCore.Qt.ControlModifier,
                        QtCore.Qt.Key_Control)
    MODIFIER_KEYS[2] = ('ctrl', QtCore.Qt.MetaModifier,
                        QtCore.Qt.Key_Meta)


def fn_name():
    return sys._getframe(1).f_code.co_name

DEBUG = False

cursord = {
    cursors.MOVE: QtCore.Qt.SizeAllCursor,
    cursors.HAND: QtCore.Qt.PointingHandCursor,
    cursors.POINTER: QtCore.Qt.ArrowCursor,
    cursors.SELECT_REGION: QtCore.Qt.CrossCursor,
    }


def draw_if_interactive():
    """
    Is called after every pylab drawing command
    """
    if matplotlib.is_interactive():
        figManager = Gcf.get_active()
        if figManager is not None:
            figManager.canvas.draw_idle()

# make place holder
qApp = None


def _create_qApp():
    """
    Only one qApp can exist at a time, so check before creating one.
    """
    global qApp

    if qApp is None:
        if DEBUG:
            print("Starting up QApplication")
        app = QtWidgets.QApplication.instance()
        if app is None:
            # check for DISPLAY env variable on X11 build of Qt
            if is_pyqt5():
                try:
                    from PyQt5 import QtX11Extras
                    is_x11_build = True
                except ImportError:
                    is_x11_build = False
            else:
                is_x11_build = hasattr(QtGui, "QX11Info")
            if is_x11_build:
                display = os.environ.get('DISPLAY')
                if display is None or not re.search(r':\d', display):
                    raise RuntimeError('Invalid DISPLAY variable')

            qApp = QtWidgets.QApplication([str(" ")])
            qApp.lastWindowClosed.connect(qApp.quit)
        else:
            qApp = app

    if is_pyqt5():
        try:
            qApp.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
            qApp.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        except AttributeError:
            pass


class Show(ShowBase):
    def mainloop(self):
        # allow KeyboardInterrupt exceptions to close the plot window.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        global qApp
        qApp.exec_()


show = Show()


def new_figure_manager(num, *args, **kwargs):
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
        self._timer.timeout.connect(self._on_timer)
        self._timer_set_interval()

    def _timer_set_single_shot(self):
        self._timer.setSingleShot(self._single)

    def _timer_set_interval(self):
        self._timer.setInterval(self._interval)

    def _timer_start(self):
        self._timer.start()

    def _timer_stop(self):
        self._timer.stop()


class FigureCanvasQT(QtWidgets.QWidget, FigureCanvasBase):

    # map Qt button codes to MouseEvent's ones:
    buttond = {QtCore.Qt.LeftButton: 1,
               QtCore.Qt.MidButton: 2,
               QtCore.Qt.RightButton: 3,
               # QtCore.Qt.XButton1: None,
               # QtCore.Qt.XButton2: None,
               }

    def __init__(self, figure):
        if DEBUG:
            print('FigureCanvasQt qt5: ', figure)
        _create_qApp()

        # NB: Using super for this call to avoid a TypeError:
        # __init__() takes exactly 2 arguments (1 given) on QWidget
        # PyQt5
        # The need for this change is documented here
        # http://pyqt.sourceforge.net/Docs/PyQt5/pyqt4_differences.html#cooperative-multi-inheritance
        super(FigureCanvasQT, self).__init__(figure=figure)
        self.figure = figure
        self.setMouseTracking(True)
        w, h = self.get_width_height()
        self.resize(w, h)

    @property
    def _dpi_ratio(self):
        # Not available on Qt4 or some older Qt5.
        try:
            return self.devicePixelRatio()
        except AttributeError:
            return 1

    def get_width_height(self):
        w, h = FigureCanvasBase.get_width_height(self)
        return int(w / self._dpi_ratio), int(h / self._dpi_ratio)

    def enterEvent(self, event):
        FigureCanvasBase.enter_notify_event(self, guiEvent=event)

    def leaveEvent(self, event):
        QtWidgets.QApplication.restoreOverrideCursor()
        FigureCanvasBase.leave_notify_event(self, guiEvent=event)

    def mouseEventCoords(self, pos):
        """Calculate mouse coordinates in physical pixels

        Qt5 use logical pixels, but the figure is scaled to physical
        pixels for rendering.   Transform to physical pixels so that
        all of the down-stream transforms work as expected.

        Also, the origin is different and needs to be corrected.

        """
        dpi_ratio = self._dpi_ratio
        x = pos.x()
        # flip y so y=0 is bottom of canvas
        y = self.figure.bbox.height / dpi_ratio - pos.y()
        return x * dpi_ratio, y * dpi_ratio

    def mousePressEvent(self, event):
        x, y = self.mouseEventCoords(event.pos())
        button = self.buttond.get(event.button())
        if button is not None:
            FigureCanvasBase.button_press_event(self, x, y, button,
                                                guiEvent=event)
        if DEBUG:
            print('button pressed:', event.button())

    def mouseDoubleClickEvent(self, event):
        x, y = self.mouseEventCoords(event.pos())
        button = self.buttond.get(event.button())
        if button is not None:
            FigureCanvasBase.button_press_event(self, x, y,
                                                button, dblclick=True,
                                                guiEvent=event)
        if DEBUG:
            print('button doubleclicked:', event.button())

    def mouseMoveEvent(self, event):
        x, y = self.mouseEventCoords(event)
        FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=event)
        # if DEBUG: print('mouse move')

    def mouseReleaseEvent(self, event):
        x, y = self.mouseEventCoords(event)
        button = self.buttond.get(event.button())
        if button is not None:
            FigureCanvasBase.button_release_event(self, x, y, button,
                                                  guiEvent=event)
        if DEBUG:
            print('button released')

    def wheelEvent(self, event):
        x, y = self.mouseEventCoords(event)
        # from QWheelEvent::delta doc
        if event.pixelDelta().x() == 0 and event.pixelDelta().y() == 0:
            steps = event.angleDelta().y() / 120
        else:
            steps = event.pixelDelta().y()

        if steps != 0:
            FigureCanvasBase.scroll_event(self, x, y, steps, guiEvent=event)
            if DEBUG:
                print('scroll event: delta = %i, '
                      'steps = %i ' % (event.delta(), steps))

    def keyPressEvent(self, event):
        key = self._get_key(event)
        if key is None:
            return
        FigureCanvasBase.key_press_event(self, key, guiEvent=event)
        if DEBUG:
            print('key press', key)

    def keyReleaseEvent(self, event):
        key = self._get_key(event)
        if key is None:
            return
        FigureCanvasBase.key_release_event(self, key, guiEvent=event)
        if DEBUG:
            print('key release', key)

    def resizeEvent(self, event):
        w = event.size().width() * self._dpi_ratio
        h = event.size().height() * self._dpi_ratio
        if DEBUG:
            print('resize (%d x %d)' % (w, h))
            print("FigureCanvasQt.resizeEvent(%d, %d)" % (w, h))
        dpival = self.figure.dpi
        winch = w / dpival
        hinch = h / dpival
        self.figure.set_size_inches(winch, hinch, forward=False)
        FigureCanvasBase.resize_event(self)
        self.draw_idle()
        QtWidgets.QWidget.resizeEvent(self, event)

    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minumumSizeHint(self):
        return QtCore.QSize(10, 10)

    def _get_key(self, event):
        if event.isAutoRepeat():
            return None

        event_key = event.key()
        event_mods = int(event.modifiers())  # actually a bitmask

        # get names of the pressed modifier keys
        # bit twiddling to pick out modifier keys from event_mods bitmask,
        # if event_key is a MODIFIER, it should not be duplicated in mods
        mods = [name for name, mod_key, qt_key in MODIFIER_KEYS
                if event_key != qt_key and (event_mods & mod_key) == mod_key]
        try:
            # for certain keys (enter, left, backspace, etc) use a word for the
            # key, rather than unicode
            key = SPECIAL_KEYS[event_key]
        except KeyError:
            # unicode defines code points up to 0x0010ffff
            # QT will use Key_Codes larger than that for keyboard keys that are
            # are not unicode characters (like multimedia keys)
            # skip these
            # if you really want them, you should add them to SPECIAL_KEYS
            MAX_UNICODE = 0x10ffff
            if event_key > MAX_UNICODE:
                return None

            key = unichr(event_key)
            # qt delivers capitalized letters.  fix capitalization
            # note that capslock is ignored
            if 'shift' in mods:
                mods.remove('shift')
            else:
                key = key.lower()

        mods.reverse()
        return '+'.join(mods + [key])

    def new_timer(self, *args, **kwargs):
        """
        Creates a new backend-specific subclass of
        :class:`backend_bases.Timer`.  This is useful for getting
        periodic events through the backend's native event
        loop. Implemented only for backends with GUIs.

        optional arguments:

        *interval*
            Timer interval in milliseconds

        *callbacks*
            Sequence of (func, args, kwargs) where func(*args, **kwargs)
            will be executed by the timer every *interval*.

    """
        return TimerQT(*args, **kwargs)

    def flush_events(self):
        global qApp
        qApp.processEvents()

    def start_event_loop(self, timeout):
        FigureCanvasBase.start_event_loop_default(self, timeout)

    start_event_loop.__doc__ = \
                             FigureCanvasBase.start_event_loop_default.__doc__

    def stop_event_loop(self):
        FigureCanvasBase.stop_event_loop_default(self)

    stop_event_loop.__doc__ = FigureCanvasBase.stop_event_loop_default.__doc__


class MainWindow(QtWidgets.QMainWindow):
    closing = QtCore.Signal()

    def closeEvent(self, event):
        self.closing.emit()
        QtWidgets.QMainWindow.closeEvent(self, event)


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
        self.window.closing.connect(canvas.close_event)
        self.window.closing.connect(self._widgetclosed)

        self.window.setWindowTitle("Figure %d" % num)
        image = os.path.join(matplotlib.rcParams['datapath'],
                             'images', 'matplotlib.png')
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

        # add text label to status bar
        self.statusbar_label = QtWidgets.QLabel()
        self.window.statusBar().addWidget(self.statusbar_label)

        self.toolbar = self._get_toolbar(self.canvas, self.window)
        if self.toolbar is not None:
            self.window.addToolBar(self.toolbar)
            self.toolbar.message.connect(self.statusbar_label.setText)
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
            self.canvas.draw_idle()

        def notify_axes_change(fig):
            # This will be called whenever the current axes is changed
            if self.toolbar is not None:
                self.toolbar.update()
        self.canvas.figure.add_axobserver(notify_axes_change)
        self.window.raise_()

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
        if matplotlib.rcParams['toolbar'] == 'toolbar2':
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
        if QtWidgets.QApplication.instance() is None:
            return
        if self.window._destroying:
            return
        self.window._destroying = True
        self.window.destroyed.connect(self._widgetclosed)

        if self.toolbar:
            self.toolbar.destroy()
        if DEBUG:
            print("destroy figure manager")
        self.window.close()

    def get_window_title(self):
        return six.text_type(self.window.windowTitle())

    def set_window_title(self, title):
        self.window.setWindowTitle(title)


class NavigationToolbar2QT(NavigationToolbar2, QtWidgets.QToolBar):
    message = QtCore.Signal(str)

    def __init__(self, canvas, parent, coordinates=True):
        """ coordinates: should we show the coordinates on the right? """
        self.canvas = canvas
        self.parent = parent
        self.coordinates = coordinates
        self._actions = {}
        """A mapping of toolitem method names to their QActions"""

        QtWidgets.QToolBar.__init__(self, parent)
        NavigationToolbar2.__init__(self, canvas)

    def _icon(self, name):
        if is_pyqt5():
            name = name.replace('.png', '_large.png')
        pm = QtGui.QPixmap(os.path.join(self.basedir, name))
        if hasattr(pm, 'setDevicePixelRatio'):
            pm.setDevicePixelRatio(self.canvas._dpi_ratio)
        return QtGui.QIcon(pm)

    def _init_toolbar(self):
        self.basedir = os.path.join(matplotlib.rcParams['datapath'], 'images')

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
                if text == 'Subplots':
                    a = self.addAction(self._icon("qt4_editor_options.png"),
                                       'Customize', self.edit_parameters)
                    a.setToolTip('Edit axis, curve and image parameters')

        self.buttons = {}

        # Add the x,y location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = QtWidgets.QLabel("", self)
            self.locLabel.setAlignment(
                    QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
            self.locLabel.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                      QtWidgets.QSizePolicy.Ignored))
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)

        # reference holder for subplots_adjust window
        self.adj_window = None

        # Esthetic adjustments - we need to set these explicitly in PyQt5
        # otherwise the layout looks different - but we don't want to set it if
        # not using HiDPI icons otherwise they look worse than before.
        if is_pyqt5():
            self.setIconSize(QtCore.QSize(24, 24))
            self.layout().setSpacing(12)

    if is_pyqt5():
        # For some reason, self.setMinimumHeight doesn't seem to carry over to
        # the actual sizeHint, so override it instead in order to make the
        # aesthetic adjustments noted above.
        def sizeHint(self):
            size = super(NavigationToolbar2QT, self).sizeHint()
            size.setHeight(max(48, size.height()))
            return size

    def edit_parameters(self):
        allaxes = self.canvas.figure.get_axes()
        if not allaxes:
            QtWidgets.QMessageBox.warning(
                self.parent, "Error", "There are no axes to edit.")
            return
        if len(allaxes) == 1:
            axes = allaxes[0]
        else:
            titles = []
            for axes in allaxes:
                name = (axes.get_title() or
                        " - ".join(filter(None, [axes.get_xlabel(),
                                                 axes.get_ylabel()])) or
                        "<anonymous {} (id: {:#x})>".format(
                            type(axes).__name__, id(axes)))
                titles.append(name)
            item, ok = QtWidgets.QInputDialog.getItem(
                self.parent, 'Customize', 'Select axes:', titles, 0, False)
            if ok:
                axes = allaxes[titles.index(six.text_type(item))]
            else:
                return

        figureoptions.figure_edit(axes, self)

    def _update_buttons_checked(self):
        # sync button checkstates to match active mode
        self._actions['pan'].setChecked(self._active == 'PAN')
        self._actions['zoom'].setChecked(self._active == 'ZOOM')

    def pan(self, *args):
        super(NavigationToolbar2QT, self).pan(*args)
        self._update_buttons_checked()

    def zoom(self, *args):
        super(NavigationToolbar2QT, self).zoom(*args)
        self._update_buttons_checked()

    def dynamic_update(self):
        self.canvas.draw_idle()

    def set_message(self, s):
        self.message.emit(s)
        if self.coordinates:
            self.locLabel.setText(s)

    def set_cursor(self, cursor):
        if DEBUG:
            print('Set cursor', cursor)
        self.canvas.setCursor(cursord[cursor])

    def draw_rubberband(self, event, x0, y0, x1, y1):
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0

        w = abs(x1 - x0)
        h = abs(y1 - y0)

        rect = [int(val)for val in (min(x0, x1), min(y0, y1), w, h)]
        self.canvas.drawRectangle(rect)

    def remove_rubberband(self):
        self.canvas.drawRectangle(None)

    def configure_subplots(self):
        image = os.path.join(matplotlib.rcParams['datapath'],
                             'images', 'matplotlib.png')
        dia = SubplotToolQt(self.canvas.figure, self.parent)
        dia.setWindowIcon(QtGui.QIcon(image))
        dia.exec_()

    def save_figure(self, *args):
        filetypes = self.canvas.get_supported_filetypes_grouped()
        sorted_filetypes = list(six.iteritems(filetypes))
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

        fname, filter = _getSaveFileName(self.parent,
                                         "Choose a filename to save to",
                                 start, filters, selectedFilter)
        if fname:
            if startpath == '':
                # explicitly missing key or empty str signals to use cwd
                matplotlib.rcParams['savefig.directory'] = startpath
            else:
                # save dir for next time
                savefig_dir = os.path.dirname(six.text_type(fname))
                matplotlib.rcParams['savefig.directory'] = savefig_dir
            try:
                self.canvas.print_figure(six.text_type(fname))
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error saving file", six.text_type(e),
                    QtWidgets.QMessageBox.Ok, QtWidgets.QMessageBox.NoButton)


class SubplotToolQt(SubplotTool, UiSubplotTool):
    def __init__(self, targetfig, parent):
        UiSubplotTool.__init__(self, None)

        self.targetfig = targetfig
        self.parent = parent
        self.donebutton.clicked.connect(self.close)
        self.resetbutton.clicked.connect(self.reset)
        self.tightlayout.clicked.connect(self.functight)

        # constraints
        self.sliderleft.valueChanged.connect(self.sliderright.setMinimum)
        self.sliderright.valueChanged.connect(self.sliderleft.setMaximum)
        self.sliderbottom.valueChanged.connect(self.slidertop.setMinimum)
        self.slidertop.valueChanged.connect(self.sliderbottom.setMaximum)

        self.defaults = {}
        for attr in ('left', 'bottom', 'right', 'top', 'wspace', 'hspace', ):
            val = getattr(self.targetfig.subplotpars, attr)
            self.defaults[attr] = val
            slider = getattr(self, 'slider' + attr)
            txt = getattr(self, attr + 'value')
            slider.setMinimum(0)
            slider.setMaximum(1000)
            slider.setSingleStep(5)
            # do this before hooking up the callbacks
            slider.setSliderPosition(int(val * 1000))
            txt.setText("%.2f" % val)
            slider.valueChanged.connect(getattr(self, 'func' + attr))
        self._setSliderPositions()

    def _setSliderPositions(self):
        for attr in ('left', 'bottom', 'right', 'top', 'wspace', 'hspace', ):
            slider = getattr(self, 'slider' + attr)
            slider.setSliderPosition(int(self.defaults[attr] * 1000))

    def funcleft(self, val):
        if val == self.sliderright.value():
            val -= 1
        val /= 1000.
        self.targetfig.subplots_adjust(left=val)
        self.leftvalue.setText("%.2f" % val)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def funcright(self, val):
        if val == self.sliderleft.value():
            val += 1
        val /= 1000.
        self.targetfig.subplots_adjust(right=val)
        self.rightvalue.setText("%.2f" % val)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def funcbottom(self, val):
        if val == self.slidertop.value():
            val -= 1
        val /= 1000.
        self.targetfig.subplots_adjust(bottom=val)
        self.bottomvalue.setText("%.2f" % val)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def functop(self, val):
        if val == self.sliderbottom.value():
            val += 1
        val /= 1000.
        self.targetfig.subplots_adjust(top=val)
        self.topvalue.setText("%.2f" % val)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def funcwspace(self, val):
        val /= 1000.
        self.targetfig.subplots_adjust(wspace=val)
        self.wspacevalue.setText("%.2f" % val)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def funchspace(self, val):
        val /= 1000.
        self.targetfig.subplots_adjust(hspace=val)
        self.hspacevalue.setText("%.2f" % val)
        if self.drawon:
            self.targetfig.canvas.draw_idle()

    def functight(self):
        self.targetfig.tight_layout()
        self._setSliderPositions()
        self.targetfig.canvas.draw_idle()

    def reset(self):
        self.targetfig.subplots_adjust(**self.defaults)
        self._setSliderPositions()
        self.targetfig.canvas.draw_idle()


def error_msg_qt(msg, parent=None):
    if not is_string_like(msg):
        msg = ','.join(map(str, msg))

    QtWidgets.QMessageBox.warning(None, "Matplotlib",
                                  msg, QtGui.QMessageBox.Ok)


def exception_handler(type, value, tb):
    """Handle uncaught exceptions
    It does not catch SystemExit
    """
    msg = ''
    # get the filename attribute if available (for IOError)
    if hasattr(value, 'filename') and value.filename is not None:
        msg = value.filename + ': '
    if hasattr(value, 'strerror') and value.strerror is not None:
        msg += value.strerror
    else:
        msg += six.text_type(value)

    if len(msg):
        error_msg_qt(msg)

FigureCanvas = FigureCanvasQT
FigureManager = FigureManagerQT
