from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.externals import six
from matplotlib.externals.six import unichr
import os
import re
import signal
import sys

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
try:
    import matplotlib.backends.qt_editor.figureoptions as figureoptions
except ImportError:
    figureoptions = None

from .qt_compat import QtCore, QtWidgets, _getSaveFileName, __version__
from matplotlib.backends.qt_editor.formsubplottool import UiSubplotTool

from .backend_qt5 import (backend_version, SPECIAL_KEYS, SUPER, ALT, CTRL,
                        SHIFT, MODIFIER_KEYS, fn_name, cursord,
                        draw_if_interactive, _create_qApp, show, TimerQT,
                        MainWindow, FigureManagerQT, NavigationToolbar2QT,
                        SubplotToolQt, error_msg_qt, exception_handler)

from .backend_qt5 import FigureCanvasQT as FigureCanvasQT5

DEBUG = False


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


class FigureCanvasQT(FigureCanvasQT5):

    def __init__(self, figure):
        if DEBUG:
            print('FigureCanvasQt qt4: ', figure)
        _create_qApp()

        # Note different super-calling style to backend_qt5
        QtWidgets.QWidget.__init__(self)
        FigureCanvasBase.__init__(self, figure)
        self.figure = figure
        self.setMouseTracking(True)
        self._idle = True
        w, h = self.get_width_height()
        self.resize(w, h)

    def wheelEvent(self, event):
        x = event.x()
        # flipy so y=0 is bottom of canvas
        y = self.figure.bbox.height - event.y()
        # from QWheelEvent::delta doc
        steps = event.delta()/120
        if (event.orientation() == QtCore.Qt.Vertical):
            FigureCanvasBase.scroll_event(self, x, y, steps)
            if DEBUG:
                print('scroll event: delta = %i, '
                      'steps = %i ' % (event.delta(), steps))


FigureCanvas = FigureCanvasQT
FigureManager = FigureManagerQT
