from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six import unichr
import os
import re
import signal
import sys

from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
    FigureCanvasBase, FigureManagerBase, NavigationToolbar2, TimerBase,
    cursors)
from matplotlib.figure import Figure
from matplotlib.widgets import SubplotTool

from .qt_compat import QtCore, QtWidgets, _getSaveFileName, __version__

from .backend_qt5 import (
    backend_version, SPECIAL_KEYS, SUPER, ALT, CTRL, SHIFT, MODIFIER_KEYS,
    cursord, _create_qApp, _BackendQT5, TimerQT, MainWindow, FigureManagerQT,
    NavigationToolbar2QT, SubplotToolQt, error_msg_qt, exception_handler)
from .backend_qt5 import FigureCanvasQT as FigureCanvasQT5

DEBUG = False


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

        # Key auto-repeat enabled by default
        self._keyautorepeat = True

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


@_BackendQT5.export
class _BackendQT4(_BackendQT5):
    FigureCanvas = FigureCanvasQT
