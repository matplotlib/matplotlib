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

    def _fake_super_fcq(self, figure):
        # Note different super-calling style to backend_qt5
        QtWidgets.QWidget.__init__(self)
        # Do not call this here (even though it looks like we should!)
        # because it will be called in the Agg backend and we want to 'break'
        # the MI diamond.  With PyQt5 (which in cooperative) this is handled by
        # super
        # FigureCanvasBase.__init__(self, figure)

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
