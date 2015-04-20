from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure

from .qt_compat import QtCore, QtWidgets, __version__

from .backend_qt5 import (backend_version, _create_qApp, FigureManagerQT)
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
        # hide until we can test and fix
        # self.startTimer(backend_IdleEvent.milliseconds)
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
