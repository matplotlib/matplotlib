"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import os  # not used
import sys
import ctypes
import warnings

import matplotlib
from matplotlib.figure import Figure

from .backend_qt5agg import NavigationToolbar2QTAgg
from .backend_qt5agg import FigureCanvasQTAggBase

from .backend_agg import FigureCanvasAgg
from .backend_qt4 import QtCore
from .backend_qt4 import FigureManagerQT
from .backend_qt4 import FigureCanvasQT
from .backend_qt4 import NavigationToolbar2QT
##### not used
from .backend_qt4 import show
from .backend_qt4 import draw_if_interactive
from .backend_qt4 import backend_version
######
from matplotlib.cbook import mplDeprecation

DEBUG = False

_decref = ctypes.pythonapi.Py_DecRef
_decref.argtypes = [ctypes.py_object]
_decref.restype = None


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    if DEBUG:
        print('backend_qt4agg.new_figure_manager')
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasQTAgg(figure)
    return FigureManagerQT(canvas, num)


class FigureCanvasQTAgg(FigureCanvasQTAggBase,
                        FigureCanvasQT, FigureCanvasAgg):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Public attribute

      figure - A Figure instance
   """

    def __init__(self, figure):
        if DEBUG:
            print('FigureCanvasQtAgg: ', figure)
        FigureCanvasQT.__init__(self, figure)
        FigureCanvasAgg.__init__(self, figure)
        self._drawRect = None
        self.blitbox = None
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        # it has been reported that Qt is semi-broken in a windows
        # environment.  If `self.draw()` uses `update` to trigger a
        # system-level window repaint (as is explicitly advised in the
        # Qt documentation) the figure responds very slowly to mouse
        # input.  The work around is to directly use `repaint`
        # (against the advice of the Qt documentation).  The
        # difference between `update` and repaint is that `update`
        # schedules a `repaint` for the next time the system is idle,
        # where as `repaint` repaints the window immediately.  The
        # risk is if `self.draw` gets called with in another `repaint`
        # method there will be an infinite recursion.  Thus, we only
        # expose windows users to this risk.
        if sys.platform.startswith('win'):
            self._priv_update = self.repaint
        else:
            self._priv_update = self.update


FigureCanvas = FigureCanvasQTAgg
FigureManager = FigureManagerQT
