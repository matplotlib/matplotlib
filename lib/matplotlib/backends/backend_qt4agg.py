"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from .backend_agg import FigureCanvasAgg
from .backend_qt4 import (
    QtCore, _BackendQT4, FigureCanvasQT, FigureManagerQT, NavigationToolbar2QT)
from .backend_qt5agg import FigureCanvasQTAggBase


class FigureCanvasQTAggBase(FigureCanvasQTAggBase):
    def _fake_super_fcqab(self, figure):
        FigureCanvasAgg.__init__(self, figure)


class FigureCanvasQTAgg(FigureCanvasQTAggBase, FigureCanvasQT):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Attributes
    ----------
    figure : `matplotlib.figure.Figure`
        A high-level Figure instance

    """
    def __init__(self, figure):
        FigureCanvasQT.__init__(self, figure)
        FigureCanvasQTAggBase.__init__(self, figure)


@_BackendQT4.export
class _BackendQT4Agg(_BackendQT4):
    FigureCanvas = FigureCanvasQTAgg
