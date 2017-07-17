"""
Render to qt from agg
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import matplotlib
from matplotlib.figure import Figure

from .backend_agg import FigureCanvasAgg
from .backend_qt4 import (
    QtCore, FigureCanvasQT, FigureManagerQT, NavigationToolbar2QT,
    backend_version, draw_if_interactive, show)
from .backend_qt5agg import FigureCanvasQTAggBase


def new_figure_manager(num, *args, **kwargs):
    """
    Create a new figure manager instance
    """
    FigureClass = kwargs.pop('FigureClass', Figure)
    thisFig = FigureClass(*args, **kwargs)
    return new_figure_manager_given_figure(num, thisFig)


def new_figure_manager_given_figure(num, figure):
    """
    Create a new figure manager instance for the given figure.
    """
    canvas = FigureCanvasQTAgg(figure)
    return FigureManagerQT(canvas, num)


class FigureCanvasQTAgg(FigureCanvasQTAggBase, FigureCanvasQT):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc...

    Attributes
    ----------
    figure : `matplotlib.figure.Figure`
        A high-level Figure instance

    """


FigureCanvas = FigureCanvasQTAgg
FigureManager = FigureManagerQT
