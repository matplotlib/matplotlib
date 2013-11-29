from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

from . import backend_gtk3cairo_gui
from . import backend_gtk3
from matplotlib.figure import Figure


FigureCanvasGTK3Cairo = backend_gtk3cairo_gui.FigureCanvasGTK3CairoGui
FigureManagerGTK3Cairo = backend_gtk3.FigureManagerGTK3


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
    canvas = FigureCanvasGTK3Cairo(figure)
    manager = FigureManagerGTK3Cairo(canvas, num)
    return manager


FigureCanvas = FigureCanvasGTK3Cairo
FigureManager = FigureManagerGTK3Cairo
show = backend_gtk3.show
