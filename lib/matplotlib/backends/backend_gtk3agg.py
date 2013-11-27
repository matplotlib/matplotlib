from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import warnings

from . import backend_gtk3agg_gui
from . import backend_gtk3
from matplotlib.figure import Figure

if six.PY3:
    warnings.warn("The Gtk3Agg backend is not known to work on Python 3.x.")


FigureCanvasGTK3Agg = backend_gtk3agg_gui.FigureCanvasGTK3AggGui
FigureManagerGTK3Agg = backend_gtk3.FigureManagerGTK3


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
    canvas = FigureCanvasGTK3Agg(figure)
    manager = FigureManagerGTK3Agg(canvas, num)
    return manager


FigureCanvas = FigureCanvasGTK3Agg
FigureManager = FigureManagerGTK3Agg
show = backend_gtk3.show
