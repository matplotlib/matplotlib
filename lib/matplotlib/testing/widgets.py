"""
========================
Widget testing utilities
========================
Functions that are useful for testing widgets.
See also matplotlib.tests.test_widgets
"""
import matplotlib.pyplot as plt
from unittest import mock


def get_ax():
    """Creates plot and returns its axes"""
    fig, ax = plt.subplots(1, 1)
    ax.plot([0, 200], [0, 200])
    ax.set_aspect(1.0)
    ax.figure.canvas.draw()
    return ax


def do_event(tool, etype, button=1, xdata=0, ydata=0, key=None, step=1):
    """
    Trigger an event

    Parameters
    ----------
    tool : matplotlib.widgets.RectangleSelector
    etype
        the event to trigger
    xdata : int
        x coord of mouse in data coords
    ydata : int
        y coord of mouse in data coords
    button : int or str
        button pressed None, 1, 2, 3, 'up', 'down' (up and down are used
        for scroll events)
    key
        the key depressed when the mouse event triggered (see
        :class:`KeyEvent`)
    step : int
        number of scroll steps (positive for 'up', negative for 'down')
    """
    event = mock.Mock()
    event.button = button
    ax = tool.ax
    event.x, event.y = ax.transData.transform([(xdata, ydata),
                                               (xdata, ydata)])[0]
    event.xdata, event.ydata = xdata, ydata
    event.inaxes = ax
    event.canvas = ax.figure.canvas
    event.key = key
    event.step = step
    event.guiEvent = None
    event.name = 'Custom'

    func = getattr(tool, etype)
    func(event)
