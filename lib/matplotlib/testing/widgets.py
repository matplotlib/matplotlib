"""
========================
Widget testing utilities
========================

Functions that are useful for testing widgets.
See also matplotlib.tests.test_widgets
"""

import matplotlib.pyplot as plt

try:
    # mock in python 3.3+
    from unittest import mock
except ImportError:
    import mock


def get_ax():
    fig, ax = plt.subplots(1, 1)
    ax.plot([0, 200], [0, 200])
    ax.set_aspect(1.0)
    ax.figure.canvas.draw()
    return ax


def do_event(tool, etype, button=1, xdata=0, ydata=0, key=None, step=1):
    """
     *tool*
        a matplotlib.widgets.RectangleSelector instance

    *etype*
        the event to trigger

    *xdata*
        x coord of mouse in data coords

    *ydata*
        y coord of mouse in data coords

     *button*
        button pressed None, 1, 2, 3, 'up', 'down' (up and down are used
        for scroll events)

    *key*
        the key depressed when the mouse event triggered (see
        :class:`KeyEvent`)

    *step*
        number of scroll steps (positive for 'up', negative for 'down')
    """
    event = mock.Mock()
    event.button = button
    ax = tool.ax
    event.x, event.y = ax.transData.transform([(xdata, ydata),
                                               (xdata, ydata)])[00]
    event.xdata, event.ydata = xdata, ydata
    event.inaxes = ax
    event.canvas = ax.figure.canvas
    event.key = key
    event.step = step
    event.guiEvent = None
    event.name = 'Custom'

    func = getattr(tool, etype)
    func(event)
