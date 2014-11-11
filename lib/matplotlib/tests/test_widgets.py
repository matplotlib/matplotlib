from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

try:
    # mock in python 3.3+
    from unittest import mock
except ImportError:
    import mock

import matplotlib.widgets as widgets
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import cleanup


def get_event(ax, button=1, xdata=0, ydata=0, key=None, step=1):
    """
     *name*
        the event name

    *canvas*
        the FigureCanvas instance generating the event

    *guiEvent*
        the GUI event that triggered the matplotlib event

    *x*
        x position - pixels from left of canvas

    *y*
        y position - pixels from bottom of canvas

    *inaxes*
        the :class:`~matplotlib.axes.Axes` instance if mouse is over axes

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
    event.x, event.y = ax.transData.transform([(xdata, ydata),
                                               (xdata, ydata)])[00]
    event.xdata, event.ydata = xdata, ydata
    event.inaxes = ax
    event.canvas = ax.figure.canvas
    event.key = key
    event.step = step
    event.guiEvent = None
    event.name = 'Custom'
    return event


@cleanup
def check_rectangle(**kwargs):
    fig, ax = plt.subplots(1, 1)
    ax.plot([0, 200], [0, 200])
    ax.figure.canvas.draw()

    def onselect(epress, erelease):
        ax._got_onselect = True
        assert epress.xdata == 100
        assert epress.ydata == 100
        assert erelease.xdata == 200
        assert erelease.ydata == 200

    tool = widgets.RectangleSelector(ax, onselect, **kwargs)
    event = get_event(ax, xdata=100, ydata=100, button=1)
    tool.press(event)

    event = get_event(ax, xdata=125, ydata=125, button=1)
    tool.onmove(event)

    # purposely drag outside of axis for release
    event = get_event(ax, xdata=250, ydata=250, button=1)
    tool.release(event)

    assert ax._got_onselect


def test_rectangle_selector():
    check_rectangle()
    check_rectangle(drawtype='line', useblit=False)
    check_rectangle(useblit=True, button=1)
    check_rectangle(drawtype='none', minspanx=10, minspany=10)
    check_rectangle(minspanx=10, minspany=10, spancoords='pixels')
    check_rectangle(rectprops=dict(fill=True))


@cleanup
def check_span(*args, **kwargs):
    fig, ax = plt.subplots(1, 1)
    ax.plot([0, 200], [0, 200])
    ax.figure.canvas.draw()

    def onselect(vmin, vmax):
        ax._got_onselect = True
        assert vmin == 100
        assert vmax == 150

    def onmove(vmin, vmax):
        assert vmin == 100
        assert vmax == 125
        ax._got_on_move = True

    if 'onmove_callback' in kwargs:
        kwargs['onmove_callback'] = onmove

    tool = widgets.SpanSelector(ax, onselect, *args, **kwargs)
    event = get_event(ax, xdata=100, ydata=100, button=1)
    tool.press(event)

    event = get_event(ax, xdata=125, ydata=125, button=1)
    tool.onmove(event)

    event = get_event(ax, xdata=150, ydata=150, button=1)
    tool.release(event)

    assert ax._got_onselect

    if 'onmove_callback' in kwargs:
        assert ax._got_on_move


def test_span_selector():
    check_span('horizontal', minspan=10, useblit=True)
    check_span('vertical', onmove_callback=True, button=1)
    check_span('horizontal', rectprops=dict(fill=True))


@cleanup
def check_lasso_selector(**kwargs):
    fig, ax = plt.subplots(1, 1)
    ax = plt.gca()
    ax.plot([0, 200], [0, 200])
    ax.figure.canvas.draw()

    def onselect(verts):
        ax._got_onselect = True
        assert verts == [(100, 100), (125, 125), (150, 150)]

    tool = widgets.LassoSelector(ax, onselect, **kwargs)
    event = get_event(ax, xdata=100, ydata=100, button=1)
    tool.press(event)

    event = get_event(ax, xdata=125, ydata=125, button=1)
    tool.onmove(event)

    event = get_event(ax, xdata=150, ydata=150, button=1)
    tool.release(event)

    assert ax._got_onselect


def test_lasso_selector():
    check_lasso_selector()
    check_lasso_selector(useblit=False, lineprops=dict(color='red'))
    check_lasso_selector(useblit=True, button=1)
