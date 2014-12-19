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



def get_ax():
    fig, ax = plt.subplots(1, 1)
    ax.plot([0, 200], [0, 200])
    ax.set_aspect(1.0)
    ax.figure.canvas.draw()
    return ax


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
    ax = get_ax()

    def onselect(epress, erelease):
        ax._got_onselect = True
        assert epress.xdata == 100
        assert epress.ydata == 100
        assert erelease.xdata == 199
        assert erelease.ydata == 199

    tool = widgets.RectangleSelector(ax, onselect, **kwargs)
    event = get_event(ax, xdata=100, ydata=100, button=1)
    tool.press(event)

    event = get_event(ax, xdata=199, ydata=199, button=1)
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


def test_ellipse():
    """For ellipse, test out the key modifiers"""
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.EllipseSelector(ax, onselect=onselect,
                                     maxdist=10)
    tool.extents = (100, 150, 100, 150)

    # drag the rectangle
    event = get_event(ax, xdata=10, ydata=10, button=1,
                    key='alt')
    tool.press(event)
    event = get_event(ax, xdata=30, ydata=30, button=1)
    tool.onmove(event)
    tool.release(event)
    assert tool.extents == (120, 170, 120, 170)

    # create from center
    event = get_event(ax, xdata=100, ydata=100, button=1,
                    key='control')
    tool.press(event)
    event = get_event(ax, xdata=125, ydata=125, button=1)
    tool.onmove(event)
    tool.release(event)
    assert tool.extents == (75, 125, 75, 125)

    # create a square
    event = get_event(ax, xdata=10, ydata=10, button=1,
                    key='shift')
    tool.press(event)
    event = get_event(ax, xdata=35, ydata=30, button=1)
    tool.onmove(event)
    tool.release(event)
    extents = [int(e) for e in tool.extents]
    assert extents == [10, 35, 10, 35]

    # create a square from center
    event = get_event(ax, xdata=100, ydata=100, button=1,
                      key='ctrl+shift')
    tool.press(event)
    event = get_event(ax, xdata=125, ydata=130, button=1)
    tool.onmove(event)
    tool.release(event)
    extents = [int(e) for e in tool.extents]
    assert extents == [70, 130, 70, 130], extents


def test_rectangle_handles():
    ax = get_ax()

    def onselect(epress, erelease):
        pass

    tool = widgets.RectangleSelector(ax, onselect=onselect,
                                     maxdist=10)
    tool.extents = (100, 150, 100, 150)

    assert tool.corners == (
        (100, 150, 150, 100), (100, 100, 150, 150))
    assert tool.extents == (100, 150, 100, 150)
    assert tool.edge_centers == (
        (100, 125.0, 150, 125.0), (125.0, 100, 125.0, 150))
    assert tool.extents == (100, 150, 100, 150)

    # grab a corner and move it
    event = get_event(ax, xdata=100, ydata=100)
    tool.press(event)
    event = get_event(ax, xdata=120, ydata=120)
    tool.onmove(event)
    tool.release(event)
    assert tool.extents ==  (120, 150, 120, 150)

    # grab the center and move it
    event = get_event(ax, xdata=132, ydata=132)
    tool.press(event)
    event = get_event(ax, xdata=120, ydata=120)
    tool.onmove(event)
    tool.release(event)
    assert tool.extents ==  (108, 138, 108, 138)

    # create a new rectangle
    event = get_event(ax, xdata=10, ydata=10)
    tool.press(event)
    event = get_event(ax, xdata=100, ydata=100)
    tool.onmove(event)
    tool.release(event)
    assert tool.extents == (10, 100, 10, 100)


@cleanup
def check_span(*args, **kwargs):
    ax = get_ax()

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
    ax = get_ax()

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
