#!/usr/bin/env python
"""
An example of how to interact with the plotting canvas by connecting
to move and click events
"""
from matplotlib.matlab import *

t = arange(0.0, 1.0, 0.01)
s = sin(2*pi*t)
ax = subplot(111)
ax.plot(t,s)

canvas = get_current_fig_manager().canvas

def on_move(event):
    # get the x and y pixel coords
    x, y = event.x, event.y

    if event.inaxes is not None:
        print 'data coords', event.xdata, event.ydata

def on_click(event):
    # get the x and y coords, flip y from top to bottom
    x, y = event.x, event.y
    if event.button==1:
        if event.inaxes is not None:
            print 'data coords', event.xdata, event.ydata

#canvas.mpl_connect('motion_notify_event', on_move)
canvas.mpl_connect('button_press_event', on_click)



show()
