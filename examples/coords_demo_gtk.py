"""
An example of how to interact with the plotting canvas by connecting
to move and click events
"""
import matplotlib
matplotlib.use('GTK')
from matplotlib.matlab import *

t = arange(0.0, 1.0, 0.01)
s = sin(2*pi*t)
ax = subplot(111)
ax.plot(t,s)

canvas = get_current_fig_manager().canvas

def on_move(widget, event):
    # get the x and y coords, flip y from top to bottom
    height = canvas.figure.bbox.y.interval()
    x, y = event.x, height-event.y

    if ax.in_axes(x, y):
        # transData transforms data coords to display coords.  Use the
        # inverse method to transform back
        t = ax.xaxis.transData.inverse_positions(x)
        val = ax.yaxis.transData.inverse_positions(y)
        print t, val

def on_click(widget, event):
    # get the x and y coords, flip y from top to bottom
    height = canvas.figure.bbox.y.interval()
    x, y = event.x, height-event.y
    if event.button==1:
        if ax.in_axes(x, y):
            # transData transforms data coords to display coords.  Use the
            # inverse method to transform back
            t = ax.xaxis.transData.inverse_positions(x)
            val = ax.yaxis.transData.inverse_positions(y)
        
            print t, val


canvas.connect('motion_notify_event', on_move)
canvas.connect('button_press_event', on_click)



show()
