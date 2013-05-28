#!/usr/bin/env python

"""
An example of how to interact with the plotting canvas by connecting
to move and click events
"""
from __future__ import print_function
import sys
from pylab import *

t = arange(0.0, 1.0, 0.01)
s = sin(2*pi*t)
fig, ax = plt.subplots()
ax.plot(t,s)


def on_move(event):
    # get the x and y pixel coords
    x, y = event.x, event.y

    if event.inaxes:
        ax = event.inaxes  # the axes instance
        print ('data coords %f %f' % (event.xdata, event.ydata))

def on_click(event):
    # get the x and y coords, flip y from top to bottom
    x, y = event.x, event.y
    if event.button==1:
        if event.inaxes is not None:
            print ('data coords %f %f' % (event.xdata, event.ydata))

binding_id = connect('motion_notify_event', on_move)
connect('button_press_event', on_click)

if "test_disconnect" in sys.argv:
    print ("disconnecting console coordinate printout...")
    disconnect(binding_id)

show()
