#!/usr/bin/env python
"""
Show how to connect to keypress events
"""
import numpy as n
from pylab import figure, show

def press(event):
    print 'press', event.key
    if event.key=='x':
        visible = xl.get_visible()
        xl.set_visible(not visible)
        fig.canvas.draw()

fig = figure()
ax = fig.add_subplot(111)

fig.canvas.mpl_connect('key_press_event', press)

ax.plot(n.random.rand(12), n.random.rand(12), 'go')
xl = ax.set_xlabel('easy come, easy go')

show()
