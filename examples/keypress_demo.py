#!/usr/bin/env python
"""
Show how to connect to keypress events

Note, on the wx backend on some platforms (eg linux), you have to
first click on the figure before the keypress events are activated.
If you know how to fix this, please email us!
"""
from pylab import *

def press(event):
    print 'press', event.key
    if event.key=='g':
        grid()
        draw()
    
connect('key_press_event', press)

title('press g to toggle grid')
plot(rand(12), rand(12), 'go')
show()
