#!/usr/bin/env python
"""
Hold the pointer over an object and press "p" to pick it.  When
picked it will turn red 

Note this algorithm calculates distance to the vertices of the
polygon, so if you want to pick a patch, click on the edge!

"""
from pylab import *
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def pick(event):
    if event.key=='p' and event.inaxes is not None:
        ax = event.inaxes
        a = ax.pick(event.x, event.y)
        
	if isinstance(a, Text):
            a.set_color('r')
        elif isinstance(a, Line2D):
            a.set_markerfacecolor('r')
	elif isinstance(a, Patch):
            a.set_facecolor('r')
        draw()
            
    
connect('key_press_event', pick)

ax = subplot(111)
title('Put mouse over object and press "p" to pick it')

for i in range(20):
    x, y = rand(2)
    text(x,y,'hi!')

for i in range(5):
    x = rand(10)
    y = rand(10)
    plot(x,y,'go')

for i in range(5):
    x = rand()
    y = rand()
    center = x,y
    p = Circle(center, radius=.1)
    ax.add_patch(p)
    

show()
