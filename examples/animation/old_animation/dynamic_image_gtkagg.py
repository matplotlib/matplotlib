#!/usr/bin/env python

from __future__ import print_function
"""
An animated image
"""
import time

import gobject
import gtk

import matplotlib
matplotlib.use('GTKAgg')

from pylab import *

fig = figure(1)
a = subplot(111)
x = arange(120.0)*2*pi/120.0
x = resize(x, (100,120))
y = arange(100.0)*2*pi/100.0
y = resize(y, (120,100))
y = transpose(y)
z = sin(x) + cos(y)
im = a.imshow( z, cmap=cm.jet)#, interpolation='nearest')

manager = get_current_fig_manager()
cnt = 0
tstart = time.time()
def updatefig(*args):
    global x, y, cnt, start
    x += pi/15
    y += pi/20
    z = sin(x) + cos(y)
    im.set_array(z)
    manager.canvas.draw()
    cnt += 1
    if cnt==50:
        print('FPS', cnt/(time.time() - tstart))
        return False
    return True

cnt = 0

gobject.idle_add(updatefig)
show()
