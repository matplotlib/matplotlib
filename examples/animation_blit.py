#!/usr/bin/env python

# For detailed comments on animation and the techniques used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation
import sys
import time

import gtk, gobject

import matplotlib
matplotlib.use('GTKAgg')
import numpy as npy
import pylab as p


ax = p.subplot(111)
canvas = ax.figure.canvas

p.subplots_adjust(left=0.3, bottom=0.3) # check for flipy bugs
p.grid() # to ensure proper background restore

# create the initial line
x = npy.arange(0,2*npy.pi,0.01)
line, = p.plot(x, npy.sin(x), animated=True, lw=2)

# for profiling
tstart = time.time()

def update_line(*args):
    if update_line.background is None:
        update_line.background = canvas.copy_from_bbox(ax.bbox)

    # restore the clean slate background
    canvas.restore_region(update_line.background)
    # update the data
    line.set_ydata(npy.sin(x+update_line.cnt/10.0))
    # just draw the animated artist
    try:
        ax.draw_artist(line)
    except AssertionError:
        return
    # just redraw the axes rectangle
    canvas.blit(ax.bbox)

    if update_line.cnt==1000:
        # print the timing info and quit
        print 'FPS:' , 1000/(time.time()-tstart)
        sys.exit()

    update_line.cnt += 1
    return True

update_line.cnt = 0
update_line.background = None
gobject.idle_add(update_line)
p.show()
