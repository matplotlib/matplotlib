#!/usr/bin/env python

# For detailed comments on animation and the techniques used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation
import time

import gtk, gobject

import matplotlib
matplotlib.use('GTKAgg')

import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111)
canvas = fig.canvas

fig.subplots_adjust(left=0.3, bottom=0.3) # check for flipy bugs
ax.grid() # to ensure proper background restore

# create the initial line
x = np.arange(0,2*np.pi,0.01)
line, = ax.plot(x, np.sin(x), animated=True, lw=2)
canvas.draw()

# for profiling
tstart = time.time()

def on_draw(event):
    background = canvas.copy_from_bbox(ax.bbox)
    if on_draw.background is None:
        gobject.idle_add(update_line)

    on_draw.background = background

on_draw.background = None

fig.canvas.mpl_connect('draw_event', on_draw)

def update_line(*args):
    if on_draw.background is None:
        return True

    print 'frame', update_line.cnt

    # restore the clean slate background
    canvas.restore_region(on_draw.background)
    # update the data
    line.set_ydata(np.sin(x+update_line.cnt/10.0))

    # just draw the animated artist
    ax.draw_artist(line)

    # just redraw the axes rectangle
    canvas.blit(ax.bbox)

    if update_line.cnt==1000:
        # print the timing info and quit
        print 'FPS:' , 1000/(time.time()-tstart)
        gtk.mainquit()
        raise SystemExit

    update_line.cnt += 1
    return True

update_line.cnt = 0



plt.show()
