# For detailed comments on animation and the techniqes used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation

from __future__ import print_function

# The number of blits() to make before exiting
NBLITS = 1000

import matplotlib
matplotlib.use('WXAgg')
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

import wx
import sys
import pylab as p
import numpy as npy
import time


# allow the user to disable the WXAgg accelerator from the command line
if '--no-accel' in sys.argv:
    import matplotlib.backends.backend_wxagg
    matplotlib.backends.backend_wxagg._use_accelerator(False)


fig, ax = plt.subplots()
canvas = fig.canvas


p.subplots_adjust(left=0.3, bottom=0.3) # check for flipy bugs
p.grid() # to ensure proper background restore

# create the initial line
x = npy.arange(0,2*npy.pi,0.01)
line, = p.plot(x, npy.sin(x), animated=True, lw=2)

# for profiling
tstart = time.time()
blit_time = 0.0

def update_line(*args):
    global blit_time

    if update_line.background is None:
        update_line.background = canvas.copy_from_bbox(ax.bbox)

    # restore the clean slate background
    canvas.restore_region(update_line.background)
    # update the data
    line.set_ydata(npy.sin(x+update_line.cnt/10.0))
    # just draw the animated artist
    ax.draw_artist(line)
    # just redraw the axes rectangle

    t = time.time()
    canvas.blit(ax.bbox)
    blit_time += time.time() - t

    if update_line.cnt == NBLITS:
        # print the timing info and quit
        frame_time = time.time() - tstart
        print('%d frames: %.2f seconds' % (NBLITS, frame_time))
        print('%d blits:  %.2f seconds' % (NBLITS, blit_time))
        print()
        print('FPS: %.2f' % (NBLITS/frame_time))
        print('BPS: %.2f' % (NBLITS/blit_time))
        sys.exit()

    update_line.cnt += 1
    wx.WakeUpIdle()



update_line.cnt = 0
update_line.background = None
wx.EVT_IDLE(wx.GetApp(), update_line)
p.show()
