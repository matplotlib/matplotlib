# For detailed comments on animation and the techniqes used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation
import matplotlib
matplotlib.use('WXAgg')
matplotlib.rcParams['toolbar'] = None

import wx
import sys
import pylab as p
import matplotlib.numerix as nx
import time


# allow the user to disable the WXAgg accelerator from the command line
if '--no-accel' in sys.argv:
    import matplotlib.backends.backend_wxagg
    matplotlib.backends.backend_wxagg._use_accelerator(False)


ax = p.subplot(111)
canvas = ax.figure.canvas


# create the initial line
x = nx.arange(0,2*nx.pi,0.01)
line, = p.plot(x, nx.sin(x), animated=True, lw=2)

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
    line.set_ydata(nx.sin(x+update_line.cnt/10.0))  
    # just draw the animated artist
    ax.draw_artist(line)
    # just redraw the axes rectangle

    t = time.time()
    canvas.blit(ax.bbox) 
    blit_time += time.time() - t

    if update_line.cnt==200:
        # print the timing info and quit
        frame_time = time.time() - tstart
        print '200 frames: %.2f seconds' % frame_time
        print '200 blits:  %.2f seconds' % blit_time
        print
        print 'FPS: %.2f' % (200/frame_time)
        print 'BPS: %.2f' % (200/blit_time)
        sys.exit()

    update_line.cnt += 1
    wx.WakeUpIdle()

update_line.cnt = 0
update_line.background = None
wx.EVT_IDLE(wx.GetApp(), update_line)
p.show()
