# For detailed comments on animation and the techniqes used here, see
# the wiki entry
# http://www.scipy.org/wikis/topical_software/MatplotlibAnimation
import matplotlib
matplotlib.use('GTKAgg')
import sys
import gtk, gobject
import pylab as p
import matplotlib.numerix as nx
import time

ax = p.subplot(111)
canvas = ax.figure.canvas

# for profiling
tstart = time.time()

# create the initial line
x = nx.arange(0,2*nx.pi,0.01)
line, = p.plot(x, nx.sin(x), animated=True)

# save the clean slate background -- everything but the animated line
# is drawn and saved in the pixel buffer background
background = None

def snap_background(self):
    global background
    background = canvas.copy_from_bbox(ax.bbox)
    return True

p.connect('draw_event', snap_background)

def update_line(*args):
    if background is None: return True
    # restore the clean slate background
    canvas.restore_region(background)
    # update the data
    line.set_ydata(nx.sin(x+update_line.cnt/10.0))  
    # just draw the animated artist
    ax.draw_artist(line)
    # just redraw the axes rectangle
    canvas.blit(ax.bbox) 
    
    if update_line.cnt==200:
        # print the timing info and quit
        print 'FPS:' , 200/(time.time()-tstart)
        sys.exit()

    update_line.cnt += 1
    return True
update_line.cnt = 0

gobject.idle_add(update_line)
p.show()
