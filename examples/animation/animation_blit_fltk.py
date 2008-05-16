import sys
import fltk
import matplotlib
matplotlib.use('FltkAgg')
import pylab as p
import numpy as npy
import time


# save the clean slate background -- everything but the animated line
# is drawn and saved in the pixel buffer background
class animator:
    def __init__(self,ax):
        self.ax=ax
        self.canvas=ax.figure.canvas
        self.canvas.mpl_connect('draw_event',self.clear)
        self.cnt=0
        self.background=None

        # for profiling
        self.tstart = time.time()

    def clear(self,event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def update(self,ptr):
        # restore the clean slate background
        if self.background is None:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.canvas.restore_region(self.background)
        # update the data
        line.set_ydata(npy.sin(x+self.cnt/10.0))
        # just draw the animated artist
        self.ax.draw_artist(line)
        # just redraw the axes rectangle
        self.canvas.blit(ax.bbox)
        self.cnt+=1
        if self.cnt==1000:
            # print the timing info and quit
            print 'FPS:' , 1000/(time.time()-self.tstart)
            sys.exit()
        return True

ax = p.subplot(111)
p.subplots_adjust(left=0.3, bottom=0.3) # check for flipy bugs
p.grid() # to ensure proper background restore
# create the initial line
x = npy.arange(0,2*npy.pi,0.01)
line, = p.plot(x, npy.sin(x), animated=True)
p.draw()
anim=animator(ax)

fltk.Fl.add_idle(anim.update)
fltk.Fl.run()
