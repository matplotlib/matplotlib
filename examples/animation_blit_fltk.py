import sys
import fltk
import pylab as p
import matplotlib.numerix as nx
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
    def clear(self,event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)   
    def update(self,ptr):
        # restore the clean slate background
        if self.background is None:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)  
        self.canvas.restore_region(self.background)
        # update the data
        line.set_ydata(nx.sin(x+self.cnt/10.0))  
        # just draw the animated artist
        self.ax.draw_artist(line)
        # just redraw the axes rectangle
        self.canvas.blit(ax.bbox) 
        self.cnt+=1
        if self.cnt==1000:
            # print the timing info and quit
            print 'FPS:' , 200/(time.time()-tstart)
            sys.exit()
        return True
            
# for profiling
tstart = time.time()

ax = p.subplot(111)
# create the initial line
x = nx.arange(0,2*nx.pi,0.01)
line, = p.plot(x, nx.sin(x), animated=True)
p.draw()
anim=animator(ax)    

fltk.Fl.add_idle(anim.update)
fltk.Fl.run()
