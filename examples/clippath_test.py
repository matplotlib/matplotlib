from matplotlib.pyplot import figure, show
import matplotlib.transforms as transforms
from matplotlib.patches import RegularPolygon
import matplotlib.agg as agg
from numpy import arange, sin, pi
from numpy.random import rand

class ClipWindow:
    def __init__(self, ax, line):
        self.ax = ax
        ax.set_title('drag polygon around to test clipping')
        self.canvas = ax.figure.canvas
        self.line = line
        self.poly = RegularPolygon(
            (200, 200), numVertices=10, radius=100,
            facecolor='yellow', alpha=0.25,
            transform=transforms.identity_transform())

        ax.add_patch(self.poly)
        self.canvas.mpl_connect('button_press_event', self.onpress)
        self.canvas.mpl_connect('button_release_event', self.onrelease)
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.x, self.y = None, None


    def onpress(self, event):
        self.x, self.y = event.x, event.y

    def onrelease(self, event):
        self.x, self.y = None, None

    def onmove(self, event):

        if self.x is None: return
        dx = event.x - self.x
        dy = event.y - self.y
        self.x, self.y = event.x, event.y
        x, y = self.poly.xy
        x += dx
        y += dy
        #print self.y, event.y, dy, y
        self.poly.xy = x,y
        self._clip()

    def _clip(self):
        fig = self.ax.figure
        l,b,w,h = fig.bbox.get_bounds()
        path = agg.path_storage()

        for i, xy in enumerate(self.poly.get_verts()):
            x,y = xy
            y = h-y
            if i==0: path.move_to(x,y)
            else:    path.line_to(x,y)
        path.close_polygon()
        self.line.set_clip_path(path)
        self.canvas.draw_idle()


fig = figure(figsize=(8,8))
ax = fig.add_subplot(111)
t = arange(0.0, 4.0, 0.01)
s = 2*sin(2*pi*8*t)

line, = ax.plot(t, 2*(rand(len(t))-0.5), 'b-')
clipwin = ClipWindow(ax, line)
show()
