"""
Show how to use a lasso to select a set of points and get the indices
of the selected points.  A callback is used to change the color of the
selected points

This is currently a proof-of-concept implementation (though it is
usable as is).  There will be some refinement of the API and the
inside polygon detection routine.
"""
from matplotlib.widgets import Lasso
import matplotlib.mlab
from matplotlib.nxutils import points_inside_poly
from matplotlib.colors import colorConverter
from matplotlib.collections import RegularPolyCollection

from matplotlib.pyplot import figure, show
from numpy import nonzero
from numpy.random import rand

class Datum:
    colorin = colorConverter.to_rgba('red')
    colorout = colorConverter.to_rgba('green')
    def __init__(self, x, y, include=False):
        self.x = x
        self.y = y
        if include: self.color = self.colorin
        else: self.color = self.colorout


class LassoManager:
    def __init__(self, ax, data):
        self.axes = ax
        self.canvas = ax.figure.canvas
        self.data = data

        self.Nxy = len(data)

        self.facecolors = [d.color for d in data]
        self.xys = [(d.x, d.y) for d in data]

        self.collection = RegularPolyCollection(
            fig.dpi, 6, sizes=(100,),
            facecolors=self.facecolors,
            offsets = self.xys,
            transOffset = ax.transData)

        ax.add_collection(self.collection)

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

    def callback(self, verts):
        ind = nonzero(points_inside_poly(self.xys, verts))[0]
        for i in range(self.Nxy):
            if i in ind:
                self.facecolors[i] = Datum.colorin
            else:
                self.facecolors[i] = Datum.colorout

        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso

    def onpress(self, event):
        if self.canvas.widgetlock.locked(): return
        if event.inaxes is None: return
        self.lasso = Lasso(event.inaxes, (event.xdata, event.ydata), self.callback)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)

data = [Datum(*xy) for xy in rand(100, 2)]

fig = figure()
ax = fig.add_subplot(111, xlim=(0,1), ylim=(0,1), autoscale_on=False)
lman = LassoManager(ax, data)

show()
