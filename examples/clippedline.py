"""
Clip a line according to the current xlimits, and change the marker
style when zoomed in
"""

from matplotlib.lines import Line2D
import numpy as npy
from pylab import figure, show

class ClippedLine(Line2D):
    """
    Clip the xlimits to the axes view limits -- this example assumes x is sorted
    """

    def __init__(self, ax, *args, **kwargs):
        Line2D.__init__(self, *args, **kwargs)
        self.ax = ax


    def set_data(self, *args, **kwargs):
        Line2D.set_data(self, *args, **kwargs)
        self.xorig = npy.array(self._x)
        self.yorig = npy.array(self._y)

    def draw(self, renderer):
        xlim = self.ax.get_xlim()

        ind0, ind1 = npy.searchsorted(self.xorig, xlim)
        self._x = self.xorig[ind0:ind1]
        self._y = self.yorig[ind0:ind1]
        N = len(self._x)
        if N<1000:
            self._marker = 's'
            self._linestyle = '-'
        else:
            self._marker = None
            self._linestyle = '-'


        Line2D.draw(self, renderer)


fig = figure()
ax = fig.add_subplot(111, autoscale_on=False)

t = npy.arange(0.0, 100.0, 0.01)
s = npy.sin(2*npy.pi*t)
line = ClippedLine(ax, t, s, color='g', ls='-', lw=2)
ax.add_line(line)
ax.set_xlim(10,30)
ax.set_ylim(-1.1,1.1)
show()


