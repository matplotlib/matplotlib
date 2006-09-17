"""
Some examples of how to annotate various artists.  


See matplotlib.text.Annotation for details
"""
from pylab import figure, show, nx
from matplotlib.patches import Rectangle, CirclePolygon, Ellipse
from matplotlib.text import Annotation

fig = figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1,5), ylim=(-1,5))

rect = Rectangle((0.5, 0.5), 1, 3, alpha=0.3)
ax.add_patch(rect)

t = nx.arange(0.0, 5.0, 0.01)
s = nx.sin(2*nx.pi*t)
line, = ax.plot(t, s, lw=3, color='purple')

a = Annotation(rect, 'A: rect', loc=('outside right', 'outside top'),
               color='blue')
ax.add_artist(a)

b = Annotation(rect, 'B: rect', loc=('inside left', 'inside top'),
               autopad=8, color='blue')
ax.add_artist(b)

c = Annotation(rect, 'C: rect', loc=('center', 'center'), color='blue')
ax.add_artist(c)

d = Annotation(ax, 'D: axes', loc=('inside right', 'inside bottom'),
               color='red')
ax.add_artist(d)

e = Annotation(ax, 'E: an axes title', loc=('center', 'outside top'),
               color='red')
ax.add_artist(e)

f = Annotation(fig, 'F: a figure title', loc=('center', 'inside top'),
               autopad=10, size=16, color='green')
ax.add_artist(f)

g = Annotation(line, 'G: localmax', loc=(1/4., 1), padx=0, pady=3,
               ha='left', va='bottom', color='purple', size=18)
ax.add_artist(g)
show()
