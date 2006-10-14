"""
Some examples of how to annotate various artists.  


See matplotlib.text.Annotation for details
"""
from pylab import figure, show, nx
from matplotlib.patches import Rectangle, CirclePolygon, Ellipse
from matplotlib.text import Annotation

fig = figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1,5), ylim=(-3,5))

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

d = Annotation(ax, 'bottom corner', loc=('inside right', 'inside bottom'),
               color='red', autopad=40, lineprops=dict(lw=2, color='red', shrink=4))
ax.add_artist(d)

e = Annotation(ax, 'E: an axes title', loc=('center', 'outside top'),
               color='red')
ax.add_artist(e)

f = Annotation(fig, 'F: a figure title', loc=('center', 'inside top'),
               autopad=10, size=16, color='green')
ax.add_artist(f)

g = Annotation(line, 'localmax', loc=(2.25, 1), padx=20, pady=80, 
               color='black', size=18,
               lineprops=dict(lw=2, color='k', shrink=5., xalign='center'))
ax.add_artist(g)

fig.savefig('annotation_demo')


# here are some annotations using various coordinate systems.  If you
# pass in loc=(x,y) where x,y are scalars, then you can specify the
# following strings for the coordinate system
#  'figure points'   : points from the lower left corner of the figure
#  'figure pixels'   : pixels from the lower left corner of the figure
#  'figure fraction' : 0,0 is lower left of figure and 1,1 is upper, right
#  'axes points'     : points from lower left corner of axes
#  'axes pixels'     : pixels from lower left corner of axes
#  'axes fraction'   : 0,1 is lower left of axes and 1,1 is upper right
#  'data'            : use the coordinate system of the object being annotated (default)


fig = figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1,5), ylim=(-3,5))

t = nx.arange(0.0, 5.0, 0.01)
s = nx.sin(2*nx.pi*t)
line, = ax.plot(t, s, lw=3, color='purple')

a = Annotation(ax, 'A: center', loc=(.5, .5),  coords='axes fraction')
ax.add_artist(a)

b = Annotation(fig, 'B: pixels', loc=(20, 20),  coords='figure pixels')
ax.add_artist(b)

c = Annotation(fig, 'C: points', loc=(100, 300),  coords='figure points')
ax.add_artist(c)

d = Annotation(line, 'D: data', loc=(1, 2),  coords='data')
ax.add_artist(d)

# use positive points or pixels to specify from left, bottom
e = Annotation(fig, 'E: a figure title (fraction)', loc=(.05, .95),  coords='figure fraction',
               horizontalalignment='left', verticalalignment='top',
               fontsize=20)
ax.add_artist(e)

# use negative points or pixels to specify from right, top
f = Annotation(fig, 'F: a figure title (points)', loc=(-10, -10),  coords='figure points',
               horizontalalignment='right', verticalalignment='top',
               fontsize=20)
ax.add_artist(f)

fig.savefig('annotation_demo2')
show()
