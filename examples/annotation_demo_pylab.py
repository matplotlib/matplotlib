"""
Some examples of how to annotate various artists.  


See matplotlib.text.Annotation for details
"""
from pylab import *
from matplotlib.patches import Rectangle

fig = figure()
ax = subplot(111, autoscale_on=False, xlim=(-1,5), ylim=(-3,5))

rect = Rectangle((0.5, 0.5), 1, 3, alpha=0.3)
ax.add_patch(rect)

t = nx.arange(0.0, 5.0, 0.01)
s = nx.sin(2*nx.pi*t)
line, = plot(t, s, lw=3, color='purple')

annotate(rect, 'A: rect', loc=('outside right', 'outside top'),
               color='blue')

annotate(rect, 'B: rect', loc=('inside left', 'inside top'),
         autopad=8, color='blue')

annotate(rect, 'C: rect', loc=('center', 'center'), color='blue')

annotate(ax, 'bottom corner', loc=('inside right', 'inside bottom'),
         color='red', autopad=40, lineprops=dict(lw=2, color='red', shrink=4))

annotate(ax, 'E: an axes title', loc=('center', 'outside top'), color='red')


annotate(fig, 'F: a figure title', loc=('center', 'inside top'),
         autopad=10, size=16, color='green')

annotate(line, 'localmax', loc=(2.25, 1), padx=20, pady=80, 
         color='black', size=18,
         lineprops=dict(lw=2, color='k', shrink=5., xalign='center'))

fig.savefig('annotation_demo')
show()
