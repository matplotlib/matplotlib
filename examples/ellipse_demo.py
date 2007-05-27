from pylab import figure, show, nx
from matplotlib.patches import Ellipse
rand = nx.mlab.rand

NUM = 250

ells = [Ellipse(xy=rand(2)*10, width=rand(), height=rand(), angle=rand()*360)
        for i in xrange(NUM)]

fig = figure()
ax = fig.add_subplot(111, aspect='equal')
for e in ells:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    e.set_alpha(rand())
    e.set_facecolor(rand(3))

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

fig.savefig('../figures/ellipse_demo.eps')
fig.savefig('../figures/ellipse_demo.png')

show()
