from matplotlib.matlab import *
from matplotlib.patches import Polygon

def fill_between(ax, x, y1, y2, **kwargs):
    # add x,y2 in reverse order
    verts = zip(x,y1) + [(x[i], y2[i]) for i in range(len(x)-1,-1,-1)]
    poly = Polygon(verts, **kwargs)
    ax.add_patch(poly)
    ax.autoscale_view()
    return poly

x = arange(0, 2, 0.01)
y1 = sin(2*pi*x)
y2 = sin(4*pi*x) + 2
ax = gca()

p = fill_between(ax, x, y1, y2, facecolor='g')
p.set_alpha(0.5)
show()
                 
