from pylab import figure, show, nx
from matplotlib.patches import RegularPolygon
import matplotlib.agg as agg

fig = figure()
ax = fig.add_subplot(111)
t = nx.arange(0.0, 4.0, 0.01)
s = 2*nx.sin(2*nx.pi*8*t)
line, = ax.plot(t, s)
line2, = ax.plot(t, 0.5*s)
line3, = ax.plot(t, 4*s)
markers, = ax.plot(t, 2*(nx.mlab.rand(len(t))-0.5), 'bo')
path = agg.path_storage()
poly = RegularPolygon( (2, 0.), numVertices=8, radius=1.5)
#for i, xy in enumerate(ax.transData.seq_xy_tups(poly.get_verts())):
for i, xy in enumerate(ax.transData.seq_xy_tups(poly.get_verts())):
    if i==0: path.move_to(*xy)
    else:    path.line_to(*xy)
path.close_polygon()
line.set_clip_path(path)
line2.set_clip_path(path)
line3.set_clip_path(path)
show()
