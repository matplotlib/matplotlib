from pylab import figure, nx, show

# unit area ellipse
rx, ry = 3., 1.
area = rx * ry * nx.pi
theta = nx.arange(0, 2*nx.pi+0.01, 0.1)
verts = zip(rx/area*nx.cos(theta), ry/area*nx.sin(theta))

x,y,s,c = nx.mlab.rand(4, 30)
s*= 10**2. 

fig = figure()
ax = fig.add_subplot(111)
ax.scatter(x,y,s,c,marker=None,verts =verts)

show()
