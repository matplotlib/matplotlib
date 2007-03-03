from pylab import fig, nx, show
fig = figure()
ax = fig,add_subplot(111)
t = nx.arange(0.0,3.01,0.01)
s = nx.sin(2*nx.pi*t)
c = nx.sin(4*nx.pi*t)
ax.fill(t, s, 'blue', t, c, 'green', alpha=0.2)
show()
