from pylab import figure, nx, show
fig = figure()
ax = fig.add_subplot(111)
t = nx.arange(0.0,3.01,0.01)
s = nx.sin(2*nx.pi*t)
c = nx.sin(4*nx.pi*t)
ax.fill(t, s, 'b', t, c, 'g', alpha=0.2)
show()
