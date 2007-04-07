from pylab import figure, show, nx

# scatter creates a RegPolyCollection
fig = figure()
ax = fig.add_subplot(111)
N = 100
x, y = 0.9*nx.mlab.rand(2,N)
area = nx.pi*(10 * nx.mlab.rand(N))**2 # 0 to 10 point radiuses
ax.scatter(x,y,s=area, marker='^', c='r', label='scatter')
ax.legend()
fig.savefig('legend_unit_polycoll')

# vlines creates a LineCollection
fig = figure()
ax = fig.add_subplot(111)
t = nx.arange(0.0, 1.0, 0.05)
ax.vlines(t, [0], nx.sin(2*nx.pi*t), label='vlines')
ax.legend()
fig.savefig('legend_unit_linecoll')
show()
