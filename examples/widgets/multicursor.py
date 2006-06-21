from matplotlib.widgets import MultiCursor
from pylab import figure, show, nx

t = nx.arange(0.0, 2.0, 0.01)
s1 = nx.sin(2*nx.pi*t)
s2 = nx.sin(4*nx.pi*t)
fig = figure()
ax1 = fig.add_subplot(211)
ax1.plot(t, s1)


ax2 = fig.add_subplot(212, sharex=ax1)
ax2.plot(t, s2)

multi = MultiCursor(fig.canvas, (ax1, ax2), color='r', lw=1)
show()
