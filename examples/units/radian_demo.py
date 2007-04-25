from basic_units import radians, degrees, cos
from pylab import figure, show, nx

x = nx.arange(0, 15, 0.01) * radians

fig = figure()

ax = fig.add_subplot(211)
ax.plot(x, cos(x), xunits=radians)

ax = fig.add_subplot(212)
ax.plot(x, cos(x), xunits=degrees)

show()

