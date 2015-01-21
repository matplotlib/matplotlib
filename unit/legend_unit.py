from __future__ import print_function
from pylab import figure, show, np

Ntests = 3
t = np.arange(0.0, 1.0, 0.05)
s = np.sin(2 * np.pi * t)

# scatter creates a RegPolyCollection
fig = figure()
ax = fig.add_subplot(Ntests, 1, 1)
N = 100
x, y = 0.9 * np.random.rand(2, N)
area = np.pi * (10 * np.random.rand(N)) ** 2  # 0 to 10 point radiuses
ax.scatter(x, y, s=area, marker='^', c='r', label='scatter')
ax.legend()

# vlines creates a LineCollection
ax = fig.add_subplot(Ntests, 1, 2)
ax.vlines(t, [0], np.sin(2 * np.pi * t), label='vlines')
ax.legend()

# vlines creates a LineCollection
ax = fig.add_subplot(Ntests, 1, 3)
ax.plot(t, s, 'b-', lw=2, label='a line')
ax.legend()

fig.savefig('legend_unit')
show()
