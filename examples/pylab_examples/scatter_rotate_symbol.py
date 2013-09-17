import matplotlib.pyplot as plt
from numpy import arange, pi, rad2deg
from numpy.random import rand
from matplotlib.markers import TICKRIGHT

rx, ry = 3., 1.
area = rx * ry * pi
theta = rad2deg(arange(0, 2*pi+0.01, 0.1))


x, y, s, c = rand(4, 30)
s *= 20**2.

fig, ax = plt.subplots()
ax.scatter(x, y, s, c, marker=TICKRIGHT, a=theta)

plt.show()
