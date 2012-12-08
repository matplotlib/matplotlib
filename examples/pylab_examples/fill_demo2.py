import matplotlib.pyplot as plt
from numpy import arange, sin, pi

fig, ax = plt.subplots()
t = arange(0.0,3.01,0.01)
s = sin(2*pi*t)
c = sin(4*pi*t)
ax.fill(t, s, 'b', t, c, 'g', alpha=0.2)
plt.show()
