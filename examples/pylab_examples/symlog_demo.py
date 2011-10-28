#!/usr/bin/env python
from pylab import *

dt = 0.01
x = arange(-50.0, 50.0, dt)
y = arange(0, 100.0, dt)

subplot(311)
plot(x, y)
xscale('symlog')
ylabel('symlogx')
grid(True)
gca().xaxis.grid(True, which='minor')  # minor grid on too

subplot(312)
plot(y, x)
yscale('symlog')
ylabel('symlogy')


subplot(313)
plot(x, np.sin(x / 3.0))
xscale('symlog')
yscale('symlog', linthreshy=0.015)
grid(True)
ylabel('symlog both')

show()
