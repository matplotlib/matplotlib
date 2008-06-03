#!/usr/bin/env python
from pylab import *

dt = 0.01
t = arange(dt, 20.0, dt)

subplot(311)
semilogy(t, exp(-t/5.0))
ylabel('semilogy')
grid(True)

subplot(312)
semilogx(t, sin(2*pi*t))
ylabel('semilogx')



grid(True)
gca().xaxis.grid(True, which='minor')  # minor grid on too

subplot(313)
loglog(t, 20*exp(-t/10.0), basex=4)
grid(True)
ylabel('loglog base 4 on x')
savefig('log_demo')
show()
