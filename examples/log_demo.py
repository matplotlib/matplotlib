#!/usr/bin/env python
from matplotlib.matlab import *

dt = 0.01
t = arange(dt, 20.0, dt)

subplot(311)
semilogy(t, exp(-t/5.0))
title('semilogy')

subplot(312)
semilogx(t, sin(2*pi*t))
title('semilogx')

subplot(313)
loglog(t, 20*exp(-t/10.0))
gca().set_xscale('log',base=4)
gca().set_yscale('log',base=8,subs=[2,4,6])
grid(True)
title('loglog with custom base and subs logscaling')
#savefig('log_demo')
show()
