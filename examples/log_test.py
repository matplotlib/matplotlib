#!/usr/bin/env python
from matplotlib.matlab import *

dt = 0.01
t = arange(dt, 200.0, dt)

semilogx(t, exp(-t/5.0))
grid(True)
#savefig('log_demo')
show()
