#!/usr/bin/env python
from matplotlib.matlab import *

figure(1)
t = arange(0.0, 1.0, 0.01)
s = sin(2*2*pi*t)
plot(t, s)

xlabel('time (s)')
ylabel('voltage (mV)')
title('About as simple as it gets, folks')
grid(True)
#axis([0,1,-1,1])
savefig('simple_plot', dpi=300)

show()
