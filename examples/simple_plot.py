#!/usr/bin/env python

from pylab import *

t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)
plot(t, s)

xlabel('time (s)')
ylabel('voltage (mV)')
title('About as simple as it gets, folks')
grid(True)
#savefig('simple_plot.png')
savefig('simple_plot')

show()
