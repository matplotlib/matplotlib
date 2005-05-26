#!/usr/bin/env python
from pylab import *
figure(1)
ax = axes([0.1, 0.1, 0.8, 0.7])
t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)+2
plot(t, s)

xlabel(r'\bf{time (s)}')
ylabel(r'\it{voltage (mV)}',fontsize=16)
title(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!", 
        {'bbox':{'fc':0.8,'pad':0}}, fontsize=16, color='r')
grid(True)
savefig('tex_demo.eps')

show()
