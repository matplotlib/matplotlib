#!/usr/bin/env python
from pylab import *
figure(1)
ax = axes([0.1, 0.1, 0.8, 0.7])
t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)
plot(t, s)

xlabel('time (s)')
ylabel('voltage (mV)')
##title(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!", 
##        fontsize=20)
title(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty{-e^{i\pi} \over 2^n}$!", 
        fontsize=20)
grid(True)
savefig('tex_demo')

show()
