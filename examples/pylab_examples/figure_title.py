#!/usr/bin/env python
from matplotlib.font_manager import FontProperties
from pylab import *

def f(t):
    s1 = cos(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, 0.1)
t2 = arange(0.0, 5.0, 0.02)
t3 = arange(0.0, 2.0, 0.01)


subplot(121)
plot(t1, f(t1), 'bo', t2, f(t2), 'k')
title('subplot 1')
ylabel('Damped oscillation')
suptitle('This is a somewhat long figure title', fontsize=16)


subplot(122)
plot(t3, cos(2*pi*t3), 'r--')
xlabel('time (s)')
title('subplot 2')
ylabel('Undamped')

show()

