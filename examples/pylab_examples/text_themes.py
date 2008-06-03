#!/usr/bin/env python
from pylab import *

font = {'family'     : 'serif',
        'color'      : 'r',
        'weight' : 'normal',
        'size'   : 12,
        }

def f(t):
    s1 = cos(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, 0.1)
t2 = arange(0.0, 5.0, 0.02)

plot(t1, f(t1), 'bo', t2, f(t2), 'k')
title('Damped exponential decay', font, size='large', color='r')
text(2, 0.65, r'$\cos(2 \pi t) \exp(-t)$', color='k')
xlabel('time (s)', font, style='italic')
ylabel('voltage (mV)', font)

#savefig('text_themes')
show()
