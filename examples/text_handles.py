#!/usr/bin/env python
#Controlling the properties of axis text using handles

# See examples/text_themes.py for a more elegant, pythonic way to control
# fonts.  After all, if we were slaves to matlab , we wouldn't be
# using python!

from matplotlib.matlab import *


def f(t):
    s1 = sin(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, 0.1)
t2 = arange(0.0, 5.0, 0.02)


subplot(111)
plot(t1, f(t1), 'bo', t2, f(t2), 'k')
text(3.0, 0.6, 'f(t) = exp(-t) sin(2 pi t)')
ttext = title('Fun with text!')
ytext = ylabel('Damped oscillation')
xtext = xlabel('time (s)')

set(ttext, size='large', color='r', style='italic')
set(xtext, size='medium', name='courier', weight='bold', color='g')
set(ytext, size='medium', name='helvetica', weight='light', color='b')
show()
