#!/usr/bin/env python
#Controlling the properties of axis text using handles

# See examples/text_themes.py for a more elegant, pythonic way to control
# fonts.  After all, if we were slaves to MATLAB , we wouldn't be
# using python!

from pylab import *


def f(t):
    s1 = sin(2*pi*t)
    e1 = exp(-t)
    return multiply(s1,e1)

t1 = arange(0.0, 5.0, 0.1)
t2 = arange(0.0, 5.0, 0.02)


fig, ax = plt.subplots()
plot(t1, f(t1), 'bo', t2, f(t2), 'k')
text(3.0, 0.6, 'f(t) = exp(-t) sin(2 pi t)')
ttext = title('Fun with text!')
ytext = ylabel('Damped oscillation')
xtext = xlabel('time (s)')

setp(ttext, size='large', color='r', style='italic')
setp(xtext, size='medium', name='courier', weight='bold', color='g')
setp(ytext, size='medium', name='helvetica', weight='light', color='b')
show()
