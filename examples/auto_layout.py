#!/usr/bin/env python
"""
Example: simple line plot.
Show how to make and save a simple line plot with labels, title and grid
"""
from pylab import *

t = arange(0.0, 1.0+0.01, 0.01)
s = cos(2*2*pi*t)
ax1 = subplot(211)
plot(t, s, '-', lw=2)

xlabel('xlabel for top axes')
ylabel('ylabel on the right')
title('About as simple as it gets, folks')
grid(True)
ax1.yaxis.set_label_position('right')
ax1.xaxis.set_ticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
for label in ax1.get_xticklabels():
    label.set_rotation(45)

ax2 = subplot(212)
plot(t, s, '-', lw=2)
grid(True)
xlabel('xlabel for bottom axes (the ticks are on the top for no good reason)')
ylabel('I\'m a lefty')
ax2.xaxis.set_label_position('bottom')
ax2.xaxis.set_ticks_position('top')


#savefig('simple_plot.png')
savefig('simple_plot')

show()
