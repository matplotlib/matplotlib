"""

Demonstrate how to do two plots on the same axes with different left
right scales.


The trick is to use *2 different axes*.  Turn the axes rectangular
frame off on the 2nd axes to keep it from obscuring the first.
Manually set the tick locs and labels as desired.  You can use
separate matplotlib.ticker formatters and locators as desired since
the two axes are independent

To do the same with different x scales, use the xaxis instance and
call tick_bottom and tick_top in place of tick_left and tick_right

"""

from matplotlib.matlab import *

ax1 = subplot(111)
t = arange(0.0, 10.0, 0.01)
s1 = exp(t)
plot(t, s1, 'b-')
ax1.yaxis.tick_left()


# turn off the 2nd axes rectangle with frameon kwarg
ax2 = subplot(111, frameon=False) 
s2 = sin(2*pi*t)
plot(t, s2, 'r.')
ax2.yaxis.tick_right()


xlabel('time (s)')

show()
