#!/usr/bin/env python

from pylab import *
figure(1)
plot(10*rand(12), 'o')
xlim(0,15)
xticks([2, 4, 8, 12], ('John', 'Hunter', 'Was', 'Here'))

ylim(-1,10)
yticks(range(8))

figure(2)
t = arange(-1,2, .01)
s = sin(2*pi*t)
plot(t,s)
# draw a thick red hline at y=0 that spans the xrange
l = axhline(linewidth=4, color='r')

# draw a default hline at y=1 that spans the xrange
l = axhline(y=1)

# draw a default vline at x=1 that spans the xrange
l = axvline(x=1)

# draw a thick blue vline at x=0 that spans the the upper quadrant of
# the yrange
l = axvline(x=0, ymin=0.75, linewidth=4, color='b')

# draw a default hline at y=.5 that spans the the middle half of
# the axes
l = axhline(y=.5, xmin=0.25, xmax=0.75)

p = axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)

p = axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

axis([-1,2,-1,2])


show()
