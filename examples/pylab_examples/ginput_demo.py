# -*- noplot -*-

from __future__ import print_function

from pylab import arange, plot, sin, ginput, show
t = arange(10)
plot(t, sin(t))
print("Please click")
x = ginput(3)
print("clicked",x)
show()
