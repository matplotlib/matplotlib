# use masked arrays to plot a line with different colors by y-value
import matplotlib.numerix.ma as ma
from matplotlib.numerix import logical_or
from pylab import  plot, show, arange, sin, pi

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)

upper = 0.77
lower = -0.77


supper = ma.masked_where(s < upper, s)
slower = ma.masked_where(s > lower, s)
smiddle = ma.masked_where(logical_or(s<lower, s>upper), s)

plot(t, slower, 'r', t, smiddle, 'b', t, supper, 'g')
show()
