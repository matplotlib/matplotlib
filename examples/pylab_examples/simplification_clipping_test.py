from pylab import *

t = arange(0.0, 2.0, 0.01)
s = sin(2*pi*t)
plot(t, s, linewidth=1.0)

ylim((-0.20, -0.28))

title('Should see four lines extending from bottom to top')
grid(True)
show()
