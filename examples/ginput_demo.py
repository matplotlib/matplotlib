from pylab import arange, plot, sin, ginput, show
t = arange(10)
plot(t, sin(t))
print "Please click"
ginput(3, verbose=True)
show()
