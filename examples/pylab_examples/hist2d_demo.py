from pylab import *
x = randn(1000)
y = randn(1000)+5

#normal distribution center at x=0 and y=5
hist2d(x,y,bins=40)
show()
