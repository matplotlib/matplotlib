from pylab import subplot, sin, pi, arange, setp, show
ax = subplot(111)
t = arange(0.0,3.0,0.01)
s = sin(2*pi*t)
c = sin(4*pi*t)

p = ax.fill(t,s,'b',t,c,'g')
setp(p, alpha=0.2)
ax.vlines( [1.5], -1.0, 1.0 )
show()
