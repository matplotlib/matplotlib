from pylab import *


y1 = arange(10)
y1.shape = 1,10

y2 = arange(10)
y2.shape = 10,1

subplot(411)
plot(y1)
subplot(412)
plot(y2)

#print ravel(y1), ravel(y2), len(y1.shape), len(y2.shape)
subplot(413)
plot(y1, y2)

subplot(414)
X = rand(10,10)
plot(X[:,1], X[1,:], 'o')
#savefig('shapes')
show()
