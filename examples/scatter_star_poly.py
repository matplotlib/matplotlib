import pylab

x = pylab.rand(10)
y = pylab.rand(10)

pylab.subplot(321)
pylab.scatter(x,y,s=80,marker=">")

pylab.subplot(322)
pylab.scatter(x,y,s=80,marker=(5,0))

verts = zip([-1.,1.,1.],[-1.,-1.,1.])
pylab.subplot(323)
pylab.scatter(x,y,s=80,marker=(verts,0))
# equivalent:
#pylab.scatter(x,y,s=80,marker=None, verts=verts)

pylab.subplot(324)
pylab.scatter(x,y,s=80,marker=(5,1))

pylab.subplot(325)
pylab.scatter(x,y,s=80,marker='+')

pylab.subplot(326)
pylab.scatter(x,y,s=80,marker=(5,2), edgecolor='g')

pylab.show()
