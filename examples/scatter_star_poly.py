import pylab

x = pylab.nx.mlab.rand(10)
y = pylab.nx.mlab.rand(10)

pylab.subplot(221)
pylab.scatter(x,y,s=80,marker=">")

pylab.subplot(222)
pylab.scatter(x,y,s=80,marker=(5,0))

verts = zip([-1.,1.,1.],[-1.,-1.,1.])
pylab.subplot(223)
pylab.scatter(x,y,s=80,marker=(verts,0))
# equivalent:
#pylab.scatter(x,y,s=80,marker=None, verts=verts)

pylab.subplot(224)
pylab.scatter(x,y,s=80,marker=(5,1))

pylab.show()
