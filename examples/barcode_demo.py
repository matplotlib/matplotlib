from pylab import figure, show, cm, nx

axprops = dict(xticks=[], yticks=[])
barprops = dict(aspect='auto', cmap=cm.binary, interpolation='nearest')

fig = figure()

# a vertical barcode
x = nx.mlab.rand(500,1)
x[x>0.8] = 1.
x[x<=0.8] = 0.
ax = fig.add_axes([0.1, 0.3, 0.1, 0.6], **axprops)
ax.imshow(x, **barprops)


# a horizontal barcode
x = nx.mlab.rand(1,500)
x[x>0.8] = 1.
x[x<=0.8] = 0.
ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
ax.imshow(x, **barprops)

fig.savefig('barcode.png', dpi=100)
show()

