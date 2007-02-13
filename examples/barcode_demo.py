from pylab import figure, show, cm, nx

# the bar
x = nx.mlab.rand(500)
x[x>0.7] = 1.
x[x<=0.7] = 0.

axprops = dict(xticks=[], yticks=[])
barprops = dict(aspect='auto', cmap=cm.binary, interpolation='nearest')

fig = figure()

# a vertical barcode
x.shape = len(x), 1
ax = fig.add_axes([0.1, 0.3, 0.1, 0.6], **axprops)
ax.imshow(x, **barprops)


# a horizontal barcode
x.shape = 1, len(x)
ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
ax.imshow(x, **barprops)

fig.savefig('barcode.png', dpi=100)
show()

