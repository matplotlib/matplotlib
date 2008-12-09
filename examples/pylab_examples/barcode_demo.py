from matplotlib.pyplot import figure, show, cm
from numpy import where
from numpy.random import rand

# the bar
x = where(rand(500)>0.7, 1.0, 0.0)

axprops = dict(xticks=[], yticks=[])
barprops = dict(aspect='auto', cmap=cm.binary, interpolation='nearest')

fig = figure()

# a vertical barcode -- this is broken at present
x.shape = len(x), 1
ax = fig.add_axes([0.1, 0.3, 0.1, 0.6], **axprops)
ax.imshow(x, **barprops)

x = x.copy()
# a horizontal barcode
x.shape = 1, len(x)
ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
ax.imshow(x, **barprops)


show()

