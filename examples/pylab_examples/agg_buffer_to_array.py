import matplotlib
matplotlib.use('Agg')
from pylab import figure, show
import numpy as np

# make an agg figure
fig = figure()
ax = fig.add_subplot(111)
ax.plot([1,2,3])
ax.set_title('a simple figure')
fig.canvas.draw()

# grab rhe pixel buffer and dumpy it into a numpy array
buf = fig.canvas.buffer_rgba(0,0)
l, b, w, h = fig.bbox.bounds
X = np.fromstring(buf, np.uint8)
X.shape = h,w,4

# now display the array X as an Axes in a new figure
fig2 = figure()
ax2 = fig2.add_subplot(111, frameon=False)
ax2.imshow(X)
show()
