"""
This example shows how to use a path patch to draw a bunch of
rectangles.  The technique of using lots of Rectangle instances, or
the faster method of using PolyCollections, were implemented before we
had proper paths with moveto/lineto, closepoly etc in mpl.  Now that
we have them, we can draw collections of regularly shaped objects with
homogeous properties more efficiently with a PathCollection.  This
example makes a histogram -- its more work to set up the vertex arrays
at the outset, but it should be much faster for large numbers of
objects
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

fig = plt.figure()
ax = fig.add_subplot(111)

# histogram our data with numpy
data = np.random.randn(1000)
n, bins = np.histogram(data, 50)


# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

XY = np.zeros((nrects, 4, 2))
XY[:,0,0] = left
XY[:,0,1] = bottom

XY[:,1,0] = left
XY[:,1,1] = top

XY[:,2,0] = right
XY[:,2,1] = top

XY[:,3,0] = right
XY[:,3,1] = bottom



barpath = path.Path.make_compound_path_from_polys(XY)
print barpath.codes[:7], barpath.codes[-7:]
patch = patches.PathPatch(barpath, facecolor='blue', edgecolor='gray', alpha=0.8)
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())

plt.show()
