import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Path = mpath.Path

fig = plt.figure()
ax = fig.add_subplot(111)

pathdata = [
    (Path.MOVETO, (0, 0)),
    (Path.CURVE4, (-1, 0)),
    (Path.CURVE4, (-1, 1)),
    (Path.CURVE4, (0, 1)),
    (Path.LINETO, (2, 1)),
    (Path.CURVE4, (3, 1)),
    (Path.CURVE4, (3, 0)),
    (Path.CURVE4, (2, 0)),
    (Path.CLOSEPOLY, (0, 0)),
    ]

codes, verts = zip(*pathdata)
path = mpath.Path(verts, codes)
patch = mpatches.PathPatch(path, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)


ax.set_xlim(-5,5)
ax.set_ylim(-5,5)

plt.show()


