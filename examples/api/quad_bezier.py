import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
pp1 = mpatches.PathPatch(
    mpath.Path([(0, 0), (1, 0), (1, 1), (0, 0)], [1, 3, 3, 5]),
    fc="none", transform=ax.transData)

ax.add_patch(pp1)
ax.plot([0.75], [0.25], "ro")
ax.set_title('The red point should be on the path')

plt.draw()

