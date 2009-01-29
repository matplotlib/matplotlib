"""
Clipping to arbitrary patches and paths
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches


fig = plt.figure()
ax = fig.add_subplot(111, frameon=False, xticks=[], yticks=[])

im = ax.imshow(np.random.rand(10,10))

patch = patches.Circle((300,300), radius=100)
im.set_clip_path(patch)

plt.show()


