"""
==========
Hatches Styles Demo
==========

Hatching (pattern filled polygons) is supported currently in the PS,
PDF, SVG and Agg backends only.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

fig, ax = plt.subplots()

x = np.repeat([1, 4, 7, 10, 13], 2)
y = np.tile([2, 5], 5)

pos = np.column_stack((x, y))
text_pos = pos + [0.9, -0.5]

hashes = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

for h in range(len(hashes)):
    ax.add_patch(Rectangle(pos[h], 2, 2, fill=False, hatch=hashes[h]))
    ax.text(text_pos[h][0], text_pos[h][1], hashes[h], fontsize=15)

plt.axis('equal')
plt.axis('off')
plt.tight_layout()

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

import matplotlib
matplotlib.patches
matplotlib.patches.Rectangle
matplotlib.axes.Axes.add_patch
matplotlib.axes.Axes.text
