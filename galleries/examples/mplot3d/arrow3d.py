"""
=============
3D arrow plot
=============

Demonstrates plotting arrows in a 3D space.

Here we plot two arrows from the same start point to different
end points. The properties of the second arrow is changed by passing
additional parameters other than ``end`` and ``start`` to
`.patches.FancyArrowPatch`.
"""

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the start and end points of the arrow
start = np.array([0, 0, 0])
end = np.array([1, 1, 1])

# Create the arrow
ax.arrow3d(end, start)

end1 = np.array([1, 2, 3])
# Passing additional keyword arguments to control properties of the arrow.
# If the `start` parameter is not passed, the arrow is drawn from (0, 0, 0).
ax.arrow3d(end1, mutation_scale=20, color='r', arrowstyle='->', linewidth=2)

plt.show()

# %%
# .. tags::
#    plot-type: 3D,
#    level: beginner
