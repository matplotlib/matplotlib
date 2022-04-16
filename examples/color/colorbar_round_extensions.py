"""
==============================
Colorbar with round extensions
==============================

This example demonstrates how to change the shape of colorbar extensions.
"""
import numpy as np
import matplotlib.pyplot as plt

# setup some generic data
N = 37
x, y = np.mgrid[:N, :N]
Z = (np.cos(x*0.2) + np.sin(y*0.3))

cmap = plt.get_cmap("plasma")

fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)

pos = ax1.imshow(Z, cmap=cmap, interpolation='bicubic')

# setting the *extend_shape* keyword to 'round' will round
# the corners of the extensions
fig.colorbar(pos, ax=ax1, extend='both', extend_shape='round', aspect=15)

ax1.set_title('extend_shape="round"')

pos = ax2.imshow(Z, cmap=cmap, interpolation='bicubic')

# *extend_shape* can be a tuple with different shapes for the minimum
# and maximum extensions of the colorbar
# the *rounding_size* keyword controls the radius of the corners
fig.colorbar(pos, ax=ax2, extend='both', extend_shape=('round', 'tri'),
             rounding_size=0.5, aspect=15)
ax2.set_title('extend_shape=("round", "tri")')
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
