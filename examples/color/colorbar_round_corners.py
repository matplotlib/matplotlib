"""
===========================
Colorbar with round corners
===========================

This example demonstrates how to create colorbars with round corners.
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

# with the *round* keyword we specify which edges
# should have round corners
fig.colorbar(pos, ax=ax1, round='both', aspect=15)

ax1.set_title('round="both"')

pos = ax2.imshow(Z, cmap=cmap, interpolation='bicubic')

# the *rounding_size* keyword controls the radius of the corners
fig.colorbar(pos, ax=ax2, round='max', rounding_size=0.5, aspect=15)
ax2.set_title('round="max", rounding_size=0.5')
plt.show()

############################################################################
# Extended colorbar with round corners
# ------------------------------------
#
# The extensions of a colorbar can also be round
# or even mixed (one extension triangular and the other round).

# specify color for extensions
cmap = plt.cm.gray.with_extremes(over='b', under='g')

fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)

pos = ax1.imshow(Z, cmap=cmap, vmin=-1.5, vmax=1.5, interpolation='bicubic')

fig.colorbar(pos, ax=ax1, extend='both', location='bottom', round='both')

ax1.set_title('extend="both", round="both"')

pos = ax2.imshow(Z, cmap=cmap, vmin=-1.5, vmax=1.5, interpolation='bicubic')

# the *extendfrac* keyword can affect the length of the
# round patch as well
fig.colorbar(pos, ax=ax2, location='bottom',
             extend='both', extendfrac=(0.1, 0.05),
             round='min', rounding_size=0.5)

ax2.set_title('extend="both", round="min"')
plt.show()

#############################################################################
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
