"""
=========================================
Using a quiver key with bbox on a map
=========================================

This example demonstrates how to use the `bbox` argument in
`.Axes.quiverkey` to add a background box to the key label.

The vector is plotted on a global map using `cartopy`, and the key is shown
in normalized axes coordinates (`coordinates='axes'`). The background box
enhances readability, especially over detailed geographic backgrounds.

A rounded, padded white box is used for the quiver key to ensure it stands out
on the map.
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

fig = plt.figure(figsize=(12, 6))

# Create axes with a map projection
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines()

# Sample vector (in geographic coordinates)
q = ax.quiver([0], [0], [1], [1], transform=ccrs.PlateCarree())

# Add a quiver key with a visible bbox inside the map area
qk = ax.quiverkey(q, X=0.85, Y=0.1, U=1, label='1 unit',
                  labelpos='E',
                  coordinates='axes',  # normalized axes coordinates (0–1)
                  bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.show()
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.quiverkey` / `matplotlib.pyplot.quiverkey`
#    - `matplotlib.patches.FancyBboxPatch`
