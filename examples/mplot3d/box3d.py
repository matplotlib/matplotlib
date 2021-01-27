"""
=================================
3D surface plot using Matplotlib
=================================
by: Iury T. Simoes-Sousa (iuryt)

The strategy is to select the data from each surface and plot
contours separately.
To use this feature you need to have gridded coordinates.

The contour plot from Matplotlib has zdir argument that defines
the normal coordinate to the plotted surface.

The offset argument defines the offset applied to the contourf surface.
"""

import matplotlib.pyplot as plt
import numpy as np

# Define dimensions
Nx, Ny, Nz = 100, 300, 500
X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))

# Create fake data
da = (((X+100)**2 + (Y-20)**2 + 2*Z)/1000+1)

vmin = da.min()
vmax = da.max()
# Key arguments for contour plots
kw = {
    'vmin': vmin,
    'vmax': vmax,
    'levels': np.linspace(vmin, vmax, 10),
    'cmap': 'viridis',
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111, projection='3d')

# Plot contour surfaces
_ = ax.contourf(
    X[:, :, 0], Y[:, :, 0], da[:, :, 0],
    zdir='z', offset=0, **kw
)
_ = ax.contourf(
    X[0, :, :], da[0, :, :], Z[0, :, :],
    zdir='y', offset=0, **kw
)
C = ax.contourf(
    da[:, -1, :], Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), **kw
)
# --


# Set limits of the plot from coord limits
ax.set(
    xlim=[X.min(), X.max()],
    ylim=[Y.min(), Y.max()],
    zlim=[Z.min(), Z.max()],
)

color = '0.4'  # color of the line of the corners
# Get xlim,ylim and zlim
xlim = np.array(ax.get_xlim())
ylim = np.array(ax.get_ylim())
zlim = np.array(ax.get_zlim())

# Plot corners
ax.plot(
    xlim*0+xlim[1], ylim, zlim*0, color,
    linewidth=1, zorder=1e4,
)
ax.plot(
    xlim, ylim*0+ylim[0], zlim*0, color,
    linewidth=1, zorder=1e4,
)
ax.plot(
    xlim*0+xlim[1], ylim*0+ylim[0], zlim, color,
    linewidth=1, zorder=1e4,
)

# Set labels and zticks
ax.set(
    xlabel='\n X [km]',
    ylabel='\n Y [km]',
    zlabel='\n Z [m]',
    zticks=[0, -150, -300, -450],
)

# Set distance and angle view
ax.view_init(40, -30)
ax.dist = 11

# Colorbar
fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Name [units]')
