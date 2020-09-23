"""
==================
Rasterization Demo
==================

Rasterization is a method where an image described in a vector graphics
format is being converted into a raster image (pixels).

Individual artists can be rasterized for saving to a vector backend
such as PDF, SVG, or PS as embedded images.  This can be useful to
reduce the file size of large artists, while maintaining the
advantages of vector graphics for other artists such as the axes
and annotations.  For instance a complicated `~.Axes.pcolormesh` or
`~.Axes.contourf` can be made significantly simpler by rasterizing.
Note that the size and resolution of the rasterized artist is
controlled by its physical size and the value of the ``dpi`` kwarg
passed to `~.Figure.savefig`.
"""

import numpy as np
import matplotlib.pyplot as plt

d = np.arange(100).reshape(10, 10)  # the values to be color-mapped
x, y = np.meshgrid(np.arange(11), np.arange(11))

theta = 0.25*np.pi
xx = x*np.cos(theta) - y*np.sin(theta)  # rotate x by -theta
yy = x*np.sin(theta) + y*np.cos(theta)  # rotate y by -theta

# Plot the rasterized and non-rasterized plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True)

# Create a pseudocolor non-rastertized plot with a non-regular rectangular grid
ax1.set_aspect(1)
ax1.pcolormesh(xx, yy, d)
ax1.set_title("No Rasterization")

# Create a pseudocolor rastertized plot with a non-regular rectangular grid
ax2.set_aspect(1)
ax2.set_title("Rasterization")
m = ax2.pcolormesh(xx, yy, d, rasterized=True)

# Create a pseudocolor non-rastertized plot with a non-regular rectangular
# grid and an overlapped "Text"
ax3.set_aspect(1)
ax3.pcolormesh(xx, yy, d)
ax3.text(0.5, 0.5, "Text", alpha=0.2,
         va="center", ha="center", size=50, transform=ax3.transAxes)
ax3.set_title("No Rasterization")

# Create a pseudocolor rastertized plot with a non-regular rectangular
# grid and an overlapped "Text"
ax4.set_aspect(1)
m = ax4.pcolormesh(xx, yy, d, zorder=-20)
ax4.text(0.5, 0.5, "Text", alpha=0.2, zorder=-15,
         va="center", ha="center", size=50, transform=ax4.transAxes)
# Set zorder value below which artists will be rasterized
ax4.set_rasterization_zorder(-10)
ax4.set_title("Rasterization z$<-10$")
# ax2.title.set_rasterized(True) # should display a warning

# Save files in pdf and eps format
plt.savefig("test_rasterization.pdf", dpi=150)
plt.savefig("test_rasterization.eps", dpi=150)

if not plt.rcParams["text.usetex"]:
    plt.savefig("test_rasterization.svg", dpi=150)
    # svg backend currently ignores the dpi
