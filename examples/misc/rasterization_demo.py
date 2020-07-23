"""
==================
Rasterization Demo
==================
"""

# Rasterization is a method where an image described in a vector graphics
# format is being converted into a raster image (pixels).


# This demo is showing how you may want to balance most things such as text
# and lines being vectorized to get all the benefits of vector graphics, but
# turning some artists into rasterized images. This method will save
# computational resources because embedding a 2k by 2k mesh in SVG
# results in a huge file that takes a long time to both save and load.


import numpy as np
import matplotlib.pyplot as plt

# Set 'd' to be the scalar 2-D array where the values will be color-mapped
d = np.arange(100).reshape(10, 10)
# x, y return 2-D grid coordinates based on the coordinates contained
x, y = np.meshgrid(np.arange(11), np.arange(11))

theta = 0.25*np.pi
xx = x*np.cos(theta) - y*np.sin(theta)  # rotates x by -theta
yy = x*np.sin(theta) + y*np.cos(theta)  # rotates y by -theta

# Plot the rasterized and non-rasterized plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# Create a pseudocolor non-rastertized plot with a non-regular rectangular grid
ax1.set_aspect(1)
ax1.pcolormesh(xx, yy, d)
ax1.set_title("No Rasterization")

# Create a pseudocolor rastertized plot with a non-regular rectangular grid
ax2.set_aspect(1)
ax2.set_title("Rasterization")
m = ax2.pcolormesh(xx, yy, d)
# Force rasterized drawing in vector backend output
m.set_rasterized(True)

# Create a pseudocolor non-rastertized plot with a non-regular rectangular grid and an overlapped "Text"
ax3.set_aspect(1)
ax3.pcolormesh(xx, yy, d)
ax3.text(0.5, 0.5, "Text", alpha=0.2,
         va="center", ha="center", size=50, transform=ax3.transAxes)
ax3.set_title("No Rasterization")

# Create a pseudocolor rastertized plot with a non-regular rectangular grid and an overlapped "Text"
ax4.set_aspect(1)
m = ax4.pcolormesh(xx, yy, d)
m.set_zorder(-20)
ax4.text(0.5, 0.5, "Text", alpha=0.2,
         zorder=-15,
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
