"""
==================
Image Antialiasing
==================

Images are represented by discrete pixels, either on the screen or in an
image file.  When data that makes up the image has a different resolution
than its representation on the screen we will see aliasing effects.

The default image interpolation in Matplotlib is 'antialiased'.  This uses a
hanning interpolation for reduced aliasing in most situations. Only when there
is upsampling by a factor of 1, 2 or >=3 is 'nearest' neighbor interpolation
used.

Other anti-aliasing filters can be specified in `.Axes.imshow` using the
*interpolation* kwarg.
"""

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# First we generate an image with varying frequency content:
x = np.arange(500) / 500 - 0.5
y = np.arange(500) / 500 - 0.5

X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
f0 = 10
k = 250
a = np.sin(np.pi * 2 * (f0 * R + k * R**2 / 2))


###############################################################################
# The following images are subsampled from 1000 data pixels to 604 rendered
# pixels. The Moire patterns in the "nearest" interpolation are caused by the
# high-frequency data being subsampled.  The "antialiased" image
# still has some Moire patterns as well, but they are greatly reduced.
fig, axs = plt.subplots(1, 2, figsize=(7, 4), constrained_layout=True)
for n, interp in enumerate(['nearest', 'antialiased']):
    im = axs[n].imshow(a, interpolation=interp, cmap='gray')
    axs[n].set_title(interp)
plt.show()

###############################################################################
# Even up-sampling an image will lead to Moire patterns unless the upsample
# is an integer number of pixels.
fig, ax = plt.subplots(1, 1, figsize=(5.3, 5.3))
ax.set_position([0, 0, 1, 1])
im = ax.imshow(a, interpolation='nearest', cmap='gray')
plt.show()

###############################################################################
# The patterns aren't as bad, but still benefit from anti-aliasing
fig, ax = plt.subplots(1, 1, figsize=(5.3, 5.3))
ax.set_position([0, 0, 1, 1])
im = ax.imshow(a, interpolation='antialiased', cmap='gray')
plt.show()

###############################################################################
# If the small Moire patterns in the default "hanning" antialiasing are
# still undesireable, then we can use other filters.
fig, axs = plt.subplots(1, 2, figsize=(7, 4), constrained_layout=True)
for n, interp in enumerate(['hanning', 'lanczos']):
    im = axs[n].imshow(a, interpolation=interp, cmap='gray')
    axs[n].set_title(interp)
plt.show()


#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions and methods is shown
# in this example:

import matplotlib
matplotlib.axes.Axes.imshow
