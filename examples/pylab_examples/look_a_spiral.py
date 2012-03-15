"""
This small script shows two very useful things : how to display an
image, i.e., a matrix of values, and how to plot contour lines, that
is, lines following a constant value in the matrix. Using hold(True),
we can overlay the two images, and in fact, any other line.

The imshow(z) method works by assigning a color to the values of the
matrix z using a colormap. In the following example, the default
colormap jet is used. The colormap basically consists of an index
ranging from 0 to 1, to which rgb values are assigned. There exists a
range of other colormaps similar to those in Matlab (TM), and it is
not so difficult to create one from scratch. In the example, two
options are passed with imshow : extent and origin. The extent
argument simply tells matplotlib what the coordinates of the image are
so it can label the x and y axis correctly. The origin option can take
one of two values, 'upper' or 'lower'. The default is 'upper', which
will just display the matrix as it is, with the (1,1) element in the
upper left corner. With the 'lower' argument, the matrix is mirrored
so the (1,1) element goes in the lower left corner. There are a lot
more options to control imshow's behavior, such as cmap (colormap),
norm (the normalization method), interpolation, alpha
(transparency). With some tweaking, you'll get the result you want.

# Formulas from C. Pickover

"""

import numpy as np
import matplotlib.pyplot as plt

# Creating the grid of coordinates x,y 
x,y = np.ogrid[-1.:1.:.01, -1.:1.:.01]

z = 3*y*(3*x**2-y**2)/4 + .5*np.cos(6*np.pi * np.sqrt(x**2 +y**2) + np.arctan2(x,y))

plt.hold(True)
# Creating image
plt.imshow(z, origin='lower', extent=[-1,1,-1,1])

# Plotting contour lines
plt.contour(z, origin='lower', extent=[-1,1,-1,1])

plt.xlabel('x')
plt.ylabel('y')
plt.title('A spiral !')

# Adding a line plot slicing the z matrix just for fun. 
plt.plot(x[:], z[50, :])

plt.show()
