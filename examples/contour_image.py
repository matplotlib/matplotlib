#!/usr/bin/env python
'''
Test combinations of contouring, filled contouring, and image plotting.
For contour labelling, see contour_demo.py.

The emphasis in this demo is on showing how to make contours register
correctly on images, and on how to get both of them oriented as
desired.  In particular, note the usage of the "origin" and "extent"
keyword arguments to imshow and contour.
'''
from pylab import *

#Default delta is large because that makes it fast, and it illustrates
# the correct registration between image and contours.
delta = 0.5

extent = (-3,4,-4,3)

x = arange(-3.0, 4.001, delta)
y = arange(-4.0, 3.001, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = (Z1 - Z2) * 10

levels = arange(-2.0,1.6,0.4)

figure()


subplot(2,2,1)

levs1, colls = contourf(X, Y, Z, levels,
                        cmap=cm.jet,
                        )
#If we want lines as well as filled regions, we need to call
# contour separately; don't try to change the edgecolor or edgewidth
# of the polygons in the collections returned by contourf.
# Use levels output from previous call to guarantee they are the same.
levs2, colls2 = contour(X, Y, Z, levs1,
                        colors = 'k',
                        hold='on')
# We don't really need dashed contour lines to indicate negative
# regions, so let's turn them off.
for c in colls2:
    c.set_linestyle('solid')

# It is easier here to make a separate call to contour than
# to set up an array of colors and linewidths.
# We are making a thick green line as a zero contour.
# Specify the zero level as a tuple with only 0 in it.
levs3, colls3 = contour(X, Y, Z, (0,),
                        colors = 'g',
                        linewidths = 2,
                        hold='on')
title('Filled contours')
#colorbar()
hot()
# To Do: make a discrete colorbar to match filled contours.

subplot(2,2,2)

imshow(Z, extent=extent)
v = axis()
contour(Z, levels, hold='on', colors = 'k', origin='upper', extent=extent)
axis(v)
title("Image, origin 'upper'")

subplot(2,2,3)

imshow(Z, origin='lower', extent=extent)
v = axis()
contour(Z, levels, hold='on', colors = 'k', origin='lower', extent=extent)
axis(v)
title("Image, origin 'lower'")

subplot(2,2,4)

# We will use the interpolation "nearest" here to show the actual
# image pixels.
# Note that the contour lines don't extend to the edge of the box.
# This is intentional. The Z values are defined at the center of each
# image pixel (each color block on the following subplot), so the
# domain that is contoured does not extend beyond these pixel centers.
imshow(Z, interpolation='nearest', extent=extent)
v = axis()
contour(Z, levels, hold='on', colors = 'k', origin='image', extent=extent)
axis(v)
ylim = get(gca(), 'ylim')
setp(gca(), ylim=ylim[::-1])
title("Image, origin from rc, reversed y-axis")

#savefig('contour_image')

show()
