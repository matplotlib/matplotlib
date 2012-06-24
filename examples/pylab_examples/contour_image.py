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

levels = arange(-2.0, 1.601, 0.4) # Boost the upper limit to avoid truncation
                                  # errors.

norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=-abs(Z).max())
cmap = cm.PRGn

figure()


subplot(2,2,1)

cset1 = contourf(X, Y, Z, levels,
                        cmap=cm.get_cmap(cmap, len(levels)-1),
                        norm=norm,
                        )
# It is not necessary, but for the colormap, we need only the
# number of levels minus 1.  To avoid discretization error, use
# either this number or a large number such as the default (256).

#If we want lines as well as filled regions, we need to call
# contour separately; don't try to change the edgecolor or edgewidth
# of the polygons in the collections returned by contourf.
# Use levels output from previous call to guarantee they are the same.
cset2 = contour(X, Y, Z, cset1.levels,
                        colors = 'k',
                        hold='on')
# We don't really need dashed contour lines to indicate negative
# regions, so let's turn them off.
for c in cset2.collections:
    c.set_linestyle('solid')

# It is easier here to make a separate call to contour than
# to set up an array of colors and linewidths.
# We are making a thick green line as a zero contour.
# Specify the zero level as a tuple with only 0 in it.
cset3 = contour(X, Y, Z, (0,),
                colors = 'g',
                linewidths = 2,
                hold='on')
title('Filled contours')
colorbar(cset1)
#hot()


subplot(2,2,2)

imshow(Z, extent=extent, cmap=cmap, norm=norm)
v = axis()
contour(Z, levels, hold='on', colors = 'k',
        origin='upper', extent=extent)
axis(v)
title("Image, origin 'upper'")

subplot(2,2,3)

imshow(Z, origin='lower', extent=extent, cmap=cmap, norm=norm)
v = axis()
contour(Z, levels, hold='on', colors = 'k',
        origin='lower', extent=extent)
axis(v)
title("Image, origin 'lower'")

subplot(2,2,4)

# We will use the interpolation "nearest" here to show the actual
# image pixels.
# Note that the contour lines don't extend to the edge of the box.
# This is intentional. The Z values are defined at the center of each
# image pixel (each color block on the following subplot), so the
# domain that is contoured does not extend beyond these pixel centers.
im = imshow(Z, interpolation='nearest', extent=extent, cmap=cmap, norm=norm)
v = axis()
contour(Z, levels, hold='on', colors = 'k',
        origin='image', extent=extent)
axis(v)
ylim = get(gca(), 'ylim')
setp(gca(), ylim=ylim[::-1])
title("Image, origin from rc, reversed y-axis")
colorbar(im)

show()
