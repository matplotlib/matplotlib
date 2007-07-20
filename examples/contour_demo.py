#!/usr/bin/env python
'''
Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also contour_image.py.
'''
from pylab import *

rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

delta = 0.025
x = arange(-3.0, 3.0, delta)
y = arange(-2.0, 2.0, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
# difference of Gaussians
Z = 10.0 * (Z2 - Z1)



# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
figure()
CS = contour(X, Y, Z)
clabel(CS, inline=1, fontsize=10)
title('Simplest default with labels')


# You can force all the contours to be the same color.
figure()
CS = contour(X, Y, Z, 6,
             colors='k', # negative contours will be dashed by default
             )
clabel(CS, fontsize=9, inline=1)
title('Single color - negative contours dashed')

# You can set negative contours to be solid instead of dashed:
rcParams['contour.negative_linestyle'] = 'solid'
figure()
CS = contour(X, Y, Z, 6,
             colors='k', # negative contours will be dashed by default
             )
clabel(CS, fontsize=9, inline=1)
title('Single color - negative contours solid')


# And you can manually specify the colors of the contour
figure()
CS = contour(X, Y, Z, 6,
             linewidths=arange(.5, 4, .5),
             colors=('r', 'green', 'blue', (1,1,0), '#afeeee', '0.5')
             )
clabel(CS, fontsize=9, inline=1)
title('Crazy lines')


# Or you can use a colormap to specify the colors; the default
# colormap will be used for the contour lines
figure()
im = imshow(Z, interpolation='bilinear', origin='lower',
            cmap=cm.gray, extent=(-3,3,-2,2))
levels = arange(-1.2, 1.6, 0.2)
CS = contour(Z, levels,
             origin='lower',
             linewidths=2,
             extent=(-3,3,-2,2))

#Thicken the zero contour.
zc = CS.collections[6]
setp(zc, linewidth=4)

clabel(CS, levels[1::2],  # label every second level
       inline=1,
       fmt='%1.1f',
       fontsize=14)

# make a colorbar for the contour lines
CB = colorbar(CS, shrink=0.8, extend='both')

title('Lines with colorbar')
hot()  # Now change the colormap for the contour lines and colorbar
flag()

# We can still add a colorbar for the image, too.
CBI = colorbar(im, orientation='horizontal', shrink=0.8)

# This makes the original colorbar look a bit out of place,
# so let's improve its position.

l,b,w,h = gca().get_position()
ll,bb,ww,hh = CB.ax.get_position()
CB.ax.set_position([ll, b+0.1*h, ww, h*0.8])


#savefig('contour_demo')
show()
