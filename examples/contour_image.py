#!/usr/bin/env python
'''
contour_image.py [options]

Test combinations of contouring, filled contouring, and image plotting.
'''
from pylab import *
import matplotlib.numerix
print 'Using', matplotlib.numerix.which[0]

from optparse import OptionParser
parser = OptionParser(usage=__doc__.rstrip())
parser.add_option("-b", "--badmask", dest="badmask", default="none",
                  help="'none', 'edge', 'interior'; default is 'none'")
parser.add_option("-d", "--delta", dest="delta", type="float", default=0.5,
                  help="grid increment in x and y; default is 0.5")
parser.add_option("-s", "--save", dest="save", default=None, metavar="FILE",
                  help="Save to FILE; default is to not save.")

#Default delta is large because that makes it fast, and it illustrates
# the correct registration between image and contours.

import sys
# We have to strip out the numerix arguments before passing the
# input arguments to the parser.
Args = [arg for arg in sys.argv if arg not in ('--Numeric', '--numarray')]
options, args = parser.parse_args(Args)
delta = options.delta
badmask = options.badmask

x = y = arange(-3.0, 3.01, delta)
X, Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = (Z1 - Z2) * 10
Zbm = Z.copy()

ny, nx = shape(Z)

Badmask = zeros(shape(Z))
if badmask == 'none':
    Badmask = None
elif badmask == 'edge':
    Badmask[:ny/5,:] = 1
    Zbm[:ny/5,:] = 100      # test ability to ignore bad values
elif badmask == 'interior':
    Badmask[ny/4:ny/2, nx/4:nx/2] = 1
    Zbm[ny/4:ny/2, nx/4:nx/2] = 100
    print "Interior masking works correctly for line contours only..."
else:
    raise ValueError("badmask must be 'none', 'edge', or 'interior'")

figure(1)
subplot(2,2,1)
levels, colls = contourf(X, Y, Zbm, 10, # [-1, -0.1, 0, 0.1],
                        cmap=cm.jet,
                        badmask = Badmask
                        )
# Use levels output from previous call to guarantee they are the same.
levs2, colls2 = contour(X, Y, Zbm, levels,
                        colors = 'r',
                        badmask = Badmask,
                        hold='on')

levs3, colls3 = contour(X, Y, Zbm, (0,),
                        colors = 'g',
                        linewidths = 2,
                        hold='on')
title('Filled contours')
#colorbar()
# Major reworking of the colorbar mechanism is needed for filled contours!

subplot(2,2,2)
imshow(Z)
v = axis()
contour(Z, levels, hold='on', colors = 'r', origin='upper')
axis(v)
title("Image, origin 'upper'")

subplot(2,2,3)
imshow(Z, origin='lower')
v = axis()
contour(Z, levels, hold='on', colors = 'r', origin='lower')
axis(v)
title("Image, origin 'lower'")

subplot(2,2,4)
imshow(Z, interpolation='nearest')
v = axis()
contour(Z, levels, hold='on', colors = 'r', origin='image')
axis(v)
ylim = get(gca(), 'ylim')
set(gca(), ylim=ylim[::-1])
title("Image, origin from rc, reversed y-axis")

if options.save is not None:
    savefig(options.save)

show()

