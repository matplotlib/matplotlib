#!/usr/bin/env python
'''
contour_image.py [options]

Test combinations of contouring, filled contouring, and image plotting.
For contour labelling, see contour_demo.py.
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
parser.add_option("-e", "--extent", dest = "extent", type="int", default=0,
                  help="""For subplots 2-4, use extent: \
specify number 1 through 4 for any of 4 possibilities.""")
parser.add_option("-f", "--figure", dest = "fignum", type="int", default=0,
                  metavar="FIGNUM",
                  help="""Plot subplot FIGNUM as a full-size plot; FIGNUM \
must be in the range 1-4.""")

#Default delta is large because that makes it fast, and it illustrates
# the correct registration between image and contours.

import sys
# We have to strip out the numerix arguments before passing the
# input arguments to the parser.
Args = [arg for arg in sys.argv if arg not in ('--Numeric', '--numarray')]
options, args = parser.parse_args(Args)
delta = options.delta
badmask = options.badmask

extents = ((-3,4,-4,3), (-3,4,3,-4), (4,-3,-4,3), (4,-3,3,-4))
if options.extent == 0:
    extent = None
elif options.extent <= 4 and options.extent > 0:
    extent = extents[options.extent - 1]
    print "Using extent ", extent, "to change axis mapping on subplots 2-4"
else:
    raise ValueError("extent must be integer, 1-4")

fignum = options.fignum

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

levels = arange(-1.2,1.5,0.4)

figure()


if fignum == 0:
    subplot(2,2,1)

if fignum == 0 or fignum == 1:
    levs1, colls = contourf(X, Y, Zbm, 10,
                            cmap=cm.jet,
                            badmask = Badmask
                            )
    #If we want lines as well as filled regions, we need to call
    # contour separately; don't try to change the edgecolor or edgewidth
    # of the polygons in the collections returned by contourf.
    # Use levels output from previous call to guarantee they are the same.
    levs2, colls2 = contour(X, Y, Zbm, levs1,
                            colors = 'k',
                            badmask = Badmask,
                            hold='on')
    # We don't really need dashed contour lines to indicate negative
    # regions, so let's turn them off.
    for c in colls2:
        c.set_linestyle('solid')

    # It is easier here to make a separate call to contour than
    # to set up an array of colors and linewidths.
    levs3, colls3 = contour(X, Y, Zbm, (0,),
                            colors = 'g',
                            linewidths = 2,
                            hold='on')
    title('Filled contours')
    colorbar()
    hot()
    # Major reworking of the colorbar mechanism is needed for filled contours!

if fignum == 0:
    subplot(2,2,2)

if fignum == 0 or fignum == 2:
    imshow(Z, extent=extent)
    v = axis()
    contour(Z, levels, hold='on', colors = 'k', origin='upper', extent=extent)
    axis(v)
    title("Image, origin 'upper'")

if fignum == 0:
    subplot(2,2,3)

if fignum == 0 or fignum == 3:
    imshow(Z, origin='lower', extent=extent)
    v = axis()
    contour(Z, levels, hold='on', colors = 'k', origin='lower', extent=extent)
    axis(v)
    title("Image, origin 'lower'")

if fignum == 0:
    subplot(2,2,4)

if fignum == 0 or fignum == 4:
    imshow(Z, interpolation='nearest', extent=extent)
    v = axis()
    contour(Z, levels, hold='on', colors = 'k', origin='image', extent=extent)
    axis(v)
    ylim = get(gca(), 'ylim')
    set(gca(), ylim=ylim[::-1])
    title("Image, origin from rc, reversed y-axis")

if options.save is not None:
    savefig(options.save)

show()
