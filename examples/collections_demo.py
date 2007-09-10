#!/usr/bin/env python
'''Demonstration of LineCollection, PolyCollection, and
RegularPolyCollection with autoscaling.

For the first two subplots, we will use spirals.  Their
size will be set in plot units, not data units.  Their positions
will be set in data units by using the "offsets" and "transOffset"
kwargs of the LineCollection and PolyCollection.

The third subplot will make regular polygons, with the same
type of scaling and positioning as in the first two.

The last subplot illustrates the use of "offsets=(xo,yo)",
that is, a single tuple instead of a list of tuples, to generate
successively offset curves, with the offset given in data
units.  This behavior is available only for the LineCollection.

'''

import matplotlib.pyplot as P
from matplotlib import collections, axes, transforms
from matplotlib.colors import colorConverter
import numpy as N

nverts = 50
npts = 100

# Make some spirals
r = N.array(range(nverts))
theta = N.array(range(nverts)) * (2*N.pi)/(nverts-1)
xx = r * N.sin(theta)
yy = r * N.cos(theta)
spiral = zip(xx,yy)

# Make some offsets
xo = N.random.randn(npts)
yo = N.random.randn(npts)
xyo = zip(xo, yo)

# Make a list of colors cycling through the rgbcmyk series.
colors = [colorConverter.to_rgba(c) for c in ('r','g','b','c','y','m','k')]

fig = P.figure()

a = fig.add_subplot(2,2,1)
col = collections.LineCollection([spiral], offsets=xyo,
                                transOffset=a.transData)
    # Note: the first argument to the collection initializer
    # must be a list of sequences of x,y tuples; we have only
    # one sequence, but we still have to put it in a list.
a.add_collection(col, autolim=True)
    # autolim=True enables autoscaling.  For collections with
    # offsets like this, it is neither efficient nor accurate,
    # but it is good enough to generate a plot that you can use
    # as a starting point.  If you know beforehand the range of
    # x and y that you want to show, it is better to set them
    # explicitly, leave out the autolim kwarg (or set it to False),
    # and omit the 'a.autoscale_view()' call below.

# Make a transform for the line segments such that their size is
# given in points:
trans = transforms.scale_transform(fig.dpi/transforms.Value(72.),
                                    fig.dpi/transforms.Value(72.))
col.set_transform(trans)  # the points to pixels transform
col.set_color(colors)

a.autoscale_view()  # See comment above, after a.add_collection.
a.set_title('LineCollection using offsets')


# The same data as above, but fill the curves.

a = fig.add_subplot(2,2,2)

col = collections.PolyCollection([spiral], offsets=xyo,
                                transOffset=a.transData)
a.add_collection(col, autolim=True)
trans = transforms.scale_transform(fig.dpi/transforms.Value(72.),
                                    fig.dpi/transforms.Value(72.))
col.set_transform(trans)  # the points to pixels transform
col.set_color(colors)


a.autoscale_view()
a.set_title('PolyCollection using offsets')


# 7-sided regular polygons

a = fig.add_subplot(2,2,3)

col = collections.RegularPolyCollection(fig.dpi, 7,
                                        sizes = N.fabs(xx)*10, offsets=xyo,
                                        transOffset=a.transData)
a.add_collection(col, autolim=True)
trans = transforms.scale_transform(fig.dpi/transforms.Value(72.),
                                    fig.dpi/transforms.Value(72.))
col.set_transform(trans)  # the points to pixels transform
col.set_color(colors)
a.autoscale_view()
a.set_title('RegularPolyCollection using offsets')


# Simulate a series of ocean current profiles, successively
# offset by 0.1 m/s so that they form what is sometimes called
# a "waterfall" plot or a "stagger" plot.

a = fig.add_subplot(2,2,4)

nverts = 60
ncurves = 20
offs = (0.1, 0.0)

yy = N.linspace(0, 2*N.pi, nverts)
ym = N.amax(yy)
xx = (0.2 + (ym-yy)/ym)**2 * N.cos(yy-0.4) * 0.5
segs = []
for i in range(ncurves):
    xxx = xx + 0.02*N.random.randn(nverts)
    curve = zip(xxx, yy*100)
    segs.append(curve)

col = collections.LineCollection(segs, offsets=offs)
a.add_collection(col, autolim=True)
col.set_color(colors)
a.autoscale_view()
a.set_title('Successive data offsets')
a.set_xlabel('Zonal velocity component (m/s)')
a.set_ylabel('Depth (m)')
# Reverse the y-axis so depth increases downward
a.set_ylim(a.get_ylim()[::-1])


P.show()


