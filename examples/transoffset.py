#!/usr/bin/env python

'''
This illustrates the use of transforms.offset_copy to
make a transform that positions a drawing element such as
a text string at a specified offset in screen coordinates
(dots or inches) relative to a location given in any
coordinates.

Every Artist--the mpl class from which classes such as
Text and Line are derived--has a transform that can be
set when the Artist is created, such as by the corresponding
pylab command.  By default this is usually the Axes.transData
transform, going from data units to screen dots.  We can
use the offset_copy function to make a modified copy of
this transform, where the modification consists of an
offset.
'''

import pylab as P
from matplotlib.transforms import offset_copy

X = P.arange(7)
Y = P.rand(7)

ax = P.subplot(1,1,1)

# If we want the same offset for each text instance,
# we only need to make one transform.  To get the
# transform argument to offset_copy, we need to make the axes
# first; the subplot command above is one way to do this.
transOffset = offset_copy(ax.transData, fig=P.gcf(),
                            x = 0.05, y=0.10, units='inches')

for x, y in zip(X, Y):
    P.plot((x,),(y,), 'ro')
    P.text(x, y, '%0.2f, %0.2f' % (x,y), transform=transOffset)

P.figure()


# offset_copy works for polar plots also, but one can't simply
# make an axes with subplot and then use the polar command to plot
# in it.  (This is a bug.)  One way to get around this while
# sticking with the pylab interface is to grab the transform
# after the first polar() command.
first = True
for x, y in zip(X, Y):
    L = P.polar((x,),(y,), 'ro')
    if first:
        transOffset = offset_copy(P.gca().transData, fig=P.gcf(),
                                y = 6, units='dots')
        first = False
    P.text(x, y, '%0.2f, %0.2f' % (x,y),
                transform=transOffset,
                horizontalalignment='center',
                verticalalignment='bottom')


P.show()

