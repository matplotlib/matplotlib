#!/usr/bin/env python
"""
The default drawing order for axes is patches, lines, text.  This
order is determined by the zorder attribute.  The following default
are set

Artist                      Z-order
Patch / PatchCollection      1
Line2D / LineCollection      2
Text                         3

You can change the order for individual artists by setting the zorder.

In the fist subplot below, the lines are drawn above the patch
collection from the scatter, which is the default.

In the subplot below, the order is reversed
"""

from pylab import *
x = rand(20); y = rand(20)

subplot(211)
plot(x, y, 'r', lw=3)
scatter(x,y,s=120)

subplot(212)
plot(x, y, 'r', zorder=1, lw=3)
scatter(x,y,s=120, zorder=2)

show()
