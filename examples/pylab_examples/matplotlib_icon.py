#!/usr/bin/env python
"""
make the matplotlib svg minimization icon
"""
import matplotlib
matplotlib.use('SVG')
from pylab import *

rc('grid', ls='-', lw=2, color='k')
fig = figure(figsize=(1, 1), dpi=72)
axes([0.025, 0.025, 0.95, 0.95], axisbg='y')

t = arange(0, 2, 0.05)
s = sin(2*pi*t)
plot(t,s, linewidth=4)
axis([-.2, 2.2, -1.2, 1.2])

grid(True)
setp(gca(), xticklabels=[], yticklabels=[])
#savefig('../images/matplotlib.svg', facecolor='0.75')

