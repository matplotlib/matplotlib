#!/usr/bin/env python
"""
make the matplotlib svg minimization icon
"""
import matplotlib
#matplotlib.use('Svg')
from pylab import *

rc('grid', ls='-', lw=2, color='k')
fig = figure(figsize=(1, 1), dpi=72)
axes([0.025, 0.025, 0.95, 0.95], axisbg='#bfd1d4')

t = arange(0, 2, 0.05)
s = sin(2*pi*t)
plot(t,s, linewidth=4, color="#ca7900")
axis([-.2, 2.2, -1.2, 1.2])

# grid(True)
setp(gca(), xticklabels=[], yticklabels=[])
savefig('matplotlib', facecolor='0.75')

