#!/usr/bin/env python
# a stacked bar plot with errorbars
from matplotlib.matlab import *

N = 5
menMeans   = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd     = (2, 3, 4, 1, 2)
womenStd   = (3, 5, 2, 3, 3)
ind = arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = bar(ind, menMeans,   width, color='r', yerr=womenStd)
p2 = bar(ind, womenMeans, width, color='y',
         bottom=menMeans, yerr=menStd)

ylabel('Scores')
title('Scores by group and gender')
set(gca(), 'xticks', ind+(width/2))
set(gca(), 'xticklabels', ('G1', 'G2', 'G3', 'G4', 'G5') )
set(gca(), 'xlim', [-width,len(ind)])
set(gca(), 'yticks', arange(0,81,10))
legend( (p1[0], p2[0]), ('Men', 'Women') )

show()
