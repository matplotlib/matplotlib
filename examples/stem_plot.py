#!/usr/bin/env python
from pylab import *

x = linspace(0.1, 2*pi, 10)
markerline, stemlines, baseline = stem(x, cos(x), '-.')
set(markerline, 'markerfacecolor', 'b')
set(baseline, 'color','r', 'linewidth', 2) 

show()
