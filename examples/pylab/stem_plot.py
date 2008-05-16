#!/usr/bin/env python
from pylab import *

x = linspace(0.1, 2*pi, 10)
markerline, stemlines, baseline = stem(x, cos(x), '-.')
setp(markerline, 'markerfacecolor', 'b')
setp(baseline, 'color','r', 'linewidth', 2)

show()
