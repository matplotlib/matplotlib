#!/usr/bin/env python

"""
You can precisely specify dashes with an on/off ink rect sequence in
points.
"""
from pylab import *

dashes = [5,2,10,5] # 5 points on, 2 off, 3 on, 1 off

l, = plot(arange(20), '--')
l.set_dashes(dashes)

show()
