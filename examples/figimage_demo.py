#!/usr/bin/env python
"""
See pcolor_demo2 for a much faster way of generating pcolor plots
"""
from __future__ import division
from matplotlib.matlab import *
rc('axes', hold=True)
rc('image', origin='lower')
figure(1, frameon=False)
Z = arange(10000.0); Z.shape = 100,100
Z[:,20:] = 1
jet() # sets the default
im1 = figimage(Z, xo=50,  yo=50)  # you can also pass cmap=cm.jet as kwarg
im2 = figimage(Z, xo=100, yo=100, alpha=.8)
gray()  # overrides current and sets default
#savefig('figimage_demo')

show()

    
