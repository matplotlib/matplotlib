#!/usr/bin/env python

from matplotlib.matlab import *

#Z = arange(60)
#Z.shape = 6,10
#Z.shape = 10,6
#print Z
Z = rand(10,6)

c = pcolor(Z)
c.set_linewidth(4)

#savefig('pcolor_small')
show()
