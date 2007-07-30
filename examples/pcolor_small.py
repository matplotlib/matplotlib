#!/usr/bin/env python

from pylab import *

Z = rand(6,10)

subplot(2,1,1)
c = pcolor(Z)
title('default: no edges')

subplot(2,1,2)
c = pcolor(Z, edgecolors='k', linewidths=4)
title('thick edges')

show()
