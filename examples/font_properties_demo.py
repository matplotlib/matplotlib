#!/usr/bin/env python
from matplotlib.matlab import *

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 'larger',
        }

plot(arange(20))
title('something')

rc('font', **font)  # pass in the font dict as kwargs
for i in range(1,15,2):
    text(i, i, 'label %d'%i, color='g')     # uses font

rcdefaults() # restore default
xlabel('hi mom')              
show()
