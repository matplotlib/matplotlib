#!/usr/bin/env python
from matplotlib.matlab import *

plot([1,2,3,4], [1,4,9,16])
set(gca(), 'xticks', [1,2,3,4])
t = set(gca(), 'xticklabels', ['Frogs', 'Hogs', 'Bogs', 'Slogs'])
set(t, 'rotation', 'vertical')
show()
