#!/usr/bin/env python
from pylab import *

plot([1,2,3,4], [1,4,9,16])
locs, labels = xticks([1,2,3,4], ['Frogs', 'Hogs', 'Bogs', 'Slogs'])
set(labels, 'rotation', 'vertical')
show()
