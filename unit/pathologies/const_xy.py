#!/usr/bin/env python
from matplotlib.matlab import *

figure(1)
plot(arange(10), ones( (10,)))


figure(2)
plot(ones( (10,)), arange(10))

figure(3)
plot(ones( (10,)), ones( (10,)), 'o')
show()
