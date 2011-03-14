#!/usr/bin/env python

from __future__ import print_function
from pylab import *

subplot(311)
plot(arange(10), ones( (10,)))


subplot(312)
plot(ones( (10,)), arange(10))

subplot(313)
plot(ones( (10,)), ones( (10,)), 'o')
show()
