#!/usr/bin/env python
# This file generates the matplotlib web page logo

from __future__ import print_function
from pylab import *
import matplotlib.cbook as cbook

# convert data to mV
datafile = cbook.get_sample_data('membrane.dat', asfileobj=False)
print('loading', datafile)

x = 1000*0.1*fromstring(file(datafile, 'rb').read(), float32)
# 0.0005 is the sample interval
t = 0.0005*arange(len(x))
figure(1, figsize=(7,1), dpi=100)
ax = subplot(111, axisbg='y')
plot(t, x)
text(0.5, 0.5,'matplotlib', color='r',
     fontsize=40, fontname='Courier',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes,
     )
axis([1, 1.72,-60, 10])
setp(gca(), 'xticklabels', [])
setp(gca(), 'yticklabels', [])

show()
