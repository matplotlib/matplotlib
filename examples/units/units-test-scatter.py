"""
Demonstrate unit handling

basic_units is a mockup of a true units package used for testing
purposed, which illustrates the basic interface that a units package
must provide to matplotlib.

The example below shows support for unit conversions over masked
arrays.
"""
import matplotlib
matplotlib.rcParams['numerix'] = 'numpy'

import basic_units as bu
import numpy as N
from matplotlib.pylab import figure, show

# create masked array
x = N.ma.MaskedArray((1,2,3,4,5,6,7,8), N.float64, mask=(1,0,1,0,0,0,1,0))

secs = bu.BasicUnit('s', 'seconds')
hertz = bu.BasicUnit('Hz', 'Hertz')
minutes = bu.BasicUnit('min', 'minutes')

secs.add_conversion_fn(hertz, lambda x:1./x)
secs.add_conversion_factor(minutes, 1/60.0)

fig = figure()
ax1 = fig.add_subplot(3,1,1)
ax1.scatter(x, x)
ax1.set_ylabel('seconds')
ax1.axis([0,10,0,10])

ax2 = fig.add_subplot(3,1,2, sharex=ax1)
x_in_secs = secs*x

ax2.scatter(x_in_secs, x_in_secs, yunits=hertz)
ax2.set_ylabel('Hertz')
ax2.axis([0,10,0,1])

ax3 = fig.add_subplot(3,1,3, sharex=ax1)
x_in_secs = secs*x
ax3.scatter(x_in_secs, x_in_secs, yunits=hertz)
ax3.set_yunits(minutes)
ax3.axis([0,10,0,1])
ax3.set_xlabel('seconds')
ax3.set_ylabel('minutes')

#fig.savefig('units-test-scatter.png')

show()

