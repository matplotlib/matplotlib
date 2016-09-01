"""
Demonstrate unit handling

basic_units is a mockup of a true units package used for testing
purposed, which illustrates the basic interface that a units package
must provide to matplotlib.

The example below shows support for unit conversions over masked
arrays.
"""
import numpy as np
import matplotlib.pyplot as plt
from basic_units import secs, hertz, minutes

# create masked array
data = (1, 2, 3, 4, 5, 6, 7, 8)
mask = (1, 0, 1, 0, 0, 0, 1, 0)
xsecs = secs * np.ma.MaskedArray(data, mask, float)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
ax1.scatter(xsecs, xsecs)
ax1.yaxis.set_units(secs)
ax1.axis([0, 10, 0, 10])

ax2.scatter(xsecs, xsecs, yunits=hertz)
ax2.axis([0, 10, 0, 1])

ax3.scatter(xsecs, xsecs, yunits=hertz)
ax3.yaxis.set_units(minutes)
ax3.axis([0, 10, 0, 1])

fig.tight_layout()
plt.show()
