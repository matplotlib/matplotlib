"""
plot using a variety of cm vs inches conversions.  The example shows
how default unit instrospection works (ax1), how various keywords can
be used to set the x and y units to override the defaults (ax2, ax3,
ax4) and how one can set the xlimits using scalars (ax3, current units
assumed) or units (conversions applied to get the numbers to current
units)

"""
from basic_units import cm, inch
import matplotlib.pyplot as plt
import numpy

cms = cm * numpy.arange(0, 10, 2)

fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(cms, cms)

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(cms, cms, xunits=cm, yunits=inch)

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(cms, cms, xunits=inch, yunits=cm)
ax3.set_xlim(3, 6)  # scalars are interpreted in current units

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(cms, cms, xunits=inch, yunits=inch)
#fig.savefig('simple_conversion_plot.png')
ax4.set_xlim(3*cm, 6*cm)  # cm are converted to inches

plt.show()
