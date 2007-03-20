import matplotlib
matplotlib.rcParams['numerix'] = 'numpy'

import basic_units as bu
import numpy as N
from pylab import figure, show
from matplotlib.cbook import iterable

cm = bu.BasicUnit('cm', 'centimeters')
inch = bu.BasicUnit('inch', 'inches')

inch.add_conversion_factor(cm, 2.54)
cm.add_conversion_factor(inch, 1/2.54)

lengths_cm = cm*N.arange(0, 10, 0.5)

# iterator test
print 'Testing iterators...'
for length in lengths_cm:
  print length

print 'Iterable() = ' + `iterable(lengths_cm)`

print 'cm', lengths_cm
print 'toinch', lengths_cm.convert_to(inch)
print 'toval', lengths_cm.convert_to(inch).get_value()

fig = figure()
ax1 = fig.add_subplot(2,2,1)
ax1.plot(lengths_cm, 2.0*lengths_cm, xunits=cm, yunits=cm)
ax1.set_xlabel('in centimeters')
ax1.set_ylabel('in centimeters')


ax2 = fig.add_subplot(2,2,2)
ax2.plot(lengths_cm, lengths_cm, xunits=cm, yunits=inch)
ax2.set_xlabel('in centimeters')
ax2.set_ylabel('in inches')

ax3 = fig.add_subplot(2,2,3)
ax3.plot(lengths_cm, 2.0*lengths_cm, xunits=inch, yunits=cm)
ax3.set_xlabel('in inches')
ax3.set_ylabel('in centimeters')


ax4 = fig.add_subplot(2,2,4)
ax4.plot(lengths_cm, 2.0*lengths_cm, xunits=inch, yunits=inch)
ax4.set_xlabel('in inches')
ax4.set_ylabel('in inches')
fig.savefig('simple_conversion_plot.png')

show()
