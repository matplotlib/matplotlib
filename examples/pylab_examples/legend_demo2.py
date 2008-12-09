#!/usr/bin/env python
#
# Make a legend for specific lines.
from pylab import *

t1 = arange(0.0, 2.0, 0.1)
t2 = arange(0.0, 2.0, 0.01)

# note that plot returns a list of lines.  The "l1, = plot" usage
# extracts the first element of the list inot l1 using tuple
# unpacking.  So l1 is a Line2D instance, not a sequence of lines
l1,    = plot(t2, exp(-t2))
l2, l3 = plot(t2, sin(2*pi*t2), '--go', t1, log(1+t1), '.')
l4,    = plot(t2, exp(-t2)*sin(2*pi*t2), 'rs-.')

legend( (l2, l4), ('oscillatory', 'damped'), 'upper right', shadow=True)
xlabel('time')
ylabel('volts')
title('Damped oscillation')
#axis([0,2,-1,1])
show()



