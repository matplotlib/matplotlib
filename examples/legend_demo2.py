# Make a legend for specific lines
from matplotlib.matlab import *

t1 = arange(0.0, 2.0, 0.1)
t2 = arange(0.0, 2.0, 0.01)

l1 = plot(t2, exp(-t2))
l2, l3 = plot(t2, sin(2*pi*t2), '--go', t1, log(1+t1), '.')
l4 = plot(t2, exp(-t2)*sin(2*pi*t2), 'rs-.')


legend( (l2, l4), ('oscillatory', 'damped'), 'upper right')

show()



