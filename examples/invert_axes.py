"""

You can use decreasing axes by flipping the normal order of the axis
limits

"""
from matplotlib.matlab import *

t = arange(0.01, 5.0, 0.01)
s = exp(-t)
plot(t, s)

set(gca(), 'xlim', [5,0])  # decreasing time

xlabel('decreasing time (s)')
ylabel('voltage (mV)')
title('Should be growing...')
grid(True)

show()
