from matplotlib.matlab import *

t = arange(0.1, 4, 0.1)
s = exp(-t)
e = 0.1*randn(len(s))

errorbar(t, s, e, fmt='o')
xlabel('Distance (m)')
ylabel('Height (m)')
title('Mean an standard error as a function of distance')
show()

