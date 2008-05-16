import matplotlib
matplotlib.use('GTKAgg')
from matplotlib.backends.backend_gtk import DialogLineprops

import numpy as npy
from pylab import figure, show

def f(t):
    s1 = npy.cos(2*npy.pi*t)
    e1 = npy.exp(-t)
    return npy.multiply(s1,e1)

t1 = npy.arange(0.0, 5.0, 0.1)
t2 = npy.arange(0.0, 5.0, 0.02)
t3 = npy.arange(0.0, 2.0, 0.01)

fig = figure()
ax = fig.add_subplot(111)
l1,  = ax.plot(t1, f(t1), 'bo', label='line 1')
l2,  = ax.plot(t2, f(t2), 'k--', label='line 2')

dlg = DialogLineprops([l1,l2])
dlg.show()
show()

