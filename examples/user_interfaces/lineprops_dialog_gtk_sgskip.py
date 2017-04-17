"""
====================
Lineprops Dialog GTK
====================

"""
import matplotlib
matplotlib.use('GTKAgg')
from matplotlib.backends.backend_gtk import DialogLineprops

import numpy as np
import matplotlib.pyplot as plt


def f(t):
    s1 = np.cos(2*np.pi*t)
    e1 = np.exp(-t)
    return np.multiply(s1, e1)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01)

fig, ax = plt.subplots()
l1, = ax.plot(t1, f(t1), 'bo', label='line 1')
l2, = ax.plot(t2, f(t2), 'k--', label='line 2')

dlg = DialogLineprops([l1, l2])
dlg.show()
plt.show()
