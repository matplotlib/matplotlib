"""
make the matplotlib svg minimization icon
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rc('grid', ls='-', lw=2, color='k')
fig = plt.figure(figsize=(1, 1), dpi=72)
plt.axes([0.025, 0.025, 0.95, 0.95], facecolor='#bfd1d4')

t = np.arange(0, 2, 0.05)
s = np.sin(2*np.pi*t)
plt.plot(t, s, linewidth=4, color="#ca7900")
plt.axis([-.2, 2.2, -1.2, 1.2])

ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.show()
