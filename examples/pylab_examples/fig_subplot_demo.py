"""
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

plt.close('all')

# Just a figure and one subplot
f, ax = plt.fig_subplot()
ax.plot(x, y)
ax.set_title('Simple plot')

# Two subplots, grab the whole fig_axes list
fax = plt.fig_subplot(2, sharex=True)
fax[1].plot(x, y)
fax[1].set_title('Sharing X axis')
fax[2].scatter(x, y)

# Two subplots, unpack the output immediately
f, ax1, ax2 = plt.fig_subplot(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# Three subplots sharing both x/y axes
f, ax1, ax2, ax3 = plt.fig_subplot(3, sharex=True, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing both axes')
ax2.scatter(x, y)
ax3.scatter(x, 2*y**2-1,color='r')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

# Four polar axes
plt.fig_subplot(2, 2, subplot_kw=dict(polar=True))

plt.show()
