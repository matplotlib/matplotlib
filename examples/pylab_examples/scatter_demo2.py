"""
make a scatter plot with varying color and size arguments
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

intc = mlab.csv2rec('mpl_examples/data/intc.csv')

delta1 = np.diff(intc.adj_close)/intc.adj_close[:-1]

# size in points ^2
volume = (15*intc.volume[:-2]/intc.volume[0])**2
close = 0.003*intc.close[:-2]/0.003*intc.open[:-2]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.75)

#ticks = arange(-0.06, 0.061, 0.02)
#xticks(ticks)
#yticks(ticks)

ax.set_xlabel(r'$\Delta_i$', fontsize=20)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=20)
ax.set_title('Volume and percent change')
ax.grid(True)

plt.show()



