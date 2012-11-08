"""
Illustrate the API for changing the cycle of colors used
when plotting multiple lines on a single Axes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

yy = np.arange(24)
yy.shape = 6,4

mpl.rc('lines', linewidth=4)
mpl.rcParams['axes.color_cycle'] = ['r', 'g', 'b', 'c']

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(yy)
ax1.set_title('Changed default color cycle to rgbc')

ax2.set_color_cycle(['c', 'm', 'y', 'k'])
ax2.plot(yy)
ax2.set_title('This axes only, cycle is cmyk')

plt.show()


