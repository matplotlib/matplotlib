"""
==============
Fill plot demo
==============

First example showcases the most basic fill plot a user can do with matplotlib.

Second example shows a few optional features:

    * Multiple curves with a single command.
    * Setting the fill color.
    * Setting the opacity (alpha value).
"""
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(0, 2 * np.pi, 500)
y1 = np.sin(x1)
y2 = np.sin(3 * x1)

fig, (ax1, ax3) = plt.subplots(1,2, sharey=True)

ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax1.set_xticklabels( ['$0$', r'$\frac{\pi}{2}$', r'$\pi$',r'$\frac{3\pi}{2}$',r'$2\pi$'] )
ax1.set_title('fill_demo_feature')
ax1.set_xlabel('Time',labelpad=2)
ax1.set_ylabel('Amplitude')
ax1.fill(x1, y1, 'b', x1, y2, 'r', alpha=0.3)
ax1.xaxis.set_label_coords(0.5, -0.1)


x3 = np.linspace(0, 1, 500)
y3 = np.sin(4 * np.pi * x3) * np.exp(-5 * x3)

ax3.set_title('fill_demo')
ax3.set_xlabel('Time', labelpad=8)
ax3.fill(x3, y3, zorder=10)
ax3.grid(True, zorder=5)
ax3.xaxis.set_label_coords(0.5, -0.1)

plt.show()
