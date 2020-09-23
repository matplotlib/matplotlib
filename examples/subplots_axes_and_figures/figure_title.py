"""
============
Figure title
============

Each subplot can have its own title (`.Axes.set_title`). Additionally,
`.Figure.suptitle` adds a centered title at the top of the figure.
"""
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0.0, 5.0, 501)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
ax1.plot(x, np.cos(6*x) * np.exp(-x))
ax1.set_title('damped')
ax1.set_xlabel('time (s)')
ax1.set_ylabel('amplitude')

ax2.plot(x, np.cos(6*x))
ax2.set_xlabel('time (s)')
ax2.set_title('undamped')

fig.suptitle('Different types of oscillations', fontsize=16)

plt.show()
