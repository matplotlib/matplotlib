"""
===================
Brighten color demo
===================

Demo of `~.brighten_color` utility.

"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

fig, ax = plt.subplots()

lightfacs = [-1., -0.5, 0., 0.5, 0.75, 1.0]
N = len(lightfacs)
y = np.linspace(0., 1., N+1) * N - 0.5
for n, lightfac in enumerate(lightfacs):
    brightened_color = mcolors.brighten_color('blue', lightfac)
    ax.fill_between([0, 1], [y[n], y[n]], [y[n+1], y[n+1]],
            facecolor=brightened_color, edgecolor='k')

ax.set_yticklabels([''] + lightfacs)

ax.set_xlim([0, 1])
ax.set_ylim(np.min(y), np.max(y))
ax.set_ylabel('Brightening Fraction')
ax.set_xticks([])
ax.set_title('Brightening of Color Blue')
plt.show()
