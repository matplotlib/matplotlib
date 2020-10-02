"""
======
Spines
======

This demo compares:

- normal axes, with spines on all four sides;
- an axes with spines only on the left and bottom;
- an axes using custom bounds to limit the extent of the spine.
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 2 * np.pi, 100)
y = 2 * np.sin(x)

# Constrained layout makes sure the labels don't overlap the axes.
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, constrained_layout=True)

ax0.plot(x, y)
ax0.set_title('normal spines')

ax1.plot(x, y)
ax1.set_title('bottom-left spines')

# Hide the right and top spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')

ax2.plot(x, y)

# Only draw spine between the y-ticks
ax2.spines['left'].set_bounds(-1, 1)
# Hide the right and top spines
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

plt.show()
