"""
====================
Line-style reference
====================

Reference for line-styles included with Matplotlib.
"""

import matplotlib.pyplot as plt


# Plot all line styles.
fig, ax = plt.subplots()

linestyles = ['-', '--', '-.', ':']
for y, linestyle in enumerate(linestyles):
    ax.text(-0.1, y, repr(linestyle),
            horizontalalignment='center', verticalalignment='center')
    ax.plot([y, y], linestyle=linestyle, linewidth=3, color='tab:blue')

ax.set_axis_off()
ax.set_title('line styles')

plt.show()
