"""
=====================
Twin Axes with zorder
=====================

`~matplotlib.axes.Axes.twinx` and `~matplotlib.axes.Axes.twiny` accept a
*zorder* keyword argument that controls whether the twin Axes is drawn in front
of or behind the original Axes.

Matplotlib also automatically manages background patch visibility for twinned
Axes groups so that only the bottom-most Axes has a visible background patch
(respecting ``frameon``). This avoids the background of a higher-zorder twin
Axes covering artists drawn on the underlying Axes.
"""

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(0, 10, 400)
y_main = np.sin(x)
y_twin = 0.4 * np.cos(x) + 0.6

fig, ax = plt.subplots()

# Put the twin Axes behind the original Axes.
ax2 = ax.twinx(zorder=ax.get_zorder() - 1)

# Draw something broad on the twin Axes so that the stacking is obvious.
ax2.fill_between(x, 0, y_twin, color="C1", alpha=0.35, label="twin fill")
ax2.plot(x, y_twin, color="C1", lw=6, alpha=0.8)

# Draw overlapping artists on the main Axes; they appear on top.
ax.scatter(x[::8], y_main[::8], s=35, color="C0", edgecolor="k", linewidth=0.5,
           zorder=3, label="main scatter")
ax.plot(x, y_main, color="C0", lw=4)

ax.set_xlabel("x")
ax.set_ylabel("main y")
ax2.set_ylabel("twin y")
ax.set_title("Twin Axes drawn behind the main Axes using zorder")

fig.tight_layout()
plt.show()
