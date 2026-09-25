"""
===========================
Cursor Layered Architecture
===========================

This example demonstrates the Figure-level arbitrary-layer architecture.

When `useblit=True` is provided to a `Cursor` and the backend supports layers
(such as `QtAgg`), the cursor artists are automatically drawn in a dedicated
`'widgets'` layer. When the mouse moves, the heavy scatter plot is fully
cached in memory, and only the lightweight `'widgets'` layer is redrawn.
This provides a completely smooth interactive experience even with a
large amount of data.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Cursor

x = np.random.normal(5, 2, 1_000_000)
y = np.random.normal(5, 2, 1_000_000)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Cursor drawn in a separate layer")

ax.scatter(x, y, alpha=0.1, color='blue', s=1)

cursor = Cursor(ax, color='red', linewidth=1, useblit=True)

plt.show()
