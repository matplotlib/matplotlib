"""
======================
Unshare and share axis
======================

The example shows how to share and unshare axes after they are created.
"""

import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.01, 5.0, 0.01)
s1 = np.sin(2 * np.pi * t)
s2 = np.exp(-t)
s3 = np.sin(4 * np.pi * t)

ax1 = plt.subplot(311)
plt.plot(t, s1)

ax2 = plt.subplot(312)
plt.plot(t, s2)

ax3 = plt.subplot(313)
plt.plot(t, s3)

ax1.share_x_axes(ax2)
ax1.share_y_axes(ax2)

# Share both axes.
ax3.share_axes(ax1)
plt.xlim(0.01, 5.0)

ax3.unshare_y_axes()
ax2.unshare_x_axes()

plt.show()
