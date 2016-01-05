import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Example 2
x = np.arange(0, 1, .01)
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(6, 6))
fig.text(0.5, 0.975, 'The new formatter, default settings',
         horizontalalignment='center',
         verticalalignment='top')

ax1.plot(x * 1e5 + 1e10, x * 1e-10 + 1e-5)
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.yaxis.set_major_formatter(ScalarFormatter())

ax2.plot(x * 1e5, x * 1e-4)
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.yaxis.set_major_formatter(ScalarFormatter())

ax3.plot(-x * 1e5 - 1e10, -x * 1e-5 - 1e-10)
ax3.xaxis.set_major_formatter(ScalarFormatter())
ax3.yaxis.set_major_formatter(ScalarFormatter())

ax4.plot(-x * 1e5, -x * 1e-4)
ax4.xaxis.set_major_formatter(ScalarFormatter())
ax4.yaxis.set_major_formatter(ScalarFormatter())

# Example 3
x = np.arange(0, 1, .01)
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(6, 6))
fig.text(0.5, 0.975, 'The new formatter, no numerical offset',
         horizontalalignment='center',
         verticalalignment='top')

ax1.plot(x * 1e5 + 1e10, x * 1e-10 + 1e-5)
ax1.xaxis.set_major_formatter(ScalarFormatter(use_offset=False))
ax1.yaxis.set_major_formatter(ScalarFormatter(use_offset=False))

ax2.plot(x * 1e5, x * 1e-4)
ax2.xaxis.set_major_formatter(ScalarFormatter(use_offset=False))
ax2.yaxis.set_major_formatter(ScalarFormatter(use_offset=False))

ax3.plot(-x * 1e5 - 1e10, -x * 1e-5 - 1e-10)
ax3.xaxis.set_major_formatter(ScalarFormatter(use_offset=False))
ax3.yaxis.set_major_formatter(ScalarFormatter(use_offset=False))

ax4.plot(-x * 1e5, -x * 1e-4)
ax4.xaxis.set_major_formatter(ScalarFormatter(use_offset=False))
ax4.yaxis.set_major_formatter(ScalarFormatter(use_offset=False))

# Example 4
x = np.arange(0, 1, .01)
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(6, 6))
fig.text(0.5, 0.975, 'The new formatter, with mathtext',
         horizontalalignment='center',
         verticalalignment='top')

ax1.plot(x * 1e5 + 1e10, x * 1e-10 + 1e-5)
ax1.xaxis.set_major_formatter(ScalarFormatter(use_mathtext=True))
ax1.yaxis.set_major_formatter(ScalarFormatter(use_mathtext=True))

ax2.plot(x * 1e5, x * 1e-4)
ax2.xaxis.set_major_formatter(ScalarFormatter(use_mathtext=True))
ax2.yaxis.set_major_formatter(ScalarFormatter(use_mathtext=True))

ax3.plot(-x * 1e5 - 1e10, -x * 1e-5 - 1e-10)
ax3.xaxis.set_major_formatter(ScalarFormatter(use_mathtext=True))
ax3.yaxis.set_major_formatter(ScalarFormatter(use_mathtext=True))

ax4.plot(-x * 1e5, -x * 1e-4)
ax4.xaxis.set_major_formatter(ScalarFormatter(use_mathtext=True))
ax4.yaxis.set_major_formatter(ScalarFormatter(use_mathtext=True))
plt.show()
