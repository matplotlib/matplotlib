"""
==================
Errorbar Subsample
==================

Demo for the errorevery keyword to show data full accuracy data plots with
few errorbars.
"""

import numpy as np
import matplotlib.pyplot as plt

# example data
x = np.arange(0.1, 4, 0.1)
y = np.exp(np.vstack([-1.0 * x, -.5 * x]))

# example variable error bar values
yerr = 0.1 + 0.1 * np.sqrt(np.vstack([x, x/2]))


# Now switch to a more OO interface to exercise more features.
fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12, 6))
ax = axs[0]
for i in range(2):
    ax.errorbar(x, y[i], yerr=yerr[i])

ax.set_title('all errorbars')

ax = axs[1]
for i in range(2):
    ax.errorbar(x, y[i], yerr=yerr[i], errorevery=6)
ax.set_title('only every 6th errorbar')

ax = axs[2]
for i in range(2):
    ax.errorbar(x, y[i], yerr=yerr[i], errorevery=(3 * i, 6))
ax.set_title('second series shifted by 3')

fig.suptitle('Errorbar subsampling for better appearance')

plt.show()
