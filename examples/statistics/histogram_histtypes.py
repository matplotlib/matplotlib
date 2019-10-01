"""
================================================================
Demo of the histogram function's different ``histtype`` settings
================================================================

* Histogram with step curve that has a color fill.
* Histogram with custom and unequal bin widths.

Selecting different bin counts and sizes can significantly affect the
shape of a histogram. The Astropy docs have a great section on how to
select these parameters:
http://docs.astropy.org/en/stable/visualization/histogram.html
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)

mu = 200
sigma = 25
x = np.random.normal(mu, sigma, size=100)

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 4))

ax0.hist(x, 20, density=True, histtype='stepfilled', facecolor='g', alpha=0.75)
ax0.set_title('stepfilled')

ax1.hist(x, 20, density=True, histtype='step', facecolor='g', alpha=0.75)
ax1.set_title('step')

# Create a histogram by providing the bin edges (unequally spaced).
bins = [100, 150, 180, 195, 205, 220, 250, 300]
ax2.hist(x, bins, density=True, histtype='bar', rwidth=0.8)
ax2.set_title('bar, unequal bins')

fig.tight_layout()
plt.show()
