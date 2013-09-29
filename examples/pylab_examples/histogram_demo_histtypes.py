import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab


mu = 200
sigma = 25
n_bins = 50
x = mu + sigma*np.random.randn(10000)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))

n, bins, patches = ax0.hist(x, n_bins, normed=1, histtype='stepfilled',
                            facecolor='g', alpha=0.75)
# Add a line showing the expected distribution.
y = mlab.normpdf(bins, mu, sigma)
ax0.plot(bins, y, 'k--', linewidth=1.5)
ax0.set_title('stepfilled')

# Create a histogram by providing the bin edges (unequally spaced).
bins = [100, 150, 180, 195, 205, 220, 250, 300]
n, bins, patches = ax1.hist(x, bins, normed=1, histtype='bar', rwidth=0.8)
ax1.set_title('unequal bins')

plt.tight_layout()
plt.show()
