
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf


mu, sigma = 200, 25
x = mu + sigma*np.random.randn(10000)

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 3))

n, bins, patches = ax0.hist(x, 50, normed=1, histtype='stepfilled',
                            facecolor='g', alpha=0.75)

# Add a line showing the expected distribution.
y = normpdf( bins, mu, sigma)
ax0.plot(bins, y, 'k--', linewidth=1.5)

# Create a histogram by providing the bin edges (unequally spaced).
bins = [100,150,180,195,205,220,250,300]
n, bins, patches = ax1.hist(x, bins, normed=1, histtype='bar', rwidth=0.8)

n, bins, patches = ax2.hist(x, 50, normed=1, histtype='step', cumulative=True)

# Add a line showing the expected distribution.
y = normpdf( bins, mu, sigma).cumsum()
y /= y[-1]
ax2.plot(bins, y, 'k--', linewidth=1.5)

# Create a second data-set with a smaller standard deviation.
sigma2 = 15.
x = mu + sigma2*np.random.randn(10000)

n, bins, patches = ax2.hist(x, bins=bins, normed=1, histtype='step', cumulative=True)

# Add a line showing the expected distribution.
y = normpdf( bins, mu, sigma2).cumsum()
y /= y[-1]
ax2.plot(bins, y, 'r--', linewidth=1.5)

# Overlay a reverted cumulative histogram.
n, bins, patches = ax2.hist(x, bins=bins, normed=1,
    histtype='step', cumulative=-1)


ax2.grid(True)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()
