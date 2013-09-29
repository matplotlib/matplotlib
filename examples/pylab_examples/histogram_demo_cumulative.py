import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab

mu, sigma = 200, 25
x = mu + sigma*np.random.randn(10000)
n, bins, patches = plt.hist(x, 50, normed=1, histtype='step', cumulative=True)

# Add a line showing the expected distribution.
y = mlab.normpdf( bins, mu, sigma).cumsum()
y /= y[-1]
plt.plot(bins, y, 'k--', linewidth=1.5)

# Create a second data-set with a smaller standard deviation.
sigma2 = 15.
x = mu + sigma2*np.random.randn(10000)

n, bins, patches = plt.hist(x, bins=bins, normed=1, histtype='step', cumulative=True)

# Add a line showing the expected distribution.
y = mlab.normpdf( bins, mu, sigma2).cumsum()
y /= y[-1]
plt.plot(bins, y, 'r--', linewidth=1.5)

# Overlay a reverted cumulative histogram.
n, bins, patches = plt.hist(x, bins=bins, normed=1,
    histtype='step', cumulative=-1)

plt.grid(True)
plt.ylim(0, 1.05)
plt.title('cumulative step')

plt.show()
