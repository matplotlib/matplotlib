import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf


mu, sigma = 200, 25
x = mu + sigma*np.random.randn(10000)

n, bins, patches = plt.hist(x, 50, normed=1, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

# Add a line showing the expected distribution.
y = normpdf( bins, mu, sigma)
plt.plot(bins, y, 'k--', linewidth=1.5)


plt.figure()

# Create a histogram by providing the bin edges (unequally spaced).
bins = [100,125,150,160,170,180,190,200,210,220,230,240,250,275,300]
n, bins, patches = plt.hist(x, bins, normed=1, histtype='bar', rwidth=0.8)

plt.figure()

n, bins, patches = plt.hist(x, 50, normed=1, histtype='step', cumulative=True)

# Add a line showing the expected distribution.
y = normpdf( bins, mu, sigma).cumsum()
y /= y[-1]
plt.plot(bins, y, 'k--', linewidth=1.5)

# Create a second data-set with a smaller standard deviation.
sigma2 = 15.
x = mu + sigma2*np.random.randn(10000)

n, bins, patches = plt.hist(x, bins=bins, normed=1, histtype='step', cumulative=True)

# Add a line showing the expected distribution.
y = normpdf( bins, mu, sigma2).cumsum()
y /= y[-1]
plt.plot(bins, y, 'r--', linewidth=1.5)

# Overlay a reverted cumulative histogram.
n, bins, patches = plt.hist(x, bins=bins, normed=1,
    histtype='step', cumulative=-1)


plt.grid(True)
plt.ylim(0, 1.05)


plt.figure()

x = mu + sigma*np.random.randn(1000,3)

n, bins, patches = plt.hist(x, 10, normed=1, histtype='bar',
                            color=['crimson', 'burlywood', 'chartreuse'],
                            label=['Crimson', 'Burlywood', 'Chartreuse'])
plt.legend()


plt.figure()
n, bins, patches = plt.hist(x, 10, normed=1, histtype='bar', stacked=True)

plt.figure()
n, bins, patches = plt.hist(x, 10, histtype='step', stacked=True, fill=True)

# Make a multiple-histogram of data-sets with different length.
x0 = mu + sigma*np.random.randn(10000)
x1 = mu + sigma*np.random.randn(7000)
x2 = mu + sigma*np.random.randn(3000)

w0 = np.ones_like(x0)
w0[:len(x0)/2] = 0.5
w1 = np.ones_like(x1)
w1[:len(x1)/2] = 0.5
w2 = np.ones_like(x2)
w2[:len(x2)/2] = 0.5


plt.figure()

n, bins, patches = plt.hist( [x0,x1,x2], 10, weights=[w0, w1, w2], histtype='bar')

plt.show()
