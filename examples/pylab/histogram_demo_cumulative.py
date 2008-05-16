#!/usr/bin/env python
from pylab import *

mu, sigma = 100, 25
x = mu + sigma*randn(10000)

# the histogram of the data
n, bins, patches = hist(x, 50, normed=1, histtype='step', cumulative=True)
setp(patches, 'facecolor', 'g', 'alpha', 0.75)

# add a 'best fit' line
y = normpdf( bins, mu, sigma).cumsum()
y /= y[-1]
l = plot(bins, y, 'k--', linewidth=1.5)

# overlay the first histogram with a second one
# were the data has a smaller standard deviation
mu, sigma = 100, 10
x = mu + sigma*randn(10000)

n, bins, patches = hist(x, bins=bins, normed=1, histtype='step', cumulative=True)
setp(patches, 'facecolor', 'r', 'alpha', 0.25)

# add a 'best fit' line
y = normpdf( bins, mu, sigma).cumsum()
y /= y[-1]
l = plot(bins, y, 'k--', linewidth=1.5)

grid(True)
ylim(0, 1.1)

#savefig('histogram_demo',dpi=72)
show()