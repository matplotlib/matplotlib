#!/usr/bin/env python
from pylab import *

mu, sigma = 100, 15
x = mu + sigma*randn(10000)

# the histogram of the data
n, bins, patches = hist(x, 50, normed=1, histtype='step')
setp(patches, 'facecolor', 'g', 'alpha', 0.75)

# add a 'best fit' line
y = normpdf( bins, mu, sigma)
l = plot(bins, y, 'k--', linewidth=1)


# overlay the first histogram with a second one
# were the data has a smaller standard deviation
mu, sigma = 100, 5
x = mu + sigma*randn(10000)

n, bins, patches = hist(x, 50, normed=1, histtype='step')
setp(patches, 'facecolor', 'r', 'alpha', 0.25)

y = normpdf( bins, mu, sigma)
l = plot(bins, y, 'k--', linewidth=1)

axis([40, 160, 0, 0.09])
grid(True)

#savefig('histogram_demo',dpi=72)
show()
