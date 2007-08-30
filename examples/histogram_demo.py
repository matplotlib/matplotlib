#!/usr/bin/env python
from pylab import *

mu, sigma = 100, 15
x = mu + sigma*randn(10000)

# the histogram of the data
n, bins, patches = hist(x, 50, normed=1)
setp(patches, 'facecolor', 'g', 'alpha', 0.75)

# add a 'best fit' line
y = normpdf( bins, mu, sigma)
l = plot(bins, y, 'r--')
setp(l, 'linewidth', 1)

xlabel('Smarts')
ylabel('Probability')
title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
axis([40, 160, 0, 0.03])
grid(True)

#savefig('histogram_demo',dpi=72)
show()
