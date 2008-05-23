#!/usr/bin/env python
import pylab as P

mu, sigma = 100, 15
x = mu + sigma*P.randn(10000)

# the histogram of the data
n, bins, patches = P.hist(x, 50, normed=1)
P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)

# add a 'best fit' line
y = P.normpdf( bins, mu, sigma)
l = P.plot(bins, y, 'r--')
P.setp(l, 'linewidth', 1)

P.xlabel('Smarts')
P.ylabel('Probability')
P.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
P.axis([40, 160, 0, 0.03])
P.grid(True)

#P.savefig('histogram_demo',dpi=72)
P.show()
