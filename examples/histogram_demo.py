from matplotlib.matlab import *

mu, sigma = 100, 15
x = mu + sigma*randn(10000)

# the histogram of the data
n, bins, patches = hist(x, 50, normed=1)

# add a 'best fit' line
y = normpdf( bins, mu, sigma)
l = plot(bins, y, 'r--')
set(l, 'linewidth', 1)

xlabel('Smarts')
ylabel('Probability')
#title(r'$\rm{Histogram of IQ: }\mu=100, \sigma=15$')
title('Histogram of IQ: mu=100, sigma=15')
#savefig('histogram_demo',dpi=72)
show()
