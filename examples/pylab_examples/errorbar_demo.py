#!/usr/bin/env python
from pylab import *

t = arange(0.1, 4, 0.1)
s = exp(-t)
e = 0.1*abs(randn(len(s)))
f = 0.1*abs(randn(len(s)))
g = 2*e
h = 2*f

figure()
errorbar(t, s, e, fmt='o')             # vertical symmetric

figure()
errorbar(t, s, None, f, fmt='o')       # horizontal symmetric

figure()
errorbar(t, s, e, f, fmt='o')          # both symmetric

figure()
errorbar(t, s, [e,g], [f,h], fmt='--o')  # both asymmetric

figure()
errorbar(t, s, [e,g], f, fmt='o', ecolor='g')      # both mixed

figure()
errorbar(t, s, e, [f,h], fmt='o')      # both mixed

figure()
errorbar(t, s, [e,g], fmt='o')         # vertical asymmetric

figure()
errorbar(t, s, yerr=e, fmt='o')        # named

figure()
errorbar(t, s, xerr=f, fmt='o')        # named
xlabel('Distance (m)')
ylabel('Height (m)')
title('Mean and standard error as a function of distance')

figure()
ax = subplot(111)
ax.set_yscale('log')
errorbar(t, s+2, e, f, fmt='o')          # both symmetric

#savefig('errorbar_demo')
show()
