from matplotlib.matlab import *

t = arange(0.1, 4, 0.1)
s = exp(-t)
e = 0.1*abs(randn(len(s)))
f = 0.1*abs(randn(len(s)))
g = 2*e
h = 2*f

figure(1)  
#errorbar(t, s, e, fmt='o')             # vertical symmetric
#errorbar(t, s, None, f, fmt='o')       # horizontal symmetric
errorbar(t, s, e, f, fmt='o')          # both symmetric
#errorbar(t, s, [e,g], [f,h], fmt='o')  # both asymmetric
#errorbar(t, s, [e,g], f, fmt='o')      # both mixed
#errorbar(t, s, e, [f,h], fmt='o')      # both mixed
#errorbar(t, s, [e,g], fmt='o')         # vertical asymmetric
#errorbar(t, s, yerr=e, fmt='o')        # named
#errorbar(t, s, xerr=f, fmt='o')        # named
xlabel('Distance (m)')
ylabel('Height (m)')
title('Mean and standard error as a function of distance')


#savefig('errorbar_demo')
show()
