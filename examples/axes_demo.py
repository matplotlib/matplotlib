# Demonstrate arbitrary placement of axes
from matplotlib.matlab import *

dt = 0.001
t = arange(0.0, 10.0, dt)
r = exp(-t[:1000]/0.05)      # impulse response
x = randn(len(t))
s = convolve(x,r,mode=2)*dt  # colored noise
s = s[:len(x)]               # remove the decay tail which will skew
                             # the probability distribution

# I'm just using the if 1 thing to break up the different regions of
# the code visually.

# plot the noise
if 1:
    plot(t, s)
    axis([0, 1, 1.1*min(s), 2*max(s) ])
    xlabel('time (s)')
    ylabel('current (nA)')
    title('Gaussian white noise convolved with an exponential function')

# Make a histogram probability density inset
if 1:
    a = axes([.65, .6, .2, .2], axisbg='y')
    n, bins, patches = hist(s, 400, normed=1)
    title('Probability')
    set(a, 'xticks', [])
    set(a, 'yticks', [])

# Make a histogram probability density inset
if 1:
    a = axes([0.2, 0.6, .2, .2], axisbg='y')
    plot(t[:len(r)], r)
    title('Impulse response')
    set(a, 'xlim', [0,.2])
    set(a, 'xticks', [])
    set(a, 'yticks', [])

show()
