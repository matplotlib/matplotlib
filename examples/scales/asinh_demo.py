"""
============
Asinh Demo
============
"""

import numpy
import matplotlib.pyplot as plt

# Prepare sample values for variations on y=x graph:
x = numpy.linspace(-3, 6, 100)

# Compare "symlog" and "asinh" behaviour on sample y=x graph:
fig1 = plt.figure()
ax0, ax1 = fig1.subplots(1, 2, sharex=True)

ax0.plot(x, x)
ax0.set_yscale('symlog')
ax0.grid()
ax0.set_title('symlog')

ax1.plot(x, x)
ax1.set_yscale('asinh')
ax1.grid()
ax1.set_title(r'$sinh^{-1}$')


# Compare "asinh" graphs with different scale parameter "a0":
fig2 = plt.figure()
axs = fig2.subplots(1, 3, sharex=True)
for ax, a0 in zip(axs, (0.2, 1.0, 5.0)):
    ax.set_title('a0={:.3g}'.format(a0))
    ax.plot(x, x, label='y=x')
    ax.plot(x, 10*x, label='y=10x')
    ax.plot(x, 100*x, label='y=100x')
    ax.set_yscale('asinh', a0=a0)
    ax.grid()
    ax.legend(loc='best', fontsize='small')

plt.show()
