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


# Compare "asinh" graphs with different scale parameter "linear_width":
fig2 = plt.figure()
axs = fig2.subplots(1, 3, sharex=True)
for ax, a0 in zip(axs, (0.2, 1.0, 5.0)):
    ax.set_title('linear_width={:.3g}'.format(a0))
    ax.plot(x, x, label='y=x')
    ax.plot(x, 10*x, label='y=10x')
    ax.plot(x, 100*x, label='y=100x')
    ax.set_yscale('asinh', linear_width=a0)
    ax.grid()
    ax.legend(loc='best', fontsize='small')


# Compare "symlog" and "asinh" scalings
# on 2D Cauchy-distributed random numbers:
fig3 = plt.figure()
ax = fig3.subplots(1, 1)
r = numpy.tan(numpy.random.uniform(-numpy.pi / 2.02, numpy.pi / 2.02,
                                   size=(5000,)))
th = numpy.random.uniform(0, 2*numpy.pi, size=r.shape)

ax.scatter(r * numpy.cos(th), r * numpy.sin(th), s=4, alpha=0.5)
ax.set_xscale('asinh')
ax.set_yscale('symlog')
ax.set_xlabel('asinh')
ax.set_ylabel('symlog')
ax.set_title('2D Cauchy random deviates')
ax.grid()


plt.show()
