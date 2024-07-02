"""
=======================
Histogram normalization
=======================

Histogram normalization rescales data into probabilities and therefore is a popular
technique for comparing populations of different sizes or histograms computed using
different bin edges. For more information on using `.Axes.hist` see
:ref:`histogram_features`.

Irregularly spaced bins
-----------------------
In this example, the bins below ``x=-1.25`` are six times wider than the rest of the
bins ::

  dx = 0.1
  xbins = np.hstack([np.arange(-4, -1.25, 6*dx), np.arange(-1.25, 4, dx)])

By normalizing by density, we preserve the shape of the distribution, whereas if we do
not, then the wider bins have much higher counts than the thinner bins:
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(19680801)

xdata = rng.normal(size=1000)
xpdf = np.arange(-4, 4, 0.1)
pdf = 1 / (np.sqrt(2 * np.pi)) * np.exp(-xpdf**2 / 2)

dx = 0.1
xbins = np.hstack([np.arange(-4, -1.25, 6*dx), np.arange(-1.25, 4, dx)])

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained')

fig.suptitle("Histogram with irregularly spaced bins")


ax['False'].hist(xdata, bins=xbins, density=False, histtype='step', label='Counts')
ax['False'].plot(xpdf, pdf * len(xdata) * dx, label=r'$N\,f_X(x)\,\delta x_0$',
                 alpha=.5)

ax['True'].hist(xdata, bins=xbins, density=True, histtype='step', label='density')
ax['True'].plot(xpdf, pdf, label='$f_X(x)$', alpha=.5)


ax['False'].set(xlabel='x [V]', ylabel='Count per bin', title="density=False")

# add the bin widths on the minor axes to highlight irregularity
ax['False'].set_xticks(xbins, minor=True)
ax['False'].legend()

ax['True'].set(xlabel='x [$V$]', ylabel='Probability density [$V^{-1}$]',
               title="density=True")
ax['False'].set_xticks(xbins, minor=True)
ax['True'].legend()


# %%
# Different bin widths
# --------------------
#
# Here we use normalization to compare histograms with binwidths of 0.1, 0.4, and 1.2:

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained')

fig.suptitle("Comparing histograms with different bin widths")
# expected PDF
ax['True'].plot(xpdf, pdf, '--', label='$f_X(x)$', color='k')

for nn, dx in enumerate([0.1, 0.4, 1.2]):
    xbins = np.arange(-4, 4, dx)
    # expected histogram:
    ax['False'].plot(xpdf, pdf*1000*dx, '--', color=f'C{nn}', alpha=.5)
    ax['False'].hist(xdata, bins=xbins, density=False, histtype='step', label=dx)

    ax['True'].hist(xdata, bins=xbins, density=True, histtype='step')

ax['False'].set(xlabel='x [$V$]', ylabel='Count per bin',
                title="density=False")
ax['True'].set(xlabel='x [$V$]', ylabel='Probability density [$V^{-1}$]',
               title='density=True')
ax['False'].legend(fontsize='small', title='bin width:')
# %%
# Populations of different sizes
# ------------------------------
#
# Here we compare the distribution of ``xdata`` with a population of 1000, and
# ``xdata2`` with 100 members. We demonstrate using *density* to generate the
# probability density function(`pdf`_) and *weight* to generate an analog to the
# probability mass function (`pmf`_).
#
# .. _pdf: https://en.wikipedia.org/wiki/Probability_density_function
# .. _pmf: https://en.wikipedia.org/wiki/Probability_mass_function

xdata2 = rng.normal(size=100)

fig, ax = plt.subplot_mosaic([['no_norm', 'density', 'weight']], layout='constrained')

fig.suptitle("Comparing histograms of populations of different sizes")

xbins = np.arange(-4, 4, 0.25)

for xd in [xdata, xdata2]:
    ax['no_norm'].hist(xd, bins=xbins, histtype='step')
    ax['density'].hist(xd, bins=xbins, histtype='step', density=True)
    ax['weight'].hist(xd, bins=xbins, histtype='step', weights=np.ones(len(xd))/len(xd),
                      label=f'N={len(xd)}')


ax['no_norm'].set(xlabel='x [$V$]', ylabel='Counts', title='No normalization')
ax['density'].set(xlabel='x [$V$]',
                  ylabel='Probability density [$V^{-1}$]', title='Density=True')
ax['weight'].set(xlabel='x bins [$V$]', ylabel='Counts / N', title='Weight = 1/N')

ax['weight'].legend(fontsize='small')

plt.show()

# %%
#
# .. tags:: plot-type: histogram
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
#    - `matplotlib.axes.Axes.set`
#    - `matplotlib.axes.Axes.legend`
#
