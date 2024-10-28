"""
.. redirect-from:: /gallery/statistics/histogram_features

.. _histogram_features:

=========
Histogram
=========

Histograms are used to visualize the distribution of a dataset across a set of fixed
bins. Histograms are created by defining the bin edges, sorting the values in a dataset
into the bins, and then counting or summing how much data is in each bin. The
`.Axes.hist` method provides the following parameters for defining bins and counting:

:bins: choose the number of bins or pass a list of bin edges
:density: normalize the histogram so that its integral is one,
:weights: assign how much each value contributes to its bin count

The Matplotlib ``hist`` method forwards the data and the *bins*, *range*, *density* and
*weights* parameters to `numpy.histogram` and plots the results; therefore, users should
consult the numpy documentation for a definitive guide.

In this example, 9 numbers between 1 and 4 are sorted into 3 bins:
"""

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(19680801)

xdata = np.array([1.2, 2.3, 3.3, 3.1, 1.7, 3.4, 2.1, 1.25, 1.3])
xbins = np.array([1, 2, 3, 4])

# changing the style of the histogram bars just to make it
# very clear where the boundaries of the bins are:
style = {'facecolor': 'none', 'edgecolor': 'C0', 'linewidth': 3}
fw, fh = (6, 3)

fig, ax = plt.subplots(layout='constrained', figsize=(fw, fh))

# count the number of values in xdata between each value in xbins
ax.hist(xdata, bins=xbins, **style)

# plot the xdata events
ax.eventplot(xdata, color='C1', alpha=.5)

# add labels and set the x tick marks to bin edges
ax.set(xlabel='x bins (dx=1.0)', xticks=xbins,
       ylabel='count per bin', title='Histogram')

# %%
# bin widths
# ==========
#
# Changing the bin size changes the shape of this sparse histogram, so it is a
# good idea to choose bins with some care with respect to your data. The `.Axes.hist`
# *bins* parameter accepts either the number of bins or a list of bin edges.
#
#
# Fixed bin edges
# ---------------
#
# Here the bins are set to the list of edges [1, 1.5, 2, 2.5, 3, 3.5, 4].
#
# .. code-block:: python
#
#    xbins = np.arange(1, 4.5, 0.5)
#    ax.hist(xdata, bins=xbins, **style)
#
# As illustrated, this generates a histogram where the bins are half as wide as the
# previous example:

xbins = np.arange(1, 4.5, 0.5)

fig, ax = plt.subplots(layout='constrained', figsize=(fw, fh))

ax.hist(xdata, bins=xbins, **style)

ax.eventplot(xdata, color='C1', alpha=.5)

ax.set(xlabel='x bins (dx=0.5)', xticks=xbins, ylabel='count',
       title='Histogram with fixed bin edges')

# %%
#
# Number of bins
# --------------
#
# We can also let numpy (via Matplotlib) choose the bins automatically:
#
# .. code-block:: python
#
#    ax['auto'].hist(xdata)
#
# or specify the number of bins, the edges of which are then chosen automatically:
#
# .. code-block:: python
#
#    ax['n4'].hist(xdata, bins=4)
#
# In the following example, we show both methods of choosing the bins:

fig, ax = plt.subplot_mosaic([['auto'], ['n4']],
                             sharey=True, layout='constrained', figsize=(fw, fh))
fig.suptitle("Dynamically computed bins")


_, bin1, _ = ax['auto'].hist(xdata, **style)
_, bin2, _ = ax['n4'].hist(xdata, bins=4, **style)


ax['auto'].eventplot(xdata, color='C1', alpha=.5)
ax['n4'].eventplot(xdata, color='C1', alpha=.5)

ax['auto'].set(xlabel='x bins', xticks=bin1, ylabel='count',
               title='bins="auto"')
ax['n4'].set(xlabel='x bins', xticks=bin2, ylabel='count',
             title='bins=4')
# %%
# The `.Axes.hist` method returns the data being binned, the bin edges, and the list
# of artists used to create the histogram. Here we use the returned bins to generate the
# xticks.
#
# Computing bin contents
# ======================
#
# Counts-per-bin is the default length of each bar in the histogram. However, the
# *density* and *weights* parameters can be used to change how the contents of each bin
# are counted, and therefore *density* and *weights* affect the length of each bar.
#
# density
# -------
#
# We can normalize the bar lengths as a probability density function using
# the ``density`` parameter.  The value (count) attached to each bar is divided by the
# total number of data points *and* the width of the bin, and thus the values
# *integrate* to one when integrating across the full range of data.
#
# e.g. ::
#
#     density = counts / (sum(counts) * np.diff(bins))
#     np.sum(density * np.diff(bins)) == 1
#
# This normalization is how `probability density functions
# <https://en.wikipedia.org/wiki/Probability_density_function>`_ are defined in
# statistics.
#
#
# To normalize the bar length, we set *density* to True:
#
# .. code-block:: python
#
#    ax['True'].hist(xdata, bins=xbins, density=True, **style)
#
# As shown, setting the density kwarg only changes the count:

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained',
                             figsize=(fw, fh))
fig.suptitle('Histogram normalization using density')


ax['False'].hist(xdata, bins=xbins, **style)
# Normalize the histogram
ax['True'].hist(xdata, bins=xbins, density=True, **style)


ax['False'].set(xlabel='x bins (dx=0.5)', xticks=xbins,
                ylabel='count per bin', title='density=False')

ax['True'].set(xlabel='x bins (dx=0.5 $V$)', xticks=xbins,
               ylabel='Probability density [$V^{-1}$])', title='density=True')


# The usefulness of this normalization is a little more clear when we draw from
# a known distribution and try to compare with theory.  So, choose 1000 points
# from a `normal distribution
# <https://en.wikipedia.org/wiki/Normal_distribution>`_, and also calculate the
# known probability density function:

xdata = rng.normal(size=1000)
xpdf = np.arange(-4, 4, 0.1)
pdf = 1 / (np.sqrt(2 * np.pi)) * np.exp(-xpdf**2 / 2)

# %%
# If we don't use ``density=True``, we need to scale the expected probability
# distribution function by both the length of the data and the width of the
# bins to fit it to the histogram:
#
# .. code-block:: python
#
#    ax['False'].hist(xdata, bins=xbins, density=False, histtype='step', label='counts')
#    ax['False'].plot(xpdf, pdf * len(xdata) * dx,
#                           label=r'$N\,f_X(x)\,\delta x$', alpha=.5)
#
# while we do not need to scale the pdf to fit it to the normalized histogram
#
# .. code-block:: python
#
#    ax['True'].hist(xdata, bins=xbins, density=True, histtype='step', label='density')
#    ax['True'].plot(xpdf, pdf, label='$f_X(x)$', alpha=.5)
#
# The effect of this scaling is that the two plots look identical, but are on different
# y-axis scales:

dx = 0.1
xbins = np.arange(-4, 4, dx)

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained',
                             figsize=(fw, fh))
fig.suptitle("Histogram normalization using scaling")

ax['False'].hist(xdata, bins=xbins, density=False, histtype='step', label='counts')
# scale and plot the expected pdf:
ax['False'].plot(xpdf, pdf * len(xdata) * dx, label=r'$N\,f_X(x)\,\delta x$', alpha=.5)


ax['True'].hist(xdata, bins=xbins, density=True, histtype='step', label='density')
ax['True'].plot(xpdf, pdf, label='$f_X(x)$', alpha=.5)


ax['False'].set(xlabel='x [$V$]', ylabel='Count per bin', title="density=False")
ax['False'].legend()
ax['True'].set(xlabel='x [$V$]', ylabel='Probability density [$V^{-1}$]',
               title="density=True")
ax['True'].legend()

# %%
# weights
# -------
#
# We can change how much each data value contributes to its bin computation by passing
# in a list of weights, one per value, to the *weights* parameter. Here we set the
# weights to only count values less than 2
#
# .. code-block:: Python
#
#    xdata = np.array([1.2, 2.3, 3.3, 3.1, 1.7, 3.4, 2.1, 1.25, 1.3])
#    weights =  (xdata<2).astype(int)

xdata = np.array([1.2, 2.3, 3.3, 3.1, 1.7, 3.4, 2.1, 1.25, 1.3])
xbins = np.arange(1, 4.5, 0.5)

fig, ax = plt.subplot_mosaic([['False', 'True']], layout='constrained',
                             figsize=(fw, fh))
fig.suptitle('Compute histograms using weights')


ax['False'].hist(xdata, bins=xbins, **style)
# Normalize the histogram
ax['True'].hist(xdata, bins=xbins, weights=(xdata < 2).astype(int), **style)


ax['False'].set(xlabel='x bins (dx=0.5)', xticks=xbins,
                ylabel='count per bin', title="unweighted")

ax['True'].set(xlabel='x bins (dx=0.5 $V$)', xticks=xbins,
               ylabel='weighted count per bin', title='weights=int(x)')

# %%
# Sometimes people want to normalize so that the sum of counts is one.  This is
# analogous to a `probability mass function
# <https://en.wikipedia.org/wiki/Probability_mass_function>`_ for a discrete
# variable where the sum of probabilities for all the values equals one.
#
# Using ``hist``, we can get this normalization if we set the *weights* to 1/N.
#
# .. code-block:: python
#
#    ax.hist(xdata, bins=xbins, weights=np.ones(len(xdata))/len(xdata),
#            histtype='step', label=f'{dx}')
#
# Note that the amplitude of this normalized histogram still depends on
# width and/or number of bins:

fig, ax = plt.subplots(layout='constrained', figsize=(fw, fh))

for nn, dx in enumerate([0.1, 0.4, 1.2]):
    xbins = np.arange(-4, 4, dx)
    ax.hist(xdata, bins=xbins, weights=np.ones(len(xdata))/len(xdata),
                   histtype='step', label=f'{dx}')

ax.set(xlabel='x [$V$]', ylabel='Bin count / N',
       title="Histogram normalization using weights")
ax.legend(fontsize='small', title='bin width:')

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
