"""

Statistical plots
=================

Matplotlib has a few of the most common statistical plots built in, such as
histograms, boxplots, and violin plots. These methods take data, compute
statistics, and then visualize the statistics.  When possible, these functions
will automatically calculate the statistics for you, such as binning and
aggregating the data for histograms, or quartiles and outliers for boxplots.
Statistical computation is usually done by underlying `numpy` methods.

"""

# %%
# hist
# ----
#
# The `~.axes.Axes.hist` method is used to plot a histogram.  The data is
# binned and the frequency of each bin is plotted.  The default number of bins
# is 10, but this can be adjusted with the *bins* keyword.

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = (5, 4)

rng = np.random.default_rng(19680801)
# collect samples from the normal distribution"
data = rng.standard_normal(1000)

fig, axs = plt.subplots(2, 1)
ax = axs[0]
ax.plot(data, '.')
ax.set_title('data')
ax.set_ylabel('data values')
ax.set_xlabel('sample number')
ax = axs[1]
ax.hist(data, bins=30)
ax.set_title('histogram of the data')
ax.set_xlabel('data values')
ax.set_ylabel('frequency')

# %%
# Sometime it is useful to normalize the histogram so that the total area under
# the histogram is 1.  This is known as a probability density function (pdf).
# The *density* argument to `~.axes.Axes.hist` can be set to ``True`` to
# normalize the histogram.  The total area of the histogram will sum to 1.

fig, ax = plt.subplots()
ax.hist(data, bins=30, density=True)
ax.set_title('normalized histogram of the data')
ax.set_xlabel('data values')
ax.set_ylabel('probability density')

# %%
# Other normalizations are possible using the *weights* argument. See
# :ref:`histogram_normalization` for more details.
#
# It is possible to plot multiple histograms in the same figure by passing
# a 2D array to ``hist``.  The columns of the array are the data values for
# each histogram.

# make three sets of data, with each set in a column of the array:
data = rng.standard_normal((1000, 3))

fig, ax = plt.subplots()
ax.hist(data, bins=30, density=True, label=['set1', 'set2', 'set3'])
ax.legend(fontsize='small')
ax.set_title('multiple histograms')
ax.set_xlabel('data values')
ax.set_ylabel('probability density')

# %%
# .. seealso::
#
#  There are many styling and processing options for histograms.  For details,
#  see:
#
#  - :ref:`histogram_normalization`
#  - :ref:`histogram_histtypes`
#  - :ref:`histogram_multihist`


# %%
# hist2d and hexbin
# -----------------
#
# If the data has two coordinates, eg arrays of :math:`(x_i, y_i)` pairs, you
# can use the `~.axes.Axes.hist2d` or `~.axes.Axes.hexbin` methods to visualize
# the frequency that the data occurs in a 2D space.

# make data: correlated + noise
x = rng.normal(size=5000)
y = 1.2 * x + rng.normal(size=5000) / 2

# plot:
fig, axs = plt.subplots(2, 1, figsize=(5, 7), sharex=True, sharey=True)
ax = axs[0]
ax.plot(x, y, '.', alpha=0.5, markeredgecolor='none')
ax.set_title('data')
ax.set_ylabel('y')

ax = axs[1]
bins = (np.arange(-3, 3.1, 0.2), np.arange(-3, 3.1, 0.2))
N, xbins, ybin, pc = ax.hist2d(x, y, bins=bins, vmax=30, cmap='plasma')
ax.contour(xbins[1:], ybin[1:], N.T, levels=4, colors='c', alpha=0.7)
ax.set_ylabel('y')
ax.set_xlabel('x')

ax.set(xlim=(-3, 3), ylim=(-3, 3))
fig.colorbar(pc, ax=ax, label='counts')

# %%
# Note that `~.axes.Axes.hist2d` returns the histogram values ``N`` above, but
# does so transposed from the typical Matplotlib convention.  We have used this
# transposed form to plot the contours of the density over the two-dimensional
# histogram.
#
# The *density* argument to `~.axes.Axes.hist2d` can be set to ``True`` to plot
# a normalized 2D histogram.  Again the normalization is such that the area
# integral of the histogram is one.  Like with `~.axes.Axes.hist`, other
# normalizations are possible using the *weights* argument.

fig, axs = plt.subplots(2, 1, figsize=(5, 7), sharex=True, sharey=True)
ax = axs[0]
ax.plot(x, y, '.', alpha=0.5, markeredgecolor='none')
ax.set_title('data')
ax.set_ylabel('y')

ax = axs[1]
bins = (np.arange(-3, 3.1, 0.2), np.arange(-3, 3.1, 0.2))
N, xbins, ybin, pc = ax.hist2d(x, y, bins=bins, density=True, cmap='plasma')
ax.set_ylabel('y')
ax.set_xlabel('x')

ax.set(xlim=(-3, 3), ylim=(-3, 3))
fig.colorbar(pc, ax=ax, label='Probability density')

# %%
# The `~.axes.Axes.hexbin` method is similar to `~.axes.Axes.hist2d`, but uses
# hexagonal bins instead of rectangular bins.  This can be useful when the data
# is sparse or the distribution is uneven.  The hexagonal bins are arranged in
# a hexagonal lattice, and the density of data in each hexagon is computed.

fig, ax = plt.subplots()

ax.hexbin(x, y, cmap='plasma', gridsize=20)
ax.set_ylabel('y')
ax.set_xlabel('x')
ax.set_title('hexbin plot')

ax.set(xlim=(-3, 3), ylim=(-3, 3))

# %%
# .. seealso::
#
#   - :ref:`hexbin_demo`
#
# boxplot and violinplot
# ----------------------
#
# The `~.axes.Axes.boxplot` and `~.axes.Axes.violinplot` methods are used to
# visualize the distribution of data in discrete bins.  If :math:`X_{ij}` is
# the data set, the statistics are calculated for each column in
# :math:`X_{ij}`, and a box or violin is drawn for each column.
#
# The boxplot usually shows the quartiles of the data distribution in each
# column, and is typically used to see if data sets are statistically distinct.
# A violin plot is similar to a boxplot, but shows the density of the data at
# different values.  The width of the violin is proportional to the density of
# the data at that value, using a smoothed `kernel density estimation
# <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ of the #
# underlying distribution.
#
# Below, we compare the data plotted as dots with a degree of transparency to
# give a feel for the raw data in each column.  A histogram of all three data
# sets is also shown, demonstrating the distribution of the data.  The boxplot
# and violin plot are then shown for the same data.  By default,
# `~.axes.Axes.boxplot` shows the median, quartiles, and outliers, while
# `~.axes.Axes.violinplot` shows a kernel density estimate of the data.

# make three categories of data
data = np.zeros((100, 3))
data[:, 0] = rng.normal(loc=0, scale=1, size=100)
data[:, 1] = rng.normal(loc=1.1, scale=1.5, size=100)
data[:, 2] = rng.normal(loc=-1.5, scale=0.7, size=100)

fig, axs = plt.subplot_mosaic([['plot', 'hist'], ['box', 'violin']], figsize=(5, 5))

ax = axs['plot']
for i in range(3):
    ax.plot(i + 1 + 0 * data[:, i], data[:, i], '.', label=f'col. {i}', alpha=0.2,
            markeredgecolor='none')
ax.set_xlabel('column number')
ax.set_ylabel('data values')
ax.set_title('plot of data')
ax.set_xlim(0.5, 3.5)

ax = axs['hist']
ax.hist(data, bins=np.arange(-4, 4.1, 0.25), density=True,
        label=['column 0', 'column 1', 'column 2'], histtype='step',
        orientation='horizontal')
ax.set_ylabel('data values')
ax.set_xlabel('probability density')
ax.legend(fontsize='small')
ax.set_title('histograms')

ax = axs['box']
ax.boxplot(data)
ax.set_xlabel('column number')
ax.set_ylabel('data values')
ax.set_title('boxplot')

ax = axs['violin']
ax.violinplot(data)
ax.set_xlabel('column number')
ax.set_ylabel('data values')
ax.set_title('violin plot')

# %%
#
# `~.axes.Axes.boxplot` and `~.axes.Axes.violinplot` can be customized in many
# ways.  For example, the *showmeans*, *showmedians*, *showextrema*, and
# *showcaps* arguments can be used to show or hide the mean, median, extrema,
# and caps.  The *vert* argument can be used to make the boxplot horizontal,
# and the *positions* argument can be used to set the positions of the boxes.
# The *widths* argument can be used to set the width of the boxes.
#
# .. seealso::
#
#   - :ref:`boxplot_demo`
#   - :ref:`boxplot_color`
#   - :ref:`boxplot_artists`
#   - :ref:`boxplot_vs_violin`
#   - :ref:`violinplot`
#   - :ref:`customized_violin`
#
# eventplot
# ---------
#
# An `~.axes.Axes.eventplot` draws a vertical line at every data point in a 1D
# array, :math:`x_i`, often to compare even timing.  Usually the events are
# passed into the method as a sequence of  1D arrays because the events being
# timed usually have a different number of occurrences.
#
# Note in the below, we can label the events by making *lineoffsets* a list of
# strings. This will make the y-axis a :ref:`categorical
# <categorical_variables>` axis, with the labels given in the list.

# make some data

len = [100, 75]
dataA = np.cumsum(rng.rayleigh(scale=0.1, size=30))
dataB = np.cumsum(rng.rayleigh(scale=0.1 * 3 / 2, size=20))
data = [dataA, dataB]

fig, ax = plt.subplots()
ax.eventplot(data, orientation='horizontal', linelengths=0.9,
             color=['C0', 'C1'], lineoffsets=['data A', 'data B'])

ax.set_xlabel('time')
ax.set_ylabel('event type')

# %%
# .. seealso::
#  - :ref:`eventplot_demo`
#
# .. admonition:: References
#
#   The use of the following functions, methods, classes and modules is shown
#   in this example:
#
#   - `matplotlib.axes.Axes.hist` / `matplotlib.pyplot.hist`
#   - `matplotlib.axes.Axes.hist2d` / `matplotlib.pyplot.hist2d`
#   - `matplotlib.axes.Axes.hexbin` / `matplotlib.pyplot.hexbin`
#   - `matplotlib.axes.Axes.boxplot` / `matplotlib.pyplot.boxplot`
#   - `matplotlib.axes.Axes.violinplot` / `matplotlib.pyplot.violinplot`
#   - `matplotlib.axes.Axes.eventplot` / `matplotlib.pyplot.eventplot`
#   - `matplotlib.axes.Axes.plot` / `matplotlib.pyplot.plot`
